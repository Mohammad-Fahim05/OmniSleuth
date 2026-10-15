"""Unit tests for Phase C1 — OpenRouter provider, configuration, and retry semantics.

All tests are deterministic and use mocks: no live API calls, no API keys, and no
network access.
"""

from unittest.mock import MagicMock, patch

import pytest

from omnisleuth.core.config import get_settings, reset_settings
from omnisleuth.core.exceptions import ConfigurationError
from omnisleuth.llm import (
    LLMGateway,
    LLMRequest,
    LLMResponse,
    OpenRouterProvider,
    SemanticCache,
    UsageMetrics,
)
from omnisleuth.llm.errors import (
    LLMErrorCategory,
    LLMProviderError,
    classify_llm_error,
    redact_literals,
)
from omnisleuth.llm.retry import execute_with_retry, is_transient_error

# Test fixture only, assembled at runtime so secret scans never flag a
# key-shaped literal in the repository. It is not a credential.
TEST_API_KEY = "sk-" + "or-v1-" + "TESTFIXTURENOTAREALKEY"


@pytest.fixture
def openrouter_env(monkeypatch):
    """Configure OpenRouter environment settings and reset the settings singleton."""
    monkeypatch.setenv("OPENROUTER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    reset_settings()
    yield get_settings()
    reset_settings()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Provider configuration and base URL
# ─────────────────────────────────────────────────────────────────────────────

def test_openrouter_default_base_url(monkeypatch):
    """OpenRouter defaults to the documented OpenAI-compatible base URL."""
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    reset_settings()
    provider = OpenRouterProvider(api_key=TEST_API_KEY)

    assert provider.effective_base_url == "https://openrouter.ai/api/v1"
    assert provider.api_key == TEST_API_KEY


def test_openrouter_base_url_from_settings(openrouter_env):
    """Base URL and model are read from configuration, not hard-coded."""
    provider = OpenRouterProvider()

    assert provider.effective_base_url == "https://openrouter.ai/api/v1"
    assert provider.model == "meta-llama/llama-3.1-8b-instruct"


def test_openrouter_base_url_trailing_slash_normalized(monkeypatch):
    """A trailing slash is normalized away so the SDK joins paths correctly."""
    reset_settings()
    provider = OpenRouterProvider(
        api_key=TEST_API_KEY, base_url="https://openrouter.ai/api/v1/"
    )
    assert provider.effective_base_url == "https://openrouter.ai/api/v1"


def test_openrouter_model_override_via_request(openrouter_env):
    """A per-request model override wins over the configured default."""
    provider = OpenRouterProvider()

    with patch("openai.OpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"intent": "query"}'))],
            usage=None,
        )

        req = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="anthropic/claude-3.5-sonnet",
            temperature=0.2,
            max_tokens=64,
            timeout_seconds=5.0,
        )
        res = provider.generate(req, request_id="llm_req_test")

    # Base URL is passed to the OpenAI-compatible client.
    assert mock_client_cls.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    # Per-request overrides are forwarded untouched.
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "anthropic/claude-3.5-sonnet"
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 64
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]

    assert res.provider == "openrouter"
    assert res.model == "anthropic/claude-3.5-sonnet"
    assert res.fallback_used is False
    assert res.error is None
    assert res.usage.total_tokens is None  # no usage metadata invented
    assert res.latency_ms >= 0.0
    assert res.request_id == "llm_req_test"


def test_openrouter_usage_metrics_captured(openrouter_env):
    """Usage and latency metrics are reported using the existing models."""
    provider = OpenRouterProvider()

    with patch("openai.OpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="plan text"))],
            usage=MagicMock(prompt_tokens=120, completion_tokens=30, total_tokens=150),
        )
        res = provider.generate(
            LLMRequest(messages=[{"role": "user", "content": "hi"}]), request_id="llm_req_x"
        )

    assert res.usage.input_tokens == 120
    assert res.usage.output_tokens == 30
    assert res.usage.total_tokens == 150
    assert res.usage.is_actual is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. Environment configuration and model selection
# ─────────────────────────────────────────────────────────────────────────────

def test_openrouter_env_configuration_and_model_selection(openrouter_env):
    """OPENROUTER_* environment variables drive provider selection."""
    settings = openrouter_env
    assert settings.openrouter_api_key == TEST_API_KEY
    assert settings.openrouter_model == "meta-llama/llama-3.1-8b-instruct"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"

    gateway = LLMGateway()
    assert gateway.is_configured is True
    provider = gateway._get_provider("openrouter")
    assert isinstance(provider, OpenRouterProvider)
    assert provider.model == "meta-llama/llama-3.1-8b-instruct"


def test_openrouter_uses_configured_model_in_gateway(openrouter_env):
    """The configured OpenRouter model is used when the request omits one."""
    gateway = LLMGateway()
    provider = gateway._get_provider("openrouter")
    assert provider.model == "meta-llama/llama-3.1-8b-instruct"


def test_openrouter_gateway_falls_back_without_credentials(monkeypatch):
    """Without OPENROUTER_API_KEY the gateway stays offline instead of erroring."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    reset_settings()

    gateway = LLMGateway()
    assert gateway.is_configured is False

    res = gateway.generate(LLMRequest(prompt="Total revenue by category"))
    assert res.fallback_used is True
    assert res.provider == "fallback"
    assert gateway.is_fallback_response(res) is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Missing API key: no network request, no crash
# ─────────────────────────────────────────────────────────────────────────────

def test_openrouter_missing_key_makes_no_network_request(monkeypatch):
    """A missing API key raises a permanent configuration error before any I/O."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reset_settings()
    provider = OpenRouterProvider()

    with patch("openai.OpenAI") as mock_client_cls:
        with pytest.raises(ConfigurationError):
            provider.generate(
                LLMRequest(messages=[{"role": "user", "content": "hi"}]),
                request_id="llm_req_test",
            )

    # No HTTP client was ever constructed.
    mock_client_cls.assert_not_called()


def test_openrouter_missing_key_is_not_retryable(monkeypatch):
    """A missing key is permanent, so retries abort immediately."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reset_settings()

    provider = OpenRouterProvider()
    calls = {"n": 0}

    def _call():
        calls["n"] += 1
        return provider.generate(
            LLMRequest(messages=[{"role": "user", "content": "hi"}]), request_id="r"
        )

    with pytest.raises(ConfigurationError):
        execute_with_retry(_call, max_retries=3, base_delay=0.001)

    assert calls["n"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Retry classification: permanent vs transient
# ─────────────────────────────────────────────────────────────────────────────

def test_authentication_errors_are_not_retried():
    """401/403/invalid key failures are permanent."""
    assert is_transient_error(Exception("401 Unauthorized invalid_api_key")) is False
    assert is_transient_error(Exception("403 Forbidden")) is False
    assert classify_llm_error(Exception("401 Unauthorized")).category == LLMErrorCategory.AUTHENTICATION


def test_invalid_request_errors_are_not_retried():
    """400-level invalid request failures are permanent."""
    assert is_transient_error(Exception("400 Bad Request invalid model")) is False
    assert classify_llm_error(Exception("422 Unprocessable Entity")).category == LLMErrorCategory.INVALID_REQUEST


def test_transient_failures_are_retried_with_existing_bounds():
    """Rate limits, server errors, timeouts, and network errors are retried."""
    assert is_transient_error(Exception("429 Rate limit exceeded")) is True
    assert is_transient_error(Exception("503 Service Unavailable")) is True
    assert is_transient_error(Exception("Connection timed out")) is True
    assert classify_llm_error(Exception("429")).category == LLMErrorCategory.RATE_LIMIT
    assert classify_llm_error(Exception("503")).category == LLMErrorCategory.SERVER_ERROR
    assert classify_llm_error(Exception("connection reset by peer")).category == LLMErrorCategory.NETWORK_ERROR
    assert classify_llm_error(Exception("read timed out")).category == LLMErrorCategory.TIMEOUT


def test_transient_retry_respects_max_retries_bound():
    """The existing bounded backoff and retry limits are preserved."""
    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        raise Exception("429 Rate limit exceeded")

    with pytest.raises(Exception):
        execute_with_retry(_flaky, max_retries=3, base_delay=0.001)

    assert calls["n"] == 4  # initial attempt + 3 retries


def test_permanent_error_aborts_after_single_attempt():
    """Permanent failures abort immediately without sleeping."""
    calls = {"n": 0}

    def _perm():
        calls["n"] += 1
        raise Exception("401 Unauthorized")

    with pytest.raises(Exception, match="401 Unauthorized"):
        execute_with_retry(_perm, max_retries=3, base_delay=0.001)

    assert calls["n"] == 1


def test_transient_then_success_returns_retry_count():
    """Transient failure then success reports the retry count."""
    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise Exception("502 Bad Gateway")
        return "ok"

    result, retries = execute_with_retry(_flaky, max_retries=3, base_delay=0.001)
    assert result == "ok"
    assert retries == 1
    assert calls["n"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 5. Secret redaction
# ─────────────────────────────────────────────────────────────────────────────

def test_error_messages_do_not_leak_api_keys():
    """API keys and Authorization headers are redacted from error categories."""
    msg = (
        f"Request failed: Authorization: Bearer {TEST_API_KEY} "
        f"(api_key={TEST_API_KEY}, model=unknown)"
    )

    # Pattern-based guardrail redaction (covers OpenRouter-shaped keys) ...
    assert TEST_API_KEY not in redact_literals(msg)
    # ... and explicit literal redaction of the configured key value.
    info = classify_llm_error(Exception(msg), secrets=[TEST_API_KEY])
    assert TEST_API_KEY not in info.message
    assert "[REDACTED_SECRET]" in info.message


def test_provider_error_sanitizes_message_before_raising():
    """A provider failure is re-raised with a sanitized message."""
    secret = TEST_API_KEY
    exc = LLMProviderError(
        message=redact_literals(f"401 error with key {secret}", secrets=[secret]),
        category=LLMErrorCategory.AUTHENTICATION,
    )
    assert secret not in str(exc)
    assert exc.category == LLMErrorCategory.AUTHENTICATION


def test_openrouter_provider_error_does_not_leak_key(openrouter_env):
    """A live-style provider failure never echoes the API key in its message."""
    secret = TEST_API_KEY
    provider = OpenRouterProvider()

    with patch("openai.OpenAI") as mock_client_cls:
        mock_client_cls.return_value.chat.completions.create.side_effect = Exception(
            f"401 Unauthorized (api_key={secret})"
        )
        with pytest.raises(LLMProviderError) as excinfo:
            provider.generate(
                LLMRequest(messages=[{"role": "user", "content": "hi"}]),
                request_id="llm_req_test",
            )

    assert secret not in str(excinfo.value)
    assert excinfo.value.category == LLMErrorCategory.AUTHENTICATION
    assert "401" in str(excinfo.value)


def test_gateway_fallback_error_message_is_sanitized(openrouter_env):
    """Gateway fallback errors are sanitized before being surfaced to callers."""
    secret = TEST_API_KEY

    class LeakyProvider:
        def generate(self, request, request_id):
            raise Exception(f"Connection error while sending Authorization: Bearer {secret}")

    gateway = LLMGateway(provider=LeakyProvider())
    with patch.object(gateway.settings, "llm_max_retries", 1), \
         patch.object(gateway.settings, "llm_retry_base_delay", 0.001):
        res = gateway.generate(LLMRequest(prompt="hi", dataset_version="ds1"))

    assert res.fallback_used is True
    assert secret not in (res.error or "")
    assert secret not in str(gateway.list_traces()[-1].error or "")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Existing provider behaviour preserved
# ─────────────────────────────────────────────────────────────────────────────

def test_openai_and_azure_providers_still_resolve(openrouter_env, monkeypatch):
    """Switching provider keeps routing OpenAI/Azure through the existing classes."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdefghij")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-test-0123456789abcdefghij")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    reset_settings()

    gateway = LLMGateway()
    from omnisleuth.llm.providers import AzureOpenAIProvider, OpenAIProvider

    assert isinstance(gateway._get_provider("openai"), OpenAIProvider)
    assert isinstance(gateway._get_provider("azure"), AzureOpenAIProvider)
    assert isinstance(gateway._get_provider("openrouter"), OpenRouterProvider)
    # Unknown provider name stays offline rather than raising.
    assert gateway._get_provider("does-not-exist").__class__.__name__ == "DeterministicFallbackProvider"


def test_openai_missing_key_is_offline_safe(monkeypatch):
    """The existing OpenAI provider also fails closed without a key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_settings()

    from omnisleuth.llm.providers import OpenAIProvider

    with patch("openai.OpenAI") as mock_client_cls:
        with pytest.raises(ConfigurationError):
            OpenAIProvider().generate(
                LLMRequest(messages=[{"role": "user", "content": "hi"}]), request_id="r"
            )
    mock_client_cls.assert_not_called()


def test_cache_key_includes_provider_identity():
    """Two providers serving the same model name cannot share cached responses."""
    cache = SemanticCache(enabled=True)

    class RecordingProvider:
        def __init__(self):
            self.calls = 0

        def generate(self, request, request_id):
            self.calls += 1
            return LLMResponse(
                request_id=request_id,
                text="plan",
                provider=request.provider or "custom",
                model=request.model or "default",
                latency_ms=1.0,
                usage=UsageMetrics(input_tokens=1, output_tokens=1, total_tokens=2, is_actual=True),
            )

    provider = RecordingProvider()
    gateway = LLMGateway(provider=provider, cache=cache)

    req_a = LLMRequest(prompt="same question", provider="openai", model="gpt-4o-mini", dataset_version="ds1")
    req_b = LLMRequest(prompt="same question", provider="openrouter", model="gpt-4o-mini", dataset_version="ds1")

    gateway.generate(req_a)
    gateway.generate(req_b)

    assert provider.calls == 2  # provider identity is part of the cache key
