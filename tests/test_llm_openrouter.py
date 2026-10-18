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


def test_audit_live_smoke_grounded_currency_and_percentages_only():
    """Test (T-003F Task 6): audit/live_smoke.py extracts only currency and percentage values."""
    from omnisleuth.evidence.grounding import extract_currency_and_percentages

    text = (
        "In 'category', 'Electronics' ranked #1 and contributed the largest share to total revenue "
        "with 45.2% ($12,345.67 of $27,313.44 total) across 50 orders with -3.5% discount."
    )
    numbers = extract_currency_and_percentages(text)

    # Must contain currency and percentage values
    assert 12345.67 in numbers
    assert 27313.44 in numbers
    assert 45.2 in numbers
    assert -3.5 in numbers

    # Must NOT contain ranks (1) or counts (50)
    assert 1.0 not in numbers
    assert 50.0 not in numbers


# ─────────────────────────────────────────────────────────────────────────────
# Defect C: Deep-investigation grounding checker tests
# ─────────────────────────────────────────────────────────────────────────────

def test_grounding_matches_valid_financial_amount_and_percentage():
    """Defect C Case 1: Financial amount and percentage matching evidence is grounded."""
    from omnisleuth.evidence.grounding import check_grounding
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    from omnisleuth.evidence.store import EvidenceStore

    store = EvidenceStore(dataset_id="ds_test", dataset_version="v1")
    store.add_evidence(
        EvidenceItem(
            evidence_id="ev_trend",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="calculate_trend",
            content={"overall_delta": -25000.0, "overall_change_pct": -23.19},
        )
    )

    answer = "Calculated overall revenue delta was -$25,000.00 (-23.19%)."
    grounded, answer_nums, ev_nums = check_grounding(answer, store)
    assert grounded is True
    assert -25000.0 in answer_nums
    assert -23.19 in answer_nums


def test_grounding_rejects_unsupported_financial_claims():
    """Defect C Case 2: An unsupported financial amount or percentage remains ungrounded."""
    from omnisleuth.evidence.grounding import check_grounding
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    from omnisleuth.evidence.store import EvidenceStore

    store = EvidenceStore(dataset_id="ds_test", dataset_version="v1")
    store.add_evidence(
        EvidenceItem(
            evidence_id="ev_calc",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="calc",
            content={"revenue": 5000.0, "pct": 10.0},
        )
    )

    # Fabricated financial claims
    answer = "Revenue was $999,999.00 with a decline of -85.5%."
    grounded, answer_nums, ev_nums = check_grounding(answer, store)
    assert grounded is False


def test_grounding_distinguishes_non_financial_confidence_scores_from_claims():
    """Defect C Case 3: Confidence scores (e.g. 80%) are not treated as financial claims.

    Reproduces the smoke test failure where deep-investigation report containing
    -23.19% and (Confidence: 80%), (Confidence: 85%), (Confidence: 100%) failed grounding.
    """
    from omnisleuth.evidence.grounding import check_grounding, extract_currency_and_percentages
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    from omnisleuth.evidence.store import EvidenceStore

    store = EvidenceStore(dataset_id="ds_test", dataset_version="v1")
    store.add_evidence(
        EvidenceItem(
            evidence_id="ev_trend",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="calculate_trend",
            content={"overall_delta": -25000.0, "overall_change_pct": -23.19},
        )
    )

    deep_report = (
        "# Multi-Agent Investigation Report\n"
        "## 1. Verified Analytical Claims\n"
        "- ✅ [SUPPORTED]: Overall revenue delta across observed period was -$25,000.00 (-23.19%).\n"
        "## 2. Tested Hypotheses\n"
        "- 🟢 **Hypothesis**: Segment variance drove change.\n"
        "  - **Status**: `supported` (Confidence: 80%)\n"
        "- 🟢 **Hypothesis**: Anomalies affected performance.\n"
        "  - **Status**: `supported` (Confidence: 85%)\n"
        "- ⚪ **Hypothesis**: Baseline stability.\n"
        "  - **Status**: `inconclusive` (Confidence: 100%)\n"
    )

    extracted = extract_currency_and_percentages(deep_report)
    assert -25000.0 in extracted
    assert -23.19 in extracted
    # Confidence percentages must NOT be extracted as financial claims
    assert 80.0 not in extracted
    assert 85.0 not in extracted
    assert 100.0 not in extracted

    grounded, answer_nums, ev_nums = check_grounding(deep_report, store)
    assert grounded is True


def test_grounding_rejects_answer_with_no_checkable_financial_claims():
    """Defect C Case 4: An answer with no checkable financial claims must NOT silently pass."""
    from omnisleuth.evidence.grounding import check_grounding
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    from omnisleuth.evidence.store import EvidenceStore

    store = EvidenceStore(dataset_id="ds_test", dataset_version="v1")
    store.add_evidence(
        EvidenceItem(
            evidence_id="ev_calc",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="calc",
            content={"revenue": 5000.0},
        )
    )

    # Non-financial narrative with no checkable currency or percentage claims
    answer = "The investigation completed successfully and all hypotheses were evaluated."
    grounded, answer_nums, ev_nums = check_grounding(answer, store)
    assert grounded is False
    assert len(answer_nums) == 0


def test_grounding_excludes_actual_report_metadata_and_rag_formula():
    """Defect C Follow-up Case 1: Exclude confidence, completeness, and formula multiplier metadata.

    Reproduces the smoke test failure where deep-investigation report contained
    valid claim -23.19%, hypothesis confidence scores (80%, 85%, 90%), and
    RAG formula definition '(Delta_segment / Delta_total) * 100%' which extracted
    100.0% and failed evidence matching.
    """
    from omnisleuth.evidence.grounding import check_grounding, extract_currency_and_percentages
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    from omnisleuth.evidence.store import EvidenceStore

    store = EvidenceStore(dataset_id="ds_test", dataset_version="v1")
    store.add_evidence(
        EvidenceItem(
            evidence_id="ev_trend",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="calculate_trend",
            content={"overall_delta": -54539.65, "overall_change_pct": -23.19},
        )
    )

    actual_report = (
        "# Multi-Agent Investigation Report\n"
        '**Question:** *"Why did revenue decline in March?"*\n'
        "## 1. Verified Analytical Claims (Programmatically Validated)\n"
        "- ✅ **[SUPPORTED]**: Calculated overall revenue delta across the observed period was -$54,539.65 (-23.19%).\n"
        "## 2. Tested Hypotheses\n"
        "- 🔴 **Hypothesis**: Changes in specific segments contributed to variance.\n"
        "  - **Status**: `rejected` (Confidence: 80%)\n"
        "- 🟢 **Hypothesis**: Statistical anomaly events affected performance.\n"
        "  - **Status**: `supported` (Confidence: 85%)\n"
        "- 🟢 **Hypothesis**: Operational factors provide context.\n"
        "  - **Status**: `supported` (Confidence: 90%)\n"
        "## 3. Contextual RAG Business Evidence\n"
        "- 📄 **Executive KPI Definitions & Methodology** (Score: 0.98):\n"
        '  > *"# Executive KPI Definitions & Business Metrics\n'
        "- **Segment Contribution**: The proportion of metric change attributable to an individual dimension: `(Delta_segment / Delta_total) * 100%`.\n"
        '- **Return Rate**: Percentage of processed orders flagged with `is_returned = True`."*\n'
    )

    extracted = extract_currency_and_percentages(actual_report)
    assert -54539.65 in extracted
    assert -23.19 in extracted
    # Confidence and formula scaling metadata must NOT be extracted
    assert 80.0 not in extracted
    assert 85.0 not in extracted
    assert 90.0 not in extracted
    assert 100.0 not in extracted

    grounded, answer_nums, ev_nums = check_grounding(actual_report, store)
    assert grounded is True


def test_grounding_supports_legitimate_financial_percentage_of_100():
    """Defect C Follow-up Case 2: A genuine 100% financial claim remains checkable and grounded."""
    from omnisleuth.evidence.grounding import check_grounding, extract_currency_and_percentages
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    from omnisleuth.evidence.store import EvidenceStore

    store = EvidenceStore(dataset_id="ds_test", dataset_version="v1")
    store.add_evidence(
        EvidenceItem(
            evidence_id="ev_share",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="calculate_share",
            content={"segment": "North", "revenue": 100000.0, "share_pct": 100.0},
        )
    )

    answer = "North region accounted for $100,000.00 (100.0% of total revenue)."
    extracted = extract_currency_and_percentages(answer)
    assert 100000.0 in extracted
    assert 100.0 in extracted

    grounded, answer_nums, ev_nums = check_grounding(answer, store)
    assert grounded is True
    assert 100.0 in answer_nums


def test_grounding_rejects_unsupported_100_percent_financial_claim():
    """Defect C Follow-up Case 3: An unsupported 100% financial claim is extracted and fails."""
    from omnisleuth.evidence.grounding import check_grounding, extract_currency_and_percentages
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    from omnisleuth.evidence.store import EvidenceStore

    store = EvidenceStore(dataset_id="ds_test", dataset_version="v1")
    store.add_evidence(
        EvidenceItem(
            evidence_id="ev_share",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="calculate_share",
            content={"segment": "North", "revenue": 25000.0, "share_pct": 25.0},
        )
    )

    answer = "North region accounted for $25,000.00 (100.0% of total revenue)."
    extracted = extract_currency_and_percentages(answer)
    assert 100.0 in extracted

    grounded, answer_nums, ev_nums = check_grounding(answer, store)
    assert grounded is False
    assert 100.0 in answer_nums


def test_grounding_rejects_answer_with_only_metadata_filtered():
    """Defect C Follow-up Case 4: Answer with only confidence/completeness metadata does not pass."""
    from omnisleuth.evidence.grounding import check_grounding, extract_currency_and_percentages
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    from omnisleuth.evidence.store import EvidenceStore

    store = EvidenceStore(dataset_id="ds_test", dataset_version="v1")
    store.add_evidence(
        EvidenceItem(
            evidence_id="ev_any",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="calc",
            content={"metric": 100.0},
        )
    )

    answer = (
        "Analysis finished. Status: supported (Confidence: 100%). "
        "Dataset completeness is 100.0% (100% complete). "
        "Calculated using formula: (Delta / Total) * 100%."
    )
    extracted = extract_currency_and_percentages(answer)
    assert len(extracted) == 0

    grounded, answer_nums, ev_nums = check_grounding(answer, store)
    # Must remain ungrounded because there are no checkable financial claims
    assert grounded is False
    assert len(answer_nums) == 0


def test_grounding_fails_when_metadata_filtered_but_financial_claim_unsupported():
    """Defect C Follow-up Case 5: Checker does not pass merely because metadata was filtered out."""
    from omnisleuth.evidence.grounding import check_grounding, extract_currency_and_percentages
    from omnisleuth.evidence.models import EvidenceItem, EvidenceType
    from omnisleuth.evidence.store import EvidenceStore

    store = EvidenceStore(dataset_id="ds_test", dataset_version="v1")
    store.add_evidence(
        EvidenceItem(
            evidence_id="ev_trend",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="calc",
            content={"overall_change_pct": -23.19},
        )
    )

    # Has metadata AND an unsupported financial claim (-95.0%)
    answer = (
        "Overall revenue declined by -95.0% across the period. "
        "Hypothesis evaluation: (Confidence: 100%), ratio * 100%."
    )
    extracted = extract_currency_and_percentages(answer)
    assert -95.0 in extracted
    assert 100.0 not in extracted

    grounded, answer_nums, ev_nums = check_grounding(answer, store)
    assert grounded is False
