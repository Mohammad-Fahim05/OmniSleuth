"""Provider abstractions for LLM Gateway: Abstract Base, Deterministic Fallback, OpenAI, Azure OpenAI, OpenRouter."""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional
from omnisleuth.core.config import get_settings
from omnisleuth.core.exceptions import ConfigurationError
from omnisleuth.core.logging import get_logger
from omnisleuth.llm.errors import LLMProviderError, classify_llm_error, redact_literals
from omnisleuth.llm.models import LLMRequest, LLMResponse
from omnisleuth.llm.tracing import generate_request_id
from omnisleuth.llm.usage import calculate_usage_and_cost

logger = get_logger("llm.providers")

# Default base URL for the OpenAI-compatible OpenRouter API.
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _as_message_payload(request: LLMRequest) -> List[Dict[str, str]]:
    """Convert a request into an OpenAI-compatible chat message payload."""
    if request.messages:
        return [{"role": m.role, "content": m.content} for m in request.messages]
    return [{"role": "user", "content": request.get_prompt_text()}]


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(self, request: LLMRequest, request_id: str) -> LLMResponse:
        """Execute request and return structured LLMResponse."""


class DeterministicFallbackProvider(BaseProvider):
    """Offline-first deterministic provider.

    Returns structured rule-based responses formatted to match requested prompts/objectives,
    with fallback_used=True, provider='fallback', and model='deterministic'.
    """

    def generate(self, request: LLMRequest, request_id: str) -> LLMResponse:
        t0 = time.perf_counter()
        prompt_text = request.get_prompt_text()

        # Build clean fallback text
        fallback_text = (
            f"[OFFLINE DETERMINISTIC SYNTHESIS]\n"
            f"Objective: {prompt_text[:200]}\n"
            f"OmniSleuth generated a rule-grounded offline response without external LLM API dependencies."
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        usage = calculate_usage_and_cost(provider="fallback")

        return LLMResponse(
            request_id=request_id,
            text=fallback_text,
            provider="fallback",
            model="deterministic",
            latency_ms=round(elapsed_ms, 2),
            fallback_used=True,
            cached=False,
            retry_count=0,
            usage=usage,
            error=None,
        )


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation with timeout and token usage tracking."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.openai_api_key
        self.model = model or self.settings.openai_model

    def generate(self, request: LLMRequest, request_id: str) -> LLMResponse:
        if not self.api_key:
            # Offline-safe: fail without performing any network request.
            raise ConfigurationError("OpenAI API key not configured. Set OPENAI_API_KEY or use LLM_PROVIDER=fallback.")

        model_to_use = request.model or self.model
        if not model_to_use:
            raise ConfigurationError("OpenAI model not configured. Set OPENAI_MODEL to an available model id.")

        timeout_sec = request.timeout_seconds or self.settings.llm_timeout_seconds
        t0 = time.perf_counter()

        try:
            # We import openai dynamically if installed
            import openai

            client = openai.OpenAI(api_key=self.api_key, timeout=timeout_sec)

            completion = client.chat.completions.create(
                model=model_to_use,
                messages=_as_message_payload(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            text_out = completion.choices[0].message.content or ""

            in_tok = completion.usage.prompt_tokens if completion.usage else None
            out_tok = completion.usage.completion_tokens if completion.usage else None
            tot_tok = completion.usage.total_tokens if completion.usage else None

            usage = calculate_usage_and_cost(
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=tot_tok,
                provider="openai",
            )

            return LLMResponse(
                request_id=request_id,
                text=text_out,
                provider="openai",
                model=model_to_use,
                latency_ms=round(elapsed_ms, 2),
                fallback_used=False,
                cached=False,
                usage=usage,
                error=None,
            )

        except ConfigurationError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            info = classify_llm_error(exc, secrets=[self.api_key])
            # Only the sanitized category/message is logged; credentials are never emitted.
            logger.warning(
                f"OpenAIProvider error after {elapsed_ms:.1f}ms "
                f"[category={info.category.value} retryable={info.retryable}]: {info.message}"
            )
            raise LLMProviderError(
                message=info.message,
                category=info.category,
                status_code=info.status_code,
                provider="openai",
            ) from exc


class OpenRouterProvider(BaseProvider):
    """OpenRouter provider using the OpenAI-compatible chat completions API.

    OpenRouter is entirely optional: when `OPENROUTER_API_KEY` is missing the
    provider raises a permanent, offline-safe error *before* any network request
    is attempted, so callers fall back to deterministic planning.
    """

    PROVIDER_NAME = "openrouter"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.openrouter_api_key
        self.model = model or self.settings.openrouter_model
        self.base_url = (base_url or self.settings.openrouter_base_url or OPENROUTER_DEFAULT_BASE_URL).rstrip("/")

    @property
    def effective_base_url(self) -> str:
        """Resolved OpenRouter base URL (never a secret)."""
        return self.base_url or OPENROUTER_DEFAULT_BASE_URL

    def generate(self, request: LLMRequest, request_id: str) -> LLMResponse:
        if not self.api_key:
            # Offline-safe: fail without performing any network request.
            raise ConfigurationError(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY or use LLM_PROVIDER=fallback."
            )

        model_to_use = request.model or self.model
        if not model_to_use:
            raise ConfigurationError(
                "OpenRouter model not configured. Set OPENROUTER_MODEL to an available OpenRouter model id."
            )

        timeout_sec = request.timeout_seconds or self.settings.llm_timeout_seconds
        t0 = time.perf_counter()

        try:
            # OpenRouter exposes an OpenAI-compatible API, so the existing OpenAI
            # client is reused with only the base URL (and optional app headers) changed.
            import openai

            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.effective_base_url,
                timeout=timeout_sec,
                default_headers={"X-Title": str(self.settings.app_name)},
            )

            completion = client.chat.completions.create(
                model=model_to_use,
                messages=_as_message_payload(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            text_out = completion.choices[0].message.content or ""

            in_tok = completion.usage.prompt_tokens if completion.usage else None
            out_tok = completion.usage.completion_tokens if completion.usage else None
            tot_tok = completion.usage.total_tokens if completion.usage else None

            usage = calculate_usage_and_cost(
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=tot_tok,
                provider=self.PROVIDER_NAME,
            )

            return LLMResponse(
                request_id=request_id,
                text=text_out,
                provider=self.PROVIDER_NAME,
                model=model_to_use,
                latency_ms=round(elapsed_ms, 2),
                fallback_used=False,
                cached=False,
                usage=usage,
                error=None,
            )

        except ConfigurationError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            info = classify_llm_error(exc, secrets=[self.api_key])
            # Only the sanitized category/message is logged; credentials are never emitted.
            logger.warning(
                f"OpenRouterProvider error after {elapsed_ms:.1f}ms "
                f"[category={info.category.value} retryable={info.retryable}]: {info.message}"
            )
            raise LLMProviderError(
                message=info.message,
                category=info.category,
                status_code=info.status_code,
                provider=self.PROVIDER_NAME,
            ) from exc


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI API provider implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.azure_openai_api_key
        self.endpoint = endpoint or self.settings.azure_openai_endpoint
        self.deployment = deployment or self.settings.azure_openai_deployment
        self.api_version = api_version or self.settings.azure_openai_api_version

    def generate(self, request: LLMRequest, request_id: str) -> LLMResponse:
        if not self.api_key or not self.endpoint or not self.deployment:
            # Offline-safe: fail without performing any network request.
            raise ConfigurationError("Azure OpenAI credentials or endpoint not fully configured.")

        timeout_sec = request.timeout_seconds or self.settings.llm_timeout_seconds
        t0 = time.perf_counter()

        try:
            import openai

            client = openai.AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                timeout=timeout_sec,
            )

            completion = client.chat.completions.create(
                model=self.deployment,
                messages=_as_message_payload(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            text_out = completion.choices[0].message.content or ""

            in_tok = completion.usage.prompt_tokens if completion.usage else None
            out_tok = completion.usage.completion_tokens if completion.usage else None
            tot_tok = completion.usage.total_tokens if completion.usage else None

            usage = calculate_usage_and_cost(
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=tot_tok,
                provider="azure",
            )

            return LLMResponse(
                request_id=request_id,
                text=text_out,
                provider="azure",
                model=self.deployment,
                latency_ms=round(elapsed_ms, 2),
                fallback_used=False,
                cached=False,
                usage=usage,
                error=None,
            )

        except ConfigurationError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            info = classify_llm_error(exc, secrets=[self.api_key])
            # Only the sanitized category/message is logged; credentials are never emitted.
            logger.warning(
                f"AzureOpenAIProvider error after {elapsed_ms:.1f}ms "
                f"[category={info.category.value} retryable={info.retryable}]: {info.message}"
            )
            raise LLMProviderError(
                message=info.message,
                category=info.category,
                status_code=info.status_code,
                provider="azure",
            ) from exc
