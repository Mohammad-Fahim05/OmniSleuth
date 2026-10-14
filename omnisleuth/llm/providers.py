"""Provider abstractions for LLM Gateway: Abstract Base, Deterministic Fallback, OpenAI, and Azure OpenAI."""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, Optional
from omnisleuth.core.config import get_settings
from omnisleuth.core.logging import get_logger
from omnisleuth.llm.models import LLMRequest, LLMResponse
from omnisleuth.llm.tracing import generate_request_id
from omnisleuth.llm.usage import calculate_usage_and_cost

logger = get_logger("llm.providers")


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
            raise ValueError("OpenAI API key not configured.")

        model_to_use = request.model or self.model
        timeout_sec = request.timeout_seconds or self.settings.llm_timeout_seconds

        t0 = time.perf_counter()

        try:
            # We import openai dynamically if installed
            import openai

            client = openai.OpenAI(api_key=self.api_key, timeout=timeout_sec)

            if request.messages:
                messages_payload = [{"role": m.role, "content": m.content} for m in request.messages]
            else:
                messages_payload = [{"role": "user", "content": request.get_prompt_text()}]

            completion = client.chat.completions.create(
                model=model_to_use,
                messages=messages_payload,
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

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(f"OpenAIProvider error: {exc}")
            raise exc


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
            raise ValueError("Azure OpenAI credentials or endpoint not fully configured.")

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

            if request.messages:
                messages_payload = [{"role": m.role, "content": m.content} for m in request.messages]
            else:
                messages_payload = [{"role": "user", "content": request.get_prompt_text()}]

            completion = client.chat.completions.create(
                model=self.deployment,
                messages=messages_payload,
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

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(f"AzureOpenAIProvider error: {exc}")
            raise exc
