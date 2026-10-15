"""Centralized LLM Gateway coordinating providers, timeouts, retries, fallback, tracing, and caching."""

import time
from typing import Dict, List, Optional
from omnisleuth.core.config import get_settings
from omnisleuth.core.logging import get_logger
from omnisleuth.llm.cache import SemanticCache
from omnisleuth.llm.models import LLMRequest, LLMResponse, TraceRecord
from omnisleuth.llm.providers import (
    AzureOpenAIProvider,
    BaseProvider,
    DeterministicFallbackProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from omnisleuth.llm.retry import RetryExhaustedError, execute_with_retry
from omnisleuth.llm.tracing import LLMTracer, generate_request_id, sanitize_text

logger = get_logger("llm.gateway")

# Provider name that unambiguously identifies the offline deterministic fallback.
FALLBACK_PROVIDER_NAME = "fallback"


class LLMGateway:
    """Centralized production-grade LLM Gateway preserving offline-first operation."""

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        cache: Optional[SemanticCache] = None,
        tracer: Optional[LLMTracer] = None,
    ):
        self.settings = get_settings()
        self.tracer = tracer or LLMTracer()
        self.cache = cache or SemanticCache()
        self.fallback_provider = DeterministicFallbackProvider()

        # Configured primary provider
        self._custom_provider = provider

    @property
    def is_configured(self) -> bool:
        """Return True if a primary LLM provider is configured with credentials."""
        if self._custom_provider:
            return True
        prov = self.settings.llm_provider.lower()
        if prov == "openai" and bool(self.settings.openai_api_key):
            return True
        if prov == "azure" and bool(self.settings.azure_openai_api_key):
            return True
        if prov == "openrouter" and bool(self.settings.openrouter_api_key):
            return True
        return False

    def _get_provider(self, requested_provider_name: Optional[str] = None) -> BaseProvider:
        """Resolve active provider instance based on settings or request override."""
        if self._custom_provider:
            return self._custom_provider

        prov_name = (requested_provider_name or self.settings.llm_provider).lower()

        if prov_name == "openai" and self.settings.openai_api_key:
            return OpenAIProvider()
        elif prov_name == "azure" and self.settings.azure_openai_api_key:
            return AzureOpenAIProvider()
        elif prov_name == "openrouter" and self.settings.openrouter_api_key:
            return OpenRouterProvider()

        # Fall back to offline deterministic provider
        return self.fallback_provider

    def is_fallback_response(self, response: LLMResponse) -> bool:
        """Return True if a response was produced by the deterministic fallback.

        Callers (e.g. LLMPlanner) MUST use this to distinguish a genuine LLM
        answer from offline fallback text; fallback text is never a valid model
        output and must not be parsed as if the model had produced it.
        """
        return bool(
            response.fallback_used
            or response.provider == FALLBACK_PROVIDER_NAME
            or response.model == "deterministic"
            or (response.error is not None)
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Execute structured LLM request through central gateway pipeline:

        1. Assign request ID.
        2. Check semantic cache.
        3. Execute provider call with retry and timeout.
        4. If provider fails or credentials missing, fall back safely.
        5. Record usage & audit trace.
        6. Return structured LLMResponse.
        """
        request_id = generate_request_id()
        t0 = time.perf_counter()

        # 1. Semantic Cache Lookup
        cached_res = self.cache.get(request)
        if cached_res:
            cached_res.request_id = request_id  # update trace ID
            self.tracer.log_request(request, cached_res)
            return cached_res

        # 2. Resolve provider
        active_provider = self._get_provider(request.provider)
        max_retries = self.settings.llm_max_retries
        base_delay = self.settings.llm_retry_base_delay
        max_delay = self.settings.llm_retry_max_delay

        retry_count = 0
        response: Optional[LLMResponse] = None

        try:
            # 3. Execute call with exponential backoff retries
            def _call():
                return active_provider.generate(request, request_id)

            response, retry_count = execute_with_retry(
                _call,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
            )
            response.retry_count = retry_count

        except RetryExhaustedError as exc:
            # 4. On exhausted retries, record attempt count and fall back
            retry_count = exc.attempts
            logger.warning(f"LLMGateway primary provider ({active_provider.__class__.__name__}) exhausted {retry_count} retries: {exc}. Activating deterministic fallback.")
            response = self.fallback_provider.generate(request, request_id)
            response.fallback_used = True
            response.retry_count = retry_count
            response.error = sanitize_text(f"Primary provider failed ({exc}). Activated offline fallback.")

        except Exception as exc:
            # 4. On provider failure or missing key, fall back to offline provider
            logger.warning(f"LLMGateway primary provider ({active_provider.__class__.__name__}) failed: {exc}. Activating deterministic fallback.")
            response = self.fallback_provider.generate(request, request_id)
            response.fallback_used = True
            response.retry_count = retry_count
            response.error = sanitize_text(f"Primary provider failed ({exc}). Activated offline fallback.")

        # 5. Log audit trace & store in cache
        self.tracer.log_request(request, response)
        if not response.fallback_used and not response.error:
            self.cache.put(request, response)

        return response

    def list_traces(self) -> List[TraceRecord]:
        """Return all recorded trace records."""
        return self.tracer.list_traces()

    def get_traces_for_investigation(self, investigation_id: str) -> List[TraceRecord]:
        """Retrieve trace records associated with an investigation."""
        return self.tracer.get_traces_for_investigation(investigation_id)
