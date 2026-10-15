"""Centralized LLM Gateway module for OmniSleuth."""

from omnisleuth.llm.cache import SemanticCache
from omnisleuth.llm.errors import (
    LLMErrorCategory,
    LLMErrorInfo,
    LLMProviderError,
    classify_llm_error,
    redact_literals,
)
from omnisleuth.llm.gateway import LLMGateway
from omnisleuth.llm.models import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    TraceRecord,
    UsageMetrics,
)
from omnisleuth.llm.providers import (
    AzureOpenAIProvider,
    BaseProvider,
    DeterministicFallbackProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from omnisleuth.llm.tracing import LLMTracer, generate_request_id, sanitize_text
from omnisleuth.llm.usage import calculate_usage_and_cost

__all__ = [
    "LLMGateway",
    "LLMRequest",
    "LLMResponse",
    "LLMMessage",
    "UsageMetrics",
    "TraceRecord",
    "BaseProvider",
    "DeterministicFallbackProvider",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "OpenRouterProvider",
    "SemanticCache",
    "LLMTracer",
    "generate_request_id",
    "calculate_usage_and_cost",
    "sanitize_text",
    "LLMErrorCategory",
    "LLMErrorInfo",
    "LLMProviderError",
    "classify_llm_error",
    "redact_literals",
]
