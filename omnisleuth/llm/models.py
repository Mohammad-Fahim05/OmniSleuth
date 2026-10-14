"""Structured Pydantic models for LLM requests, responses, usage, and tracing."""

from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """An individual message in a chat-based LLM request."""

    role: str = Field(description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(description="Text content of the message")


class LLMRequest(BaseModel):
    """Structured provider-independent LLM request."""

    prompt: Optional[str] = Field(default=None, description="Single text prompt (if not using messages)")
    messages: Optional[List[LLMMessage]] = Field(default=None, description="Structured chat messages")
    model: Optional[str] = Field(default=None, description="Target model override")
    provider: Optional[str] = Field(default=None, description="Target provider override ('openai', 'azure', 'fallback')")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    timeout_seconds: Optional[float] = Field(default=None, ge=0.1)
    investigation_id: Optional[str] = Field(default=None, description="Associated investigation ID for tracing")
    dataset_version: Optional[str] = Field(default=None, description="Dataset version hash for cache isolation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context or caller metadata")

    def get_prompt_text(self) -> str:
        """Extract unified text representation of prompt/messages."""
        if self.prompt:
            return self.prompt
        if self.messages:
            return "\n".join([f"{m.role}: {m.content}" for m in self.messages])
        return ""


class UsageMetrics(BaseModel):
    """Token usage and estimated cost breakdown."""

    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    estimated_cost_usd: Optional[float] = Field(default=None, ge=0.0)
    is_actual: bool = Field(default=True, description="True if provided by LLM API, False if estimated/N/A")


class LLMResponse(BaseModel):
    """Structured response returned by the LLM Gateway."""

    request_id: str = Field(description="Unique request trace ID")
    text: str = Field(description="Generated answer or completion text")
    provider: str = Field(description="Provider that produced response ('openai', 'azure', 'fallback')")
    model: str = Field(description="Model name used")
    latency_ms: float = Field(ge=0.0, description="Total latency in milliseconds")
    fallback_used: bool = Field(default=False, description="True if deterministic fallback was activated")
    cached: bool = Field(default=False, description="True if response came from semantic cache")
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")
    usage: Optional[UsageMetrics] = Field(default=None, description="Token usage and pricing metrics")
    error: Optional[str] = Field(default=None, description="Error message if failure occurred")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TraceRecord(BaseModel):
    """Complete audit record for a single LLM request execution."""

    request_id: str
    timestamp: str
    investigation_id: Optional[str] = None
    dataset_version: Optional[str] = None
    provider: str
    model: str
    operation: str = "generate"
    latency_ms: float
    success: bool
    fallback_used: bool
    cached: bool
    retry_count: int
    usage: Optional[UsageMetrics] = None
    error: Optional[str] = None
