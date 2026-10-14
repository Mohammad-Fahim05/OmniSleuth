"""Request tracing and audit logging for LLM Gateway calls."""

from datetime import datetime, timezone
import uuid
from typing import Dict, List, Optional
from omnisleuth.core.logging import get_logger
from omnisleuth.guardrails.patterns import SECRET_LEAKAGE_PATTERNS
from omnisleuth.llm.models import LLMRequest, LLMResponse, TraceRecord

logger = get_logger("llm.tracing")


def generate_request_id() -> str:
    """Generate a unique, traceable LLM request identifier."""
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:8]
    return f"llm_req_{now_str}_{short_uuid}"


def sanitize_text(text: str) -> str:
    """Sanitize secrets and API keys from traced text."""
    sanitized = text
    for _, pattern in SECRET_LEAKAGE_PATTERNS:
        sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
    return sanitized


class LLMTracer:
    """Maintains trace history and correlates LLM requests with investigations."""

    def __init__(self):
        self._traces: List[TraceRecord] = []

    def log_request(
        self,
        request: LLMRequest,
        response: LLMResponse,
    ) -> TraceRecord:
        """Create and store a trace record for an LLM call."""
        record = TraceRecord(
            request_id=response.request_id,
            timestamp=response.timestamp,
            investigation_id=request.investigation_id,
            dataset_version=request.dataset_version,
            provider=response.provider,
            model=response.model,
            latency_ms=response.latency_ms,
            success=response.error is None,
            fallback_used=response.fallback_used,
            cached=response.cached,
            retry_count=response.retry_count,
            usage=response.usage,
            error=response.error,
        )
        self._traces.append(record)

        cost_str = f"${response.usage.estimated_cost_usd:.6f}" if (response.usage and response.usage.estimated_cost_usd is not None) else "N/A"
        logger.info(
            f"LLM Trace [{response.request_id}] provider={response.provider} model={response.model} "
            f"latency={response.latency_ms:.1f}ms fallback={response.fallback_used} cached={response.cached} "
            f"cost={cost_str} inv_id={request.investigation_id or 'none'}"
        )
        return record

    def get_traces_for_investigation(self, investigation_id: str) -> List[TraceRecord]:
        """Retrieve all LLM trace records correlated with a specific investigation ID."""
        return [t for t in self._traces if t.investigation_id == investigation_id]

    def list_traces(self) -> List[TraceRecord]:
        """Return all recorded LLM trace records."""
        return list(self._traces)
