"""Pydantic data models and schemas for formal input and output guardrails."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GuardrailCategory(str, Enum):
    """Categorical classification of guardrail evaluations."""

    SAFE = "safe"
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    OFF_TOPIC = "off_topic"
    ARBITRARY_CODE = "arbitrary_code"
    SECRET_LEAKAGE = "secret_leakage"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    MISSING_EVIDENCE = "missing_evidence"
    UNSUPPORTED_CAUSALITY = "unsupported_causality"
    INVALID_OUTPUT = "invalid_output"
    INVALID_INPUT = "invalid_input"


class GuardrailSeverity(str, Enum):
    """Severity level of a guardrail violation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailDecision(str, Enum):
    """Actionable decision from guardrail evaluation."""

    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    REWRITE = "rewrite"


class TopicScope(str, Enum):
    """Scope classification for input queries."""

    ON_TOPIC = "on_topic"
    OFF_TOPIC = "off_topic"
    UNCERTAIN = "uncertain"


class GuardrailResult(BaseModel):
    """Structured result of an input or output guardrail evaluation."""

    allowed: bool = True
    decision: GuardrailDecision = GuardrailDecision.ALLOW
    category: GuardrailCategory = GuardrailCategory.SAFE
    severity: GuardrailSeverity = GuardrailSeverity.LOW
    reason: str = "Passed all safety and grounding guardrails."
    user_message: Optional[str] = None
    matched_rule: Optional[str] = None
    redacted_content: Optional[str] = None
    calibrated_content: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GuardedAnswer(BaseModel):
    """Strictly structured and verified final investigation output."""

    answer: str = Field(description="Direct narrative response to the business inquiry")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Referenced evidence items with IDs and provenance")
    calculation: List[str] = Field(default_factory=list, description="Deterministic calculations and formulas used")
    interpretation: str = Field(default="", description="Calibrated business interpretation without unsupported causality")
    confidence: Optional[float] = Field(default=None, description="Confidence score bounded between 0.0 and 1.0")
    limitations: List[str] = Field(default_factory=list, description="Explicit boundaries and data limitations")
    guardrail_result: Optional[GuardrailResult] = None
