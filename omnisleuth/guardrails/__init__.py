"""Formal Input and Output Guardrails subsystem."""

from omnisleuth.guardrails.input_guardrails import InputGuardrailEngine
from omnisleuth.guardrails.models import (
    GuardedAnswer,
    GuardrailCategory,
    GuardrailDecision,
    GuardrailResult,
    GuardrailSeverity,
    TopicScope,
)
from omnisleuth.guardrails.output_guardrails import OutputGuardrailEngine
from omnisleuth.guardrails.patterns import (
    ARBITRARY_CODE_PATTERNS,
    JAILBREAK_PATTERNS,
    OFF_TOPIC_PATTERNS,
    ON_TOPIC_KEYWORDS,
    PROMPT_INJECTION_PATTERNS,
    SECRET_LEAKAGE_PATTERNS,
    UNSUPPORTED_CAUSAL_PHRASES,
)

__all__ = [
    "InputGuardrailEngine",
    "OutputGuardrailEngine",
    "GuardrailResult",
    "GuardedAnswer",
    "GuardrailCategory",
    "GuardrailSeverity",
    "GuardrailDecision",
    "TopicScope",
    "PROMPT_INJECTION_PATTERNS",
    "JAILBREAK_PATTERNS",
    "ARBITRARY_CODE_PATTERNS",
    "OFF_TOPIC_PATTERNS",
    "ON_TOPIC_KEYWORDS",
    "SECRET_LEAKAGE_PATTERNS",
    "UNSUPPORTED_CAUSAL_PHRASES",
]
