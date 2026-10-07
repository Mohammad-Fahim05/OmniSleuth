"""Deterministic input guardrails detecting prompt injections, jailbreaks, code execution, and off-topic requests."""

import re
from typing import Optional, Set, Tuple
from omnisleuth.core.logging import get_logger
from omnisleuth.guardrails.models import (
    GuardrailCategory,
    GuardrailDecision,
    GuardrailResult,
    GuardrailSeverity,
    TopicScope,
)
from omnisleuth.guardrails.patterns import (
    ARBITRARY_CODE_PATTERNS,
    JAILBREAK_PATTERNS,
    OFF_TOPIC_PATTERNS,
    ON_TOPIC_KEYWORDS,
    PROMPT_INJECTION_PATTERNS,
)

logger = get_logger("guardrails.input")


class InputGuardrailEngine:
    """Pre-planning input guardrail validator enforcing system boundaries and safety."""

    def __init__(self):
        self.on_topic_keywords: Set[str] = set(ON_TOPIC_KEYWORDS)

    def _normalize(self, text: str) -> str:
        """Normalize query text for consistent, robust regex pattern evaluation."""
        lowered = text.lower().strip()
        # Collapse multiple spaces and trim surrounding punctuation
        cleaned = re.sub(r"\s+", " ", lowered)
        return cleaned

    def check_prompt_injection(self, text: str) -> Optional[GuardrailResult]:
        """Check for prompt injection and system override attempts."""
        normalized = self._normalize(text)
        for rule_name, pattern in PROMPT_INJECTION_PATTERNS:
            if pattern.search(normalized):
                logger.warning(f"Input Guardrail Block: Prompt injection detected via rule '{rule_name}'")
                return GuardrailResult(
                    allowed=False,
                    decision=GuardrailDecision.BLOCK,
                    category=GuardrailCategory.PROMPT_INJECTION,
                    severity=GuardrailSeverity.HIGH,
                    reason=f"Prompt injection pattern detected ({rule_name}).",
                    user_message=(
                        "I can help investigate the uploaded business dataset and supporting documents, "
                        "but I cannot override the system's investigation rules or reveal internal prompts."
                    ),
                    matched_rule=rule_name,
                )
        return None

    def check_jailbreak(self, text: str) -> Optional[GuardrailResult]:
        """Check for adversarial jailbreak, developer mode, or constraint bypass attempts."""
        normalized = self._normalize(text)
        for rule_name, pattern in JAILBREAK_PATTERNS:
            if pattern.search(normalized):
                logger.warning(f"Input Guardrail Block: Jailbreak attempt detected via rule '{rule_name}'")
                return GuardrailResult(
                    allowed=False,
                    decision=GuardrailDecision.BLOCK,
                    category=GuardrailCategory.JAILBREAK,
                    severity=GuardrailSeverity.HIGH,
                    reason=f"Jailbreak pattern detected ({rule_name}).",
                    user_message=(
                        "I operate strictly as an evidence-grounded data investigation system "
                        "and cannot disable safety constraints or operate unrestricted."
                    ),
                    matched_rule=rule_name,
                )
        return None

    def check_arbitrary_code(self, text: str) -> Optional[GuardrailResult]:
        """Check for requests asking to run arbitrary Python code or shell commands."""
        normalized = self._normalize(text)

        for rule_name, pattern in ARBITRARY_CODE_PATTERNS:
            if pattern.search(normalized):
                # False-positive guard: If asking about product names or schema columns containing technical words
                is_legitimate_query = (
                    any(q_word in normalized for q_word in ["which product", "how many", "count", "where category", "contains the word", "named"])
                    and not any(exec_word in normalized for exec_word in ["exec(", "eval(", "os.system", "subprocess", "run this python code", "execute this"])
                )
                if is_legitimate_query:
                    continue

                logger.warning(f"Input Guardrail Block: Arbitrary code execution request detected via rule '{rule_name}'")
                return GuardrailResult(
                    allowed=False,
                    decision=GuardrailDecision.BLOCK,
                    category=GuardrailCategory.ARBITRARY_CODE,
                    severity=GuardrailSeverity.CRITICAL,
                    reason=f"Arbitrary code execution request detected ({rule_name}).",
                    user_message=(
                        "Arbitrary code execution is strictly prohibited. OmniSleuth only executes "
                        "pre-approved, deterministic allowlisted analytical tools."
                    ),
                    matched_rule=rule_name,
                )
        return None

    def check_topic_scope(self, text: str) -> Tuple[TopicScope, Optional[GuardrailResult]]:
        """Classify input query into ON_TOPIC, OFF_TOPIC, or UNCERTAIN."""
        normalized = self._normalize(text)
        words = set(re.findall(r"\b\w{2,}\b", normalized))

        # Check explicit off-topic patterns
        for rule_name, pattern in OFF_TOPIC_PATTERNS:
            if pattern.search(normalized):
                # Check if it also has significant dataset analytical intent
                overlap = words.intersection(self.on_topic_keywords)
                if len(overlap) < 2:
                    logger.info(f"Input Guardrail Block: Off-topic inquiry detected via rule '{rule_name}'")
                    result = GuardrailResult(
                        allowed=False,
                        decision=GuardrailDecision.BLOCK,
                        category=GuardrailCategory.OFF_TOPIC,
                        severity=GuardrailSeverity.LOW,
                        reason=f"Off-topic inquiry outside data investigation scope ({rule_name}).",
                        user_message=(
                            "This request is outside OmniSleuth's scope. I can investigate questions "
                            "related to the uploaded business dataset and supporting business documents."
                        ),
                        matched_rule=rule_name,
                    )
                    return TopicScope.OFF_TOPIC, result

        # Check domain relevance
        if any(w in self.on_topic_keywords for w in words):
            return TopicScope.ON_TOPIC, None

        # If neutral/uncertain, permit exploration
        return TopicScope.UNCERTAIN, None

    def validate_input(self, question: str) -> GuardrailResult:
        """Execute full input guardrail pipeline: Injection -> Jailbreak -> Code -> Scope."""
        clean_q = question.strip() if question else ""
        if not clean_q:
            return GuardrailResult(
                allowed=False,
                decision=GuardrailDecision.BLOCK,
                category=GuardrailCategory.OFF_TOPIC,
                severity=GuardrailSeverity.LOW,
                reason="Empty question string provided.",
                user_message="Please provide a valid analytical question regarding the dataset.",
            )

        # 1. Prompt Injection
        inj_res = self.check_prompt_injection(clean_q)
        if inj_res:
            return inj_res

        # 2. Jailbreak
        jb_res = self.check_jailbreak(clean_q)
        if jb_res:
            return jb_res

        # 3. Arbitrary Code Execution
        code_res = self.check_arbitrary_code(clean_q)
        if code_res:
            return code_res

        # 4. Scope Classification
        scope, off_res = self.check_topic_scope(clean_q)
        if off_res:
            return off_res

        # Allowed
        return GuardrailResult(
            allowed=True,
            decision=GuardrailDecision.ALLOW,
            category=GuardrailCategory.SAFE,
            severity=GuardrailSeverity.LOW,
            reason="Input verified as safe, on-topic analytical inquiry.",
        )
