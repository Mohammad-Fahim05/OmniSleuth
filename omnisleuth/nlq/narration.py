"""Optional narrative rewriting of templated answers (Phase D).

Disabled by default. When enabled, the templated answer may be rewritten by the
LLM **only if every number in the rewrite is grounded in the evidence already
minted by ``PlanExecutor``**. Any ungrounded number discards the rewrite and the
templated answer is shown instead. No new number is ever computed here.
"""

import re
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from omnisleuth.core.logging import get_logger
from omnisleuth.llm.models import LLMRequest
from omnisleuth.core.exceptions import ConfigurationError

logger = get_logger("nlq.narration")

MAX_NARRATION_CHARS = 1200
_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


class NarrationResult(BaseModel):
    """Outcome of an optional narration attempt."""

    used: bool = False
    text: Optional[str] = None
    reason: Optional[str] = None
    grounded_numbers: int = 0
    ungrounded_numbers: List[float] = Field(default_factory=list)


def extract_numbers(text: str) -> List[float]:
    """Extract numeric literals from text, ignoring thousands separators."""
    numbers: List[float] = []
    for match in _NUMBER_RE.finditer(text or ""):
        raw = match.group(0).replace(",", "")
        try:
            numbers.append(float(raw))
        except ValueError:  # pragma: no cover - regex guarantees parseability
            continue
    return numbers


def _evidence_numbers(evidence_store: Any) -> set:
    """Collect every numeric value present in the evidence store."""
    numbers = set()

    def _walk(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            numbers.add(round(float(value), 2))
        elif isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _walk(v)

    for ev in evidence_store.list_evidence():
        _walk(ev.content)
    return numbers


def is_grounded(number: float, evidence_numbers: set, tolerance_pct: float = 1.0) -> bool:
    """True when a number matches an evidence value within tolerance."""
    for candidate in evidence_numbers:
        if candidate == 0 and number == 0:
            return True
        if candidate == 0:
            continue
        if abs(number - candidate) / abs(candidate) * 100.0 <= tolerance_pct:
            return True
    return False


def narrate(
    template_answer: str,
    gateway: Any,
    evidence_store: Any,
    question: str = "",
    tolerance_pct: float = 1.0,
) -> NarrationResult:
    """Attempt an LLM rewrite, keeping it only when every number is grounded."""
    if not template_answer or evidence_store is None:
        return NarrationResult(used=False, reason="no template answer or evidence store")

    try:
        prompt = (
            "Rewrite the following analytical answer in clear business language. "
            "Use ONLY the numbers already present in the answer. Do not add, remove, round, "
            "estimate, or invent any number, percentage, or date. "
            f"Keep it under {MAX_NARRATION_CHARS} characters.\n\n"
            f"Question: {question}\n\nAnswer:\n{template_answer[:MAX_NARRATION_CHARS]}"
        )
        response = gateway.generate(
            LLMRequest(prompt=prompt, temperature=0.0, max_tokens=400)
        )
    except ConfigurationError as exc:
        logger.info(f"Narration skipped ({exc})")
        return NarrationResult(used=False, reason="llm_not_configured")
    except Exception as exc:  # provider failures never break answering
        logger.warning(f"Narration attempt failed: {type(exc).__name__}")
        return NarrationResult(used=False, reason="llm_error")

    if getattr(gateway, "is_fallback_response", None) and gateway.is_fallback_response(response):
        return NarrationResult(used=False, reason="fallback_response")

    text = (response.text or "").strip()
    if not text:
        return NarrationResult(used=False, reason="empty_narration")
    if len(text) > MAX_NARRATION_CHARS * 2:
        return NarrationResult(used=False, reason="narration_too_long")

    evidence_numbers = _evidence_numbers(evidence_store)
    template_numbers = set(round(n, 2) for n in extract_numbers(template_answer))
    narration_numbers = extract_numbers(text)

    grounded = 0
    ungrounded: List[float] = []
    for number in narration_numbers:
        if number in template_numbers or is_grounded(number, evidence_numbers, tolerance_pct):
            grounded += 1
        else:
            ungrounded.append(round(number, 2))

    if ungrounded:
        logger.info(
            "Narration discarded: %d ungrounded number(s) [%s]",
            len(ungrounded),
            ungrounded[:5],
        )
        return NarrationResult(
            used=False,
            reason="ungrounded_numbers",
            grounded_numbers=grounded,
            ungrounded_numbers=ungrounded[:10],
        )

    return NarrationResult(used=True, text=text, grounded_numbers=grounded)
