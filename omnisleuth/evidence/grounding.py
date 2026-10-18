"""Grounding verification helpers for comparing answer claims against evidence stores."""

import re
from typing import Any, List, Optional, Set, Tuple


def _walk_numbers(value: Any) -> List[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out += _walk_numbers(v)
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out += _walk_numbers(v)
        return out
    return []


_CURRENCY_RE = re.compile(r"(?:^|[^\w])([+-]?)\s*\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)")
_PERCENTAGE_RE = re.compile(r"([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)\s*%")
_NON_CLAIM_SCORE_PRE_RE = re.compile(
    r"(?:\b(confidence|score|threshold|completeness)(?:\s+(?:is|of|level|score))?[^\w\n\r]*|[\)\w`\]]\s*[*×/]\s*)$",
    re.IGNORECASE,
)
_NON_CLAIM_SCORE_POST_RE = re.compile(
    r"^\s*[*_`)]*\s*(?:confidence|score|complete|completeness)\b",
    re.IGNORECASE,
)


def extract_currency_and_percentages(text: str) -> List[float]:
    """Extract only currency ($...) and percentage (...%) values from answer text.

    Counts, ranks, IDs, confidence ratings, model scores, and formula definitions
    are excluded so GROUNDED checks focus strictly on financial and proportional computations.
    """
    if not text:
        return []
    numbers = []

    # 1. Currency numbers (e.g. $1,234.56, -$50.00)
    for match in _CURRENCY_RE.finditer(text):
        sign, val_str = match.groups()
        cleaned = val_str.replace(",", "")
        try:
            val = float(cleaned)
            if sign == "-":
                val = -val
            numbers.append(val)
        except ValueError:
            continue

    # 2. Percentage numbers (e.g. 45.2%, +12.5%, -3.4%)
    for match in _PERCENTAGE_RE.finditer(text):
        # Exclude confidence ratings, scores, completeness metadata, or formula multipliers
        pre_ctx = text[max(0, match.start() - 30) : match.start()]
        post_ctx = text[match.end() : min(len(text), match.end() + 30)]
        if _NON_CLAIM_SCORE_PRE_RE.search(pre_ctx) or _NON_CLAIM_SCORE_POST_RE.search(post_ctx):
            continue

        val_str = match.group(1).replace(",", "")
        try:
            numbers.append(float(val_str))
        except ValueError:
            continue

    return numbers


def check_grounding(
    answer: str,
    evidence_store: Optional[Any],
) -> Tuple[bool, List[float], Set[float]]:
    """Determine whether all checkable financial claims in an answer match evidence.

    Returns:
        (is_grounded, answer_numbers, evidence_numbers)

    Rules:
        - Financial claims ($ amounts and analytical percentages) must match
          deterministic evidence numbers within tolerance (0.02 or 2%).
        - Non-financial numbers (confidence scores, ranks, counts) are excluded.
        - Answers with NO checkable financial claims are NOT marked grounded
          (avoids vacuous truth).
    """
    evidence_numbers: Set[float] = set()
    if evidence_store:
        for ev in evidence_store.list_evidence():
            evidence_numbers |= {round(v, 2) for v in _walk_numbers(ev.content)}

    answer_numbers = extract_currency_and_percentages(answer)
    if not answer_numbers:
        return False, [], evidence_numbers

    is_grounded = all(
        any(abs(n - e) <= max(0.02, abs(e) * 0.02) for e in evidence_numbers)
        for n in answer_numbers
    )
    return is_grounded, answer_numbers, evidence_numbers
