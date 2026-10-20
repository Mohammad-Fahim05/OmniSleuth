"""Programmatic claim validation engine enforcing deterministic evidence grounding."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from omnisleuth.core.logging import get_logger
from omnisleuth.evidence.models import Claim, ClaimStatus, EvidenceItem, EvidenceType
from omnisleuth.evidence.store import EvidenceStore

logger = get_logger("evidence.validator")

# Prohibited definite causal phrases (Rule 6)
UNSUPPORTED_CAUSAL_PATTERNS = [
    (r"\bcaused\b", "contributed to"),
    (r"\bwas caused by\b", "was associated with"),
    (r"\bdirectly caused\b", "strongly contributed to"),
    (r"\bthe sole reason\b", "a primary observed factor"),
    (r"\bproves that\b", "suggests that"),
    (r"\bdefinitely resulted from\b", "likely coincided with"),
    (r"\bguaranteed cause\b", "observed contributor"),
]

# Regex to extract numbers (including percentages, negative numbers, decimals)
NUMERIC_REGEX = re.compile(r"[-+]?\d*\.?\d+(?:%)?")


class ValidationResult(BaseModel):
    """Detailed programmatic validation outcome for an individual claim."""

    claim_id: str
    original_statement: str
    status: ClaimStatus
    matched_evidence_ids: List[str] = Field(default_factory=list)
    verified_numbers: List[float] = Field(default_factory=list)
    unverified_numbers: List[float] = Field(default_factory=list)
    detected_causal_issues: List[str] = Field(default_factory=list)
    calibrated_statement: Optional[str] = None
    validation_explanation: str


class ClaimValidator:
    """Deterministic validator comparing text claims against evidence store records."""

    def __init__(self, numerical_tolerance_pct: float = 1.0):
        """Initialize validator.

        Args:
            numerical_tolerance_pct: Relative percentage tolerance (e.g. 1.0 = ±1.0%) for numerical matching.
        """
        self.numerical_tolerance_pct = numerical_tolerance_pct

    def extract_numbers_from_text(self, text: str) -> List[float]:
        """Extract all floating-point and percentage values from text."""
        # 1. Strip date patterns like YYYY-MM-DD or YYYY-MM so calendar years/months aren't treated as metric claims
        cleaned = re.sub(r"\b\d{4}-\d{2}(?:-\d{2})?\b", "", text)
        # 2. Remove comma thousands separators between digits (e.g. 1,000 or $398,325.40)
        cleaned = re.sub(r"(?<=\d),(?=\d{3}(?:\b|[^\d]))", "", cleaned)
        # 3. Strip numbered list prefixes (e.g. '1. North', '2. West')
        cleaned = re.sub(r"(?:^|(?<=[\s,;(]))\d+\.\s+(?=[A-Za-z])", "", cleaned)
        # 4. Strip group count descriptors (e.g. '(4 groups)', 'showing top 10 of 15 groups')
        cleaned = re.sub(r"\b(?:\d+\s+groups?|showing top \d+ of \d+ groups)\b", "", cleaned, flags=re.IGNORECASE)

        matches = NUMERIC_REGEX.findall(cleaned)
        numbers: List[float] = []
        for m in matches:
            clean_str = m.rstrip("%")
            try:
                val = float(clean_str)
                numbers.append(val)
            except ValueError:
                continue
        return numbers

    def check_unsupported_causality(self, text: str) -> Tuple[bool, List[str], str]:
        """Detect prohibited definite causal language and generate calibrated rewrite.

        Returns:
            Tuple of (has_causal_issue, detected_keywords, calibrated_statement)
        """
        detected: List[str] = []
        calibrated = text

        for pattern, replacement in UNSUPPORTED_CAUSAL_PATTERNS:
            if re.search(pattern, calibrated, re.IGNORECASE):
                matched = re.findall(pattern, calibrated, re.IGNORECASE)
                detected.extend(matched)
                calibrated = re.sub(pattern, replacement, calibrated, flags=re.IGNORECASE)

        has_issue = len(detected) > 0
        return has_issue, detected, calibrated

    def _collect_all_numbers_from_evidence(self, evidence_items: List[EvidenceItem]) -> Set[float]:
        """Recursively harvest all numbers from stored evidence content."""
        numbers: Set[float] = set()

        def extract_recursive(data: Any):
            if isinstance(data, (int, float)) and not isinstance(data, bool):
                numbers.add(round(float(data), 2))
                numbers.add(round(abs(float(data)), 2))
            elif isinstance(data, dict):
                for v in data.values():
                    extract_recursive(v)
            elif isinstance(data, list):
                for item in data:
                    extract_recursive(item)

        for ev in evidence_items:
            extract_recursive(ev.content)

        return numbers

    def validate_claim(self, claim: Claim, store: EvidenceStore) -> ValidationResult:
        """Programmatically validate a claim against the EvidenceStore."""
        statement = claim.statement
        claim_numbers = self.extract_numbers_from_text(statement)
        if not claim_numbers and claim.extracted_numbers:
            claim_numbers = list(claim.extracted_numbers)
        elif not claim.extracted_numbers and claim_numbers:
            claim.extracted_numbers = list(claim_numbers)
        has_causal, causal_words, calibrated_text = self.check_unsupported_causality(statement)

        evidence_items = store.list_evidence()
        if not evidence_items and claim_numbers:
            return ValidationResult(
                claim_id=claim.claim_id,
                original_statement=statement,
                status=ClaimStatus.INCONCLUSIVE,
                detected_causal_issues=causal_words,
                calibrated_statement=calibrated_text if has_causal else None,
                validation_explanation="No evidence items available in store to verify numerical claims.",
            )

        evidence_numbers = self._collect_all_numbers_from_evidence(evidence_items)

        verified_nums: List[float] = []
        unverified_nums: List[float] = []
        matched_ev_ids: Set[str] = set()

        for c_num in claim_numbers:
            matched = False
            abs_c_num = abs(c_num)

            for ev_num in evidence_numbers:
                # Match with tolerance
                abs_ev_num = abs(ev_num)
                delta = abs(abs_c_num - abs_ev_num)
                rel_diff = (delta / abs_ev_num * 100.0) if abs_ev_num > 0 else delta

                if delta <= 0.05 or rel_diff <= self.numerical_tolerance_pct:
                    matched = True
                    verified_nums.append(c_num)
                    break

            if not matched:
                unverified_nums.append(c_num)

        # Find matching evidence items
        for ev in evidence_items:
            ev_nums = self._collect_all_numbers_from_evidence([ev])
            for v_num in verified_nums:
                if any(abs(abs(v_num) - abs(en)) <= 0.05 for en in ev_nums):
                    matched_ev_ids.add(ev.evidence_id)

        # Determine validation status
        if claim_numbers and unverified_nums:
            status = ClaimStatus.UNSUPPORTED
            explanation = f"Numbers {unverified_nums} could not be verified against deterministic evidence."
        elif claim_numbers and not unverified_nums:
            status = ClaimStatus.SUPPORTED
            explanation = f"All numbers {verified_nums} verified against evidence {list(matched_ev_ids)}."
            if has_causal:
                explanation += f" Warning: Causal phrases {causal_words} detected and calibrated."
        else:
            # Text-only claim with no numbers
            status = ClaimStatus.SUPPORTED if not has_causal else ClaimStatus.INCONCLUSIVE
            explanation = "Qualitative statement with no specific numerical assertions."
            if has_causal:
                explanation += f" Unsupported causal phrases detected: {causal_words}."

        # Update the claim object
        claim.status = status
        claim.supporting_evidence_ids = list(matched_ev_ids)
        claim.validation_notes = explanation

        return ValidationResult(
            claim_id=claim.claim_id,
            original_statement=statement,
            status=status,
            matched_evidence_ids=list(matched_ev_ids),
            verified_numbers=verified_nums,
            unverified_numbers=unverified_nums,
            detected_causal_issues=causal_words,
            calibrated_statement=calibrated_text if has_causal else statement,
            validation_explanation=explanation,
        )

    def validate_all(self, claims: List[Claim], store: EvidenceStore) -> List[ValidationResult]:
        """Validate a list of claims and return outcomes."""
        return [self.validate_claim(claim, store) for claim in claims]
