"""Deterministic output guardrails enforcing numerical grounding, evidence citations, causal calibration, and secret redaction."""

import re
from typing import Any, Dict, List, Optional, Tuple
from omnisleuth.core.logging import get_logger
from omnisleuth.evidence.models import Claim, ClaimStatus, EvidenceItem
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.evidence.validator import ClaimValidator, ValidationResult
from omnisleuth.guardrails.models import (
    GuardedAnswer,
    GuardrailCategory,
    GuardrailDecision,
    GuardrailResult,
    GuardrailSeverity,
)
from omnisleuth.guardrails.patterns import (
    SECRET_LEAKAGE_PATTERNS,
    UNSUPPORTED_CAUSAL_PHRASES,
)

logger = get_logger("guardrails.output")


class OutputGuardrailEngine:
    """Post-generation output guardrail validator enforcing evidence grounding and confidentiality."""

    def __init__(self, claim_validator: Optional[ClaimValidator] = None):
        self.claim_validator = claim_validator or ClaimValidator()

    def check_and_redact_secrets(self, text: str) -> Tuple[bool, str, List[str]]:
        """Detect and redact accidentally leaked credentials, tokens, or private keys."""
        detected: List[str] = []
        sanitized = text

        for rule_name, pattern in SECRET_LEAKAGE_PATTERNS:
            if pattern.search(sanitized):
                detected.append(rule_name)
                sanitized = pattern.sub("[REDACTED_CONFIDENTIAL_SECRET]", sanitized)

        has_leakage = len(detected) > 0
        if has_leakage:
            logger.warning(f"Output Guardrail: Redacted secret leakage matching rules: {detected}")

        return has_leakage, sanitized, detected

    def check_and_calibrate_causality(self, text: str) -> Tuple[bool, str, List[str]]:
        """Calibrate unsupported definitive causal terms into associative and contribution phrasing."""
        detected: List[str] = []
        calibrated = text

        for pattern_str, replacement in UNSUPPORTED_CAUSAL_PHRASES:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(calibrated):
                matched = pattern.findall(calibrated)
                detected.extend(matched)
                calibrated = pattern.sub(replacement, calibrated)

        has_causality_issue = len(detected) > 0
        return has_causality_issue, calibrated, detected

    def validate_numerical_claims(
        self,
        claims: List[Claim],
        evidence_store: EvidenceStore,
    ) -> Tuple[List[ValidationResult], bool]:
        """Validate all narrative numerical claims against deterministic evidence store records."""
        validation_results = self.claim_validator.validate_all(claims, evidence_store)
        has_unsupported = any(r.status == ClaimStatus.UNSUPPORTED for r in validation_results)
        return validation_results, has_unsupported

    def structure_final_response(
        self,
        raw_answer: str,
        evidence_store: EvidenceStore,
        validation_results: Optional[List[ValidationResult]] = None,
        confidence: Optional[float] = None,
    ) -> GuardedAnswer:
        """Construct verified, structured GuardedAnswer complying with required section schema."""
        ev_items = evidence_store.list_evidence()
        val_results = validation_results or []

        # 1. Redact any secrets
        _, clean_narrative, _ = self.check_and_redact_secrets(raw_answer)

        # 2. Calibrate causal language
        _, calibrated_narrative, _ = self.check_and_calibrate_causality(clean_narrative)

        # 3. Extract calculations and evidence references
        calc_lines: List[str] = []
        ev_refs: List[Dict[str, Any]] = []

        for ev in ev_items:
            ev_refs.append({
                "evidence_id": ev.evidence_id,
                "type": ev.evidence_type.value,
                "source": ev.source_id,
                "location": ev.location,
                "confidence": ev.confidence,
            })
            if "formula" in ev.content:
                calc_lines.append(f"{ev.content.get('metric', 'Metric')}: {ev.content['formula']}")
            elif "overall_delta" in ev.content:
                calc_lines.append(f"Delta: {ev.content['overall_delta']} ({ev.content.get('overall_change_pct', 0.0)}%)")

        # 4. Synthesize limitations
        limitations = [
            "Calculations are strictly deterministic derived from provided dataset records.",
            "Document context provides operational background, not definitive statistical causality.",
        ]

        # Calculate composite confidence
        if confidence is None:
            if ev_items:
                avg_conf = sum(e.confidence for e in ev_items) / len(ev_items)
                confidence = round(avg_conf, 2)
            else:
                confidence = 0.5

        return GuardedAnswer(
            answer=calibrated_narrative,
            evidence=ev_refs,
            calculation=calc_lines,
            interpretation="Findings describe mathematical attributions and observed document associations.",
            confidence=confidence,
            limitations=limitations,
        )

    def validate_output(
        self,
        answer_text: str,
        evidence_store: EvidenceStore,
        claims: Optional[List[Claim]] = None,
    ) -> Tuple[GuardrailResult, GuardedAnswer]:
        """Execute full output guardrail pipeline: Secrets -> Causality -> Claims -> Schema."""
        # 1. Secret Leakage
        has_secrets, sanitized_text, secret_rules = self.check_and_redact_secrets(answer_text)

        # 2. Causal Phrasing Calibration
        has_causal, calibrated_text, causal_terms = self.check_and_calibrate_causality(sanitized_text)

        # 3. Claim Validation
        val_results: List[ValidationResult] = []
        has_unsupported = False
        if claims:
            val_results, has_unsupported = self.validate_numerical_claims(claims, evidence_store)

        # 4. Check Evidence Sufficiency
        ev_items = evidence_store.list_evidence()
        if not ev_items and claims:
            result = GuardrailResult(
                allowed=False,
                decision=GuardrailDecision.BLOCK,
                category=GuardrailCategory.MISSING_EVIDENCE,
                severity=GuardrailSeverity.HIGH,
                reason="Generated response references claims but EvidenceStore contains no supporting items.",
                user_message="Investigation concluded with inconclusive evidence; no grounded findings could be verified.",
            )
            guarded_ans = self.structure_final_response(
                raw_answer="Inconclusive: No supporting evidence items found in dataset records.",
                evidence_store=evidence_store,
                confidence=0.0,
            )
            guarded_ans.guardrail_result = result
            return result, guarded_ans

        # 5. Handle Unsupported Claims
        if has_unsupported:
            result = GuardrailResult(
                allowed=True,
                decision=GuardrailDecision.REWRITE,
                category=GuardrailCategory.UNSUPPORTED_CLAIM,
                severity=GuardrailSeverity.MEDIUM,
                reason="One or more generated numerical claims were unverified against evidence and flagged.",
                calibrated_content=calibrated_text,
            )
        elif has_secrets:
            result = GuardrailResult(
                allowed=True,
                decision=GuardrailDecision.REDACT,
                category=GuardrailCategory.SECRET_LEAKAGE,
                severity=GuardrailSeverity.HIGH,
                reason=f"Secret credentials detected and redacted ({', '.join(secret_rules)}).",
                redacted_content=sanitized_text,
            )
        elif has_causal:
            result = GuardrailResult(
                allowed=True,
                decision=GuardrailDecision.REWRITE,
                category=GuardrailCategory.UNSUPPORTED_CAUSALITY,
                severity=GuardrailSeverity.LOW,
                reason=f"Definitive causal phrases ({', '.join(causal_terms)}) calibrated to associative terms.",
                calibrated_content=calibrated_text,
            )
        else:
            result = GuardrailResult(
                allowed=True,
                decision=GuardrailDecision.ALLOW,
                category=GuardrailCategory.SAFE,
                severity=GuardrailSeverity.LOW,
                reason="Output verified: grounded in evidence, calibrated language, no secrets.",
            )

        guarded_ans = self.structure_final_response(
            raw_answer=calibrated_text,
            evidence_store=evidence_store,
            validation_results=val_results,
        )
        guarded_ans.guardrail_result = result
        return result, guarded_ans
