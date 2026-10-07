"""Investigation Agent for hypothesis generation, evidence linking, and status evaluation."""

from typing import Any, Dict, List, Optional
import pandas as pd
from omnisleuth.agents.models import AgentTask
from omnisleuth.analytics.engine import DeterministicAnalyticsEngine
from omnisleuth.core.logging import get_logger
from omnisleuth.evidence.models import (
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
    InvestigationNode,
)
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.ingestion.models import DatasetMetadata

logger = get_logger("agents.investigator")


class InvestigationAgent:
    """Specialized agent deriving and evaluating hypotheses against stored evidence."""

    def __init__(self, analytics_engine: Optional[DeterministicAnalyticsEngine] = None):
        self.engine = analytics_engine or DeterministicAnalyticsEngine()

    def generate_and_test_hypotheses(
        self,
        target_metric: str,
        dimensions: List[str],
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        evidence_store: EvidenceStore,
    ) -> List[Hypothesis]:
        """Generate testable hypotheses based on findings and evaluate them against evidence."""
        logger.info(f"InvestigationAgent generating hypotheses for metric '{target_metric}' across {dimensions}")
        hypotheses: List[Hypothesis] = []

        # 1. Check if there is a trend decline/increase in evidence
        trend_ev = [ev for ev in evidence_store.list_evidence() if ev.evidence_type == EvidenceType.DETERMINISTIC_CALC and "overall_delta" in ev.content]
        has_trend = len(trend_ev) > 0
        trend_delta = trend_ev[0].content.get("overall_delta", 0.0) if has_trend else 0.0

        # Generate Dimension Hypotheses (e.g. "Category/Region contributed most to change")
        for dim in dimensions:
            hyp = Hypothesis(
                statement=f"Changes in specific segments of '{dim}' contributed substantially to observed variance in '{target_metric}'.",
                rationale=f"Categorical segment contribution breakdown for {dim}.",
                tested_with_tools=["compare_segments", "segment_contribution"],
            )

            # Look for segment evidence
            seg_ev = [
                ev for ev in evidence_store.list_evidence()
                if dim in ev.location or (isinstance(ev.content.get("segment_column"), str) and ev.content.get("segment_column") == dim)
            ]

            if seg_ev:
                # Check actual segment findings in content
                top_contributor_share = 0.0
                has_significant_variance = False
                for ev in seg_ev:
                    content = ev.content or {}
                    contribs = content.get("contributions", [])
                    if isinstance(contribs, list) and len(contribs) > 0:
                        shares = [abs(c.get("percentage_contribution", 0.0)) for c in contribs if isinstance(c, dict)]
                        if shares and max(shares) > 15.0:
                            top_contributor_share = max(shares)
                            has_significant_variance = True
                    elif "overall_delta" in content:
                        has_significant_variance = True

                if has_significant_variance:
                    hyp.status = HypothesisStatus.SUPPORTED
                    hyp.confidence = min(0.95, 0.70 + (top_contributor_share / 100.0) * 0.25)
                    hyp.findings_summary = f"Supported: Segment variance in {dim} shows significant contribution (peak segment impact: {top_contributor_share:.1f}%)."
                else:
                    hyp.status = HypothesisStatus.REJECTED
                    hyp.confidence = 0.80
                    hyp.findings_summary = f"Rejected: Segment differences in {dim} did not show substantial variance."

                hyp.evidence_ids = [e.evidence_id for e in seg_ev]
            else:
                hyp.status = HypothesisStatus.INCONCLUSIVE
                hyp.confidence = 0.30
                hyp.findings_summary = f"Inconclusive: No segment analytical evidence was computed for dimension '{dim}'."

            evidence_store.add_hypothesis(hyp)
            hypotheses.append(hyp)

        # Generate Anomaly Hypothesis
        anom_ev = [ev for ev in evidence_store.list_evidence() if ev.evidence_type == EvidenceType.ML_ANOMALY]
        if anom_ev:
            anom_count = anom_ev[0].content.get("anomaly_count", 0)
            hyp_anom = Hypothesis(
                statement=f"Statistical anomaly events in '{target_metric}' affected overall performance.",
                rationale="Univariate/multivariate outlier detection analysis.",
                status=HypothesisStatus.SUPPORTED if anom_count > 0 else HypothesisStatus.REJECTED,
                confidence=0.85,
                evidence_ids=[anom_ev[0].evidence_id],
                tested_with_tools=["detect_anomalies"],
                findings_summary=f"Detected {anom_count} anomalous data points.",
            )
            evidence_store.add_hypothesis(hyp_anom)
            hypotheses.append(hyp_anom)

        # Generate Operational RAG Context Hypothesis
        rag_ev = [ev for ev in evidence_store.list_evidence() if ev.evidence_type == EvidenceType.RAG_DOCUMENT]
        if rag_ev:
            hyp_rag = Hypothesis(
                statement="External operational factors or business events documented in company reports provide context for the metrics.",
                rationale="Retrieved quarterly business reports and operational review documents.",
                status=HypothesisStatus.SUPPORTED,
                confidence=0.90,
                evidence_ids=[e.evidence_id for e in rag_ev],
                tested_with_tools=["retrieval_agent"],
                findings_summary=f"Found {len(rag_ev)} relevant business document citations.",
            )
            evidence_store.add_hypothesis(hyp_rag)
            hypotheses.append(hyp_rag)

        # Update Investigation Tree in Evidence Store
        for hyp in hypotheses:
            node = InvestigationNode(
                node_type="hypothesis",
                title=hyp.statement[:80] + "...",
                details={
                    "hypothesis_id": hyp.hypothesis_id,
                    "status": hyp.status.value,
                    "confidence": hyp.confidence,
                    "evidence_ids": hyp.evidence_ids,
                },
            )
            evidence_store.add_node(node)

        return hypotheses

    def execute_task(
        self,
        task: AgentTask,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        evidence_store: EvidenceStore,
    ) -> AgentTask:
        """Execute investigation task."""
        target_metric = task.parameters.get("target_metric", "revenue")
        dimensions = task.parameters.get("candidate_dimensions", metadata.categorical_columns)

        hypotheses = self.generate_and_test_hypotheses(
            target_metric=target_metric,
            dimensions=dimensions,
            df=df,
            metadata=metadata,
            evidence_store=evidence_store,
        )

        task.status = "completed"
        task.result = [h.model_dump() for h in hypotheses]
        task.evidence_ids = [h.hypothesis_id for h in hypotheses]
        return task
