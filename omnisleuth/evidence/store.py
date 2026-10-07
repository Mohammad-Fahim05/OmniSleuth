"""Evidence store for tracking tool executions and verifiable analytical evidence."""

from typing import Any, Dict, List, Optional
from omnisleuth.analytics.models import (
    AggregationResult,
    MetricDecompositionResult,
    SegmentComparisonResult,
    SegmentContributionResult,
    TrendResult,
)
from omnisleuth.core.logging import get_logger
from omnisleuth.evidence.models import (
    Claim,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    InvestigationNode,
    ToolExecution,
)
from omnisleuth.ml.models import AnomalyDetectionResult
from omnisleuth.tools.base import BaseToolOutput

logger = get_logger("evidence")


class EvidenceStore:
    """Central provenance store for an investigation."""

    def __init__(self, dataset_id: str, dataset_version: str):
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        self.executions: Dict[str, ToolExecution] = {}
        self.evidence_items: Dict[str, EvidenceItem] = {}
        self.claims: Dict[str, Claim] = {}
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.nodes: Dict[str, InvestigationNode] = {}

    def record_tool_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        output: BaseToolOutput,
    ) -> ToolExecution:
        """Record an execution and mint corresponding EvidenceItems."""
        execution = ToolExecution(
            tool_name=tool_name,
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            parameters=parameters,
            result=output.data.model_dump() if hasattr(output.data, "model_dump") else output.data,
            success=output.success,
            error=output.error,
            execution_time_ms=output.execution_time_ms,
        )
        self.executions[execution.execution_id] = execution

        if output.success and output.data is not None:
            self._mint_evidence_from_tool_data(execution, output.data)

        return execution

    def _mint_evidence_from_tool_data(self, execution: ToolExecution, data: Any) -> None:
        """Extract structured evidence facts from deterministic analytics or ML results."""
        if isinstance(data, TrendResult):
            # Record trend evidence
            ev = EvidenceItem(
                evidence_type=EvidenceType.DETERMINISTIC_CALC,
                source_id=execution.execution_id,
                location=f"Trend({data.date_column} -> {data.metric_column})",
                content={
                    "metric": data.metric_column,
                    "start_period": data.start_period,
                    "end_period": data.end_period,
                    "start_value": data.start_value,
                    "end_value": data.end_value,
                    "overall_delta": data.overall_delta,
                    "overall_change_pct": data.overall_change_pct,
                    "time_series": [p.model_dump() for p in data.time_series],
                },
                confidence=1.0,
                related_tool_execution_id=execution.execution_id,
            )
            self.add_evidence(ev)

        elif isinstance(data, SegmentContributionResult):
            ev = EvidenceItem(
                evidence_type=EvidenceType.DETERMINISTIC_CALC,
                source_id=execution.execution_id,
                location=f"Contribution({data.segment_column} -> {data.metric_column})",
                content={
                    "segment_column": data.segment_column,
                    "metric": data.metric_column,
                    "total_delta": data.total_delta,
                    "total_change_pct": data.total_change_pct,
                    "contributions": [c.model_dump() for c in data.contributions],
                },
                confidence=1.0,
                related_tool_execution_id=execution.execution_id,
            )
            self.add_evidence(ev)

        elif isinstance(data, MetricDecompositionResult):
            ev = EvidenceItem(
                evidence_type=EvidenceType.DETERMINISTIC_CALC,
                source_id=execution.execution_id,
                location=f"Decomposition({data.target_metric})",
                content={
                    "target_metric": data.target_metric,
                    "target_delta": data.target_delta,
                    "target_change_pct": data.target_change_pct,
                    "factor_effects": [f.model_dump() for f in data.factor_effects],
                },
                confidence=1.0,
                related_tool_execution_id=execution.execution_id,
            )
            self.add_evidence(ev)

        elif isinstance(data, SegmentComparisonResult):
            ev = EvidenceItem(
                evidence_type=EvidenceType.DETERMINISTIC_CALC,
                source_id=execution.execution_id,
                location=f"Comparison({data.segment_column} -> {data.metric_column})",
                content={
                    "segment_column": data.segment_column,
                    "metric_column": data.metric_column,
                    "total_metric_value": data.total_metric_value,
                    "segments": [s.model_dump() for s in data.segments],
                },
                confidence=1.0,
                related_tool_execution_id=execution.execution_id,
            )
            self.add_evidence(ev)

        elif isinstance(data, AggregationResult):
            ev = EvidenceItem(
                evidence_type=EvidenceType.DETERMINISTIC_CALC,
                source_id=execution.execution_id,
                location=f"Aggregation({data.metric_column})",
                content={
                    "metric": data.metric_column,
                    "agg_func": data.agg_func.value,
                    "group_by": data.group_by_columns,
                    "grand_total": data.grand_total,
                    "records": data.records,
                },
                confidence=1.0,
                related_tool_execution_id=execution.execution_id,
            )
            self.add_evidence(ev)

        elif isinstance(data, AnomalyDetectionResult):
            ev = EvidenceItem(
                evidence_type=EvidenceType.ML_ANOMALY,
                source_id=execution.execution_id,
                location=f"Anomalies({','.join(data.feature_columns)})",
                content={
                    "method": data.method.value,
                    "anomaly_count": data.anomaly_count,
                    "anomaly_percentage": data.anomaly_percentage,
                    "anomalies": [a.model_dump() for a in data.anomalies],
                },
                confidence=0.9,
                related_tool_execution_id=execution.execution_id,
            )
            self.add_evidence(ev)

    def add_evidence(self, item: EvidenceItem) -> None:
        """Register a discrete evidence item."""
        self.evidence_items[item.evidence_id] = item
        logger.debug(f"Added evidence [{item.evidence_type.value}] ID: {item.evidence_id}")

    def get_evidence(self, evidence_id: str) -> Optional[EvidenceItem]:
        """Retrieve evidence item by ID."""
        return self.evidence_items.get(evidence_id)

    def list_evidence(self, evidence_type: Optional[EvidenceType] = None) -> List[EvidenceItem]:
        """List all evidence items, optionally filtered by type."""
        items = list(self.evidence_items.values())
        if evidence_type:
            items = [item for item in items if item.evidence_type == evidence_type]
        return items

    def add_hypothesis(self, hyp: Hypothesis) -> None:
        """Add or update an analytical hypothesis."""
        self.hypotheses[hyp.hypothesis_id] = hyp

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        return self.hypotheses.get(hypothesis_id)

    def add_claim(self, claim: Claim) -> None:
        """Add a factual/numerical claim."""
        self.claims[claim.claim_id] = claim

    def add_node(self, node: InvestigationNode) -> None:
        """Add a tree node and link to parent if specified."""
        self.nodes[node.node_id] = node
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.node_id not in parent.children_ids:
                parent.children_ids.append(node.node_id)
