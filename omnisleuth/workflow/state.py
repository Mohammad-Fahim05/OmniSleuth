"""State definition for LangGraph investigation workflows."""

from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd
from omnisleuth.evidence.models import Claim
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.evidence.validator import ValidationResult
from omnisleuth.ingestion.models import DatasetMetadata


class PlannedToolCall(TypedDict):
    """An individual tool call step in the structured investigation plan."""

    step: int
    tool_name: str
    parameters: Dict[str, Any]
    rationale: str


class InvestigationState(TypedDict):
    """Shared state passed between LangGraph nodes during an investigation."""

    # Inputs
    question: str
    dataset_metadata: DatasetMetadata
    df: pd.DataFrame
    evidence_store: EvidenceStore

    # Step 1: Question Understanding
    question_intent: str
    target_metric: Optional[str]
    target_dimension: Optional[str]
    target_time_period: Optional[str]

    # Step 2 & 3: Planning & Validation
    plan: List[PlannedToolCall]
    plan_is_valid: bool
    plan_validation_notes: Optional[str]

    # Step 4: Tool Execution
    tool_executions: List[str]  # Execution IDs

    # Step 5: Claim Extraction & Validation
    claims: List[Claim]
    validation_results: List[ValidationResult]

    # Guardrails (Phase 15)
    input_guardrail_result: Optional[Any]
    output_guardrail_result: Optional[Any]
    guarded_answer: Optional[Any]

    # Synthesis
    final_answer: str
    investigation_tree: Dict[str, Any]
    status: str
    error: Optional[str]
