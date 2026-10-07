"""Graph nodes for orchestrating deterministic data investigations."""

import re
from typing import Any, Dict, List, Optional
import pandas as pd
from omnisleuth.analytics.models import AggregationFunction
from omnisleuth.core.logging import get_logger
from omnisleuth.evidence.models import Claim, ClaimStatus, InvestigationNode
from omnisleuth.evidence.validator import ClaimValidator
from omnisleuth.guardrails import InputGuardrailEngine, OutputGuardrailEngine
from omnisleuth.tools import get_default_tool_registry
from omnisleuth.workflow.state import InvestigationState, PlannedToolCall

logger = get_logger("workflow.nodes")
_registry = get_default_tool_registry()
_validator = ClaimValidator()
_input_guard = InputGuardrailEngine()
_output_guard = OutputGuardrailEngine(claim_validator=_validator)


def input_guardrails_node(state: InvestigationState) -> Dict[str, Any]:
    """Execute pre-planning input guardrails."""
    q = state.get("question", "")
    logger.info(f"Running input guardrails on: '{q}'")
    res = _input_guard.validate_input(q)

    if not res.allowed:
        logger.warning(f"Workflow stopped by Input Guardrails: {res.reason}")
        return {
            "input_guardrail_result": res,
            "final_answer": f"⚠️ **Request Blocked by Security Guardrails**: {res.user_message}",
            "status": "blocked",
            "error": res.reason,
        }

    return {
        "input_guardrail_result": res,
        "status": "input_validated",
    }


def classify_question_node(state: InvestigationState) -> Dict[str, Any]:
    """Classify the business question into analytical intent and extract candidate metrics/dimensions."""
    if state.get("status") == "blocked":
        return {}

    q = state["question"].lower()
    meta = state["dataset_metadata"]

    logger.info(f"Classifying question: '{state['question']}'")

    # Match metric column
    target_metric = None
    for col in meta.numeric_columns:
        if col.lower() in q or (col.lower().replace("_", " ") in q):
            target_metric = col
            break
    if not target_metric and meta.numeric_columns:
        # Default to first numeric column (e.g. 'revenue' or 'sales' or 'amount')
        for pref in ["revenue", "sales", "amount", "total"]:
            for col in meta.numeric_columns:
                if pref in col.lower():
                    target_metric = col
                    break
            if target_metric:
                break
        if not target_metric:
            target_metric = meta.numeric_columns[0]

    # Match categorical dimension
    target_dim = None
    for col in meta.categorical_columns:
        if col.lower() in q or (col.lower().replace("_", " ") in q):
            target_dim = col
            break
    if not target_dim and meta.categorical_columns:
        target_dim = meta.categorical_columns[0]

    # Intent determination
    if any(w in q for w in ["why", "decline", "drop", "fell", "decrease", "increase", "trend", "change", "over time"]):
        intent = "trend_investigation"
    elif any(w in q for w in ["compare", "breakdown", "by region", "by category", "top", "distribution"]):
        intent = "segment_comparison"
    elif any(w in q for w in ["anomaly", "outlier", "unusual", "spike"]):
        intent = "anomaly_investigation"
    else:
        intent = "summary_investigation"

    return {
        "question_intent": intent,
        "target_metric": target_metric,
        "target_dimension": target_dim,
        "status": "question_classified",
    }


def inspect_dataset_node(state: InvestigationState) -> Dict[str, Any]:
    """Inspect dataset columns and ensure date/numeric requirements are met."""
    if state.get("status") == "blocked":
        return {}

    meta = state["dataset_metadata"]
    logger.info(f"Inspecting dataset: {meta.dataset_id} ({meta.row_count} rows, {meta.column_count} cols)")

    # Record root investigation node in evidence store
    root_node = InvestigationNode(
        node_type="finding",
        title=f"Investigation Goal: {state['question']}",
        details={"dataset_id": meta.dataset_id, "intent": state.get("question_intent", "unknown")},
    )
    state["evidence_store"].add_node(root_node)

    return {"status": "dataset_inspected"}


def create_plan_node(state: InvestigationState) -> Dict[str, Any]:
    """Generate a structured, allowlisted tool investigation plan based on intent."""
    if state.get("status") == "blocked":
        return {}
    intent = state.get("question_intent", "summary_investigation")
    meta = state["dataset_metadata"]
    metric = state.get("target_metric") or (meta.numeric_columns[0] if meta.numeric_columns else "value")
    dim = state.get("target_dimension") or (meta.categorical_columns[0] if meta.categorical_columns else None)
    dt_col = meta.datetime_columns[0] if meta.datetime_columns else None

    plan: List[PlannedToolCall] = []
    step = 1

    # 1. Profile dataset
    plan.append(
        PlannedToolCall(
            step=step,
            tool_name="profile_dataset",
            parameters={},
            rationale="Verify dataset quality, missing values, and statistical distributions.",
        )
    )
    step += 1

    # 2. Add trend or comparison tools depending on intent
    if dt_col and intent in ["trend_investigation", "summary_investigation"]:
        plan.append(
            PlannedToolCall(
                step=step,
                tool_name="calculate_trend",
                parameters={"date_column": dt_col, "metric_column": metric, "frequency": "M"},
                rationale=f"Calculate monthly {metric} time-series trend to identify change points.",
            )
        )
        step += 1

    if dim:
        plan.append(
            PlannedToolCall(
                step=step,
                tool_name="compare_segments",
                parameters={"segment_column": dim, "metric_column": metric},
                rationale=f"Compare total {metric} across segments of {dim}.",
            )
        )
        step += 1

    if intent == "anomaly_investigation" or intent == "trend_investigation":
        plan.append(
            PlannedToolCall(
                step=step,
                tool_name="detect_anomalies",
                parameters={"feature_columns": [metric], "method": "iqr"},
                rationale=f"Detect statistical anomalies and outliers in {metric}.",
            )
        )
        step += 1

    return {"plan": plan, "status": "plan_created"}


def validate_plan_node(state: InvestigationState) -> Dict[str, Any]:
    """Validate that all planned tools exist in the allowlist registry and stay within bounds."""
    plan = state.get("plan", [])
    if not plan:
        return {"plan_is_valid": False, "plan_validation_notes": "Plan is empty.", "status": "plan_rejected"}

    if len(plan) > 10:  # Rule 7: Bounded limit
        return {
            "plan_is_valid": False,
            "plan_validation_notes": f"Plan exceeds maximum allowed tool calls limit of 10 (got {len(plan)}).",
            "status": "plan_rejected",
        }

    for step in plan:
        tool_name = step["tool_name"]
        try:
            _registry.get(tool_name)
        except Exception as exc:
            return {
                "plan_is_valid": False,
                "plan_validation_notes": f"Disallowed tool '{tool_name}' in plan: {str(exc)}",
                "status": "plan_rejected",
            }

    return {"plan_is_valid": True, "plan_validation_notes": "All planned tools allowlisted and valid.", "status": "plan_validated"}


def execute_tools_node(state: InvestigationState) -> Dict[str, Any]:
    """Execute the allowlisted plan sequentially, recording evidence and audit records."""
    if state.get("status") == "blocked":
        return {}

    if not state.get("plan_is_valid", False):
        return {"status": "execution_skipped_invalid_plan"}

    plan = state["plan"]
    df = state["df"]
    meta = state["dataset_metadata"]
    store = state["evidence_store"]
    execution_ids: List[str] = []

    for step in plan:
        tool_name = step["tool_name"]
        params = step["parameters"]
        tool = _registry.get(tool_name)
        output = tool.run(params, df, meta)
        exec_record = store.record_tool_execution(tool_name, params, output)
        execution_ids.append(exec_record.execution_id)

    return {"tool_executions": execution_ids, "status": "tools_executed"}


def validate_claims_node(state: InvestigationState) -> Dict[str, Any]:
    """Extract statements from evidence and run programmatic claim validation."""
    if state.get("status") == "blocked":
        return {}

    store = state["evidence_store"]
    claims: List[Claim] = []

    # Generate grounded claims directly from minted evidence items
    for ev in store.list_evidence():
        if ev.content.get("overall_delta") is not None:
            delta = ev.content["overall_delta"]
            pct = ev.content.get("overall_change_pct", 0.0)
            metric = ev.content.get("metric", "metric")
            claims.append(
                Claim(
                    statement=f"Observed total {metric} change across the period was {delta} ({pct}%).",
                    extracted_numbers=[delta, pct],
                )
            )
        elif ev.content.get("grand_total") is not None:
            tot = ev.content["grand_total"]
            claims.append(
                Claim(
                    statement=f"Total aggregate value was {tot}.",
                    extracted_numbers=[tot],
                )
            )

    validation_results = _validator.validate_all(claims, store)

    return {
        "claims": claims,
        "validation_results": validation_results,
        "status": "claims_validated",
    }


def generate_answer_node(state: InvestigationState) -> Dict[str, Any]:
    """Synthesize evidence-grounded final explanation with uncertainty calibration and output guardrails."""
    if state.get("status") == "blocked":
        return {
            "final_answer": state.get("final_answer", ""),
            "status": "blocked",
        }
    store = state["evidence_store"]
    val_results = state.get("validation_results", [])
    ev_items = store.list_evidence()
    claims = state.get("claims", [])

    lines = [
        f"### Investigation Summary for: \"{state['question']}\"",
        "",
        f"**Dataset ID:** `{state['dataset_metadata'].dataset_id}` | **Rows:** {state['dataset_metadata'].row_count}",
        "",
        "#### Verified Analytical Findings:",
    ]

    for vr in val_results:
        if vr.status == ClaimStatus.SUPPORTED:
            lines.append(f"- ✅ **Supported**: {vr.calibrated_statement or vr.original_statement}")
        elif vr.status == ClaimStatus.UNSUPPORTED:
            lines.append(f"- ❌ **Unsupported (Flagged)**: {vr.original_statement} ({vr.validation_explanation})")

    lines.append("")
    lines.append("#### Evidence Provenance:")
    for ev in ev_items:
        lines.append(f"- Evidence ID `[{ev.evidence_id}]` ({ev.evidence_type.value}) at `{ev.location}` (Confidence: {ev.confidence*100:.0f}%)")

    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> Findings describe observed associations and mathematical contributions; no unsupported causal claims are made.")

    raw_text = "\n".join(lines)

    # Execute Output Guardrails (Secret Redaction, Causal Language Calibration, Schema Packaging)
    guard_res, guarded_answer = _output_guard.validate_output(
        answer_text=raw_text,
        evidence_store=store,
        claims=claims,
    )

    return {
        "final_answer": guarded_answer.answer,
        "output_guardrail_result": guard_res,
        "guarded_answer": guarded_answer,
        "status": "completed",
    }
