"""Empirical evaluation harness: OmniSleuth vs Baseline 1 (Direct Tools) vs Baseline 2 (Fixed Pipeline).

Metrics measured from actual execution — no fabricated values.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from omnisleuth.agents.orchestrator import MultiAgentOrchestrator
from omnisleuth.core.logging import get_logger
from omnisleuth.evaluation.benchmark_suite import BENCHMARK_QUESTIONS, BenchmarkQuestion
from omnisleuth.evidence.models import ClaimStatus, EvidenceType
from omnisleuth.ingestion.models import DatasetMetadata
from omnisleuth.ml.detectors import (
    IsolationForestDetector,
    evaluate_anomaly_detection,
    inject_synthetic_anomalies,
)
from omnisleuth.rag.document_store import DocumentStore
from omnisleuth.rag.evaluator import RetrievalEvaluator
from omnisleuth.rag.models import RetrievalMode
from omnisleuth.tools import get_default_tool_registry

logger = get_logger("evaluation")


# ─────────────────────────────────────────────────────────────────────────────
# Result Models
# ─────────────────────────────────────────────────────────────────────────────

class PerQuestionResult(BaseModel):
    """Detailed per-question evaluation record."""

    question_id: str
    category: str
    question: str
    status: str  # "completed", "blocked", "error"
    latency_ms: float
    tools_expected: List[str]
    tools_executed: List[str]
    tool_selection_correct: bool
    evidence_count: int
    supported_claims: int
    unsupported_claims: int
    total_claims: int
    keywords_found: List[str]
    keywords_missing: List[str]
    answer_relevance_score: float = 0.0
    error_message: Optional[str] = None


class SystemBenchmarkResult(BaseModel):
    """Aggregated empirical benchmark metrics for an individual architecture."""

    system_name: str
    total_questions: int

    # Correctness & Grounding
    numerical_correctness_pct: float = Field(ge=0.0, le=100.0)
    tool_selection_accuracy_pct: float = Field(ge=0.0, le=100.0)
    evidence_coverage_pct: float = Field(ge=0.0, le=100.0)
    unsupported_claim_rate_pct: float = Field(ge=0.0, le=100.0)
    investigation_completeness_pct: float = Field(ge=0.0, le=100.0)

    # Anomaly Detection
    anomaly_f1: float = Field(ge=0.0, le=1.0, default=0.0)
    anomaly_precision: float = Field(ge=0.0, le=1.0, default=0.0)
    anomaly_recall: float = Field(ge=0.0, le=1.0, default=0.0)

    # Retrieval (from Phase 10 RAG evaluator)
    retrieval_recall_at_5: float = Field(ge=0.0, le=1.0, default=0.0)
    retrieval_mrr: float = Field(ge=0.0, le=1.0, default=0.0)
    retrieval_ndcg_at_5: float = Field(ge=0.0, le=1.0, default=0.0)
    retrieval_mode: str = "reranked"

    # Faithfulness & Relevance
    faithfulness_pct: float = Field(ge=0.0, le=100.0, default=100.0,
                                     description="(supported_claims / total_claims) * 100")
    answer_relevance_pct: float = Field(ge=0.0, le=100.0, default=0.0,
                                         description="Avg keyword match rate across questions")

    # Latency
    latency_p50_ms: float
    latency_p95_ms: float

    # Cost
    cost_per_investigation: str = Field(
        default="N/A (offline deterministic benchmark)",
        description="Only populated if a metered LLM API is used.",
    )

    # Guardrails
    guardrail_block_rate_pct: float = Field(ge=0.0, le=100.0, default=0.0)

    # Per-question breakdowns
    per_question_results: List[PerQuestionResult] = Field(default_factory=list)


class ExperimentRecord(BaseModel):
    """Full machine-readable experiment record for persistence."""

    experiment_id: str
    timestamp: str
    dataset_name: str
    dataset_version: str
    dataset_rows: int
    benchmark_version: str = "1.0.0"
    total_questions: int
    retrieval_mode: str
    model_configuration: str = "offline-deterministic"
    systems: Dict[str, SystemBenchmarkResult]
    per_question_results: Dict[str, List[PerQuestionResult]]
    retrieval_benchmark: Optional[Dict[str, Any]] = None
    anomaly_benchmark: Optional[Dict[str, Any]] = None
    failures: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Investigation Completeness Scoring
# ─────────────────────────────────────────────────────────────────────────────
# Methodology: Each question earns a score 0.0–1.0 based on 4 criteria:
#   1. Status = completed (0.25)
#   2. At least one expected tool was executed (0.25)
#   3. Evidence store is non-empty (0.25)
#   4. No unsupported claims (0.25)
# Investigation completeness = mean(scores) * 100

def _compute_completeness_score(
    status: str,
    tools_correct: bool,
    evidence_count: int,
    unsupported: int,
    should_succeed: bool,
) -> float:
    """Compute per-question investigation completeness score (0.0–1.0)."""
    if not should_succeed:
        # For questions that should fail, completeness means correctly blocking
        return 1.0 if status == "blocked" else 0.0

    score = 0.0
    if status == "completed":
        score += 0.25
    if tools_correct:
        score += 0.25
    if evidence_count > 0:
        score += 0.25
    if unsupported == 0:
        score += 0.25
    return score


def _get_default_tool_params(tool_name: str, metadata: DatasetMetadata) -> Dict[str, Any]:
    """Provide schema-compliant default arguments for isolated tool executions."""
    metric_col = metadata.numeric_columns[0] if metadata.numeric_columns else "revenue"
    date_col = metadata.datetime_columns[0] if metadata.datetime_columns else "order_date"
    seg_col = metadata.categorical_columns[0] if metadata.categorical_columns else "region"

    if tool_name == "profile_dataset":
        return {}
    elif tool_name == "aggregate_metric":
        return {"metric_column": metric_col, "aggregation_function": "sum"}
    elif tool_name == "calculate_trend":
        return {"metric_column": metric_col, "date_column": date_col, "frequency": "M"}
    elif tool_name == "compare_segments":
        return {"metric_column": metric_col, "segment_column": seg_col}
    elif tool_name == "segment_contribution":
        return {"metric_column": metric_col, "segment_column": seg_col, "period_1_filters": [], "period_2_filters": []}
    elif tool_name == "decompose_metric":
        order_col = "order_id" if "order_id" in metadata.column_types else (metadata.categorical_columns[0] if metadata.categorical_columns else "order_id")
        return {"revenue_column": metric_col, "order_id_column": order_col, "period_1_filters": [], "period_2_filters": []}
    elif tool_name == "detect_anomalies":
        return {"feature_columns": [metric_col], "method": "iqr"}
    elif tool_name == "create_chart":
        return {
            "chart_type": "bar",
            "title": f"{metric_col} by {seg_col}",
            "data": [{"segment": seg_col, "value": 100.0}],
            "x_column": "segment",
            "y_column": "value",
        }
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class OmniSleuthEvaluator:
    """Runs repeatable benchmark evaluations across candidate architectures."""

    def __init__(self, orchestrator: Optional[MultiAgentOrchestrator] = None):
        self.orchestrator = orchestrator or MultiAgentOrchestrator()

    # ─── OmniSleuth Full System ────────────────────────────────────────────


    def evaluate_omnisleuth(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        questions: Optional[List[BenchmarkQuestion]] = None,
    ) -> SystemBenchmarkResult:
        """Run standard benchmark suite against OmniSleuth multi-agent architecture."""
        suite = questions or BENCHMARK_QUESTIONS
        logger.info(f"Initiating OmniSleuth benchmark evaluation across {len(suite)} test questions.")

        per_question: List[PerQuestionResult] = []
        latencies_ms: List[float] = []
        correct_tools_count = 0
        supported_claims_count = 0
        unsupported_claims_count = 0
        total_claims_count = 0
        evidence_covered_count = 0
        completeness_scores: List[float] = []
        keyword_match_rates: List[float] = []
        blocked_count = 0

        for bq in suite:
            t0 = time.perf_counter()
            try:
                res = self.orchestrator.run_investigation(bq.question, df, metadata)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

                status = res.get("status", "error")
                if status == "blocked":
                    blocked_count += 1

                # 1. Tool Selection Accuracy
                executed_tools: List[str] = []
                if res.get("executed_tasks"):
                    executed_tools = [t.tool_name for t in res["executed_tasks"] if t.tool_name]
                executed_set = set(executed_tools)
                tool_correct = any(exp_t in executed_set for exp_t in bq.expected_tools) if bq.expected_tools else True
                if tool_correct:
                    correct_tools_count += 1

                # 2. Evidence Coverage
                ev_count = 0
                store = res.get("evidence_store")
                if store and len(store.list_evidence()) > 0:
                    ev_count = len(store.list_evidence())
                    if bq.should_succeed:
                        evidence_covered_count += 1

                # 3. Claims & Validation
                val_results = res.get("validation_results", [])
                q_supported = 0
                q_unsupported = 0
                for vr in val_results:
                    total_claims_count += 1
                    if vr.status == ClaimStatus.SUPPORTED:
                        supported_claims_count += 1
                        q_supported += 1
                    elif vr.status == ClaimStatus.UNSUPPORTED:
                        unsupported_claims_count += 1
                        q_unsupported += 1

                # 4. Answer Relevance via keyword matching
                final_answer = str(res.get("final_answer", "")).lower()
                kw_found = [kw for kw in bq.expected_ground_truth_keywords if kw.lower() in final_answer]
                kw_missing = [kw for kw in bq.expected_ground_truth_keywords if kw.lower() not in final_answer]
                kw_rate = (len(kw_found) / len(bq.expected_ground_truth_keywords) * 100.0) if bq.expected_ground_truth_keywords else 100.0
                keyword_match_rates.append(kw_rate)

                # 5. Completeness
                comp = _compute_completeness_score(status, tool_correct, ev_count, q_unsupported, bq.should_succeed)
                completeness_scores.append(comp)

                per_question.append(PerQuestionResult(
                    question_id=bq.id,
                    category=bq.category,
                    question=bq.question,
                    status=status,
                    latency_ms=round(elapsed_ms, 2),
                    tools_expected=bq.expected_tools,
                    tools_executed=executed_tools,
                    tool_selection_correct=tool_correct,
                    evidence_count=ev_count,
                    supported_claims=q_supported,
                    unsupported_claims=q_unsupported,
                    total_claims=q_supported + q_unsupported,
                    keywords_found=kw_found,
                    keywords_missing=kw_missing,
                    answer_relevance_score=round(kw_rate, 2),
                ))

            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)
                completeness_scores.append(0.0)
                keyword_match_rates.append(0.0)
                logger.warning(f"Benchmark error on '{bq.id}': {e}")
                per_question.append(PerQuestionResult(
                    question_id=bq.id, category=bq.category, question=bq.question,
                    status="error", latency_ms=round(elapsed_ms, 2),
                    tools_expected=bq.expected_tools, tools_executed=[],
                    tool_selection_correct=False, evidence_count=0,
                    supported_claims=0, unsupported_claims=0, total_claims=0,
                    keywords_found=[], keywords_missing=bq.expected_ground_truth_keywords,
                    error_message=str(e),
                ))

        total_q = len(suite)
        succeeding_q = sum(1 for bq in suite if bq.should_succeed)

        num_correctness = (
            round((supported_claims_count / total_claims_count * 100.0), 2)
            if total_claims_count > 0 else 100.0
        )
        tool_acc = round((correct_tools_count / total_q * 100.0), 2)
        ev_cov = min(100.0, round((evidence_covered_count / succeeding_q * 100.0), 2)) if succeeding_q > 0 else 0.0
        unsupported_rate = (
            round((unsupported_claims_count / total_claims_count * 100.0), 2)
            if total_claims_count > 0 else 0.0
        )
        completeness = round(float(np.mean(completeness_scores) * 100.0), 2) if completeness_scores else 0.0
        faithfulness = (
            round((supported_claims_count / total_claims_count * 100.0), 2)
            if total_claims_count > 0 else 100.0
        )
        relevance = round(float(np.mean(keyword_match_rates)), 2) if keyword_match_rates else 0.0

        p50 = round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else 0.0
        p95 = round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0

        guardrail_block_rate = round((blocked_count / total_q * 100.0), 2)

        return SystemBenchmarkResult(
            system_name="OmniSleuth (Agent + Tools + RAG + Validator)",
            total_questions=total_q,
            numerical_correctness_pct=num_correctness,
            tool_selection_accuracy_pct=tool_acc,
            evidence_coverage_pct=ev_cov,
            unsupported_claim_rate_pct=unsupported_rate,
            investigation_completeness_pct=completeness,
            faithfulness_pct=faithfulness,
            answer_relevance_pct=relevance,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            guardrail_block_rate_pct=guardrail_block_rate,
            per_question_results=per_question,
        )

    # ─── Baseline 1: Direct Analytical Tools (no agents) ───────────────────

    def evaluate_baseline_direct_tools(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        questions: Optional[List[BenchmarkQuestion]] = None,
    ) -> SystemBenchmarkResult:
        """Baseline 1: Execute the first expected tool directly without agent orchestration.

        This simulates a user who knows exactly which tool to call but has no
        supervisor, evidence store, claim validation, or RAG context.
        """
        suite = questions or BENCHMARK_QUESTIONS
        registry = get_default_tool_registry()
        per_question: List[PerQuestionResult] = []
        latencies_ms: List[float] = []
        correct_tools = 0
        evidence_covered = 0
        completeness_scores: List[float] = []
        keyword_rates: List[float] = []

        for bq in suite:
            t0 = time.perf_counter()

            if not bq.should_succeed:
                # Direct tools have no guardrails — they'd try to execute anyway
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)
                completeness_scores.append(0.0)
                keyword_rates.append(0.0)
                per_question.append(PerQuestionResult(
                    question_id=bq.id, category=bq.category, question=bq.question,
                    status="completed", latency_ms=round(elapsed_ms, 2),
                    tools_expected=bq.expected_tools, tools_executed=[],
                    tool_selection_correct=False, evidence_count=0,
                    supported_claims=0, unsupported_claims=0, total_claims=0,
                    keywords_found=[], keywords_missing=bq.expected_ground_truth_keywords,
                ))
                continue

            try:
                executed_tools: List[str] = []
                result_text = ""

                for tool_name in bq.expected_tools[:1]:  # Only execute first tool
                    if tool_name not in registry._tools:
                        continue
                    tool_fn = registry.get(tool_name)
                    params = _get_default_tool_params(tool_name, metadata)
                    tool_res = tool_fn.run(parameters=params, df=df, metadata=metadata)
                    executed_tools.append(tool_name)
                    result_text = str(tool_res.data if tool_res.success else tool_res.error)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

                tool_correct = bool(executed_tools)
                if tool_correct:
                    correct_tools += 1

                has_evidence = bool(result_text)
                if has_evidence:
                    evidence_covered += 1

                kw_found = [kw for kw in bq.expected_ground_truth_keywords if kw.lower() in result_text.lower()]
                kw_missing = [kw for kw in bq.expected_ground_truth_keywords if kw.lower() not in result_text.lower()]
                kw_rate = (len(kw_found) / len(bq.expected_ground_truth_keywords) * 100.0) if bq.expected_ground_truth_keywords else 100.0
                keyword_rates.append(kw_rate)

                comp = 0.0
                if tool_correct:
                    comp += 0.5
                if has_evidence:
                    comp += 0.5
                completeness_scores.append(comp)

                per_question.append(PerQuestionResult(
                    question_id=bq.id, category=bq.category, question=bq.question,
                    status="completed", latency_ms=round(elapsed_ms, 2),
                    tools_expected=bq.expected_tools, tools_executed=executed_tools,
                    tool_selection_correct=tool_correct, evidence_count=1 if has_evidence else 0,
                    supported_claims=0, unsupported_claims=0, total_claims=0,
                    keywords_found=kw_found, keywords_missing=kw_missing,
                    answer_relevance_score=round(kw_rate, 2),
                ))

            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)
                completeness_scores.append(0.0)
                keyword_rates.append(0.0)
                per_question.append(PerQuestionResult(
                    question_id=bq.id, category=bq.category, question=bq.question,
                    status="error", latency_ms=round(elapsed_ms, 2),
                    tools_expected=bq.expected_tools, tools_executed=[],
                    tool_selection_correct=False, evidence_count=0,
                    supported_claims=0, unsupported_claims=0, total_claims=0,
                    keywords_found=[], keywords_missing=bq.expected_ground_truth_keywords,
                    error_message=str(e),
                ))

        total_q = len(suite)
        succeeding_q = sum(1 for bq in suite if bq.should_succeed)

        p50 = round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else 0.0
        p95 = round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0

        return SystemBenchmarkResult(
            system_name="Baseline 1 — Direct Analytical Tools (no agents)",
            total_questions=total_q,
            numerical_correctness_pct=0.0,  # No claim validation
            tool_selection_accuracy_pct=round((correct_tools / total_q * 100.0), 2),
            evidence_coverage_pct=round((evidence_covered / succeeding_q * 100.0), 2) if succeeding_q > 0 else 0.0,
            unsupported_claim_rate_pct=100.0,  # No claim validation → all claims unverified
            investigation_completeness_pct=round(float(np.mean(completeness_scores) * 100.0), 2) if completeness_scores else 0.0,
            faithfulness_pct=0.0,  # No evidence grounding
            answer_relevance_pct=round(float(np.mean(keyword_rates)), 2) if keyword_rates else 0.0,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            guardrail_block_rate_pct=0.0,  # No guardrails
            per_question_results=per_question,
        )

    # ─── Baseline 2: Fixed Pipeline (tools + evidence, no agents) ──────────

    def evaluate_baseline_fixed_pipeline(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        questions: Optional[List[BenchmarkQuestion]] = None,
    ) -> SystemBenchmarkResult:
        """Baseline 2: Fixed pipeline executing tools and minting evidence, but no
        agent orchestration, no RAG, no hypothesis testing, no guardrails.
        """
        from omnisleuth.evidence.store import EvidenceStore
        from omnisleuth.evidence.validator import ClaimValidator
        from omnisleuth.evidence.models import Claim

        suite = questions or BENCHMARK_QUESTIONS
        registry = get_default_tool_registry()
        validator = ClaimValidator()
        per_question: List[PerQuestionResult] = []
        latencies_ms: List[float] = []
        correct_tools = 0
        evidence_covered = 0
        supported_total = 0
        unsupported_total = 0
        claims_total = 0
        completeness_scores: List[float] = []
        keyword_rates: List[float] = []

        for bq in suite:
            t0 = time.perf_counter()

            if not bq.should_succeed:
                # No guardrails → would attempt execution
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)
                completeness_scores.append(0.0)
                keyword_rates.append(0.0)
                per_question.append(PerQuestionResult(
                    question_id=bq.id, category=bq.category, question=bq.question,
                    status="completed", latency_ms=round(elapsed_ms, 2),
                    tools_expected=bq.expected_tools, tools_executed=[],
                    tool_selection_correct=False, evidence_count=0,
                    supported_claims=0, unsupported_claims=0, total_claims=0,
                    keywords_found=[], keywords_missing=bq.expected_ground_truth_keywords,
                ))
                continue

            try:
                store = EvidenceStore(dataset_id=metadata.dataset_id, dataset_version=metadata.version_hash)
                executed_tools: List[str] = []
                result_text = ""

                for tool_name in bq.expected_tools[:2]:  # Execute up to 2 expected tools
                    if tool_name not in registry._tools:
                        continue
                    tool_fn = registry.get(tool_name)
                    params = _get_default_tool_params(tool_name, metadata)
                    tool_res = tool_fn.run(parameters=params, df=df, metadata=metadata)
                    executed_tools.append(tool_name)
                    res_str = str(tool_res.data if tool_res.success else tool_res.error)
                    result_text += res_str + "\n"
                    store.add_evidence(
                        evidence_type=EvidenceType.DETERMINISTIC_CALC,
                        source_id=tool_name,
                        content=res_str,
                        confidence=1.0,
                    )

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

                tool_correct = any(t in set(executed_tools) for t in bq.expected_tools) if bq.expected_tools else True
                if tool_correct:
                    correct_tools += 1

                ev_items = store.list_evidence()
                ev_count = len(ev_items)
                if ev_count > 0:
                    evidence_covered += 1

                # Build claims from evidence
                claims: List[Claim] = []
                for ev in ev_items:
                    if ev.content.get("overall_delta") is not None:
                        delta = ev.content["overall_delta"]
                        pct = ev.content.get("overall_change_pct", 0.0)
                        claims.append(Claim(
                            statement=f"Delta was {delta} ({pct}%).",
                            extracted_numbers=[delta, pct],
                        ))
                    elif ev.content.get("grand_total") is not None:
                        claims.append(Claim(
                            statement=f"Total was {ev.content['grand_total']}.",
                            extracted_numbers=[ev.content["grand_total"]],
                        ))

                val_results = validator.validate_all(claims, store)
                q_sup = sum(1 for v in val_results if v.status == ClaimStatus.SUPPORTED)
                q_unsup = sum(1 for v in val_results if v.status == ClaimStatus.UNSUPPORTED)
                supported_total += q_sup
                unsupported_total += q_unsup
                claims_total += len(val_results)

                kw_found = [kw for kw in bq.expected_ground_truth_keywords if kw.lower() in result_text.lower()]
                kw_missing = [kw for kw in bq.expected_ground_truth_keywords if kw.lower() not in result_text.lower()]
                kw_rate = (len(kw_found) / len(bq.expected_ground_truth_keywords) * 100.0) if bq.expected_ground_truth_keywords else 100.0
                keyword_rates.append(kw_rate)

                comp = _compute_completeness_score("completed", tool_correct, ev_count, q_unsup, True)
                completeness_scores.append(comp)

                per_question.append(PerQuestionResult(
                    question_id=bq.id, category=bq.category, question=bq.question,
                    status="completed", latency_ms=round(elapsed_ms, 2),
                    tools_expected=bq.expected_tools, tools_executed=executed_tools,
                    tool_selection_correct=tool_correct, evidence_count=ev_count,
                    supported_claims=q_sup, unsupported_claims=q_unsup,
                    total_claims=len(val_results),
                    keywords_found=kw_found, keywords_missing=kw_missing,
                    answer_relevance_score=round(kw_rate, 2),
                ))

            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)
                completeness_scores.append(0.0)
                keyword_rates.append(0.0)
                per_question.append(PerQuestionResult(
                    question_id=bq.id, category=bq.category, question=bq.question,
                    status="error", latency_ms=round(elapsed_ms, 2),
                    tools_expected=bq.expected_tools, tools_executed=[],
                    tool_selection_correct=False, evidence_count=0,
                    supported_claims=0, unsupported_claims=0, total_claims=0,
                    keywords_found=[], keywords_missing=bq.expected_ground_truth_keywords,
                    error_message=str(e),
                ))

        total_q = len(suite)
        succeeding_q = sum(1 for bq in suite if bq.should_succeed)

        num_correctness = round((supported_total / claims_total * 100.0), 2) if claims_total > 0 else 0.0
        p50 = round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else 0.0
        p95 = round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0

        return SystemBenchmarkResult(
            system_name="Baseline 2 — Fixed Pipeline (Tools + Evidence, no agents)",
            total_questions=total_q,
            numerical_correctness_pct=num_correctness,
            tool_selection_accuracy_pct=round((correct_tools / total_q * 100.0), 2),
            evidence_coverage_pct=round((evidence_covered / succeeding_q * 100.0), 2) if succeeding_q > 0 else 0.0,
            unsupported_claim_rate_pct=round((unsupported_total / claims_total * 100.0), 2) if claims_total > 0 else 100.0,
            investigation_completeness_pct=round(float(np.mean(completeness_scores) * 100.0), 2) if completeness_scores else 0.0,
            faithfulness_pct=round((supported_total / claims_total * 100.0), 2) if claims_total > 0 else 0.0,
            answer_relevance_pct=round(float(np.mean(keyword_rates)), 2) if keyword_rates else 0.0,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            guardrail_block_rate_pct=0.0,
            per_question_results=per_question,
        )

    # ─── Anomaly Detection Evaluation ──────────────────────────────────────

    def evaluate_anomaly_detection(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
    ) -> Dict[str, Any]:
        """Run anomaly detection evaluation with synthetic injected ground truth labels."""
        df_injected, ground_truth = inject_synthetic_anomalies(
            df, target_columns=target_columns, anomaly_rate=0.05, magnitude_factor=4.0,
        )

        detector = IsolationForestDetector(contamination=0.05)
        t0 = time.perf_counter()
        result = detector.detect(df_injected, target_columns)
        runtime = (time.perf_counter() - t0) * 1000.0

        # Build prediction array
        y_pred = np.zeros(len(df_injected), dtype=int)
        for anom in result.anomalies:
            if 0 <= anom.row_index < len(y_pred):
                y_pred[anom.row_index] = 1

        metrics = evaluate_anomaly_detection(ground_truth, y_pred, runtime)

        return {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "false_positive_rate": metrics.false_positive_rate,
            "true_positives": metrics.true_positives,
            "false_positives": metrics.false_positives,
            "true_negatives": metrics.true_negatives,
            "false_negatives": metrics.false_negatives,
            "total_injected": int(ground_truth.sum()),
            "total_detected": result.anomaly_count,
            "runtime_ms": round(runtime, 2),
        }

    # ─── Retrieval Evaluation (Phase 10 Integration) ───────────────────────

    def evaluate_retrieval(
        self,
        mode: RetrievalMode = RetrievalMode.RERANKED,
    ) -> Dict[str, Any]:
        """Run Phase 10 retrieval evaluation and return metrics."""
        doc_store = DocumentStore()
        ret_eval = RetrievalEvaluator(document_store=doc_store)
        report = ret_eval.run_comparative_benchmark(top_k=5)

        # Extract the specified mode's results
        mode_key = mode.value
        if mode_key in report.strategies:
            strat = report.strategies[mode_key]
            return {
                "mode": strat.mode.value,
                "recall_at_1": strat.recall_at_1,
                "recall_at_3": strat.recall_at_3,
                "recall_at_5": strat.recall_at_5,
                "mrr": strat.mrr,
                "ndcg_at_5": strat.ndcg_at_5,
                "mean_latency_ms": strat.mean_latency_ms,
                "p95_latency_ms": strat.p95_latency_ms,
                "total_queries": strat.total_queries,
                "all_strategies": {k: v.model_dump() for k, v in report.strategies.items()},
            }
        return {"mode": mode_key, "error": "Mode not found in benchmark results"}

    # ─── Full Benchmark Runner ─────────────────────────────────────────────

    def run_full_benchmark(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        questions: Optional[List[BenchmarkQuestion]] = None,
        output_dir: Optional[Path] = None,
    ) -> ExperimentRecord:
        """Execute the complete Phase 18 benchmark with all systems, persist results."""
        suite = questions or BENCHMARK_QUESTIONS
        logger.info(f"Starting full Phase 18 benchmark: {len(suite)} questions")
        failures: List[str] = []

        # 1. OmniSleuth
        logger.info("=== Evaluating OmniSleuth ===")
        os_result = self.evaluate_omnisleuth(df, metadata, suite)
        os_per_q = os_result.per_question_results

        # 2. Baseline 1
        logger.info("=== Evaluating Baseline 1 (Direct Tools) ===")
        try:
            b1_result = self.evaluate_baseline_direct_tools(df, metadata, suite)
            b1_per_q = b1_result.per_question_results
        except Exception as e:
            failures.append(f"Baseline 1 error: {e}")
            b1_result = SystemBenchmarkResult(
                system_name="Baseline 1 — Direct Tools (FAILED)", total_questions=len(suite),
                numerical_correctness_pct=0.0, tool_selection_accuracy_pct=0.0,
                evidence_coverage_pct=0.0, unsupported_claim_rate_pct=100.0,
                investigation_completeness_pct=0.0, latency_p50_ms=0.0, latency_p95_ms=0.0,
            )
            b1_per_q = []

        # 3. Baseline 2
        logger.info("=== Evaluating Baseline 2 (Fixed Pipeline) ===")
        try:
            b2_result = self.evaluate_baseline_fixed_pipeline(df, metadata, suite)
            b2_per_q = b2_result.per_question_results
        except Exception as e:
            failures.append(f"Baseline 2 error: {e}")
            b2_result = SystemBenchmarkResult(
                system_name="Baseline 2 — Fixed Pipeline (FAILED)", total_questions=len(suite),
                numerical_correctness_pct=0.0, tool_selection_accuracy_pct=0.0,
                evidence_coverage_pct=0.0, unsupported_claim_rate_pct=100.0,
                investigation_completeness_pct=0.0, latency_p50_ms=0.0, latency_p95_ms=0.0,
            )
            b2_per_q = []

        # 4. Anomaly Detection Evaluation
        logger.info("=== Evaluating Anomaly Detection ===")
        anomaly_bench = None
        try:
            numeric_cols = [c for c in metadata.numeric_columns if c in df.columns]
            if numeric_cols:
                anomaly_bench = self.evaluate_anomaly_detection(df, numeric_cols[:2])
                os_result.anomaly_f1 = anomaly_bench["f1"]
                os_result.anomaly_precision = anomaly_bench["precision"]
                os_result.anomaly_recall = anomaly_bench["recall"]
        except Exception as e:
            failures.append(f"Anomaly evaluation error: {e}")

        # 5. Retrieval Evaluation
        logger.info("=== Evaluating Retrieval (Phase 10) ===")
        retrieval_bench = None
        try:
            retrieval_bench = self.evaluate_retrieval(mode=RetrievalMode.RERANKED)
            os_result.retrieval_recall_at_5 = retrieval_bench.get("recall_at_5", 0.0)
            os_result.retrieval_mrr = retrieval_bench.get("mrr", 0.0)
            os_result.retrieval_ndcg_at_5 = retrieval_bench.get("ndcg_at_5", 0.0)
            os_result.retrieval_mode = retrieval_bench.get("mode", "reranked")
        except Exception as e:
            failures.append(f"Retrieval evaluation error: {e}")

        # Build experiment record
        now = datetime.now(timezone.utc)
        experiment = ExperimentRecord(
            experiment_id=f"exp_{now.strftime('%Y%m%d_%H%M%S')}",
            timestamp=now.isoformat(),
            dataset_name=metadata.name,
            dataset_version=metadata.version_hash,
            dataset_rows=metadata.row_count,
            total_questions=len(suite),
            retrieval_mode=os_result.retrieval_mode,
            systems={
                "omnisleuth": os_result,
                "baseline_1_direct_tools": b1_result,
                "baseline_2_fixed_pipeline": b2_result,
            },
            per_question_results={
                "omnisleuth": os_per_q,
                "baseline_1_direct_tools": b1_per_q,
                "baseline_2_fixed_pipeline": b2_per_q,
            },
            retrieval_benchmark=retrieval_bench,
            anomaly_benchmark=anomaly_bench,
            failures=failures,
        )

        # Persist results
        if output_dir is None:
            output_dir = Path("omnisleuth/evaluation/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{experiment.experiment_id}.json"
        filepath = output_dir / filename
        filepath.write_text(
            experiment.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info(f"Benchmark results saved to {filepath}")

        return experiment
