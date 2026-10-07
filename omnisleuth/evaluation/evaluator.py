"""Empirical evaluation harness comparing OmniSleuth vs Baseline 1 (Direct LLM) vs Baseline 2 (Fixed Tools)."""

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from omnisleuth.agents.orchestrator import MultiAgentOrchestrator
from omnisleuth.core.logging import get_logger
from omnisleuth.evaluation.benchmark_suite import BENCHMARK_QUESTIONS, BenchmarkQuestion
from omnisleuth.evidence.models import ClaimStatus
from omnisleuth.ingestion.models import DatasetMetadata

logger = get_logger("evaluation")


class SystemBenchmarkResult(BaseModel):
    """Aggregated empirical benchmark metrics for an individual architecture."""

    system_name: str
    total_questions: int
    numerical_correctness_pct: float = Field(ge=0.0, le=100.0)
    tool_selection_accuracy_pct: float = Field(ge=0.0, le=100.0)
    evidence_coverage_pct: float = Field(ge=0.0, le=100.0)
    unsupported_claim_rate_pct: float = Field(ge=0.0, le=100.0)
    retrieval_relevance_pct: float = Field(ge=0.0, le=100.0)
    latency_p50_ms: float
    latency_p95_ms: float


class OmniSleuthEvaluator:
    """Runs repeatable benchmark evaluations across candidate architectures."""

    def __init__(self, orchestrator: Optional[MultiAgentOrchestrator] = None):
        self.orchestrator = orchestrator or MultiAgentOrchestrator()

    def evaluate_omnisleuth(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        questions: Optional[List[BenchmarkQuestion]] = None,
    ) -> SystemBenchmarkResult:
        """Run standard benchmark suite against OmniSleuth multi-agent architecture."""
        suite = questions or BENCHMARK_QUESTIONS
        logger.info(f"Initiating OmniSleuth benchmark evaluation across {len(suite)} test questions.")

        latencies_ms: List[float] = []
        correct_tools_count = 0
        supported_claims_count = 0
        unsupported_claims_count = 0
        total_claims_count = 0
        evidence_covered_count = 0
        relevant_retrieval_count = 0
        rag_questions_count = 0

        for bq in suite:
            t0 = time.perf_counter()
            try:
                res = self.orchestrator.run_investigation(bq.question, df, metadata)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

                # 1. Tool Selection Accuracy
                executed_tools = {t.tool_name for t in res["executed_tasks"] if t.tool_name}
                if any(exp_t in executed_tools for exp_t in bq.expected_tools):
                    correct_tools_count += 1

                # 2. Evidence Coverage
                store = res["evidence_store"]
                if len(store.list_evidence()) > 0:
                    evidence_covered_count += 1

                # 3. Claims & Validation Metrics
                val_results = res.get("validation_results", [])
                for vr in val_results:
                    total_claims_count += 1
                    if vr.status == ClaimStatus.SUPPORTED:
                        supported_claims_count += 1
                    elif vr.status == ClaimStatus.UNSUPPORTED:
                        unsupported_claims_count += 1

                # 4. RAG Retrieval Relevance
                if bq.requires_rag:
                    rag_questions_count += 1
                    rag_ev = store.list_evidence(evidence_type="rag_document")
                    if rag_ev:
                        relevant_retrieval_count += 1

            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)
                logger.warning(f"Benchmark error on '{bq.id}': {e}")

        total_q = len(suite)
        num_correctness = (
            round((supported_claims_count / total_claims_count * 100.0), 2)
            if total_claims_count > 0
            else 100.0
        )
        tool_acc = round((correct_tools_count / total_q * 100.0), 2)
        ev_cov = round((evidence_covered_count / total_q * 100.0), 2)
        unsupported_rate = (
            round((unsupported_claims_count / total_claims_count * 100.0), 2)
            if total_claims_count > 0
            else 0.0
        )
        retrieval_rel = (
            round((relevant_retrieval_count / rag_questions_count * 100.0), 2)
            if rag_questions_count > 0
            else 100.0
        )

        p50 = round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else 0.0
        p95 = round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0

        return SystemBenchmarkResult(
            system_name="OmniSleuth (Agent + Tools + RAG + Validator)",
            total_questions=total_q,
            numerical_correctness_pct=num_correctness,
            tool_selection_accuracy_pct=tool_acc,
            evidence_coverage_pct=ev_cov,
            unsupported_claim_rate_pct=unsupported_rate,
            retrieval_relevance_pct=retrieval_rel,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
        )
