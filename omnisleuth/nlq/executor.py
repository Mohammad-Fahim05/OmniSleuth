"""Executor: Deterministic execution of validated AnalysisPlans with evidence grounding."""

import math
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from omnisleuth.agents.orchestrator import MultiAgentOrchestrator
from omnisleuth.core.logging import get_logger
from omnisleuth.evidence.models import (
    Claim,
    ClaimStatus,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
)
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.evidence.validator import ClaimValidator
from omnisleuth.ingestion.models import DatasetMetadata
from omnisleuth.ml.detectors import IQRAnomalyDetector, IsolationForestDetector
from omnisleuth.nlq.models import (
    AnalysisPlan,
    ChartSpec,
    ChartType,
    FilterOp,
    FilterSpec,
    MetricAgg,
    PlanIntent,
    QueryResult,
    TimeGrain,
)
from omnisleuth.persistence.repository import Repository

logger = get_logger("nlq.executor")


class PlanExecutor:
    """Executes validated AnalysisPlans deterministically via pandas and allowlisted tools."""

    def __init__(self, repository: Optional[Repository] = None, orchestrator: Optional[Any] = None):
        self.repo = repository or Repository()
        self.claim_validator = ClaimValidator(numerical_tolerance_pct=1.0)
        # Injectable so callers can share one orchestrator instance instead of
        # building a second (expensive) agent stack per executor.
        self.orchestrator = orchestrator or MultiAgentOrchestrator()

    def execute(
        self,
        plan: AnalysisPlan,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        question: str = "",
    ) -> QueryResult:
        """Execute validated AnalysisPlan against DataFrame and metadata."""
        # 1. Handle Unsupported or Needs Clarification
        if plan.intent == PlanIntent.UNSUPPORTED:
            return QueryResult(
                status="unsupported",
                answer=plan.unsupported_reason or "Question is outside allowlisted dataset operations.",
                plan=plan,
                plain_language_summary="Question could not be resolved to a supported dataset query.",
                planner_used=plan.planner,
                limitations=["No execution attempted; unsupported query."],
            )

        if plan.needs_clarification:
            return QueryResult(
                status="needs_clarification",
                answer=plan.needs_clarification,
                plan=plan,
                plain_language_summary="Ambiguous query requiring user clarification.",
                planner_used=plan.planner,
                limitations=["No execution attempted; clarification required."],
            )

        if plan.intent == PlanIntent.GENERAL_KNOWLEDGE:
            return QueryResult(
                status="unsupported",
                answer="General knowledge questions not answerable from your uploaded dataset. OpenRouter LLM assistance is required for general world knowledge.",
                plan=plan,
                plain_language_summary="General knowledge question outside dataset scope.",
                planner_used=plan.planner,
                limitations=["Not computed from dataset; no evidence minted."],
            )

        # 2. Delegate Deep Diagnostic / Causal Investigation
        if plan.intent == PlanIntent.DEEP_INVESTIGATION:
            return self._execute_deep_investigation(plan, df, metadata, question)

        # 3. Create Fresh EvidenceStore for This Query
        inv_id = f"inv_{metadata.dataset_id[:6]}_{uuid.uuid4().hex[:8]}"
        store = EvidenceStore(
            dataset_id=metadata.dataset_id,
            dataset_version=metadata.version_hash,
        )

        try:
            # 4. Route Execution by Plan Intent
            if plan.intent == PlanIntent.DATASET_OVERVIEW:
                return self._execute_overview(plan, df, metadata, store, inv_id)
            elif plan.intent == PlanIntent.MISSING_VALUES:
                return self._execute_missing_values(plan, df, metadata, store, inv_id)
            elif plan.intent == PlanIntent.ANOMALY:
                return self._execute_anomaly(plan, df, metadata, store, inv_id)
            elif plan.intent == PlanIntent.CORRELATION:
                return self._execute_correlation(plan, df, metadata, store, inv_id)
            elif plan.intent == PlanIntent.CONTRIBUTION:
                return self._execute_contribution(plan, df, metadata, store, inv_id)
            elif plan.intent == PlanIntent.PERIOD_COMPARISON:
                return self._execute_period_comparison(plan, df, metadata, store, inv_id)
            elif plan.intent == PlanIntent.TREND:
                return self._execute_trend(plan, df, metadata, store, inv_id)
            elif plan.intent in (PlanIntent.QUERY, PlanIntent.CHART_REQUEST):
                return self._execute_query(plan, df, metadata, store, inv_id, question)
            elif plan.intent == PlanIntent.DOCUMENTATION_QUESTION:
                return self._execute_documentation_question(plan, df, metadata, store, inv_id, question)
            else:
                return QueryResult(
                    status="unsupported",
                    answer=f"Intent '{plan.intent.value}' is not supported by deterministic executor.",
                    plan=plan,
                    plain_language_summary=f"Unsupported intent: {plan.intent.value}",
                    planner_used=plan.planner,
                )

        except Exception as exc:
            logger.error(f"Executor failed during plan execution: {exc}", exc_info=True)
            return QueryResult(
                status="failed",
                answer=f"Execution error: {exc}",
                plan=plan,
                plain_language_summary="Execution failed due to an internal error.",
                planner_used=plan.planner,
                limitations=[str(exc)],
            )

    def _execute_deep_investigation(
        self,
        plan: AnalysisPlan,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        question: str,
    ) -> QueryResult:
        """Delegate diagnostic questions to existing MultiAgentOrchestrator."""
        inv_res = self.orchestrator.run_investigation(question or "Deep diagnostic investigation", df, metadata)
        store = inv_res["evidence_store"]
        claims = inv_res.get("claims", [])
        final_answer = inv_res["final_answer"]

        # Persist results (the dataset row must exist first for the FK constraint)
        inv_id = f"inv_{metadata.dataset_id[:6]}_{uuid.uuid4().hex[:8]}"
        sid = getattr(metadata, "session_id", None)
        try:
            self.repo.save_dataset(metadata, session_id=sid)
        except Exception as exc:
            logger.warning(f"Could not persist dataset metadata for deep investigation: {exc}")
        try:
            self.repo.save_investigation_results(
                investigation_id=inv_id,
                dataset_id=metadata.dataset_id,
                question=question or "Deep diagnostic investigation",
                status="completed",
                plan_dict=plan.model_dump(),
                final_answer=final_answer,
                evidence_store=store,
                planner_used=plan.planner,
                summary_text="Multi-agent diagnostic investigation with automated hypothesis testing.",
                session_id=sid,
            )
        except Exception as exc:
            logger.warning(f"Could not persist deep investigation results: {exc}")

        return QueryResult(
            status="answered",
            answer=final_answer + "\n\n*(Executed via Multi-Agent Deep Investigation pipeline)*",
            evidence_store=store,
            claims=claims,
            plan=plan,
            plain_language_summary="Multi-agent diagnostic investigation with automated hypothesis testing.",
            planner_used="deterministic",
            limitations=["Correlation and pattern analysis; does not prove causality."],
            investigation_id=inv_id,
        )

    def _execute_overview(
        self,
        plan: AnalysisPlan,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        store: EvidenceStore,
        inv_id: str,
    ) -> QueryResult:
        """Execute dataset overview intent."""
        row_cnt = len(df)
        col_cnt = len(df.columns)
        num_cols = len(metadata.numeric_columns)
        cat_cols = len(metadata.categorical_columns)

        payload = {
            "row_count": row_cnt,
            "column_count": col_cnt,
            "numeric_columns_count": num_cols,
            "categorical_columns_count": cat_cols,
            "duplicate_rows": int(df.duplicated().sum()),
        }

        ev = EvidenceItem(
            evidence_id=f"ev_overview_{uuid.uuid4().hex[:6]}",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="dataset_overview",
            location=metadata.name,
            content=payload,
            confidence=1.0,
        )
        store.add_evidence(ev)

        statement = (
            f"Dataset '{metadata.name}' contains {row_cnt:,} rows and {col_cnt} columns "
            f"({num_cols} numeric, {cat_cols} categorical) with {payload['duplicate_rows']} duplicate rows."
        )

        claim = Claim(
            statement=statement,
            extracted_numbers=[float(row_cnt), float(col_cnt), float(payload["duplicate_rows"])],
            status=ClaimStatus.SUPPORTED,
        )
        self.claim_validator.validate_all([claim], store)
        store.add_claim(claim)

        result_table = [
            {"Column": c.name, "Type": c.data_type.value, "Unique Count": c.unique_count, "Missing": c.missing_count}
            for c in metadata.columns[:50]
        ]

        self._persist_results(inv_id, metadata, "Dataset overview", statement, plan, store)

        return QueryResult(
            status="answered",
            answer=statement,
            result_table=result_table,
            evidence_store=store,
            claims=[claim],
            plan=plan,
            plain_language_summary=f"Profiled {row_cnt:,} rows and {col_cnt} columns.",
            planner_used=plan.planner,
            investigation_id=inv_id,
        )

    def _execute_documentation_question(
        self,
        plan: AnalysisPlan,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        store: EvidenceStore,
        inv_id: str,
        question: Optional[str] = None,
    ) -> QueryResult:
        """Execute documentation / glossary inquiry with empirical dataset grounding."""
        # 1. Resolve target column from plan or question text
        target_col = plan.target_column
        if not target_col or target_col not in df.columns:
            if question:
                q_low = question.lower()
                for col in df.columns:
                    col_norm = col.lower().replace("_", " ")
                    pat = rf"\b{re.escape(col_norm)}\b|\b{re.escape(col.lower())}\b"
                    if re.search(pat, q_low):
                        target_col = col
                        break

        # Suggestion helpers
        suggested_cat = (
            metadata.categorical_columns[0]
            if getattr(metadata, "categorical_columns", None)
            else "category"
        )
        suggested_num = (
            metadata.numeric_columns[0]
            if getattr(metadata, "numeric_columns", None)
            else "revenue"
        )

        # 2. Case A: Target column found in dataset
        if target_col and target_col in df.columns:
            series = df[target_col]
            total_rows = len(df)
            non_null = int(series.count())
            missing = int(series.isna().sum())
            unique_cnt = int(series.nunique())
            dtype_str = str(series.dtype)

            col_meta = next((c for c in metadata.columns if c.name == target_col), None)
            col_type = col_meta.data_type.value if col_meta else "unspecified"

            is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

            payload: Dict[str, Any] = {
                "target_column": target_col,
                "data_type": col_type,
                "dtype": dtype_str,
                "total_rows": total_rows,
                "non_null_count": non_null,
                "missing_count": missing,
                "unique_count": unique_cnt,
            }

            table_rows = [
                {"Property": "Column Name", "Value": str(target_col)},
                {"Property": "Inferred Data Type", "Value": str(col_type)},
                {"Property": "Physical Dtype", "Value": str(dtype_str)},
                {"Property": "Total Rows", "Value": f"{total_rows}"},
                {"Property": "Populated Values", "Value": f"{non_null}"},
                {"Property": "Missing Values", "Value": f"{missing}"},
                {"Property": "Unique Values", "Value": f"{unique_cnt}"},
            ]

            if is_numeric and non_null > 0:
                min_val = round(float(series.min()), 2)
                max_val = round(float(series.max()), 2)
                mean_val = round(float(series.mean()), 2)
                median_val = round(float(series.median()), 2)

                payload.update({
                    "min": min_val,
                    "max": max_val,
                    "mean": mean_val,
                    "median": median_val,
                })

                table_rows.extend([
                    {"Property": "Min Value", "Value": f"{min_val:,.2f}"},
                    {"Property": "Max Value", "Value": f"{max_val:,.2f}"},
                    {"Property": "Mean Value", "Value": f"{mean_val:,.2f}"},
                    {"Property": "Median Value", "Value": f"{median_val:,.2f}"},
                ])

                statement = (
                    f"Column '{target_col}' has {non_null} populated values with {unique_cnt} unique values "
                    f"and {missing} missing values (min: {min_val:.2f}, max: {max_val:.2f}, mean: {mean_val:.2f})."
                )

                extracted_numbers = [
                    float(non_null),
                    float(unique_cnt),
                    float(missing),
                    float(min_val),
                    float(max_val),
                    float(mean_val),
                ]

                answer = (
                    f"In dataset '{metadata.name}', '{target_col}' is a {col_type} column ({dtype_str}) "
                    f"with {non_null:,} values ranging from {min_val:,.2f} to {max_val:,.2f} "
                    f"(mean: {mean_val:,.2f}, median: {median_val:,.2f}, unique: {unique_cnt:,}, missing: {missing:,}).\n\n"
                    f"**Data Grounding Note:** Dataset metadata and empirical distributions describe how '{target_col}' is stored "
                    f"and distributed, but do not provide domain-specific business definitions or accounting formulas "
                    f"(such as gross vs. net calculations, tax handling, or discounts). "
                    f"Please consult domain documentation for formal business definitions.\n\n"
                    f"**Suggested Analytical Questions:**\n"
                    f"- What is the total {target_col}?\n"
                    f"- What is the average {target_col} by {suggested_cat}?\n"
                    f"- What is the monthly trend of {target_col}?"
                )
            else:
                statement = (
                    f"Column '{target_col}' has {non_null} populated values with {unique_cnt} unique values "
                    f"and {missing} missing values."
                )
                extracted_numbers = [float(non_null), float(unique_cnt), float(missing)]

                samples = [str(x) for x in series.dropna().unique()[:5]]
                if samples:
                    table_rows.append({"Property": "Sample Values", "Value": ", ".join(samples)})

                answer = (
                    f"In dataset '{metadata.name}', '{target_col}' is a {col_type} column ({dtype_str}) "
                    f"with {non_null:,} populated values ({unique_cnt:,} unique, {missing:,} missing).\n\n"
                    f"**Data Grounding Note:** Column metadata reflects schema type and empirical values, "
                    f"not business glossary rules or metric definitions.\n\n"
                    f"**Suggested Analytical Questions:**\n"
                    f"- How many records are there by {target_col}?\n"
                    f"- Which {target_col} has the highest {suggested_num}?"
                )

            ev = EvidenceItem(
                evidence_id=f"ev_doc_{uuid.uuid4().hex[:6]}",
                evidence_type=EvidenceType.DETERMINISTIC_CALC,
                source_id="documentation_question",
                location=f"{metadata.name}:{target_col}",
                content=payload,
                confidence=1.0,
            )
            store.add_evidence(ev)

            claim = Claim(
                statement=statement,
                extracted_numbers=extracted_numbers,
                status=ClaimStatus.SUPPORTED,
            )
            self.claim_validator.validate_all([claim], store)
            store.add_claim(claim)

            self._persist_results(inv_id, metadata, question or f"Explain {target_col}", statement, plan, store)

            return QueryResult(
                status="answered",
                answer=answer,
                result_table=table_rows,
                evidence_store=store,
                claims=[claim],
                plan=plan,
                plain_language_summary=f"Provided schema and distribution profile for column '{target_col}' in dataset '{metadata.name}'.",
                planner_used=plan.planner,
                limitations=[
                    "Dataset metadata provides empirical data characteristics; formal business definitions require domain documentation."
                ],
                investigation_id=inv_id,
            )

        # 3. Case B: Target column NOT found in dataset (general question or unmapped metric)
        available_cols = [c.name for c in metadata.columns] if metadata.columns else list(df.columns)
        available_str = ", ".join(sorted(available_cols))

        table_rows = [
            {"Column": c.name, "Type": c.data_type.value, "Unique Count": c.unique_count, "Missing": c.missing_count}
            for c in metadata.columns[:50]
        ] if metadata.columns else []

        payload = {
            "available_columns": available_cols,
            "column_count": len(available_cols),
        }
        ev = EvidenceItem(
            evidence_id=f"ev_doc_{uuid.uuid4().hex[:6]}",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="documentation_question",
            location=metadata.name,
            content=payload,
            confidence=1.0,
        )
        store.add_evidence(ev)

        missing_mention = f" for '{target_col}'" if target_col else ""
        statement = (
            f"Business definitions and metric formulas{missing_mention} are not embedded in dataset metadata. "
            f"The dataset contains {len(available_cols)} columns."
        )
        claim = Claim(
            statement=statement,
            extracted_numbers=[float(len(available_cols))],
            status=ClaimStatus.SUPPORTED,
        )
        self.claim_validator.validate_all([claim], store)
        store.add_claim(claim)

        answer = (
            f"Business definitions and metric formulas{missing_mention} are not embedded in dataset metadata. "
            f"Dataset '{metadata.name}' contains the following available columns: {available_str}.\n\n"
            f"**Data Grounding Note:** While empirical data can compute aggregations, distributions, and trends, "
            f"formal business definitions (such as accounting policies or metric calculations) must be verified "
            f"against business domain documentation.\n\n"
            f"**Suggested Analytical Questions:**\n"
            f"- What is the total {suggested_num}?\n"
            f"- What is the average {suggested_num} by {suggested_cat}?\n"
            f"- Describe dataset overview"
        )

        self._persist_results(inv_id, metadata, question or "Documentation question", statement, plan, store)

        return QueryResult(
            status="answered",
            answer=answer,
            result_table=table_rows,
            evidence_store=store,
            claims=[claim],
            plan=plan,
            plain_language_summary=f"General dataset documentation and available columns for '{metadata.name}'.",
            planner_used=plan.planner,
            limitations=[
                "Business definitions and metric formulas are not embedded in dataset metadata."
            ],
            investigation_id=inv_id,
        )

    def _execute_missing_values(
        self,
        plan: AnalysisPlan,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        store: EvidenceStore,
        inv_id: str,
    ) -> QueryResult:
        """Execute missing values and completeness inspection."""
        missing_dict = df.isna().sum().to_dict()
        total_cells = df.size
        total_missing = sum(missing_dict.values())
        completeness = round(((total_cells - total_missing) / total_cells * 100.0), 2) if total_cells > 0 else 100.0

        ev = EvidenceItem(
            evidence_id=f"ev_missing_{uuid.uuid4().hex[:6]}",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="missing_values_audit",
            location=metadata.name,
            content={"total_missing": total_missing, "completeness_pct": completeness, "missing_by_column": missing_dict},
            confidence=1.0,
        )
        store.add_evidence(ev)

        claim = Claim(
            statement=f"Dataset completeness is {completeness}%, with {total_missing} total null values.",
            extracted_numbers=[float(completeness), float(total_missing)],
            status=ClaimStatus.SUPPORTED,
        )
        self.claim_validator.validate_all([claim], store)
        store.add_claim(claim)

        rows = [{"Column": col, "Missing Count": cnt, "Missing Pct": f"{(cnt / len(df) * 100):.1f}%"} for col, cnt in missing_dict.items() if cnt > 0]
        if not rows:
            statement = f"Dataset completeness is 100.0%. No missing values found in any of the {len(df.columns)} columns."
        else:
            statement = f"Dataset completeness is {completeness}%. Found {total_missing} missing values across {len(rows)} columns."

        self._persist_results(inv_id, metadata, "Missing values audit", statement, plan, store)

        return QueryResult(
            status="answered",
            answer=statement,
            result_table=rows or [{"Status": "Clean", "Missing Count": 0}],
            evidence_store=store,
            claims=[claim],
            plan=plan,
            plain_language_summary=f"Audit of null values and data completeness ({completeness}% complete).",
            planner_used=plan.planner,
            investigation_id=inv_id,
        )

    def _execute_anomaly(
        self,
        plan: AnalysisPlan,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        store: EvidenceStore,
        inv_id: str,
    ) -> QueryResult:
        """Execute anomaly detection on target numeric column."""
        target_col = plan.target_column or (metadata.numeric_columns[0] if metadata.numeric_columns else None)
        if not target_col:
            return QueryResult(
                status="unsupported",
                answer="No numeric columns available in dataset for anomaly detection.",
                plan=plan,
                plain_language_summary="Anomaly detection requires at least one numeric feature.",
                planner_used=plan.planner,
            )

        if len(df) < 5:
            return QueryResult(
                status="failed",
                answer=f"Insufficient data rows ({len(df)}) for statistical anomaly detection (minimum 5 rows required).",
                plan=plan,
                plain_language_summary="Dataset too small for anomaly detection.",
                planner_used=plan.planner,
                limitations=["Requires >= 5 rows."],
            )

        detector = IQRAnomalyDetector(multiplier=1.5)
        res = detector.detect(df, [target_col])

        ev = EvidenceItem(
            evidence_id=f"ev_anom_{uuid.uuid4().hex[:6]}",
            evidence_type=EvidenceType.ML_ANOMALY,
            source_id="iqr_detector",
            location=target_col,
            content={
                "anomaly_count": res.anomaly_count,
                "anomaly_percentage": res.anomaly_percentage,
                "method": "iqr",
                "target_column": target_col,
            },
            confidence=0.9,
        )
        store.add_evidence(ev)

        statement = (
            f"Identified {res.anomaly_count} statistical outlier records in '{target_col}' "
            f"({res.anomaly_percentage:.1f}% of dataset) using 1.5x IQR threshold. "
            f"Results indicate statistical deviation only and do not imply operational causality."
        )

        claim = Claim(
            statement=statement,
            extracted_numbers=[float(res.anomaly_count), float(res.anomaly_percentage)],
            status=ClaimStatus.SUPPORTED,
        )
        self.claim_validator.validate_all([claim], store)
        store.add_claim(claim)

        result_table = [a.data for a in res.anomalies[:50]]
        self._persist_results(inv_id, metadata, "Anomaly detection", statement, plan, store)

        return QueryResult(
            status="answered",
            answer=statement,
            result_table=result_table,
            evidence_store=store,
            claims=[claim],
            plan=plan,
            plain_language_summary=f"Detected {res.anomaly_count} outliers in '{target_col}'.",
            planner_used=plan.planner,
            limitations=[res.non_causal_disclaimer],
            investigation_id=inv_id,
        )

    def _execute_correlation(
        self,
        plan: AnalysisPlan,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        store: EvidenceStore,
        inv_id: str,
    ) -> QueryResult:
        """Execute correlation between two numeric metrics."""
        cols = [m.column for m in plan.metrics[:2]]
        if len(cols) < 2 or not all(c in df.columns for c in cols):
            return QueryResult(
                status="unsupported",
                answer="Correlation requires two valid numeric columns.",
                plan=plan,
                plain_language_summary="Correlation requires two numeric features.",
                planner_used=plan.planner,
            )

        valid_df = df[cols].dropna()
        if len(valid_df) < 3:
            return QueryResult(
                status="failed",
                answer="Insufficient rows with valid numbers to calculate correlation.",
                plan=plan,
                plain_language_summary="Insufficient data for correlation.",
                planner_used=plan.planner,
            )

        corr_val = float(valid_df[cols[0]].corr(valid_df[cols[1]]))
        corr_rounded = round(corr_val, 3)

        ev = EvidenceItem(
            evidence_id=f"ev_corr_{uuid.uuid4().hex[:6]}",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="pearson_correlation",
            location=f"{cols[0]} ~ {cols[1]}",
            content={"column_a": cols[0], "column_b": cols[1], "correlation": corr_rounded, "sample_size": len(valid_df)},
            confidence=1.0,
        )
        store.add_evidence(ev)

        strength = "strong" if abs(corr_val) >= 0.7 else "moderate" if abs(corr_val) >= 0.4 else "weak"
        direction = "positive" if corr_val > 0 else "negative" if corr_val < 0 else "zero"

        statement = (
            f"Pearson correlation between '{cols[0]}' and '{cols[1]}' is {corr_rounded:+.3f} "
            f"({strength} {direction} correlation, N={len(valid_df):,}). "
            f"Statistical correlation measures linear co-occurrence and does not establish causation."
        )

        claim = Claim(
            statement=statement,
            extracted_numbers=[corr_rounded, float(len(valid_df))],
            status=ClaimStatus.SUPPORTED,
        )
        self.claim_validator.validate_all([claim], store)
        store.add_claim(claim)

        result_table = [{"Feature A": cols[0], "Feature B": cols[1], "Correlation": corr_rounded, "Sample Size": len(valid_df)}]
        self._persist_results(inv_id, metadata, "Correlation analysis", statement, plan, store)

        return QueryResult(
            status="answered",
            answer=statement,
            result_table=result_table,
            evidence_store=store,
            claims=[claim],
            plan=plan,
            plain_language_summary=f"Calculated correlation between '{cols[0]}' and '{cols[1]}': {corr_rounded:+.3f}.",
            planner_used=plan.planner,
            limitations=["Correlation indicates statistical co-movement only; does not imply causality."],
            investigation_id=inv_id,
        )

    def _execute_contribution(
        self,
        plan: AnalysisPlan,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        store: EvidenceStore,
        inv_id: str,
    ) -> QueryResult:
        """Execute categorical contribution breakdown."""
        target_col = plan.target_column or (metadata.numeric_columns[0] if metadata.numeric_columns else None)
        dim_col = plan.group_by[0] if plan.group_by else (metadata.categorical_columns[0] if metadata.categorical_columns else None)

        if not target_col or not dim_col:
            return QueryResult(
                status="unsupported",
                answer="Contribution analysis requires a numeric metric and a categorical dimension.",
                plan=plan,
                plain_language_summary="Missing numeric or categorical target.",
                planner_used=plan.planner,
            )

        grouped = df.groupby(dim_col, as_index=False)[target_col].sum()
        total_val = float(grouped[target_col].sum())
        grouped["share_pct"] = (grouped[target_col] / total_val * 100.0).round(2)
        grouped = grouped.sort_values(by=target_col, ascending=False)

        top_row = grouped.iloc[0]
        top_segment = top_row[dim_col]
        top_share = top_row["share_pct"]

        ev = EvidenceItem(
            evidence_id=f"ev_contrib_{uuid.uuid4().hex[:6]}",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="segment_contribution",
            location=f"{dim_col} -> {target_col}",
            content={
                "segment_column": dim_col,
                "metric_column": target_col,
                "total_value": round(total_val, 2),
                "top_segment": str(top_segment),
                "top_share_pct": float(top_share),
                "breakdown": grouped.to_dict(orient="records"),
            },
            confidence=1.0,
        )
        store.add_evidence(ev)

        statement = (
            f"In '{dim_col}', '{top_segment}' contributed the largest share to total {target_col} "
            f"with {top_share:.1f}% (${float(top_row[target_col]):,.2f} of ${total_val:,.2f} total)."
        )

        claim = Claim(
            statement=statement,
            extracted_numbers=[float(top_share), round(float(top_row[target_col]), 2), round(total_val, 2)],
            status=ClaimStatus.SUPPORTED,
        )
        self.claim_validator.validate_all([claim], store)
        store.add_claim(claim)

        result_table = grouped.to_dict(orient="records")
        chart_spec = {"type": "bar", "x": dim_col, "y": target_col}

        self._persist_results(inv_id, metadata, "Contribution breakdown", statement, plan, store)

        return QueryResult(
            status="answered",
            answer=statement,
            result_table=result_table,
            chart_spec=chart_spec,
            evidence_store=store,
            claims=[claim],
            plan=plan,
            plain_language_summary=f"Analyzed {target_col} contribution across '{dim_col}'.",
            planner_used=plan.planner,
            investigation_id=inv_id,
        )

    def _execute_period_comparison(
        self,
        plan: AnalysisPlan,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        store: EvidenceStore,
        inv_id: str,
    ) -> QueryResult:
        """Execute comparison between two discrete time windows (period A vs period B)."""
        date_col = plan.time.date_column if (plan.time and plan.time.date_column) else (
            metadata.datetime_columns[0] if metadata.datetime_columns else "order_date"
        )
        metric_col = plan.metrics[0].column if plan.metrics else metadata.numeric_columns[0]
        p_a = plan.period_a
        p_b = plan.period_b

        dt_df = df.copy()
        dt_df["_dt"] = pd.to_datetime(dt_df[date_col], errors="coerce")

        df_a = dt_df[(dt_df["_dt"] >= pd.to_datetime(p_a.start)) & (dt_df["_dt"] <= pd.to_datetime(p_a.end))]
        df_b = dt_df[(dt_df["_dt"] >= pd.to_datetime(p_b.start)) & (dt_df["_dt"] <= pd.to_datetime(p_b.end))]

        val_a = float(df_a[metric_col].sum())
        val_b = float(df_b[metric_col].sum())
        delta = val_b - val_a
        pct_change = round((delta / val_a * 100.0), 2) if val_a != 0 else 0.0

        ev = EvidenceItem(
            evidence_id=f"ev_pcomp_{uuid.uuid4().hex[:6]}",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="period_comparison",
            location=f"{p_a.label or p_a.start} vs {p_b.label or p_b.start}",
            content={
                "metric": metric_col,
                "period_a": {"start": p_a.start, "end": p_a.end, "value": round(val_a, 2)},
                "period_b": {"start": p_b.start, "end": p_b.end, "value": round(val_b, 2)},
                "delta": round(delta, 2),
                "percentage_change": pct_change,
            },
            confidence=1.0,
        )
        store.add_evidence(ev)

        label_a = p_a.label or p_a.start
        label_b = p_b.label or p_b.start
        direction = "increased" if delta > 0 else "decreased" if delta < 0 else "remained flat"

        statement = (
            f"Comparing {label_b} against {label_a}, total {metric_col} {direction} by "
            f"{delta:+,.2f} ({pct_change:+.1f}%), from ${val_a:,.2f} to ${val_b:,.2f}."
        )

        claim = Claim(
            statement=statement,
            extracted_numbers=[round(delta, 2), pct_change, round(val_a, 2), round(val_b, 2)],
            status=ClaimStatus.SUPPORTED,
        )
        self.claim_validator.validate_all([claim], store)
        store.add_claim(claim)

        result_table = [
            {"Period": label_a, "Value": round(val_a, 2), "Rows": len(df_a)},
            {"Period": label_b, "Value": round(val_b, 2), "Rows": len(df_b)},
            {"Period": "Delta", "Value": round(delta, 2), "Rows": len(df_b) - len(df_a)},
        ]

        self._persist_results(inv_id, metadata, f"{label_a} vs {label_b} comparison", statement, plan, store)

        return QueryResult(
            status="answered",
            answer=statement,
            result_table=result_table,
            evidence_store=store,
            claims=[claim],
            plan=plan,
            plain_language_summary=f"Compared {metric_col} between {label_a} and {label_b}: {pct_change:+.1f}% change.",
            planner_used=plan.planner,
            investigation_id=inv_id,
        )

    def _execute_trend(
        self,
        plan: AnalysisPlan,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        store: EvidenceStore,
        inv_id: str,
    ) -> QueryResult:
        """Execute monthly or quarterly time-series trend calculation."""
        date_col = plan.time.date_column if plan.time else metadata.datetime_columns[0]
        metric_col = plan.metrics[0].column if plan.metrics else metadata.numeric_columns[0]
        grain_char = "Q" if (plan.time and plan.time.grain == TimeGrain.QUARTER) else "D" if (plan.time and plan.time.grain == TimeGrain.DAY) else "M"

        calc_df = df.copy()
        calc_df["_dt"] = pd.to_datetime(calc_df[date_col], errors="coerce")
        calc_df = calc_df.dropna(subset=["_dt"])

        # Apply time boundaries if specified
        if plan.time and plan.time.start:
            calc_df = calc_df[calc_df["_dt"] >= pd.to_datetime(plan.time.start)]
        if plan.time and plan.time.end:
            calc_df = calc_df[calc_df["_dt"] <= pd.to_datetime(plan.time.end)]

        calc_df["period"] = calc_df["_dt"].dt.to_period(grain_char).astype(str)
        monthly_series = calc_df.groupby("period", as_index=False)[metric_col].sum().sort_values("period")

        start_val = float(monthly_series.iloc[0][metric_col])
        end_val = float(monthly_series.iloc[-1][metric_col])
        delta = end_val - start_val
        change_pct = round((delta / start_val * 100.0), 2) if start_val != 0 else 0.0

        ts_list = [{"period": r["period"], "value": round(float(r[metric_col]), 2)} for _, r in monthly_series.iterrows()]

        ev = EvidenceItem(
            evidence_id=f"ev_trend_{uuid.uuid4().hex[:6]}",
            evidence_type=EvidenceType.DETERMINISTIC_CALC,
            source_id="calculate_trend",
            location=f"{date_col} -> {metric_col}",
            content={
                "metric": metric_col,
                "start_period": str(monthly_series.iloc[0]["period"]),
                "end_period": str(monthly_series.iloc[-1]["period"]),
                "start_value": round(start_val, 2),
                "end_value": round(end_val, 2),
                "overall_delta": round(delta, 2),
                "overall_change_pct": change_pct,
                "time_series": ts_list,
            },
            confidence=1.0,
        )
        store.add_evidence(ev)

        direction = "up" if delta > 0 else "down" if delta < 0 else "unchanged"
        statement = (
            f"Across {len(monthly_series)} periods ({monthly_series.iloc[0]['period']} to {monthly_series.iloc[-1]['period']}), "
            f"overall {metric_col} moved {direction} by {delta:+,.2f} ({change_pct:+.1f}%), "
            f"from ${start_val:,.2f} in the first period to ${end_val:,.2f} in the last period."
        )

        claim = Claim(
            statement=statement,
            extracted_numbers=[round(delta, 2), change_pct, round(start_val, 2), round(end_val, 2)],
            status=ClaimStatus.SUPPORTED,
        )
        self.claim_validator.validate_all([claim], store)
        store.add_claim(claim)

        result_table = monthly_series.to_dict(orient="records")
        chart_spec = {"type": "line", "x": "period", "y": metric_col}

        self._persist_results(inv_id, metadata, f"{metric_col} time-series trend", statement, plan, store)

        return QueryResult(
            status="answered",
            answer=statement,
            result_table=result_table,
            chart_spec=chart_spec,
            evidence_store=store,
            claims=[claim],
            plan=plan,
            plain_language_summary=f"Computed {metric_col} trend: {change_pct:+.1f}% change across {len(monthly_series)} periods.",
            planner_used=plan.planner,
            investigation_id=inv_id,
        )

    def _execute_query(
        self,
        plan: AnalysisPlan,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        store: EvidenceStore,
        inv_id: str,
        question: str,
    ) -> QueryResult:
        """Execute flexible Query plan (the workhorse: aggregations, group-bys, filters, limits)."""
        initial_rows = len(df)
        working_df = df.copy()

        # Step 1: Apply Filters via Boolean Masks
        applied_filter_notes = []
        for f in plan.filters:
            col = f.column
            if col not in working_df.columns:
                continue

            mask = self._build_filter_mask(working_df, f)
            working_df = working_df[mask]
            applied_filter_notes.append(f"{col} {f.op.value} {f.value}")

        filtered_rows = len(working_df)
        limitations: List[str] = []
        if filtered_rows < initial_rows:
            limitations.append(f"Filtered {initial_rows - filtered_rows} rows ({filtered_rows}/{initial_rows} matching).")
        if filtered_rows == 0:
            return QueryResult(
                status="answered",
                answer="No rows matched the specified filter criteria.",
                result_table=[],
                evidence_store=store,
                plan=plan,
                plain_language_summary="Query executed but returned 0 matching records.",
                planner_used=plan.planner,
                limitations=["Zero rows matched filters."],
            )

        # Step 2: Aggregation and Group-By
        result_table: List[Dict[str, Any]] = []
        statement = ""
        claims: List[Claim] = []

        if plan.group_by:
            # Grouped aggregation
            agg_dict = {}
            for m in plan.metrics:
                agg_func = "nunique" if m.agg == MetricAgg.NUNIQUE else m.agg.value
                agg_dict[m.column] = agg_func

            grouped = working_df.groupby(plan.group_by, as_index=False).agg(agg_dict)

            # Sort if requested
            if plan.sort:
                sort_col = plan.sort.by
                if sort_col in grouped.columns:
                    grouped = grouped.sort_values(by=sort_col, ascending=(plan.sort.direction.value == "asc"))
                elif plan.metrics and plan.metrics[0].column in grouped.columns:
                    grouped = grouped.sort_values(by=plan.metrics[0].column, ascending=(plan.sort.direction.value == "asc"))

            # Limit if requested
            if plan.limit:
                grouped = grouped.head(plan.limit)
            else:
                grouped = grouped.head(50)  # Safe default cap

            result_table = grouped.to_dict(orient="records")

            # Mint Evidence
            ev = EvidenceItem(
                evidence_id=f"ev_query_{uuid.uuid4().hex[:6]}",
                evidence_type=EvidenceType.DETERMINISTIC_CALC,
                source_id="flexible_query_group_by",
                location=f"group_by({', '.join(plan.group_by)})",
                content={
                    "group_by": plan.group_by,
                    "metrics": [m.model_dump() for m in plan.metrics],
                    "filters_applied": applied_filter_notes,
                    "rows_before": initial_rows,
                    "rows_after": filtered_rows,
                    "records": result_table,
                },
                confidence=1.0,
            )
            store.add_evidence(ev)

            # Format Answer
            primary_metric = plan.metrics[0].column
            primary_agg = plan.metrics[0].agg.value
            max_display = 10
            top_k = min(max_display, len(result_table))
            top_previews = []
            for i in range(top_k):
                row = result_table[i]
                dim_str = " / ".join(str(row[g]) for g in plan.group_by)
                metric_val = row[primary_metric]
                formatted_val = f"${metric_val:,.2f}" if isinstance(metric_val, (int, float)) and "rev" in primary_metric.lower() else f"{metric_val:,.2f}" if isinstance(metric_val, (int, float)) else str(metric_val)
                top_previews.append(f"{i+1}. {dim_str} ({formatted_val})")

            if len(result_table) > top_k:
                group_desc = f"showing top {top_k} of {len(result_table)} groups"
            else:
                group_desc = f"{len(result_table)} groups"

            statement = (
                f"Breakdown of {primary_agg} {primary_metric} by {', '.join(plan.group_by)} ({group_desc}): "
                + ", ".join(top_previews) + "."
            )

            # Extract numbers for Claim
            extracted_nums = []
            for i in range(top_k):
                val = result_table[i][primary_metric]
                if isinstance(val, (int, float)):
                    extracted_nums.append(round(float(val), 2))

            claim = Claim(
                statement=statement,
                extracted_numbers=extracted_nums,
                status=ClaimStatus.SUPPORTED,
            )
            self.claim_validator.validate_all([claim], store)
            store.add_claim(claim)
            claims.append(claim)

        else:
            # Scalar aggregation across filtered rows
            scalar_dict: Dict[str, Any] = {}
            for m in plan.metrics:
                s = working_df[m.column].dropna()
                if m.agg == MetricAgg.SUM:
                    val = float(s.sum())
                elif m.agg == MetricAgg.MEAN:
                    val = float(s.mean())
                elif m.agg == MetricAgg.MEDIAN:
                    val = float(s.median())
                elif m.agg == MetricAgg.COUNT:
                    val = int(len(s))
                elif m.agg == MetricAgg.NUNIQUE:
                    val = int(s.nunique())
                elif m.agg == MetricAgg.MIN:
                    val = float(s.min())
                elif m.agg == MetricAgg.MAX:
                    val = float(s.max())
                else:
                    val = float(s.sum())

                scalar_dict[f"{m.agg.value}_{m.column}"] = round(val, 2)

            result_table = [scalar_dict]

            # Mint Evidence
            ev = EvidenceItem(
                evidence_id=f"ev_scalar_{uuid.uuid4().hex[:6]}",
                evidence_type=EvidenceType.DETERMINISTIC_CALC,
                source_id="flexible_query_scalar",
                location="dataset_aggregate",
                content={
                    "scalars": scalar_dict,
                    "rows_before": initial_rows,
                    "rows_after": filtered_rows,
                    "filters_applied": applied_filter_notes,
                },
                confidence=1.0,
            )
            store.add_evidence(ev)

            # Build Answer statement
            m0 = plan.metrics[0]
            val0 = scalar_dict[f"{m0.agg.value}_{m0.column}"]
            formatted_val = f"${val0:,.2f}" if "rev" in m0.column.lower() or "price" in m0.column.lower() else f"{val0:,}"

            filter_clause = f" where {', '.join(applied_filter_notes)}" if applied_filter_notes else ""
            statement = f"Total {m0.agg.value} of '{m0.column}' across {filtered_rows:,} records{filter_clause} is {formatted_val}."

            claim = Claim(
                statement=statement,
                extracted_numbers=[round(float(val0), 2), float(filtered_rows)],
                status=ClaimStatus.SUPPORTED,
            )
            self.claim_validator.validate_all([claim], store)
            store.add_claim(claim)
            claims.append(claim)

        # Build Chart Spec if appropriate
        chart_spec = None
        if plan.chart:
            chart_spec = plan.chart.model_dump()
        elif plan.group_by and plan.metrics:
            chart_spec = {"type": "bar", "x": plan.group_by[0], "y": plan.metrics[0].column}

        self._persist_results(inv_id, metadata, question or "Flexible query", statement, plan, store)

        return QueryResult(
            status="answered",
            answer=statement,
            result_table=result_table,
            chart_spec=chart_spec,
            evidence_store=store,
            claims=claims,
            plan=plan,
            plain_language_summary=f"Calculated {', '.join(m.agg.value + ' ' + m.column for m in plan.metrics)}.",
            planner_used=plan.planner,
            limitations=limitations,
            investigation_id=inv_id,
        )

    def _build_filter_mask(self, df: pd.DataFrame, f: FilterSpec) -> pd.Series:
        """Construct boolean filter mask using pure pandas operations without eval/query."""
        s = df[f.column]
        op = f.op
        val = f.value

        if op == FilterOp.EQ:
            return s == val
        elif op == FilterOp.NE:
            return s != val
        elif op == FilterOp.GT:
            return pd.to_numeric(s, errors="coerce") > float(val)
        elif op == FilterOp.GTE:
            return pd.to_numeric(s, errors="coerce") >= float(val)
        elif op == FilterOp.LT:
            return pd.to_numeric(s, errors="coerce") < float(val)
        elif op == FilterOp.LTE:
            return pd.to_numeric(s, errors="coerce") <= float(val)
        elif op == FilterOp.IN:
            return s.isin(val)
        elif op == FilterOp.BETWEEN:
            num_s = pd.to_numeric(s, errors="coerce")
            return (num_s >= float(val[0])) & (num_s <= float(val[1]))
        elif op == FilterOp.CONTAINS:
            return s.astype(str).str.contains(str(val), case=False, na=False)
        elif op == FilterOp.IS_NULL:
            return s.isna()
        elif op == FilterOp.NOT_NULL:
            return s.notna()
        else:
            return pd.Series(True, index=df.index)

    def _persist_results(
        self,
        inv_id: str,
        metadata: DatasetMetadata,
        question: str,
        final_answer: str,
        plan: AnalysisPlan,
        store: EvidenceStore,
    ) -> None:
        """Persist investigation results into SQLite repository."""
        sid = getattr(metadata, "session_id", None) or self.repo.session_id
        try:
            self.repo.save_dataset(metadata, session_id=sid)
        except Exception:
            pass
        try:
            self.repo.save_investigation_results(
                investigation_id=inv_id,
                dataset_id=metadata.dataset_id,
                question=question,
                status="completed",
                plan_dict=plan.model_dump(),
                final_answer=final_answer,
                evidence_store=store,
                session_id=sid,
            )
        except Exception as exc:
            logger.warning(f"Could not persist NLQ query results to SQLite: {exc}")
