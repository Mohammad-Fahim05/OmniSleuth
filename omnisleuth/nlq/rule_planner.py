"""Deterministic RulePlanner mapping user questions to validated AnalysisPlans."""

import calendar
import difflib
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from omnisleuth.nlq.context import ColumnContext, SchemaContext
from omnisleuth.nlq.models import (
    AnalysisPlan,
    ChartSpec,
    ChartType,
    FilterOp,
    FilterSpec,
    MetricAgg,
    MetricSpec,
    PeriodSpec,
    PlanIntent,
    SortDirection,
    SortSpec,
    TimeGrain,
    TimeSpec,
)


class RulePlanner:
    """Deterministic, rule-based planner implementing the Planner Protocol.

    Transforms plain-English questions into allowlisted AnalysisPlan instances
    using pattern matching, synonym maps, and dataset-relative temporal parsing.
    """

    # Common business synonyms mapped to likely column concepts
    SYNONYMS: Dict[str, List[str]] = {
        "revenue": ["sales", "turnover", "income", "amount", "total_sales", "spending"],
        "order_id": ["order", "orders", "transaction", "transactions", "invoices"],
        "category": ["categories", "segment", "segments", "product_category", "type", "types", "dept", "department", "departments"],
        "region": ["regions", "territory", "territories", "geography", "location", "locations", "area", "areas", "market", "zone"],
        "customer_id": ["customer", "customers", "client", "clients", "user", "users", "buyer", "buyers", "account"],
        "order_date": ["date", "dates", "timestamp", "time", "day", "days", "month", "months", "year", "years"],
        "quantity": ["quantities", "qty", "volume", "units", "items"],
        "discount": ["discounts", "rebate", "rebates", "discount_rate", "markdown"],
        "profit": ["profits", "margin", "margins", "net_income", "earnings"],
    }

    def __init__(self, max_question_length: int = 500):
        self.max_question_length = max_question_length

    def plan(
        self,
        question: str,
        schema_ctx: SchemaContext,
        history_ctx: Optional[List[Dict[str, Any]]] = None,
    ) -> AnalysisPlan:
        """Produce an AnalysisPlan from question and schema context."""
        # 1. Sanitize untrusted question input
        sanitized_q = self._sanitize_question(question)
        if not sanitized_q:
            return AnalysisPlan(
                intent=PlanIntent.UNSUPPORTED,
                confidence=0.0,
                unsupported_reason="Empty or blank question received.",
                planner="deterministic",
            )

        q_lower = sanitized_q.lower()

        # 2. Check for general knowledge / out-of-scope inquiries
        if self._is_general_knowledge(q_lower):
            return AnalysisPlan(
                intent=PlanIntent.GENERAL_KNOWLEDGE,
                confidence=1.0,
                unsupported_reason="General knowledge question outside of dataset analytics scope.",
                planner="deterministic",
            )

        # 3. Check for documentation / glossary questions
        if any(term in q_lower for term in ("definition of", "formula for", "what is aov", "business rule")):
            return AnalysisPlan(
                intent=PlanIntent.DOCUMENTATION_QUESTION,
                confidence=0.9,
                planner="deterministic",
            )

        # 4. Check for deep diagnostic/causal investigation
        if any(term in q_lower for term in ("why did", "root cause", "explain the drop", "reason for decline", "investigate")):
            return AnalysisPlan(
                intent=PlanIntent.DEEP_INVESTIGATION,
                confidence=0.9,
                planner="deterministic",
            )

        # 5. Check for overview / dataset quality intents
        if any(term in q_lower for term in ("overview", "describe dataset", "summarize dataset", "how many rows", "dataset summary")):
            return AnalysisPlan(
                intent=PlanIntent.DATASET_OVERVIEW,
                confidence=1.0,
                planner="deterministic",
            )

        if any(term in q_lower for term in ("missing", "null", "empty values", "data quality", "completeness")):
            return AnalysisPlan(
                intent=PlanIntent.MISSING_VALUES,
                confidence=1.0,
                planner="deterministic",
            )

        # 6. Check for anomaly / outlier questions
        if any(term in q_lower for term in ("anomaly", "anomalies", "outlier", "outliers", "unusual transactions")):
            target_metric = self._resolve_target_numeric_column(q_lower, schema_ctx) or (
                schema_ctx.numeric_columns[0] if schema_ctx.numeric_columns else None
            )
            return AnalysisPlan(
                intent=PlanIntent.ANOMALY,
                target_column=target_metric,
                confidence=0.9,
                planner="deterministic",
            )

        # 7. Check for correlation questions
        if any(term in q_lower for term in ("correlation", "correlate", "relationship between")):
            num_cols = self._extract_multiple_numeric_columns(q_lower, schema_ctx)
            if len(num_cols) >= 2:
                return AnalysisPlan(
                    intent=PlanIntent.CORRELATION,
                    metrics=[MetricSpec(column=c, agg=MetricAgg.MEAN) for c in num_cols[:2]],
                    confidence=0.9,
                    planner="deterministic",
                )

        # 8. Check for contribution / decomposition questions
        if any(term in q_lower for term in ("contribution", "contribute", "contributed", "contributes", "decomposition")):
            target_metric = self._resolve_target_numeric_column(q_lower, schema_ctx) or (
                schema_ctx.numeric_columns[0] if schema_ctx.numeric_columns else None
            )
            cat_col = self._resolve_target_categorical_column(q_lower, schema_ctx)
            # Fallback: a categorical column named directly in a contribution question
            # (e.g. "Which category contributed most to revenue?") without a grouping preposition.
            if not cat_col:
                for col in schema_ctx.categorical_columns:
                    col_ctx_chk = schema_ctx.columns.get(col)
                    if col_ctx_chk and col_ctx_chk.data_type == "identifier":
                        continue
                    c_low = col.lower()
                    plural = c_low[:-1] + "ies" if c_low.endswith("y") else c_low + "s"
                    if re.search(rf"\b{re.escape(c_low)}\b", q_lower) or re.search(rf"\b{re.escape(plural)}\b", q_lower):
                        cat_col = col
                        break
            if target_metric and cat_col:
                return AnalysisPlan(
                    intent=PlanIntent.CONTRIBUTION,
                    target_column=target_metric,
                    group_by=[cat_col],
                    confidence=0.85,
                    planner="deterministic",
                )

        # 9. Check for Period Comparison (e.g. "March vs April", "2023 vs 2024")
        period_a, period_b = self._extract_period_comparison(q_lower, schema_ctx)
        if period_a and period_b:
            target_metric = self._resolve_target_numeric_column(q_lower, schema_ctx) or (
                schema_ctx.numeric_columns[0] if schema_ctx.numeric_columns else None
            )
            if target_metric and schema_ctx.date_columns:
                return AnalysisPlan(
                    intent=PlanIntent.PERIOD_COMPARISON,
                    metrics=[MetricSpec(column=target_metric, agg=MetricAgg.SUM)],
                    time=TimeSpec(date_column=schema_ctx.date_columns[0]),
                    period_a=period_a,
                    period_b=period_b,
                    confidence=0.9,
                    planner="deterministic",
                )

        # 10. Check for Trend / Time-series questions
        is_trend = any(term in q_lower for term in ("trend", "monthly", "over time", "month over month", "quarterly", "trajectory", "growth"))
        if is_trend and schema_ctx.date_columns:
            target_metric = self._resolve_target_numeric_column(q_lower, schema_ctx) or (
                schema_ctx.numeric_columns[0] if schema_ctx.numeric_columns else None
            )
            if target_metric:
                grain = TimeGrain.QUARTER if "quarter" in q_lower else TimeGrain.DAY if "daily" in q_lower else TimeGrain.MONTH
                time_filter = self._extract_time_filter(q_lower, schema_ctx)
                start_val = time_filter.get("start") if time_filter else None
                end_val = time_filter.get("end") if time_filter else None

                return AnalysisPlan(
                    intent=PlanIntent.TREND,
                    metrics=[MetricSpec(column=target_metric, agg=MetricAgg.SUM)],
                    time=TimeSpec(date_column=schema_ctx.date_columns[0], grain=grain, start=start_val, end=end_val),
                    chart=ChartSpec(type=ChartType.LINE, x=schema_ctx.date_columns[0], y=target_metric),
                    confidence=0.95,
                    planner="deterministic",
                )

        # 11. Flexible Query Workhorse (Totals, Averages, Group-Bys, Filterings, Rankings)
        plan = self._plan_flexible_query(q_lower, schema_ctx)
        if plan:
            return plan

        # 12. If question cannot be confidently resolved, return honest unsupported
        return AnalysisPlan(
            intent=PlanIntent.UNSUPPORTED,
            confidence=0.0,
            unsupported_reason=(
                f"I could not identify any specific metric, column, or analytical intent in your question: '{question}'. "
                f"Available columns: {', '.join(sorted(schema_ctx.columns.keys()))}"
            ),
            planner="deterministic",
        )

    def _sanitize_question(self, question: str) -> str:
        """Sanitize untrusted user question: cap length and strip control characters."""
        if not question or not isinstance(question, str):
            return ""
        # Cap length
        q = question[: self.max_question_length]
        # Strip control characters except newline and tab
        q = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", q)
        return q.strip()

    def _is_general_knowledge(self, q_lower: str) -> bool:
        """Identify general knowledge questions unrelated to data analytics."""
        gk_patterns = [
            r"weather\s+in",
            r"who\s+(is|was)",
            r"what\s+is\s+the\s+capital\s+of",
            r"write\s+(a\s+poem|a\s+story|a\s+song)",
            r"tell\s+me\s+a\s+joke",
            r"how\s+far\s+is",
            r"translate\s+to",
        ]
        return any(re.search(pat, q_lower) for pat in gk_patterns)

    def _plan_flexible_query(self, q_lower: str, schema_ctx: SchemaContext) -> Optional[AnalysisPlan]:
        """Construct a structured Query plan for flexible aggregation and filtering."""
        # A. Resolve Aggregation
        agg = MetricAgg.SUM
        if any(w in q_lower for w in ("distinct", "unique")):
            agg = MetricAgg.NUNIQUE
        elif any(w in q_lower for w in ("average", "mean", "avg")):
            agg = MetricAgg.MEAN
        elif any(w in q_lower for w in ("median",)):
            agg = MetricAgg.MEDIAN
        elif any(w in q_lower for w in ("count", "number of", "how many")):
            agg = MetricAgg.COUNT
        elif any(w in q_lower for w in ("maximum", "highest", "max")):
            agg = MetricAgg.MAX
        elif any(w in q_lower for w in ("minimum", "lowest", "min")):
            agg = MetricAgg.MIN

        has_agg_keyword = any(w in q_lower for w in ("total", "sum", "average", "mean", "avg", "median", "count", "how many", "distinct", "unique", "max", "maximum", "highest", "min", "minimum", "lowest", "top", "bottom"))
        has_chart_keyword = any(w in q_lower for w in ("chart", "plot", "graph", "bar", "line", "pie"))

        # B. Resolve Metric Column
        target_metric = self._resolve_target_numeric_column(q_lower, schema_ctx)

        # If user asked for an explicit column that does NOT exist in dataset (e.g. "profit")
        missing_requested = self._detect_missing_column_request(q_lower, schema_ctx)
        if missing_requested:
            return AnalysisPlan(
                intent=PlanIntent.UNSUPPORTED,
                confidence=0.0,
                unsupported_reason=(
                    f"Column '{missing_requested}' does not exist in the dataset. "
                    f"Available columns: {', '.join(sorted(schema_ctx.columns.keys()))}"
                ),
                planner="deterministic",
            )

        # For count / nunique, check if any column was mentioned directly
        if agg in (MetricAgg.COUNT, MetricAgg.NUNIQUE):
            for col_name in schema_ctx.columns.keys():
                col_low = col_name.lower()
                if (
                    re.search(rf"\b{col_low}\b", q_lower)
                    or re.search(rf"\b{col_low}s\b", q_lower)
                    or (col_low.endswith("y") and re.search(rf"\b{col_low[:-1]}ies\b", q_lower))
                ):
                    target_metric = col_name
                    break

        # C. Resolve Group By
        group_by: List[str] = []
        cat_match = self._resolve_target_categorical_column(q_lower, schema_ctx)
        if cat_match:
            # Avoid grouping by cat_match if it's already the target_metric for NUNIQUE or COUNT
            if not (agg in (MetricAgg.NUNIQUE, MetricAgg.COUNT) and target_metric == cat_match and not any(w in q_lower for w in ("by", "per", "across", "breakdown", "each"))):
                group_by.append(cat_match)

        # Look for multiple categorical columns if specified (e.g., "by category and region")
        if group_by:
            for col_name in schema_ctx.categorical_columns:
                col_ctx_grp = schema_ctx.columns.get(col_name)
                if col_name not in group_by and (col_ctx_grp is None or col_ctx_grp.data_type != "identifier"):
                    col_low = col_name.lower()
                    if (
                        re.search(rf"\band\s+{re.escape(col_low)}\b", q_lower)
                        or re.search(rf"\band\s+{re.escape(col_low)}s\b", q_lower)
                    ):
                        group_by.append(col_name)
        group_by = group_by[:3]  # Enforce max 3

        # D. Resolve Filters
        filters = self._extract_filters(q_lower, schema_ctx)

        # Guard: if no target_metric, no group_by, no filters, and no analytical keywords: return None
        if not target_metric and not group_by and not filters and not has_agg_keyword and not has_chart_keyword:
            return None

        if not target_metric:
            if agg in (MetricAgg.COUNT, MetricAgg.NUNIQUE):
                id_cols = [c for c, ctx in schema_ctx.columns.items() if ctx.data_type == "identifier"]
                target_metric = id_cols[0] if id_cols else "*"
            elif schema_ctx.numeric_columns:
                target_metric = schema_ctx.numeric_columns[0]
            else:
                return None

        # Build metric spec
        metrics = [MetricSpec(column=target_metric, agg=agg)]

        # E. Resolve Sort & Limit
        sort_spec: Optional[SortSpec] = None
        limit_val: Optional[int] = None

        top_match = re.search(r"\b(top|highest|best|first)\s+(\d+)\b", q_lower)
        bottom_match = re.search(r"\b(bottom|lowest|worst|last)\s+(\d+)\b", q_lower)

        if top_match:
            limit_val = min(50, max(1, int(top_match.group(2))))
            sort_spec = SortSpec(by=target_metric if target_metric != "*" else "count", direction=SortDirection.DESC)
        elif bottom_match:
            limit_val = min(50, max(1, int(bottom_match.group(2))))
            sort_spec = SortSpec(by=target_metric if target_metric != "*" else "count", direction=SortDirection.ASC)
        elif group_by:
            # Default sorting for grouped metrics
            sort_spec = SortSpec(by=target_metric if target_metric != "*" else "count", direction=SortDirection.DESC)

        # F. Chart selection
        chart_spec: Optional[ChartSpec] = None
        if "chart" in q_lower or "plot" in q_lower or "bar" in q_lower or group_by:
            if group_by:
                chart_spec = ChartSpec(type=ChartType.BAR, x=group_by[0], y=target_metric)

        # G. Time filter
        time_spec = None
        time_filter = self._extract_time_filter(q_lower, schema_ctx)
        if time_filter and schema_ctx.date_columns:
            time_spec = TimeSpec(
                date_column=schema_ctx.date_columns[0],
                start=time_filter.get("start"),
                end=time_filter.get("end"),
            )

        return AnalysisPlan(
            intent=PlanIntent.QUERY,
            metrics=metrics,
            group_by=group_by,
            filters=filters,
            sort=sort_spec,
            limit=limit_val,
            time=time_spec,
            chart=chart_spec,
            confidence=0.85,
            planner="deterministic",
        )

    def _resolve_target_numeric_column(self, q_lower: str, schema_ctx: SchemaContext) -> Optional[str]:
        """Identify which numeric column the question targets."""
        # 1. Exact column name in query
        for col in schema_ctx.numeric_columns:
            pattern = rf"\b{re.escape(col.lower().replace('_', ' '))}\b|\b{re.escape(col.lower())}\b"
            if re.search(pattern, q_lower):
                return col

        # 2. Check synonym maps
        for col in schema_ctx.numeric_columns:
            col_key = col.lower()
            syns = self.SYNONYMS.get(col_key, [])
            for syn in syns:
                if re.search(rf"\b{re.escape(syn)}\b", q_lower):
                    return col

        # 3. Fuzzy matching via difflib on tokens
        tokens = re.findall(r"\b[a-zA-Z_]+\b", q_lower)
        for tok in tokens:
            matches = difflib.get_close_matches(tok, schema_ctx.numeric_columns, n=1, cutoff=0.85)
            if matches:
                return matches[0]

        return None

    def _resolve_target_categorical_column(self, q_lower: str, schema_ctx: SchemaContext) -> Optional[str]:
        """Identify which categorical grouping column the question targets."""
        # 1. Check explicit prepositional patterns first
        for col in schema_ctx.categorical_columns:
            # Skip identifier columns like order_id / customer_id from general grouping
            col_ctx_chk = schema_ctx.columns.get(col)
            if col_ctx_chk and col_ctx_chk.data_type == "identifier":
                if not re.search(rf"\b(by|per|across|top|bottom)\s+{re.escape(col.lower())}\b", q_lower):
                    continue

            c_low = col.lower()
            plural = c_low[:-1] + "ies" if c_low.endswith("y") else c_low + "s"
            explicit_patterns = [
                rf"\bby\s+{re.escape(c_low)}\b",
                rf"\bby\s+{re.escape(plural)}\b",
                rf"\bper\s+{re.escape(c_low)}\b",
                rf"\bacross\s+{re.escape(c_low)}\b",
                rf"\bacross\s+{re.escape(plural)}\b",
                rf"\bfor\s+each\s+{re.escape(c_low)}\b",
                rf"\b{re.escape(c_low)}\s+breakdown\b",
                rf"\b(top|bottom|highest|lowest|best|worst)\s+(\d+)?\s*{re.escape(plural)}\b",
                rf"\b(top|bottom|highest|lowest|best|worst)\s+(\d+)?\s*{re.escape(c_low)}\b",
            ]
            if any(re.search(p, q_lower) for p in explicit_patterns):
                return col

            # Check synonyms with explicit grouping prepositions
            syns = self.SYNONYMS.get(c_low, [])
            for s in syns:
                syn_patterns = [
                    rf"\bby\s+{re.escape(s)}\b",
                    rf"\bper\s+{re.escape(s)}\b",
                    rf"\bacross\s+{re.escape(s)}\b",
                    rf"\bfor\s+each\s+{re.escape(s)}\b",
                    rf"\b{re.escape(s)}\s+breakdown\b",
                    rf"\b(top|bottom|highest|lowest|best|worst)\s+(\d+)?\s*{re.escape(s)}\b",
                ]
                if any(re.search(p, q_lower) for p in syn_patterns):
                    return col

        # 2. If question contains explicit grouping prepositions ("by", "per", "across", "breakdown")
        has_group_prep = any(w in q_lower for w in ("by ", "per ", "across ", "breakdown", "for each "))
        if has_group_prep:
            for col in schema_ctx.categorical_columns:
                col_ctx_chk2 = schema_ctx.columns.get(col)
                if col_ctx_chk2 and col_ctx_chk2.data_type == "identifier":
                    continue
                c_low = col.lower()
                plural = c_low[:-1] + "ies" if c_low.endswith("y") else c_low + "s"
                if re.search(rf"\b{re.escape(c_low)}\b", q_lower) or re.search(rf"\b{re.escape(plural)}\b", q_lower):
                    return col

        return None

    def _detect_missing_column_request(self, q_lower: str, schema_ctx: SchemaContext) -> Optional[str]:
        """Detect if the user explicitly requested a nonexistent column (e.g. 'profit')."""
        known_financial_terms = ["profit", "margin", "tax", "cost", "salary", "expense", "churn"]
        all_cols_lower = [c.lower() for c in schema_ctx.columns.keys()]

        for term in known_financial_terms:
            if re.search(rf"\b{term}\b", q_lower) and term not in all_cols_lower:
                return term
        return None

    def _extract_multiple_numeric_columns(self, q_lower: str, schema_ctx: SchemaContext) -> List[str]:
        """Extract multiple numeric columns for correlation analysis."""
        found = []
        for col in schema_ctx.numeric_columns:
            if re.search(rf"\b{col.lower()}\b", q_lower):
                found.append(col)
        return found

    def _extract_period_comparison(self, q_lower: str, schema_ctx: SchemaContext) -> Tuple[Optional[PeriodSpec], Optional[PeriodSpec]]:
        """Extract period comparison specs (e.g. 'March vs April', '2023 vs 2024')."""
        # Month vs Month pattern
        months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        m_num = {m: f"{i+1:02d}" for i, m in enumerate(months)}

        vs_match = re.search(r"\b([a-zA-Z]+)\s+(vs|versus|compared to)\s+([a-zA-Z]+)\b", q_lower)
        if vs_match:
            w1 = vs_match.group(1).lower()
            w2 = vs_match.group(3).lower()

            # Reference data max year
            ref_year = "2024"
            if schema_ctx.max_date:
                ref_year = schema_ctx.max_date[:4]

            if w1 in m_num and w2 in m_num:
                a_end_day = calendar.monthrange(int(ref_year), int(m_num[w1]))[1]
                b_end_day = calendar.monthrange(int(ref_year), int(m_num[w2]))[1]
                p_a = PeriodSpec(start=f"{ref_year}-{m_num[w1]}-01", end=f"{ref_year}-{m_num[w1]}-{a_end_day:02d}", label=w1.title())
                p_b = PeriodSpec(start=f"{ref_year}-{m_num[w2]}-01", end=f"{ref_year}-{m_num[w2]}-{b_end_day:02d}", label=w2.title())
                return p_a, p_b

        # Year vs Year pattern
        yr_match = re.search(r"\b(\d{4})\s+(vs|versus|compared to)\s+(\d{4})\b", q_lower)
        if yr_match:
            y1 = yr_match.group(1)
            y2 = yr_match.group(3)
            p_a = PeriodSpec(start=f"{y1}-01-01", end=f"{y1}-12-31", label=y1)
            p_b = PeriodSpec(start=f"{y2}-01-01", end=f"{y2}-12-31", label=y2)
            return p_a, p_b

        return None, None

    def _extract_time_filter(self, q_lower: str, schema_ctx: SchemaContext) -> Optional[Dict[str, str]]:
        """Extract temporal boundary filter relative to DATA's max date."""
        ref_year = "2024"
        if schema_ctx.max_date:
            ref_year = schema_ctx.max_date[:4]

        # Year filter: "in 2024", "for 2023"
        yr_m = re.search(r"\b(in|for|during)\s+(\d{4})\b", q_lower)
        if yr_m:
            yr = yr_m.group(2)
            return {"start": f"{yr}-01-01", "end": f"{yr}-12-31"}

        # Quarter filter: "in Q1", "Q2 2024"
        q_m = re.search(r"\b(q[1-4])(?:\s+(\d{4}))?\b", q_lower)
        if q_m:
            q_num = q_m.group(1).upper()
            yr = q_m.group(2) or ref_year
            q_bounds = {
                "Q1": (f"{yr}-01-01", f"{yr}-03-31"),
                "Q2": (f"{yr}-04-01", f"{yr}-06-30"),
                "Q3": (f"{yr}-07-01", f"{yr}-09-30"),
                "Q4": (f"{yr}-10-01", f"{yr}-12-31"),
            }
            s, e = q_bounds[q_num]
            return {"start": s, "end": e}

        # Month filter: "in March", "for May"
        months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        m_num = {m: f"{i+1:02d}" for i, m in enumerate(months)}
        for m in months:
            if re.search(rf"\b(in|for|during)\s+{m}\b", q_lower):
                mn = m_num[m]
                return {"start": f"{ref_year}-{mn}-01", "end": f"{ref_year}-{mn}-31"}

        return None

    def _extract_filters(self, q_lower: str, schema_ctx: SchemaContext) -> List[FilterSpec]:
        """Extract allowlisted filter specifications from natural language query."""
        filters: List[FilterSpec] = []

        # 1. Categorical exact equality (e.g. "for Electronics", "in North")
        for col_name, col_ctx in schema_ctx.columns.items():
            if col_ctx.data_type == "categorical":
                for sample_val in col_ctx.sample_values:
                    s_str = str(sample_val).lower()
                    if len(s_str) >= 2:
                        if s_str.isalnum():
                            m_found = bool(re.search(rf"\b{re.escape(s_str)}\b", q_lower))
                        else:
                            m_found = s_str in q_lower
                        if m_found:
                            filters.append(
                                FilterSpec(column=col_name, op=FilterOp.EQ, value=sample_val)
                            )
                            break

        # 2. Numeric threshold filters (e.g. "revenue > 500", "sales greater than 1000")
        num_patterns = [
            (r">=\s*(\d+(?:\.\d+)?)", FilterOp.GTE),
            (r"<=\s*(\d+(?:\.\d+)?)", FilterOp.LTE),
            (r">\s*(\d+(?:\.\d+)?)", FilterOp.GT),
            (r"<\s*(\d+(?:\.\d+)?)", FilterOp.LT),
            (r"greater\s+than\s+(\d+(?:\.\d+)?)", FilterOp.GT),
            (r"more\s+than\s+(\d+(?:\.\d+)?)", FilterOp.GT),
            (r"less\s+than\s+(\d+(?:\.\d+)?)", FilterOp.LT),
        ]
        for pat, op in num_patterns:
            m = re.search(pat, q_lower)
            if m and schema_ctx.numeric_columns:
                val = float(m.group(1))
                filters.append(
                    FilterSpec(column=schema_ctx.numeric_columns[0], op=op, value=val)
                )
                break

        return filters[:8]
