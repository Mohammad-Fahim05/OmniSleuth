"""Follow-up question resolution (Phase D).

A follow-up is a short, deliberately underspecified question that inherits its
metric and dimensions from an earlier question in the same session, for example
``And for 2023?`` or ``now by region``.

Rule mode supports a small, explicit set of patterns. LLM mode may handle more,
but every resolved plan — whichever planner produced it — is still validated by
``PlanValidator`` and executed by ``PlanExecutor``.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from omnisleuth.core.logging import get_logger
from omnisleuth.nlq.context import SchemaContext
from omnisleuth.nlq.models import AnalysisPlan, FilterSpec, MetricSpec, PlanIntent, TimeSpec

logger = get_logger("nlq.followup")

# A follow-up marker anchors the start of the question. Deliberately narrow:
# standalone analytical questions must never match.
_FOLLOWUP_START_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^(?:and|then)\s+(?:for|in|by|with|what|how|show|break|split|drill)\b"),
    re.compile(r"^(?:what|how)\s+about\b"),
    re.compile(r"^(?:now|next)\s+(?:show|break|split|drill|compare|filter)\b"),
    re.compile(r"^(?:now|next)\s+by\b"),
    re.compile(r"^same\s+(?:for|but|metric|number|period)\b"),
    re.compile(r"^(?:break|drill)\s+(?:that|it|this|them)\s+down\b"),
    re.compile(r"^(?:instead|also)\s+(?:show|for|by|with|in)\b"),
    re.compile(r"^and\s+for\b"),
)

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_DAYS_IN_MONTH = {2: 29, 4: 30, 6: 30, 9: 30, 11: 30}
_MONTH_ABBR = {v: k[:3] for k, v in _MONTHS.items()}

# Intents whose plans can be extended by a follow-up.
_EXTENDABLE_INTENTS = (
    PlanIntent.QUERY,
    PlanIntent.CHART_REQUEST,
    PlanIntent.TREND,
    PlanIntent.PERIOD_COMPARISON,
    PlanIntent.CONTRIBUTION,
)

MAX_HISTORY_ENTRIES_IN_PROMPT = 3


class FollowupResolution(BaseModel):
    """Outcome of follow-up resolution for a single question."""

    is_followup: bool = False
    marker: Optional[str] = None
    plan: Optional[AnalysisPlan] = None
    needs_clarification: Optional[str] = None
    applied: List[str] = Field(default_factory=list)


def detect_followup_marker(question: str) -> Optional[str]:
    """Return the matched marker text when the question looks like a follow-up."""
    q = (question or "").strip().lower()
    if not q or len(q) > 80:
        return None
    for pattern in _FOLLOWUP_START_PATTERNS:
        if pattern.match(q):
            return q[:40]
    # "by <dimension>" with no other content is also a follow-up.
    if re.match(r"^by\s+[a-z_ ]{2,30}\??$", q):
        return q
    return None


def question_names_a_metric(question: str, schema_ctx: SchemaContext) -> bool:
    """True when the question itself specifies a metric (so it is not a follow-up)."""
    q = (question or "").lower()
    for col in schema_ctx.numeric_columns:
        if col.lower() in q:
            return True
    return False


def summarize_plan(question: str, plan: AnalysisPlan, answer: Optional[str] = None) -> Dict[str, Any]:
    """Build a compact, JSON-safe summary of a completed turn for history_ctx."""
    return {
        "question": (question or "")[:200],
        "intent": plan.intent.value if hasattr(plan.intent, "value") else str(plan.intent),
        "group_by": list(plan.group_by),
        "metrics": [{"column": m.column, "agg": m.agg.value} for m in plan.metrics],
        "filters": [
            {"column": f.column, "op": f.op.value, "value": f.value} for f in plan.filters
        ],
        "time": (
            {"date_column": plan.time.date_column, "start": plan.time.start, "end": plan.time.end}
            if plan.time
            else None
        ),
        "answer": (answer or "")[:400],
    }


def summarize_result(question: str, result: Any) -> Dict[str, Any]:
    """Summarize a QueryResult (or orchestrator result) into a history entry."""
    plan = getattr(result, "plan", None)
    if plan is None and isinstance(result, dict):
        plan = result.get("plan")
    if isinstance(plan, AnalysisPlan):
        return summarize_plan(question, plan, getattr(result, "answer", ""))
    # Orchestrator results carry an investigation plan, not an AnalysisPlan.
    if isinstance(result, dict):
        return {
            "question": (question or "")[:200],
            "intent": PlanIntent.DEEP_INVESTIGATION.value,
            "group_by": [],
            "metrics": [],
            "filters": [],
            "time": None,
            "answer": (result.get("final_answer") or "")[:400],
        }
    return summarize_plan(question, AnalysisPlan(intent=PlanIntent.UNSUPPORTED))


def format_history_for_prompt(history_ctx: List[Dict[str, Any]]) -> str:
    """Bounded, JSON-safe rendering of prior turns for an LLM planner prompt."""
    recent = history_ctx[-MAX_HISTORY_ENTRIES_IN_PROMPT:]
    lines: List[str] = []
    for entry in recent:
        if not isinstance(entry, dict):
            continue
        lines.append(
            "Previous question: {q} | intent: {i} | group_by: {g} | metrics: {m}".format(
                q=entry.get("question", ""),
                i=entry.get("intent", ""),
                g=entry.get("group_by", []),
                m=entry.get("metrics", []),
            )
        )
    if not lines:
        return "(no previous questions in this session)"
    return "\n".join(lines)


def resolve_rule_followup(
    question: str,
    history_ctx: List[Dict[str, Any]],
    schema_ctx: SchemaContext,
) -> Optional[AnalysisPlan]:
    """Resolve a follow-up against the most recent extendable plan.

    Returns an AnalysisPlan (possibly carrying ``needs_clarification``) when the
    question is a follow-up, or ``None`` when it should be handled as a
    standalone question.
    """
    marker = detect_followup_marker(question)
    if marker is None:
        return None

    if not history_ctx:
        # A follow-up with nothing to follow up on: ask, do not guess.
        return AnalysisPlan(
            intent=PlanIntent.QUERY,
            confidence=0.0,
            needs_clarification=(
                f"'{question}' follows up on an earlier question, but there is no earlier question in this "
                "session to extend. Ask the analysis explicitly, for example "
                "'What is total revenue by category?'."
            ),
            planner="deterministic",
        )

    if question_names_a_metric(question, schema_ctx):
        return None

    referent = _latest_extendable(history_ctx)
    if referent is None:
        return AnalysisPlan(
            intent=PlanIntent.QUERY,
            confidence=0.0,
            needs_clarification=(
                f"'{question}' follows up on an earlier question, but the previous turn did not produce a "
                "reusable analytical plan. Ask the analysis explicitly, for example "
                "'What is total revenue by category?'."
            ),
            planner="deterministic",
        )

    q_lower = (question or "").lower()
    overrides: List[str] = []
    group_by = list(referent.get("group_by") or [])
    time_spec = _copy_time(referent.get("time"))
    metrics = [MetricSpec(**m) for m in (referent.get("metrics") or []) if "column" in m]
    filters = [FilterSpec(**f) for f in (referent.get("filters") or []) if "column" in f]

    # ── Temporal override: "and for 2023?" / "and for March?" ──────────────────
    temporal = _extract_temporal(q_lower, schema_ctx)
    if temporal is not None:
        if time_spec is None:
            date_col = schema_ctx.date_columns[0] if schema_ctx.date_columns else None
            if not date_col:
                return AnalysisPlan(
                    intent=PlanIntent.QUERY,
                    confidence=0.0,
                    needs_clarification=(
                        f"Cannot restrict to {temporal[2]} because this dataset has no date column. "
                        f"Available columns: {', '.join(sorted(schema_ctx.columns)) or 'None'}."
                    ),
                    planner="deterministic",
                )
            time_spec = TimeSpec(date_column=date_col, grain="month")
        time_spec.start, time_spec.end = temporal[0], temporal[1]
        overrides.append(f"date range set to {temporal[0]}..{temporal[1]}")
    elif _asks_for_time(q_lower) and schema_ctx.date_columns:
        time_spec = None
        overrides.append("temporal filter cleared")

    # ── Dimension override: "now by region" ───────────────────────────────────
    dimension = _extract_dimension(q_lower, schema_ctx)
    if dimension is not None:
        if dimension is False:
            available = schema_ctx.categorical_columns or sorted(schema_ctx.columns)
            return AnalysisPlan(
                intent=PlanIntent.QUERY,
                confidence=0.0,
                needs_clarification=(
                    "That follow-up name does not match a column in this dataset. "
                    f"Available categorical columns: {', '.join(available) or 'None'}."
                ),
                planner="deterministic",
            )
        group_by = [dimension]
        overrides.append(f"group_by set to {dimension}")

    if not metrics:
        return AnalysisPlan(
            intent=PlanIntent.QUERY,
            confidence=0.0,
            needs_clarification=(
                "The previous question did not define a metric to extend. Ask explicitly, for example "
                "'What is total revenue by category?'."
            ),
            planner="deterministic",
        )

    intent = PlanIntent.QUERY
    if "trend" in q_lower and time_spec is None and schema_ctx.date_columns:
        intent = PlanIntent.TREND
        time_spec = TimeSpec(date_column=schema_ctx.date_columns[0], grain="month")

    return AnalysisPlan(
        intent=intent,
        metrics=metrics,
        group_by=group_by,
        filters=filters,
        time=time_spec,
        confidence=0.75,
        followup_of=str(referent.get("question") or "")[:200] or None,
        assumptions=[f"Follow-up on: {referent.get('question', '')}"] + [f"Applied {o}." for o in overrides],
        planner="deterministic",
    )


def _latest_extendable(history_ctx: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Most recent history entry that carries a usable metric plan."""
    for entry in reversed(history_ctx):
        if not isinstance(entry, dict):
            continue
        if entry.get("intent") in (
            PlanIntent.UNSUPPORTED.value,
            PlanIntent.GENERAL_KNOWLEDGE.value,
            PlanIntent.DEEP_INVESTIGATION.value,
        ):
            continue
        if entry.get("metrics"):
            return entry
    return None


def _copy_time(time_entry: Optional[Dict[str, Any]]) -> Optional[TimeSpec]:
    if not time_entry or not isinstance(time_entry, dict):
        return None
    date_column = time_entry.get("date_column")
    if not date_column:
        return None
    return TimeSpec(
        date_column=date_column,
        grain=time_entry.get("grain") or "month",
        start=time_entry.get("start"),
        end=time_entry.get("end"),
    )


def _extract_temporal(q_lower: str, schema_ctx: SchemaContext) -> Optional[Tuple[str, str, str]]:
    """Return (start, end, label) for an explicit year or month in the question."""
    year_match = _YEAR_RE.search(q_lower)
    if year_match:
        year = int(year_match.group(1))
        return (f"{year}-01-01", f"{year}-12-31", str(year))

    for name, month in _MONTHS.items():
        if name in q_lower or _MONTH_ABBR[month] in q_lower:
            last_day = 31 if month not in _DAYS_IN_MONTH else _DAYS_IN_MONTH[month]
            year = 2024
            if schema_ctx.min_date:
                try:
                    year = int(schema_ctx.min_date[:4])
                except (TypeError, ValueError):
                    pass
            return (f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day}", name.title())
    return None


def _asks_for_time(q_lower: str) -> bool:
    return any(t in q_lower for t in ("all time", "overall", "across all", "remove the", "without the"))


def _extract_dimension(q_lower: str, schema_ctx: SchemaContext):
    """Return the dimension column named by the question, or False when unresolvable."""
    if "by " not in q_lower and "per " not in q_lower and "across " not in q_lower:
        return None

    candidates = [q_lower[q_lower.index(t) + len(t):].strip(" ?.,!") for t in ("by ", "per ", "across ") if t in q_lower]
    phrase = candidates[0] if candidates else ""
    phrase_words = phrase.replace("?", "").strip()

    for col in schema_ctx.categorical_columns:
        if col.lower() == phrase_words:
            return col
        if col.lower().rstrip("s") == phrase_words.rstrip("s"):
            return col
    for col, synonyms in _SYNONYMS.items():
        if col.lower() in phrase_words or any(s in phrase_words for s in synonyms):
            return col
    return False


# Imported late to avoid a module-level cycle with the rule planner's synonym map.
def _load_synonyms() -> Dict[str, List[str]]:
    try:
        from omnisleuth.nlq.rule_planner import RulePlanner

        return RulePlanner.SYNONYMS
    except Exception:  # pragma: no cover - defensive
        return {}


_SYNONYMS = _load_synonyms()
