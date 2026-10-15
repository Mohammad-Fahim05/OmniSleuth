"""Planner routing: one question-to-result path through planning, validation, execution.

This is the single integration point for planner selection. It reuses the
existing :class:`~omnisleuth.nlq.rule_planner.RulePlanner`,
:class:`~omnisleuth.nlq.llm_planner.LLMPlanner`,
:class:`~omnisleuth.nlq.validator.PlanValidator`, and
:class:`~omnisleuth.nlq.executor.PlanExecutor`. It introduces no second planner,
gateway, validator, or executor.

Invariants preserved:
- Deterministic planning (``rule``) is the default and requires no credentials.
- Every plan, from either planner, is validated by the shared ``PlanValidator``.
- Every plan is executed by ``PlanExecutor`` (deterministic calculations only).
- LLM planning never runs implicitly: it must be selected explicitly.
- A failed or fallback LLM response is never treated as an LLM-generated plan.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from omnisleuth.core.config import get_settings
from omnisleuth.core.exceptions import ConfigurationError
from omnisleuth.core.logging import get_logger
from omnisleuth.ingestion.models import DatasetMetadata
from omnisleuth.nlq.charting import ChartBuildResult, build_chart_from_table, infer_example_questions
from omnisleuth.nlq.context import SchemaContext, build_schema_context
from omnisleuth.nlq.executor import PlanExecutor
from omnisleuth.nlq.followup import summarize_result
from omnisleuth.nlq.llm_planner import LLMPlanner, LLMPlanningError
from omnisleuth.nlq.models import PlanIntent, QueryResult
from omnisleuth.nlq.narration import NarrationResult, narrate
from omnisleuth.nlq.planner import Planner
from omnisleuth.nlq.rule_planner import RulePlanner
from omnisleuth.nlq.validator import PlanValidator

logger = get_logger("nlq.pipeline")

RULE_MODE = "rule"
LLM_MODE = "llm"
FALLBACK_RULE = "rule"
FALLBACK_FAIL = "fail"

MODE_RULE = "rule"
MODE_LLM = "llm"


class PlanRoutingResult(BaseModel):
    """Outcome of a routed question, including which planner actually produced the plan."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    requested_mode: str = Field(description="Mode the user selected ('rule' or 'llm')")
    planner_used: str = Field(description="Planner that produced the executed plan ('rule' or 'llm')")
    fallback_used: bool = Field(
        default=False,
        description="True when LLM planning failed and the configured fallback policy applied",
    )
    fallback_reason: Optional[str] = Field(
        default=None, description="Sanitized reason why deterministic fallback was used"
    )
    llm_configured: bool = Field(
        default=False, description="True if the gateway reports a configured provider"
    )
    general_knowledge: bool = Field(
        default=False,
        description="True when the answer is general knowledge rather than a dataset computation",
    )
    deep_investigation: bool = Field(
        default=False, description="True when the question was routed to the multi-agent orchestrator"
    )
    followup_of: Optional[str] = Field(
        default=None, description="Question this follow-up extends, when applicable"
    )
    chart: Optional[ChartBuildResult] = Field(default=None, description="Chart built from the computed table")
    narration: Optional[NarrationResult] = Field(default=None, description="Optional narration outcome")
    history_entry: Dict[str, Any] = Field(default_factory=dict, description="Compact turn summary for follow-ups")
    result: QueryResult


class AnalysisPipeline:
    """Routes a question through planning, validation, and execution.

    ``rule`` mode is the safe default: no credentials are required, no network
    call can happen, and the plan comes from ``RulePlanner``. ``llm`` mode uses
    ``LLMPlanner`` but never skips validation or deterministic execution.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        fallback: Optional[str] = None,
        rule_planner: Optional[RulePlanner] = None,
        llm_planner: Optional[LLMPlanner] = None,
        validator: Optional[PlanValidator] = None,
        executor: Optional[PlanExecutor] = None,
    ):
        self.settings = get_settings()
        self.mode = self._normalize_mode(mode or self.settings.nlq_planner_mode)
        self.fallback = self._normalize_fallback(fallback or self.settings.nlq_planner_fallback)
        self.rule_planner = rule_planner or RulePlanner()
        self._llm_planner = llm_planner
        self.plan_validator = validator or PlanValidator()
        self.executor = executor or PlanExecutor()

    # ── Validation of configuration ───────────────────────────────────────────

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = str(mode or MODE_RULE).strip().lower()
        if normalized not in (MODE_RULE, MODE_LLM):
            raise ConfigurationError(
                f"Invalid NLQ planner mode '{mode}'. Supported values: '{MODE_RULE}', '{MODE_LLM}'."
            )
        return normalized

    @staticmethod
    def _normalize_fallback(fallback: str) -> str:
        normalized = str(fallback or FALLBACK_RULE).strip().lower()
        if normalized not in (FALLBACK_RULE, FALLBACK_FAIL):
            raise ConfigurationError(
                f"Invalid NLQ planner fallback policy '{fallback}'. Supported values: '{FALLBACK_RULE}', '{FALLBACK_FAIL}'."
            )
        return normalized

    # ── Planner selection ─────────────────────────────────────────────────────

    @property
    def llm_planner(self) -> LLMPlanner:
        """Lazily construct the LLM planner so ``rule`` mode never touches LLM code paths."""
        if self._llm_planner is None:
            self._llm_planner = LLMPlanner()
        return self._llm_planner

    def is_llm_available(self) -> bool:
        """True if LLM planning is selected AND a provider is actually configured."""
        if self.mode != MODE_LLM:
            return False
        try:
            return bool(self.llm_planner.gateway.is_configured)
        except Exception:  # pragma: no cover - defensive
            return False

    # ── Question to result ────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        history_ctx: Optional[List[Dict[str, Any]]] = None,
    ) -> PlanRoutingResult:
        """Answer a question: plan, validate, execute — never the reverse order."""
        schema_ctx = build_schema_context(df, metadata)
        history = list(history_ctx or [])

        if self.mode == MODE_LLM:
            return self._run_llm(question, schema_ctx, df, metadata, history)
        return self._run_rule(question, schema_ctx, df, metadata, history)

    def _finish(
        self,
        question: str,
        result: QueryResult,
        requested_mode: str,
        planner_used: str,
        schema_ctx: SchemaContext,
        fallback_used: bool = False,
        fallback_reason: Optional[str] = None,
        llm_configured: bool = False,
    ) -> PlanRoutingResult:
        """Post-process an executed result identically for both planner paths."""
        plan = result.plan

        chart = build_chart_from_table(
            result.result_table,
            requested_type=(plan.chart.type.value if plan.chart else None),
        )
        result.chart_spec = chart.spec if chart.used else None

        narration: Optional[NarrationResult] = None
        if self.settings.nlq_narration_enabled and result.status == "answered" and self.mode == MODE_LLM:
            narration = narrate(
                result.answer,
                self.llm_planner.gateway,
                result.evidence_store,
                question=question,
            )
            if narration.used and narration.text:
                result.answer = narration.text

        return PlanRoutingResult(
            requested_mode=requested_mode,
            planner_used=planner_used,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            llm_configured=llm_configured,
            general_knowledge=plan.intent == PlanIntent.GENERAL_KNOWLEDGE,
            deep_investigation=plan.intent == PlanIntent.DEEP_INVESTIGATION,
            followup_of=plan.followup_of,
            chart=chart,
            narration=narration,
            history_entry=summarize_result(question, result),
            result=result,
        )

    def _run_rule(
        self,
        question: str,
        schema_ctx: SchemaContext,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        history_ctx: Optional[List[Dict[str, Any]]] = None,
    ) -> PlanRoutingResult:
        """Deterministic path: RulePlanner -> PlanValidator -> PlanExecutor."""
        plan = self.rule_planner.plan(question, schema_ctx, history_ctx)
        validated = self.plan_validator.validate(plan, schema_ctx)
        result = self.executor.execute(validated, df, metadata, question)

        return self._finish(
            question=question,
            result=result,
            requested_mode=MODE_RULE,
            planner_used=MODE_RULE,
            schema_ctx=schema_ctx,
            llm_configured=self._gateway_configured(),
        )

    def _run_llm(
        self,
        question: str,
        schema_ctx: SchemaContext,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        history_ctx: Optional[List[Dict[str, Any]]] = None,
    ) -> PlanRoutingResult:
        """LLM path with the explicitly configured fallback policy."""
        llm_configured = self._gateway_configured()

        try:
            plan = self.llm_planner.plan(question, schema_ctx, history_ctx)
            planner_used = MODE_LLM
            fallback_used = False
            fallback_reason: Optional[str] = None
        except LLMPlanningError as exc:
            reason = self._sanitized_reason(exc)
            logger.warning(f"LLM planning failed [{getattr(exc, 'category', 'unknown')}]: {reason}")

            if self.fallback == FALLBACK_FAIL:
                fallback_plan = self.rule_planner.plan(question, schema_ctx, history_ctx)
                failed = QueryResult(
                    status="failed",
                    answer=f"LLM planning failed: {reason}",
                    plan=fallback_plan,
                    plain_language_summary="LLM planning failed and deterministic fallback is disabled.",
                    planner_used=MODE_LLM,
                    limitations=[reason],
                )
                return self._finish(
                    question=question,
                    result=failed,
                    requested_mode=MODE_LLM,
                    planner_used=MODE_LLM,
                    schema_ctx=schema_ctx,
                    fallback_used=False,
                    fallback_reason=reason,
                    llm_configured=llm_configured,
                )

            # Fallback policy 'rule': still answer deterministically, but disclose it.
            planner_used = MODE_RULE
            fallback_used = True
            fallback_reason = reason
            plan = self.rule_planner.plan(question, schema_ctx, history_ctx)

        validated = self.plan_validator.validate(plan, schema_ctx)
        result = self.executor.execute(validated, df, metadata, question)

        return self._finish(
            question=question,
            result=result,
            requested_mode=MODE_LLM,
            planner_used=planner_used,
            schema_ctx=schema_ctx,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            llm_configured=llm_configured,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _gateway_configured(self) -> bool:
        try:
            return bool(self.llm_planner.gateway.is_configured)
        except Exception:  # pragma: no cover - defensive
            return False

    @staticmethod
    def _sanitized_reason(exc: LLMPlanningError) -> str:
        """Build a short, actionable, credential-free reason for display.

        Includes the most specific validation detail available, so a user can act
        on the failure instead of seeing only a generic category.
        """
        category = str(getattr(exc, "category", "planning_error") or "planning_error")
        details = list(getattr(exc, "errors", None) or [])
        detail = str(details[-1]) if details else ""
        if len(detail) > 200:
            detail = detail[:200] + "..."
        message = f"[{category}] {detail}" if detail else f"[{category}] {exc}"
        if len(message) > 300:
            message = message[:300] + "..."
        return message

    # ── Planner protocol passthrough (keeps routing testable without a DataFrame) ──

    def plan_only(
        self,
        question: str,
        schema_ctx: SchemaContext,
        history_ctx: Optional[List[Dict[str, Any]]] = None,
    ):
        """Return the routed, validated plan without executing it.

        Useful for tests and for callers that need only the plan; execution still
        happens exclusively through ``PlanExecutor``.
        """
        history = list(history_ctx or [])
        if self.mode == MODE_LLM:
            try:
                plan = self.llm_planner.plan(question, schema_ctx, history)
            except LLMPlanningError:
                if self.fallback == FALLBACK_FAIL:
                    raise
                plan = self.rule_planner.plan(question, schema_ctx, history)
        else:
            plan = self.rule_planner.plan(question, schema_ctx, history)
        return self.plan_validator.validate(plan, schema_ctx)

    def example_questions(self, df: pd.DataFrame, metadata: DatasetMetadata) -> List[str]:
        """Example questions derived from the active dataset's real columns."""
        return infer_example_questions(build_schema_context(df, metadata))
