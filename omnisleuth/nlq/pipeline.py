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
from omnisleuth.core.rate_limiter import InMemoryRateLimiter, get_rate_limiter
from omnisleuth.guardrails import GuardrailResult, InputGuardrailEngine
from omnisleuth.ingestion.models import DatasetMetadata
from omnisleuth.nlq.charting import ChartBuildResult, build_chart_from_table, infer_example_questions
from omnisleuth.nlq.context import SchemaContext, build_schema_context
from omnisleuth.nlq.executor import PlanExecutor
from omnisleuth.nlq.followup import summarize_result
from omnisleuth.nlq.llm_planner import LLMPlanner, LLMPlanningError
from omnisleuth.nlq.models import AnalysisPlan, PlanIntent, QueryResult
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
    input_guardrail_result: Optional[GuardrailResult] = Field(
        default=None, description="Outcome of input guardrail evaluation"
    )
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
        input_guardrails: Optional[InputGuardrailEngine] = None,
        check_topic_scope: bool = False,
        rate_limiter: Optional[InMemoryRateLimiter] = None,
    ):
        self.settings = get_settings()
        self.mode = self._normalize_mode(mode or self.settings.nlq_planner_mode)
        self.fallback = self._normalize_fallback(fallback or self.settings.nlq_planner_fallback)
        self.rule_planner = rule_planner or RulePlanner()
        self._llm_planner = llm_planner
        self.plan_validator = validator or PlanValidator()
        self.executor = executor or PlanExecutor()
        max_q_len = getattr(self.settings, "nlq_max_question_length", 500)
        self.input_guardrails = input_guardrails or InputGuardrailEngine(max_question_length=max_q_len)
        self.check_topic_scope = check_topic_scope
        self.rate_limiter = rate_limiter or get_rate_limiter()

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
        session_id: Optional[str] = None,
    ) -> PlanRoutingResult:
        """Answer a question: validate input, plan, validate, execute — never the reverse order."""
        # 1. Input Guardrails validation runs before any planner, LLM, or tool invocation.
        guard_res = self.input_guardrails.validate_input(
            question, check_scope=self.check_topic_scope
        )
        if not guard_res.allowed:
            logger.warning(
                "Input guardrail blocked query '%s' [%s]: %s",
                str(question)[:60],
                guard_res.category.value,
                guard_res.reason,
            )
            blocked_plan = AnalysisPlan(
                intent=PlanIntent.UNSUPPORTED,
                confidence=0.0,
                unsupported_reason=guard_res.user_message or guard_res.reason,
                assumptions=[f"Guardrail block ({guard_res.category.value}): {guard_res.reason}"],
                planner="deterministic",
            )
            blocked_result = QueryResult(
                status="blocked",
                answer=f"⚠️ **Request Blocked by Security Guardrails**: {guard_res.user_message}",
                plan=blocked_plan,
                plain_language_summary=f"Query blocked by input guardrails: {guard_res.reason}",
                planner_used=self.mode,
                limitations=[guard_res.reason],
            )
            return PlanRoutingResult(
                requested_mode=self.mode,
                planner_used=self.mode,
                fallback_used=False,
                fallback_reason=None,
                llm_configured=self._gateway_configured(),
                general_knowledge=False,
                deep_investigation=False,
                followup_of=None,
                chart=None,
                narration=None,
                history_entry={},
                input_guardrail_result=guard_res,
                result=blocked_result,
            )

        # 2. In-memory session rate limiting (evaluated before planners, LLMs, or tools)
        effective_session_id = session_id or getattr(metadata, "session_id", None)
        if self.rate_limiter is not None and effective_session_id:
            is_allowed, count, retry_after = self.rate_limiter.check(
                "question", effective_session_id, consume=True
            )
            if not is_allowed:
                limit = self.rate_limiter.get_limit_for_action("question")
                window = self.rate_limiter.get_window_for_action("question")
                window_desc = "minute" if window <= 60.0 else "hour"
                rate_msg = (
                    f"Rate limit exceeded: Session '{effective_session_id}' reached the limit of "
                    f"{limit} questions per {window_desc}. Please try again in {retry_after:.1f}s."
                )
                logger.warning(rate_msg)
                blocked_plan = AnalysisPlan(
                    intent=PlanIntent.UNSUPPORTED,
                    confidence=0.0,
                    unsupported_reason=rate_msg,
                    assumptions=[f"Rate limit exceeded: {limit} questions per {window_desc}"],
                    planner="deterministic",
                )
                blocked_result = QueryResult(
                    status="blocked",
                    answer=f"⚠️ **Rate Limit Exceeded**: {rate_msg}",
                    plan=blocked_plan,
                    plain_language_summary=rate_msg,
                    planner_used=self.mode,
                    limitations=[rate_msg],
                )
                return PlanRoutingResult(
                    requested_mode=self.mode,
                    planner_used=self.mode,
                    fallback_used=False,
                    fallback_reason=None,
                    llm_configured=self._gateway_configured(),
                    general_knowledge=False,
                    deep_investigation=False,
                    followup_of=None,
                    chart=None,
                    narration=None,
                    history_entry={},
                    input_guardrail_result=guard_res,
                    result=blocked_result,
                )

        schema_ctx = build_schema_context(df, metadata)
        history = list(history_ctx or [])

        if self.mode == MODE_LLM:
            return self._run_llm(question, schema_ctx, df, metadata, history, guard_result=guard_res)
        return self._run_rule(question, schema_ctx, df, metadata, history, guard_result=guard_res)

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
        input_guardrail_result: Optional[GuardrailResult] = None,
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
            input_guardrail_result=input_guardrail_result,
            result=result,
        )

    def _run_rule(
        self,
        question: str,
        schema_ctx: SchemaContext,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        history_ctx: Optional[List[Dict[str, Any]]] = None,
        guard_result: Optional[GuardrailResult] = None,
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
            input_guardrail_result=guard_result,
        )

    def _run_llm(
        self,
        question: str,
        schema_ctx: SchemaContext,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        history_ctx: Optional[List[Dict[str, Any]]] = None,
        guard_result: Optional[GuardrailResult] = None,
    ) -> PlanRoutingResult:
        """LLM path with the explicitly configured fallback policy."""
        llm_configured = self._gateway_configured()

        disagreement_note: Optional[str] = None
        try:
            plan = self.llm_planner.plan(question, schema_ctx, history_ctx)
            planner_used = MODE_LLM
            fallback_used = False
            fallback_reason: Optional[str] = None

            # Optional safety net: when the rule planner returns a confident plan
            # whose intent differs from the LLM plan for the same question, record
            # a disagreement note in the result and prefer the rule plan for intents
            # in {contribution, deep_investigation}.
            try:
                rule_plan = self.rule_planner.plan(question, schema_ctx, history_ctx)
                if (
                    rule_plan
                    and rule_plan.intent != PlanIntent.UNSUPPORTED
                    and rule_plan.confidence >= 0.7
                    and rule_plan.intent != plan.intent
                ):
                    disagreement_note = (
                        f"Disagreement note: Rule planner planned intent '{rule_plan.intent.value}' "
                        f"(confidence {rule_plan.confidence:.2f}) while LLM planned '{plan.intent.value}'."
                    )
                    if rule_plan.intent in (PlanIntent.CONTRIBUTION, PlanIntent.DEEP_INVESTIGATION):
                        plan = rule_plan
                        if disagreement_note not in plan.assumptions:
                            plan.assumptions.append(disagreement_note)
                    else:
                        if disagreement_note not in plan.assumptions:
                            plan.assumptions.append(disagreement_note)
            except Exception as e:
                logger.debug(f"Safety net rule planner check failed: {e}")

        except LLMPlanningError as exc:
            disagreement_note = None
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
                    input_guardrail_result=guard_result,
                )

            # Fallback policy 'rule': still answer deterministically, but disclose it.
            planner_used = MODE_RULE
            fallback_used = True
            fallback_reason = reason
            plan = self.rule_planner.plan(question, schema_ctx, history_ctx)

        validated = self.plan_validator.validate(plan, schema_ctx)
        result = self.executor.execute(validated, df, metadata, question)
        if disagreement_note and disagreement_note not in result.limitations:
            result.limitations.append(disagreement_note)

        return self._finish(
            question=question,
            result=result,
            requested_mode=MODE_LLM,
            planner_used=planner_used,
            schema_ctx=schema_ctx,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            llm_configured=llm_configured,
            input_guardrail_result=guard_result,
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

        Guarded against malicious input before invoking any planner.
        """
        guard_res = self.input_guardrails.validate_input(
            question, check_scope=self.check_topic_scope
        )
        if not guard_res.allowed:
            return AnalysisPlan(
                intent=PlanIntent.UNSUPPORTED,
                confidence=0.0,
                unsupported_reason=guard_res.user_message or guard_res.reason,
                assumptions=[f"Guardrail block ({guard_res.category.value}): {guard_res.reason}"],
                planner="deterministic",
            )

        history = list(history_ctx or [])
        if self.mode == MODE_LLM:
            try:
                plan = self.llm_planner.plan(question, schema_ctx, history)
                try:
                    rule_plan = self.rule_planner.plan(question, schema_ctx, history)
                    if (
                        rule_plan
                        and rule_plan.intent != PlanIntent.UNSUPPORTED
                        and rule_plan.confidence >= 0.7
                        and rule_plan.intent != plan.intent
                    ):
                        note = (
                            f"Disagreement note: Rule planner planned intent '{rule_plan.intent.value}' "
                            f"(confidence {rule_plan.confidence:.2f}) while LLM planned '{plan.intent.value}'."
                        )
                        if rule_plan.intent in (PlanIntent.CONTRIBUTION, PlanIntent.DEEP_INVESTIGATION):
                            plan = rule_plan
                            if note not in plan.assumptions:
                                plan.assumptions.append(note)
                        else:
                            if note not in plan.assumptions:
                                plan.assumptions.append(note)
                except Exception as e:
                    logger.debug(f"Safety net rule planner check failed in plan_only: {e}")
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
