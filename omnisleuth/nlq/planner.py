"""Planner Protocol definition for deterministic and LLM-assisted NLQ planners."""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from omnisleuth.nlq.context import SchemaContext
from omnisleuth.nlq.models import AnalysisPlan


@runtime_checkable
class Planner(Protocol):
    """Protocol satisfied by both RulePlanner and LLMPlanner."""

    def plan(
        self,
        question: str,
        schema_ctx: SchemaContext,
        history_ctx: Optional[List[Dict[str, Any]]] = None,
    ) -> AnalysisPlan:
        """Produce an AnalysisPlan from a question and schema context."""
        ...
