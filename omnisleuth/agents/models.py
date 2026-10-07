"""Typed models for multi-agent delegation and structured planning."""

from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Specialized agent roles in OmniSleuth."""

    SUPERVISOR = "supervisor"
    DATA_ANALYST = "data_analyst"
    RETRIEVAL = "retrieval"
    INVESTIGATOR = "investigator"


class AgentTask(BaseModel):
    """A discrete unit of work delegated to a specialized agent."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    assigned_to: AgentRole
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rationale: str
    status: str = Field(default="pending")
    result: Optional[Any] = None
    evidence_ids: List[str] = Field(default_factory=list)


class InvestigationPlan(BaseModel):
    """Structured, bounded investigation plan created by Supervisor."""

    objective: str
    target_metric: str
    target_dimension: Optional[str] = None
    requires_rag: bool = False
    rag_query: Optional[str] = None
    tasks: List[AgentTask] = Field(default_factory=list)
    max_depth: int = 3
    max_tool_calls: int = 10
    max_retries: int = 2
