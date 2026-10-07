"""Specialized multi-agent system module (Supervisor, Analyst, Retriever, Investigator, Orchestrator)."""

from omnisleuth.agents.analyst import DataAnalystAgent
from omnisleuth.agents.investigator import InvestigationAgent
from omnisleuth.agents.models import (
    AgentRole,
    AgentTask,
    InvestigationPlan,
)
from omnisleuth.agents.orchestrator import MultiAgentOrchestrator
from omnisleuth.agents.retriever import RetrievalAgent
from omnisleuth.agents.supervisor import SupervisorAgent

__all__ = [
    "AgentRole",
    "AgentTask",
    "InvestigationPlan",
    "SupervisorAgent",
    "DataAnalystAgent",
    "RetrievalAgent",
    "InvestigationAgent",
    "MultiAgentOrchestrator",
]
