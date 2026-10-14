"""Supervisor Agent responsible for structured planning, delegation, and stopping rules."""

from typing import Any, Dict, List, Optional
import pandas as pd
from omnisleuth.agents.models import AgentRole, AgentTask, InvestigationPlan
from omnisleuth.core.config import get_settings
from omnisleuth.core.exceptions import BoundedLimitExceededError
from omnisleuth.core.logging import get_logger
from omnisleuth.ingestion.models import DatasetMetadata

logger = get_logger("agents.supervisor")


from omnisleuth.llm import LLMGateway, LLMRequest


class SupervisorAgent:
    """Orchestrates structured investigation planning and task delegation."""

    def __init__(self, llm_gateway: Optional[LLMGateway] = None):
        self.settings = get_settings()
        self.llm_gateway = llm_gateway or LLMGateway()

    def create_investigation_plan(
        self,
        question: str,
        metadata: DatasetMetadata,
    ) -> InvestigationPlan:
        """Create a structured, allowlisted plan delegated to specialized agents."""
        logger.info(f"Supervisor generating investigation plan for: '{question}'")
        q_lower = question.lower()

        # 1. Identify primary metric
        target_metric = None
        for col in metadata.numeric_columns:
            if col.lower() in q_lower or col.lower().replace("_", " ") in q_lower:
                target_metric = col
                break
        if not target_metric:
            target_metric = metadata.numeric_columns[0] if metadata.numeric_columns else "revenue"

        # 2. Identify candidate dimensions
        candidate_dims = metadata.categorical_columns
        date_col = metadata.datetime_columns[0] if metadata.datetime_columns else None

        # 3. Determine if RAG context is warranted
        rag_keywords = ["why", "cause", "reason", "issue", "outage", "supplier", "policy", "definition", "report", "event"]
        requires_rag = any(k in q_lower for k in rag_keywords)
        rag_query = f"{question} {target_metric}" if requires_rag else None

        tasks: List[AgentTask] = []

        # Task 1: Dataset Quality Profile (Data Analyst)
        tasks.append(
            AgentTask(
                assigned_to=AgentRole.DATA_ANALYST,
                tool_name="profile_dataset",
                parameters={},
                rationale="Audit dataset distributions and data quality before deep analysis.",
            )
        )

        # Task 2: Trend Analysis (Data Analyst)
        if date_col:
            tasks.append(
                AgentTask(
                    assigned_to=AgentRole.DATA_ANALYST,
                    tool_name="calculate_trend",
                    parameters={"date_column": date_col, "metric_column": target_metric, "frequency": "M"},
                    rationale=f"Calculate monthly time-series trend of {target_metric} to locate inflections.",
                )
            )

        # Task 3: Segment Comparisons & Contributions (Data Analyst)
        for dim in candidate_dims[:2]:  # Limit to top 2 dimensions to respect tool budget
            tasks.append(
                AgentTask(
                    assigned_to=AgentRole.DATA_ANALYST,
                    tool_name="compare_segments",
                    parameters={"segment_column": dim, "metric_column": target_metric},
                    rationale=f"Examine distribution of {target_metric} across {dim}.",
                )
            )

        # Task 4: Anomaly Detection (Data Analyst)
        tasks.append(
            AgentTask(
                assigned_to=AgentRole.DATA_ANALYST,
                tool_name="detect_anomalies",
                parameters={"feature_columns": [target_metric], "method": "iqr"},
                rationale=f"Identify statistical outlier transactions in {target_metric}.",
            )
        )

        # Task 5: Contextual Document Retrieval (Retrieval Agent)
        if requires_rag and rag_query:
            tasks.append(
                AgentTask(
                    assigned_to=AgentRole.RETRIEVAL,
                    parameters={"query": rag_query, "top_k": 3},
                    rationale="Retrieve operational reports and business definitions for domain context.",
                )
            )

        # Task 6: Hypothesis Generation & Investigation (Investigator Agent)
        tasks.append(
            AgentTask(
                assigned_to=AgentRole.INVESTIGATOR,
                parameters={"target_metric": target_metric, "candidate_dimensions": candidate_dims},
                rationale="Generate and systematically evaluate specific explanatory hypotheses.",
            )
        )

        # Enforce Rule 7 Bounded Limits
        if len(tasks) > self.settings.max_tool_calls:
            raise BoundedLimitExceededError(
                f"Generated plan has {len(tasks)} tasks, exceeding max tool calls limit of {self.settings.max_tool_calls}."
            )

        plan = InvestigationPlan(
            objective=question,
            target_metric=target_metric,
            target_dimension=candidate_dims[0] if candidate_dims else None,
            requires_rag=requires_rag,
            rag_query=rag_query,
            tasks=tasks,
            max_depth=self.settings.max_investigation_depth,
            max_tool_calls=self.settings.max_tool_calls,
            max_retries=self.settings.max_retries,
        )

        logger.info(f"Supervisor generated bounded plan with {len(tasks)} tasks.")
        return plan
