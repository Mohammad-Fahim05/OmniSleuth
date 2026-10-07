"""Data Analyst Agent executing allowlisted analytical tools and recording evidence."""

from typing import Any, Dict, Optional
import pandas as pd
from omnisleuth.agents.models import AgentTask
from omnisleuth.core.logging import get_logger
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.ingestion.models import DatasetMetadata
from omnisleuth.tools import ToolRegistry, get_default_tool_registry

logger = get_logger("agents.analyst")


class DataAnalystAgent:
    """Specialized agent executing deterministic analytical tools."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or get_default_tool_registry()

    def execute_task(
        self,
        task: AgentTask,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        evidence_store: EvidenceStore,
    ) -> AgentTask:
        """Execute an analytical tool task and record evidence."""
        if not task.tool_name:
            task.status = "failed"
            task.result = "No tool_name specified in task."
            return task

        logger.info(f"DataAnalystAgent executing tool: '{task.tool_name}' with parameters {task.parameters}")

        try:
            tool = self.registry.get(task.tool_name)
            output = tool.run(task.parameters, df, metadata)

            exec_record = evidence_store.record_tool_execution(
                tool_name=task.tool_name,
                parameters=task.parameters,
                output=output,
            )

            task.status = "completed" if output.success else "failed"
            task.result = output.data.model_dump() if hasattr(output.data, "model_dump") else output.data
            task.evidence_ids = [exec_record.execution_id]

        except Exception as exc:
            logger.error(f"DataAnalystAgent error on tool '{task.tool_name}': {exc}", exc_info=True)
            task.status = "failed"
            task.result = str(exc)

        return task
