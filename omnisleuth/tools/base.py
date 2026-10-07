"""Base classes and registry for allowlisted analytical tools."""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Type
import pandas as pd
from pydantic import BaseModel, Field
from omnisleuth.core.exceptions import ToolExecutionError
from omnisleuth.core.logging import get_logger
from omnisleuth.ingestion.models import DatasetMetadata

logger = get_logger("tools")


class BaseToolInput(BaseModel):
    """Base model for all tool input parameters."""


class BaseToolOutput(BaseModel):
    """Base model for all tool outputs."""

    success: bool = True
    error: Optional[str] = None
    execution_time_ms: float = Field(default=0.0)
    data: Optional[Any] = None


class BaseTool(ABC):
    """Abstract base class for all allowlisted tools."""

    name: str
    description: str
    input_model: Type[BaseToolInput]
    output_model: Type[BaseToolOutput]

    def run(
        self,
        parameters: Dict[str, Any],
        df: pd.DataFrame,
        metadata: DatasetMetadata,
    ) -> BaseToolOutput:
        """Validate input, execute tool deterministically, handle errors, and log timing."""
        start_time = time.perf_counter()
        logger.info(f"Executing allowlisted tool '{self.name}' on dataset '{metadata.dataset_id}'")

        try:
            # 1. Strict input validation via Pydantic
            validated_input = self.input_model(**parameters)

            # 2. Execute concrete analytical logic
            raw_output = self._execute(validated_input, df, metadata)

            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
            logger.info(f"Tool '{self.name}' completed successfully in {elapsed_ms}ms")

            if isinstance(raw_output, BaseToolOutput):
                raw_output.execution_time_ms = elapsed_ms
                return raw_output

            return BaseToolOutput(
                success=True,
                execution_time_ms=elapsed_ms,
                data=raw_output,
            )

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
            err_msg = f"Tool '{self.name}' failed: {str(exc)}"
            logger.error(err_msg, exc_info=True)
            return BaseToolOutput(
                success=False,
                error=err_msg,
                execution_time_ms=elapsed_ms,
                data=None,
            )

    @abstractmethod
    def _execute(
        self,
        params: Any,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
    ) -> Any:
        """Concrete tool implementation using deterministic engines."""


class ToolRegistry:
    """Registry maintaining only allowlisted analytical tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register an allowlisted tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered allowlisted tool: '{tool.name}'")

    def get(self, name: str) -> BaseTool:
        """Retrieve a registered tool by name or raise ToolExecutionError."""
        if name not in self._tools:
            raise ToolExecutionError(
                f"Tool '{name}' is not in allowlist. Available tools: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def list_tools(self) -> List[Dict[str, Any]]:
        """List metadata and JSON schemas for all allowlisted tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_model.model_json_schema(),
            }
            for tool in self._tools.values()
        ]
