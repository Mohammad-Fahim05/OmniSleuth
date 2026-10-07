"""LangGraph-based state machine for deterministic investigation workflows."""

from typing import Any, Callable, Dict, List, Optional
import pandas as pd
from omnisleuth.core.logging import get_logger
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.ingestion.models import DatasetMetadata
from omnisleuth.workflow.nodes import (
    classify_question_node,
    create_plan_node,
    execute_tools_node,
    generate_answer_node,
    input_guardrails_node,
    inspect_dataset_node,
    validate_claims_node,
    validate_plan_node,
)
from omnisleuth.workflow.state import InvestigationState

logger = get_logger("workflow.graph")

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    # Lightweight deterministic fallback StateGraph implementation
    START = "__start__"
    END = "__end__"

    class CompiledStateGraph:
        def __init__(self, nodes: Dict[str, Callable], edges: Dict[str, Any], start_node: Optional[str]):
            self.nodes = nodes
            self.edges = edges
            self.start_node = start_node

        def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
            state = dict(initial_state)
            current_node = self.start_node
            while current_node and current_node != END:
                node_func = self.nodes.get(current_node)
                if not node_func:
                    break
                update = node_func(state)
                if isinstance(update, dict):
                    state.update(update)
                if state.get("status") == "blocked":
                    break
                edge_val = self.edges.get(current_node)
                if isinstance(edge_val, tuple):
                    path_fn, path_map = edge_val
                    decision = path_fn(state)
                    current_node = path_map.get(decision, END)
                else:
                    current_node = edge_val
            return state

    class StateGraph:
        def __init__(self, state_schema: Any):
            self.nodes: Dict[str, Callable] = {}
            self.edges: Dict[str, Any] = {}
            self.start_node: Optional[str] = None

        def add_node(self, name: str, action: Callable) -> None:
            self.nodes[name] = action

        def add_edge(self, start_key: str, end_key: str) -> None:
            if start_key == START:
                self.start_node = end_key
            else:
                self.edges[start_key] = end_key

        def add_conditional_edges(self, source: str, path: Callable, path_map: Dict[str, str]) -> None:
            self.edges[source] = (path, path_map)

        def compile(self) -> CompiledStateGraph:
            return CompiledStateGraph(self.nodes, self.edges, self.start_node)


def _route_after_input_guardrails(state: InvestigationState) -> str:
    """Conditional routing: Halt immediately at END if input guardrail blocked the request."""
    if state.get("status") == "blocked":
        return END
    return "classify_question"


def build_investigation_graph():
    """Build and compile the Phase 8 & Phase 15 LangGraph state graph."""
    graph = StateGraph(InvestigationState)

    # 1. Add nodes
    graph.add_node("input_guardrails", input_guardrails_node)
    graph.add_node("classify_question", classify_question_node)
    graph.add_node("inspect_dataset", inspect_dataset_node)
    graph.add_node("create_plan", create_plan_node)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("validate_claims", validate_claims_node)
    graph.add_node("generate_answer", generate_answer_node)

    # 2. Add edges with conditional guardrail routing
    graph.add_edge(START, "input_guardrails")
    graph.add_conditional_edges(
        "input_guardrails",
        _route_after_input_guardrails,
        {
            "classify_question": "classify_question",
            END: END,
        },
    )
    graph.add_edge("classify_question", "inspect_dataset")
    graph.add_edge("inspect_dataset", "create_plan")
    graph.add_edge("create_plan", "validate_plan")
    graph.add_edge("validate_plan", "execute_tools")
    graph.add_edge("execute_tools", "validate_claims")
    graph.add_edge("validate_claims", "generate_answer")
    graph.add_edge("generate_answer", END)

    compiled_graph = graph.compile()
    logger.info("Compiled LangGraph investigation workflow successfully with guardrails.")
    return compiled_graph


def run_investigation(
    question: str,
    df: pd.DataFrame,
    metadata: DatasetMetadata,
) -> InvestigationState:
    """Convenience entry point to run an end-to-end investigation."""
    store = EvidenceStore(dataset_id=metadata.dataset_id, dataset_version=metadata.version_hash)
    compiled_graph = build_investigation_graph()

    initial_state: InvestigationState = {
        "question": question,
        "dataset_metadata": metadata,
        "df": df,
        "evidence_store": store,
        "input_guardrail_result": None,
        "output_guardrail_result": None,
        "guarded_answer": None,
        "question_intent": "",
        "target_metric": None,
        "target_dimension": None,
        "target_time_period": None,
        "plan": [],
        "plan_is_valid": False,
        "plan_validation_notes": None,
        "tool_executions": [],
        "claims": [],
        "validation_results": [],
        "final_answer": "",
        "investigation_tree": {},
        "status": "started",
        "error": None,
    }

    final_state = compiled_graph.invoke(initial_state)
    return final_state
