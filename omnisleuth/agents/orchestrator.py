"""Multi-agent orchestrator coordinating supervisor, analyst, retriever, and investigator."""

from typing import Any, Dict, List, Optional
import pandas as pd
from omnisleuth.agents.analyst import DataAnalystAgent
from omnisleuth.agents.investigator import InvestigationAgent
from omnisleuth.agents.models import AgentRole, AgentTask, InvestigationPlan
from omnisleuth.agents.retriever import RetrievalAgent
from omnisleuth.agents.supervisor import SupervisorAgent
from omnisleuth.core.logging import get_logger
from omnisleuth.evidence.models import Claim, ClaimStatus, EvidenceType
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.evidence.validator import ClaimValidator, ValidationResult
from omnisleuth.guardrails import InputGuardrailEngine, OutputGuardrailEngine
from omnisleuth.ingestion.models import DatasetMetadata
from omnisleuth.rag.document_store import DocumentStore
from omnisleuth.tools import ToolRegistry, get_default_tool_registry

logger = get_logger("agents.orchestrator")


class MultiAgentOrchestrator:
    """Coordinates specialized agents over typed shared state and evidence store with guardrails."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        document_store: Optional[DocumentStore] = None,
    ):
        self.registry = tool_registry or get_default_tool_registry()
        self.doc_store = document_store or DocumentStore()

        self.input_guard = InputGuardrailEngine()
        self.supervisor = SupervisorAgent()
        self.analyst = DataAnalystAgent(registry=self.registry)
        self.retriever = RetrievalAgent(document_store=self.doc_store)
        self.investigator = InvestigationAgent()
        self.validator = ClaimValidator()
        self.output_guard = OutputGuardrailEngine(claim_validator=self.validator)

    def run_investigation(
        self,
        question: str,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
    ) -> Dict[str, Any]:
        """Execute full multi-agent investigation lifecycle."""
        logger.info(f"MultiAgentOrchestrator starting investigation for: '{question}'")

        # 1. Input Guardrails Check
        input_guard_res = self.input_guard.validate_input(question)
        if not input_guard_res.allowed:
            logger.warning(f"MultiAgentOrchestrator: Blocked query '{question}' by input guardrails: {input_guard_res.reason}")
            return {
                "investigation_id": "inv_blocked",
                "question": question,
                "dataset_id": metadata.dataset_id,
                "plan": None,
                "executed_tasks": [],
                "claims": [],
                "validation_results": [],
                "evidence_store": None,
                "input_guardrail_result": input_guard_res,
                "output_guardrail_result": None,
                "guarded_answer": None,
                "final_answer": f"⚠️ **Request Blocked by Security Guardrails**: {input_guard_res.user_message}",
                "status": "blocked",
            }

        # 2. Initialize Evidence Store
        evidence_store = EvidenceStore(
            dataset_id=metadata.dataset_id,
            dataset_version=metadata.version_hash,
        )

        # 2. Step 1: Supervisor creates structured plan
        plan = self.supervisor.create_investigation_plan(question, metadata)

        # 3. Step 2 & 3: Delegate tasks to specialized agents
        executed_tasks: List[AgentTask] = []
        for task in plan.tasks:
            logger.info(f"Delegating task '{task.task_id}' to [{task.assigned_to.value}]")

            if task.assigned_to == AgentRole.DATA_ANALYST:
                res_task = self.analyst.execute_task(task, df, metadata, evidence_store)
            elif task.assigned_to == AgentRole.RETRIEVAL:
                res_task = self.retriever.execute_task(task, evidence_store)
            elif task.assigned_to == AgentRole.INVESTIGATOR:
                res_task = self.investigator.execute_task(task, df, metadata, evidence_store)
            else:
                task.status = "skipped"
                res_task = task

            executed_tasks.append(res_task)

        # 4. Step 4: Extract candidate claims from all stored evidence
        claims: List[Claim] = []
        for ev in evidence_store.list_evidence():
            if ev.content.get("overall_delta") is not None:
                delta = ev.content["overall_delta"]
                pct = ev.content.get("overall_change_pct", 0.0)
                metric = ev.content.get("metric", "metric")
                claims.append(
                    Claim(
                        statement=f"Calculated overall {metric} delta across the observed period was {delta} ({pct}%).",
                        extracted_numbers=[delta, pct],
                    )
                )
            elif ev.content.get("grand_total") is not None:
                tot = ev.content["grand_total"]
                claims.append(
                    Claim(
                        statement=f"Grand total aggregate metric value was {tot}.",
                        extracted_numbers=[tot],
                    )
                )
            elif ev.content.get("anomaly_count") is not None:
                anom_cnt = ev.content["anomaly_count"]
                claims.append(
                    Claim(
                        statement=f"Identified {anom_cnt} statistical anomaly records in the dataset.",
                        extracted_numbers=[float(anom_cnt)],
                    )
                )

        # 5. Step 5: Programmatic claim validation
        validation_results = self.validator.validate_all(claims, evidence_store)

        # 6. Step 6: Generate structured synthesis answer
        raw_final_answer = self._synthesize_answer(question, metadata, evidence_store, validation_results)

        # 7. Step 7: Output Guardrails (Secret Redaction, Causal Calibration, Schema Packaging)
        output_guard_res, guarded_answer = self.output_guard.validate_output(
            answer_text=raw_final_answer,
            evidence_store=evidence_store,
            claims=claims,
        )

        logger.info(f"MultiAgentOrchestrator completed investigation with {len(evidence_store.list_evidence())} evidence items.")

        return {
            "question": question,
            "dataset_metadata": metadata,
            "plan": plan,
            "executed_tasks": executed_tasks,
            "evidence_store": evidence_store,
            "hypotheses": list(evidence_store.hypotheses.values()),
            "claims": claims,
            "validation_results": validation_results,
            "input_guardrail_result": input_guard_res,
            "output_guardrail_result": output_guard_res,
            "guarded_answer": guarded_answer,
            "final_answer": guarded_answer.answer,
            "status": "completed",
        }

    def _synthesize_answer(
        self,
        question: str,
        metadata: DatasetMetadata,
        evidence_store: EvidenceStore,
        val_results: List[ValidationResult],
    ) -> str:
        """Construct evidence-grounded final answer with citations."""
        lines = [
            f"# Multi-Agent Investigation Report",
            f"**Question:** *\"{question}\"*",
            f"**Dataset:** `{metadata.name}` (`{metadata.dataset_id}`) | **Rows:** {metadata.row_count:,}",
            "",
            "## 1. Verified Analytical Claims (Programmatically Validated)",
        ]

        if val_results:
            for vr in val_results:
                badge = "✅" if vr.status == ClaimStatus.SUPPORTED else "⚠️" if vr.status == ClaimStatus.INCONCLUSIVE else "❌"
                lines.append(f"- {badge} **[{vr.status.value.upper()}]**: {vr.calibrated_statement or vr.original_statement}")
        else:
            lines.append("- *No discrete numerical claims generated.*")

        lines.append("")
        lines.append("## 2. Tested Hypotheses")
        for hyp in evidence_store.hypotheses.values():
            status_emoji = "🟢" if hyp.status.value == "supported" else "🔴" if hyp.status.value == "rejected" else "⚪"
            lines.append(f"- {status_emoji} **Hypothesis**: {hyp.statement}")
            lines.append(f"  - **Status**: `{hyp.status.value}` (Confidence: {hyp.confidence * 100:.0f}%)")
            if hyp.findings_summary:
                lines.append(f"  - **Evaluation**: {hyp.findings_summary}")

        lines.append("")
        lines.append("## 3. Contextual RAG Business Evidence")
        rag_items = evidence_store.list_evidence(evidence_type=EvidenceType.RAG_DOCUMENT)
        if rag_items:
            for item in rag_items:
                lines.append(f"- 📄 **{item.location}** (Score: {item.confidence:.2f}):")
                lines.append(f"  > *\"{item.content.get('excerpt', '').strip()}\"*")
        else:
            lines.append("- *No contextual business document evidence retrieved or required for this query.*")

        lines.append("")
        lines.append("## 4. Evidence Provenance & Audit Trail")
        for ev in evidence_store.list_evidence():
            lines.append(f"- `[{ev.evidence_id}]` Type: `{ev.evidence_type.value}` | Source: `{ev.source_id}` | Location: `{ev.location}`")

        lines.append("")
        lines.append("> [!NOTE]")
        lines.append("> OmniSleuth enforces deterministic computations and calibrated association language. No unverified causal claims are asserted.")

        return "\n".join(lines)
