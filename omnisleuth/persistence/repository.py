"""Repository for querying and persisting investigations, datasets, and evidence."""

from typing import Any, Dict, List, Optional
from omnisleuth.core.logging import get_logger
from omnisleuth.evidence.models import (
    Claim,
    ClaimStatus,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
    ToolExecution,
)
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.ingestion.models import DatasetMetadata
from omnisleuth.persistence.db import DatabaseManager
from omnisleuth.persistence.serializer import safe_json_dumps, safe_json_loads

logger = get_logger("persistence.repository")


class Repository:
    """Provides high-level transactional persistence operations."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def save_dataset(self, metadata: DatasetMetadata) -> None:
        """Persist dataset descriptor and schema version."""
        schema_json = safe_json_dumps([c.model_dump() for c in metadata.columns])
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO datasets 
                (dataset_id, name, file_path, version_hash, row_count, column_count, duplicate_rows)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.dataset_id,
                    metadata.name,
                    metadata.file_path,
                    metadata.version_hash,
                    metadata.row_count,
                    metadata.column_count,
                    metadata.duplicate_rows,
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO dataset_versions (version_hash, dataset_id, schema_json)
                VALUES (?, ?, ?)
                """,
                (metadata.version_hash, metadata.dataset_id, schema_json),
            )
            conn.commit()
        logger.info(f"Saved dataset metadata for '{metadata.name}' ({metadata.dataset_id})")

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Retrieve all registered datasets."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def list_investigations(self, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all registered investigations, optionally filtered by dataset."""
        with self.db.get_connection() as conn:
            if dataset_id:
                cur = conn.execute(
                    "SELECT investigation_id, dataset_id, question, status, created_at FROM investigations WHERE dataset_id = ? ORDER BY created_at DESC",
                    (dataset_id,),
                )
            else:
                cur = conn.execute(
                    "SELECT investigation_id, dataset_id, question, status, created_at FROM investigations ORDER BY created_at DESC"
                )
            return [dict(row) for row in cur.fetchall()]

    def save_investigation_results(
        self,
        investigation_id: str,
        dataset_id: str,
        question: str,
        status: str,
        plan_dict: Dict[str, Any],
        final_answer: str,
        evidence_store: EvidenceStore,
        result_table: Optional[List[Dict[str, Any]]] = None,
        chart_spec: Optional[Dict[str, Any]] = None,
        planner_used: Optional[str] = None,
        summary_text: Optional[str] = None,
    ) -> None:
        """Atomically persist investigation run, executions, evidence, hypotheses, and claims.

        ``result_table``, ``chart_spec``, ``planner_used`` and ``summary_text`` are
        optional and purely additive so existing callers keep working.
        """
        with self.db.get_connection() as conn:
            # 1. Investigations table
            conn.execute(
                """
                INSERT OR REPLACE INTO investigations
                (investigation_id, dataset_id, question, status, plan_json, final_answer,
                 result_table_json, chart_spec_json, planner_used, summary_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    investigation_id,
                    dataset_id,
                    question,
                    status,
                    safe_json_dumps(plan_dict),
                    final_answer,
                    safe_json_dumps(result_table) if result_table else None,
                    safe_json_dumps(chart_spec) if chart_spec else None,
                    planner_used,
                    summary_text,
                ),
            )

            # 2. Tool Executions
            for exec_rec in evidence_store.executions.values():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tool_executions
                    (execution_id, investigation_id, tool_name, parameters_json, result_json, success, error, execution_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        exec_rec.execution_id,
                        investigation_id,
                        exec_rec.tool_name,
                        safe_json_dumps(exec_rec.parameters),
                        safe_json_dumps(exec_rec.result) if exec_rec.result is not None else None,
                        exec_rec.success,
                        exec_rec.error,
                        exec_rec.execution_time_ms,
                    ),
                )

            # 3. Evidence Items
            for ev in evidence_store.list_evidence():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO evidence_items
                    (evidence_id, investigation_id, evidence_type, source_id, location, content_json, confidence, related_tool_execution_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ev.evidence_id,
                        investigation_id,
                        ev.evidence_type.value,
                        ev.source_id,
                        ev.location,
                        safe_json_dumps(ev.content),
                        ev.confidence,
                        ev.related_tool_execution_id,
                    ),
                )

            # 4. Hypotheses
            for hyp in evidence_store.hypotheses.values():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO hypotheses
                    (hypothesis_id, investigation_id, statement, status, confidence, findings_summary)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        hyp.hypothesis_id,
                        investigation_id,
                        hyp.statement,
                        hyp.status.value,
                        hyp.confidence,
                        hyp.findings_summary,
                    ),
                )

            # 5. Claims
            for clm in evidence_store.claims.values():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO claims
                    (claim_id, investigation_id, statement, status, validation_notes)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        clm.claim_id,
                        investigation_id,
                        clm.statement,
                        clm.status.value,
                        clm.validation_notes,
                    ),
                )

            conn.commit()
        logger.info(f"Persisted investigation '{investigation_id}' with {len(evidence_store.list_evidence())} evidence items.")

    def get_investigation(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """Fetch investigation record and associated evidence items."""
        with self.db.get_connection() as conn:
            cur = conn.execute("SELECT * FROM investigations WHERE investigation_id = ?", (investigation_id,))
            inv_row = cur.fetchone()
            if not inv_row:
                return None

            res = dict(inv_row)
            res["plan"] = safe_json_loads(res.get("plan_json"), default={})
            res["result_table"] = safe_json_loads(res.get("result_table_json"), default=None)
            res["chart_spec"] = safe_json_loads(res.get("chart_spec_json"), default=None)

            # Evidence items
            cur_ev = conn.execute("SELECT * FROM evidence_items WHERE investigation_id = ?", (investigation_id,))
            ev_list = []
            for r in cur_ev.fetchall():
                ev_d = dict(r)
                ev_d["content"] = safe_json_loads(ev_d.get("content_json"), default={})
                ev_list.append(ev_d)
            res["evidence_items"] = ev_list

            # Hypotheses
            cur_hyp = conn.execute("SELECT * FROM hypotheses WHERE investigation_id = ?", (investigation_id,))
            res["hypotheses"] = [dict(r) for r in cur_hyp.fetchall()]

            # Claims
            cur_clm = conn.execute("SELECT * FROM claims WHERE investigation_id = ?", (investigation_id,))
            res["claims"] = [dict(r) for r in cur_clm.fetchall()]

            return res
