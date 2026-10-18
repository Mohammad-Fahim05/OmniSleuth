import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid
from omnisleuth.core.exceptions import IngestionError
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
from omnisleuth.ingestion.models import ColumnSchema, DatasetMetadata
from omnisleuth.ingestion.storage import safe_cleanup_file
from omnisleuth.persistence.db import DatabaseManager
from omnisleuth.persistence.serializer import safe_json_dumps, safe_json_loads

logger = get_logger("persistence.repository")


class Repository:
    """Provides high-level transactional persistence operations."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def save_dataset(
        self,
        metadata: DatasetMetadata,
        verify_file: bool = False,
        cleanup_on_failure: bool = False,
        session_id: Optional[str] = None,
    ) -> None:
        """Persist dataset descriptor and schema version atomically with session ownership.

        Args:
            metadata: DatasetMetadata containing descriptor, schema, and file path.
            verify_file: If True, checks that the underlying file exists on disk and is readable.
            cleanup_on_failure: If True, safely deletes the underlying upload file if
                database insertion or commit fails.
            session_id: Optional owner session ID override (defaults to metadata.session_id).

        Raises:
            IngestionError: If file validation, ownership verification, or database persistence fails.
        """
        owner_session = session_id or getattr(metadata, "session_id", None)
        path = Path(metadata.file_path)

        # Enforce cross-session authorization: foreign session cannot overwrite an existing dataset
        with self.db.get_connection() as conn:
            cur = conn.execute("SELECT session_id FROM datasets WHERE dataset_id = ?", (metadata.dataset_id,))
            existing_row = cur.fetchone()
            if existing_row:
                existing_session = existing_row["session_id"]
                if existing_session is not None and existing_session != owner_session:
                    raise IngestionError(
                        f"Access denied: Dataset '{metadata.dataset_id}' belongs to another session."
                    )

        if verify_file:
            if not path.exists() or not path.is_file():
                raise IngestionError(
                    f"Cannot register dataset '{metadata.name}': referenced file does not exist on disk."
                )
            if not os.access(path, os.R_OK):
                raise IngestionError(
                    f"Cannot register dataset '{metadata.name}': referenced file is not readable."
                )

        schema_json = safe_json_dumps([c.model_dump() for c in metadata.columns])
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO datasets
                    (dataset_id, name, file_path, version_hash, row_count, column_count, duplicate_rows, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metadata.dataset_id,
                        metadata.name,
                        str(path.resolve()) if path.exists() else metadata.file_path,
                        metadata.version_hash,
                        metadata.row_count,
                        metadata.column_count,
                        metadata.duplicate_rows,
                        owner_session,
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
            logger.info(f"Saved dataset metadata for '{metadata.name}' ({metadata.dataset_id}) [session={owner_session}]")
        except Exception as exc:
            logger.error(f"Failed to persist dataset '{metadata.name}' to database: {exc}")
            if cleanup_on_failure and path.exists() and path.is_file():
                safe_cleanup_file(path)
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError(
                f"Failed to register dataset '{metadata.name}' in database."
            ) from exc

    def get_dataset(
        self,
        dataset_id: str,
        check_file: bool = True,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch dataset record by ID, optionally validating file existence and session ownership.

        Args:
            dataset_id: Unique dataset identifier.
            check_file: If True, checks whether the referenced file exists and sets 'file_exists'.
            session_id: If provided, enforces that dataset belongs to this session.

        Returns:
            Dictionary representing dataset row or None if not found or inaccessible.
        """
        with self.db.get_connection() as conn:
            if session_id is not None:
                cur = conn.execute(
                    "SELECT * FROM datasets WHERE dataset_id = ? AND session_id = ?",
                    (dataset_id, session_id),
                )
            else:
                cur = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            if check_file:
                res["file_exists"] = Path(res["file_path"]).is_file()
            return res

    def get_dataset_metadata(
        self,
        dataset_id: str,
        session_id: Optional[str] = None,
    ) -> Optional[DatasetMetadata]:
        """Reconstruct DatasetMetadata from SQLite descriptor and schema version without reparsing the file.

        Args:
            dataset_id: Unique dataset identifier.
            session_id: If provided, enforces session ownership.

        Returns:
            Reconstructed DatasetMetadata instance or None if not found or inaccessible.
        """
        record = self.get_dataset(dataset_id, check_file=False, session_id=session_id)
        if not record:
            return None

        with self.db.get_connection() as conn:
            cur = conn.execute(
                "SELECT schema_json FROM dataset_versions WHERE dataset_id = ? AND version_hash = ?",
                (dataset_id, record["version_hash"]),
            )
            ver_row = cur.fetchone()
            columns = []
            if ver_row and ver_row["schema_json"]:
                try:
                    raw_cols = safe_json_loads(ver_row["schema_json"], default=[])
                    columns = [ColumnSchema(**c) for c in raw_cols]
                except Exception as exc:
                    logger.warning(f"Could not parse stored schema_json for '{dataset_id}': {exc}")

            return DatasetMetadata(
                dataset_id=record["dataset_id"],
                name=record["name"],
                file_path=record["file_path"],
                version_hash=record["version_hash"],
                row_count=record["row_count"],
                column_count=record["column_count"],
                duplicate_rows=record["duplicate_rows"],
                columns=columns,
                session_id=record.get("session_id"),
            )

    def load_dataset_dataframe(
        self,
        dataset_id: str,
        session_id: Optional[str] = None,
    ) -> Tuple[Any, DatasetMetadata]:
        """Verify underlying file and lazily load DataFrame for a persisted dataset.

        Args:
            dataset_id: Unique dataset identifier.
            session_id: If provided, enforces session ownership.

        Returns:
            Tuple of (DataFrame, DatasetMetadata).

        Raises:
            IngestionError: If dataset is missing, file does not exist, or file is unreadable.
        """
        path = self.verify_dataset_file(dataset_id, session_id=session_id)
        metadata = self.get_dataset_metadata(dataset_id, session_id=session_id)
        if not metadata:
            raise IngestionError(f"Dataset '{dataset_id}' metadata not found in database.")

        from omnisleuth.ingestion.csv_loader import CSVLoader

        loader = CSVLoader()
        df, _ = loader.load_csv(
            file_path=path,
            dataset_name=metadata.name,
            parse_dates=True,
            session_id=session_id,
        )
        return df, metadata

    def verify_dataset_file(self, dataset_id: str, session_id: Optional[str] = None) -> Path:
        """Verify that a registered dataset's file exists and is readable.

        Args:
            dataset_id: Unique dataset identifier.
            session_id: If provided, enforces that dataset belongs to this session.

        Returns:
            Verified Path to the dataset file.

        Raises:
            IngestionError: If dataset record is missing, inaccessible, or file does not exist or cannot be read.
        """
        dataset = self.get_dataset(dataset_id, check_file=False, session_id=session_id)
        if not dataset:
            raise IngestionError(f"Dataset '{dataset_id}' not found in database.")

        path = Path(dataset["file_path"])
        if not path.exists() or not path.is_file():
            raise IngestionError(f"Dataset file for '{dataset['name']}' is missing from storage.")

        if not os.access(path, os.R_OK):
            raise IngestionError(f"Dataset file for '{dataset['name']}' is not readable.")

        return path

    def list_datasets(
        self,
        check_files: bool = True,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve registered datasets, optionally filtered by session owner.

        Args:
            check_files: If True, annotates each record with 'file_exists' boolean.
            session_id: If provided, only returns datasets belonging to this session.

        Returns:
            List of dataset records.
        """
        with self.db.get_connection() as conn:
            if session_id is not None:
                cursor = conn.execute(
                    "SELECT * FROM datasets WHERE session_id = ? ORDER BY created_at DESC",
                    (session_id,),
                )
            else:
                cursor = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC")
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                if check_files:
                    d["file_exists"] = Path(d["file_path"]).is_file()
                results.append(d)
            return results

    def delete_dataset(
        self,
        dataset_id: str,
        delete_file: bool = True,
        session_id: Optional[str] = None,
    ) -> bool:
        """Atomically delete a dataset and optionally remove its stored file if owned by session.

        Args:
            dataset_id: ID of the dataset to remove.
            delete_file: If True, safely removes the stored file if it is in uploads_dir.
            session_id: If provided, verifies ownership before deletion.

        Returns:
            True if dataset was found and deleted, False otherwise.
        """
        dataset = self.get_dataset(dataset_id, check_file=False, session_id=session_id)
        if not dataset:
            return False

        # Enforce session ownership: unowned request or mismatched session cannot delete owned dataset
        if dataset.get("session_id") is not None and (session_id is None or session_id != dataset["session_id"]):
            return False

        with self.db.get_connection() as conn:
            # Cascade deletion of child investigation records (tool executions, evidence, hypotheses, claims)
            cur = conn.execute("SELECT investigation_id FROM investigations WHERE dataset_id = ?", (dataset_id,))
            inv_ids = [row["investigation_id"] for row in cur.fetchall()]
            for inv_id in inv_ids:
                conn.execute("DELETE FROM tool_executions WHERE investigation_id = ?", (inv_id,))
                conn.execute("DELETE FROM evidence_items WHERE investigation_id = ?", (inv_id,))
                conn.execute("DELETE FROM hypotheses WHERE investigation_id = ?", (inv_id,))
                conn.execute("DELETE FROM claims WHERE investigation_id = ?", (inv_id,))

            conn.execute("DELETE FROM dataset_versions WHERE dataset_id = ?", (dataset_id,))
            conn.execute("DELETE FROM investigations WHERE dataset_id = ?", (dataset_id,))
            conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
            conn.commit()

        if delete_file and dataset.get("file_path"):
            safe_cleanup_file(dataset["file_path"])

        logger.info(f"Deleted dataset '{dataset['name']}' ({dataset_id})")
        return True

    def list_investigations(
        self,
        dataset_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve registered investigations, optionally filtered by dataset and session."""
        with self.db.get_connection() as conn:
            if dataset_id and session_id is not None:
                cur = conn.execute(
                    "SELECT investigation_id, dataset_id, question, status, created_at, session_id "
                    "FROM investigations WHERE dataset_id = ? AND session_id = ? ORDER BY created_at DESC",
                    (dataset_id, session_id),
                )
            elif dataset_id:
                cur = conn.execute(
                    "SELECT investigation_id, dataset_id, question, status, created_at, session_id "
                    "FROM investigations WHERE dataset_id = ? ORDER BY created_at DESC",
                    (dataset_id,),
                )
            elif session_id is not None:
                cur = conn.execute(
                    "SELECT investigation_id, dataset_id, question, status, created_at, session_id "
                    "FROM investigations WHERE session_id = ? ORDER BY created_at DESC",
                    (session_id,),
                )
            else:
                cur = conn.execute(
                    "SELECT investigation_id, dataset_id, question, status, created_at, session_id "
                    "FROM investigations ORDER BY created_at DESC"
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
        session_id: Optional[str] = None,
    ) -> None:
        """Atomically persist investigation run, executions, evidence, hypotheses, and claims with session isolation."""
        # Enforce target dataset ownership if session_id is specified
        if session_id is not None:
            ds = self.get_dataset(dataset_id, check_file=False, session_id=session_id)
            if not ds:
                raise IngestionError(
                    f"Access denied: Target dataset '{dataset_id}' not found in current session."
                )
        else:
            # Inherit session_id from dataset record if available
            with self.db.get_connection() as conn:
                cur = conn.execute("SELECT session_id FROM datasets WHERE dataset_id = ?", (dataset_id,))
                r = cur.fetchone()
                if r and r["session_id"]:
                    session_id = r["session_id"]

        with self.db.get_connection() as conn:
            # Enforce investigation ownership: foreign session cannot overwrite an existing investigation
            cur = conn.execute("SELECT session_id FROM investigations WHERE investigation_id = ?", (investigation_id,))
            inv_row = cur.fetchone()
            if inv_row:
                existing_inv_session = inv_row["session_id"]
                if existing_inv_session is not None and existing_inv_session != session_id:
                    raise IngestionError(
                        f"Access denied: Investigation '{investigation_id}' belongs to another session."
                    )

            # 1. Investigations table
            conn.execute(
                """
                INSERT OR REPLACE INTO investigations
                (investigation_id, dataset_id, question, status, plan_json, final_answer,
                 result_table_json, chart_spec_json, planner_used, summary_text, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    session_id,
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

    def get_investigation(
        self,
        investigation_id: str,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch investigation record and associated evidence items, optionally enforcing session ownership."""
        with self.db.get_connection() as conn:
            if session_id is not None:
                cur = conn.execute(
                    "SELECT * FROM investigations WHERE investigation_id = ? AND session_id = ?",
                    (investigation_id, session_id),
                )
            else:
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

            # Tool executions
            cur_tool = conn.execute("SELECT * FROM tool_executions WHERE investigation_id = ?", (investigation_id,))
            tool_list = []
            for r in cur_tool.fetchall():
                td = dict(r)
                td["parameters"] = safe_json_loads(td.get("parameters_json"), default={})
                td["result"] = safe_json_loads(td.get("result_json"), default=None)
                tool_list.append(td)
            res["tool_executions"] = tool_list

            return res

    def reconstruct_investigation_state(
        self,
        investigation_id: str,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct investigation state including EvidenceStore, hypotheses, and claims defensively.

        Ensures session ownership if session_id is provided. Corrupted or malformed
        evidence/hypotheses/claims are skipped or set to safe defaults without crashing rehydration.

        Args:
            investigation_id: ID of the investigation to rehydrate.
            session_id: Optional session identifier to enforce access control.

        Returns:
            Dictionary of reconstructed investigation state, or None if not found or unauthorized.
        """
        inv = self.get_investigation(investigation_id, session_id=session_id)
        if not inv:
            return None

        dataset_id = inv.get("dataset_id", "")
        ds_meta = self.get_dataset_metadata(dataset_id, session_id=session_id) if dataset_id else None
        version_hash = ds_meta.version_hash if ds_meta else "unknown"

        store = EvidenceStore(dataset_id=dataset_id, dataset_version=version_hash)

        # Reconstruct tool executions
        for t_dict in inv.get("tool_executions", []):
            try:
                texec = ToolExecution(
                    execution_id=t_dict.get("execution_id", str(uuid.uuid4())),
                    tool_name=t_dict.get("tool_name", "unknown"),
                    parameters=t_dict.get("parameters", {}),
                    result=t_dict.get("result"),
                    success=bool(t_dict.get("success", True)),
                    error=t_dict.get("error"),
                    execution_time_ms=float(t_dict.get("execution_time_ms", 0.0) or 0.0),
                )
                store.executions[texec.execution_id] = texec
            except Exception as exc:
                logger.warning(f"Skipping corrupt tool execution in {investigation_id}: {exc}")

        # Reconstruct evidence items
        for ev_dict in inv.get("evidence_items", []):
            try:
                ev_type_str = ev_dict.get("evidence_type")
                try:
                    ev_type = EvidenceType(ev_type_str)
                except (ValueError, KeyError):
                    ev_type = EvidenceType.STRUCTURED_DATA

                content = ev_dict.get("content") or {}
                ev_item = EvidenceItem(
                    evidence_id=ev_dict.get("evidence_id", str(uuid.uuid4())),
                    evidence_type=ev_type,
                    source_id=ev_dict.get("source_id", "unknown"),
                    location=ev_dict.get("location"),
                    content=content,
                    confidence=float(ev_dict.get("confidence", 1.0) or 1.0),
                    related_tool_execution_id=ev_dict.get("related_tool_execution_id"),
                )
                store.add_evidence(ev_item)
            except Exception as exc:
                logger.warning(f"Skipping corrupt evidence item in {investigation_id}: {exc}")

        # Reconstruct hypotheses
        for hyp_dict in inv.get("hypotheses", []):
            try:
                st_str = hyp_dict.get("status")
                try:
                    hyp_status = HypothesisStatus(st_str)
                except (ValueError, KeyError):
                    hyp_status = HypothesisStatus.INCONCLUSIVE

                hyp_item = Hypothesis(
                    hypothesis_id=hyp_dict.get("hypothesis_id", str(uuid.uuid4())),
                    statement=hyp_dict.get("statement", "Unspecified hypothesis"),
                    status=hyp_status,
                    confidence=float(hyp_dict.get("confidence", 1.0) or 1.0),
                    findings_summary=hyp_dict.get("findings_summary"),
                )
                store.add_hypothesis(hyp_item)
            except Exception as exc:
                logger.warning(f"Skipping corrupt hypothesis in {investigation_id}: {exc}")

        # Reconstruct claims
        for clm_dict in inv.get("claims", []):
            try:
                st_str = clm_dict.get("status")
                try:
                    clm_status = ClaimStatus(st_str)
                except (ValueError, KeyError):
                    clm_status = ClaimStatus.INCONCLUSIVE

                clm_item = Claim(
                    claim_id=clm_dict.get("claim_id", str(uuid.uuid4())),
                    statement=clm_dict.get("statement", "Unspecified claim"),
                    status=clm_status,
                    validation_notes=clm_dict.get("validation_notes"),
                )
                store.add_claim(clm_item)
            except Exception as exc:
                logger.warning(f"Skipping corrupt claim in {investigation_id}: {exc}")

        plan = inv.get("plan") if isinstance(inv.get("plan"), dict) else {}
        executed_tasks = plan.get("tasks", []) if isinstance(plan, dict) else []

        return {
            "investigation_id": inv["investigation_id"],
            "dataset_id": inv["dataset_id"],
            "dataset_metadata": ds_meta,
            "question": inv.get("question", ""),
            "status": inv.get("status", "completed"),
            "plan": plan,
            "executed_tasks": executed_tasks,
            "evidence_store": store,
            "hypotheses": store.list_hypotheses(),
            "claims": store.list_claims(),
            "validation_results": [],
            "final_answer": inv.get("final_answer", ""),
            "planner_used": inv.get("planner_used"),
            "summary_text": inv.get("summary_text"),
            "result_table": inv.get("result_table"),
            "chart_spec": inv.get("chart_spec"),
            "session_id": inv.get("session_id"),
        }
