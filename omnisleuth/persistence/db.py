"""SQLite database schema initialization and connection management."""

from pathlib import Path
import sqlite3
from typing import Optional
from omnisleuth.core.config import get_settings
from omnisleuth.core.logging import get_logger

logger = get_logger("persistence.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    version_hash TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    duplicate_rows INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    version_hash TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    schema_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
);

CREATE TABLE IF NOT EXISTS investigations (
    investigation_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    plan_json TEXT,
    final_answer TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
);

CREATE TABLE IF NOT EXISTS tool_executions (
    execution_id TEXT PRIMARY KEY,
    investigation_id TEXT,
    tool_name TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    result_json TEXT,
    success BOOLEAN NOT NULL,
    error TEXT,
    execution_time_ms REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evidence_items (
    evidence_id TEXT PRIMARY KEY,
    investigation_id TEXT,
    evidence_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    location TEXT,
    content_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    related_tool_execution_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    investigation_id TEXT,
    statement TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    findings_summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    investigation_id TEXT,
    statement TEXT NOT NULL,
    status TEXT NOT NULL,
    validation_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class DatabaseManager:
    """Manages SQLite connections and schema migrations."""

    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = Path(db_path or settings.sqlite_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with Row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def initialize_schema(self) -> None:
        """Create database tables if they do not already exist."""
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        logger.info(f"Initialized SQLite database at: {self.db_path}")
