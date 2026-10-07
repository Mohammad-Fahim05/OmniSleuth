"""Evidence provenance models, claims, hypotheses, and investigation nodes."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    """Categorical type of evidence item."""

    STRUCTURED_DATA = "structured_data"
    DETERMINISTIC_CALC = "deterministic_calc"
    ML_ANOMALY = "ml_anomaly"
    RAG_DOCUMENT = "rag_document"
    CHART = "chart"


class ClaimStatus(str, Enum):
    """Validation status of an analytical claim."""

    PENDING = "pending"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"


class HypothesisStatus(str, Enum):
    """Evaluation status of an investigation hypothesis."""

    PENDING = "pending"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class ToolExecution(BaseModel):
    """Immutable audit record of an allowlisted tool execution."""

    execution_id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:10]}")
    tool_name: str
    dataset_id: str
    dataset_version: str
    parameters: Dict[str, Any]
    result: Optional[Any] = None
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvidenceItem(BaseModel):
    """A discrete, verifiable unit of evidence grounded in data or retrieval."""

    evidence_id: str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:10]}")
    evidence_type: EvidenceType
    source_id: str = Field(description="ID of the source (e.g. execution_id, doc_id, or dataset_id)")
    location: Optional[str] = Field(default=None, description="Row index, column name, doc page, or section")
    content: Dict[str, Any] = Field(description="Structured content payload or extracted metrics")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    related_tool_execution_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Claim(BaseModel):
    """A factual or numerical statement made during an investigation."""

    claim_id: str = Field(default_factory=lambda: f"clm_{uuid.uuid4().hex[:10]}")
    statement: str
    extracted_numbers: List[float] = Field(default_factory=list)
    extracted_entities: List[str] = Field(default_factory=list)
    status: ClaimStatus = Field(default=ClaimStatus.PENDING)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    validation_notes: Optional[str] = None


class Hypothesis(BaseModel):
    """An analytical hypothesis tested during an investigation."""

    hypothesis_id: str = Field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:10]}")
    statement: str
    rationale: Optional[str] = None
    status: HypothesisStatus = Field(default=HypothesisStatus.PENDING)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    tested_with_tools: List[str] = Field(default_factory=list)
    findings_summary: Optional[str] = None


class InvestigationNode(BaseModel):
    """A node in the hierarchical investigation tree."""

    node_id: str = Field(default_factory=lambda: f"node_{uuid.uuid4().hex[:10]}")
    node_type: str = Field(description="'finding', 'hypothesis', 'evidence', 'action', 'synthesis'")
    title: str
    details: Dict[str, Any] = Field(default_factory=dict)
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    status: str = Field(default="active")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
