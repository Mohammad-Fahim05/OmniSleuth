"""Data models for contextual RAG document management and retrieval."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Categorical classification of knowledge documents."""

    DATA_DICTIONARY = "data_dictionary"
    KPI_DEFINITION = "kpi_definition"
    BUSINESS_RULE = "business_rule"
    QUARTERLY_REPORT = "quarterly_report"
    COMPANY_POLICY = "company_policy"
    METHODOLOGY = "methodology"
    GENERAL = "general"


class RetrievalMode(str, Enum):
    """Supported multi-stage retrieval strategy modes."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class DocumentChunk(BaseModel):
    """A single chunk of a knowledge document with provenance and ranking metadata."""

    chunk_id: str = Field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:8]}")
    doc_id: str
    doc_title: str
    doc_type: DocumentType
    content: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    similarity_score: float = Field(default=0.0, description="Relevance score (higher is more relevant)")
    
    # Phase 10 - Provenance and Multi-Stage Rank Metadata
    retrieval_mode: Optional[RetrievalMode] = None
    rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Formatted citation string."""
        page_str = f", p.{self.page_number}" if self.page_number else ""
        sec_str = f" [{self.section_title}]" if self.section_title else ""
        return f"{self.doc_title} ({self.doc_type.value}{page_str}{sec_str})"


class DocumentMetadata(BaseModel):
    """Metadata descriptor for an ingested document."""

    doc_id: str
    title: str
    doc_type: DocumentType
    file_path: Optional[str] = None
    total_chunks: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RetrievalResult(BaseModel):
    """Result of a contextual document retrieval operation."""

    query: str
    mode: RetrievalMode = RetrievalMode.BM25
    chunks: List[DocumentChunk] = Field(default_factory=list)
    has_relevant_evidence: bool = False
    relevance_threshold: float = 0.15
    top_score: float = 0.0
    retrieval_notes: Optional[str] = None
    latency_ms: float = 0.0
    total_candidates: int = 0
