"""Data models for automated dataset discovery and investigation tree navigation."""

from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class FindingCategory(str, Enum):
    """Categorical classification of auto-discovered findings."""

    TREND_INFLECTION = "trend_inflection"
    SEGMENT_CONCENTRATION = "segment_concentration"
    ANOMALY_SPIKE = "anomaly_spike"
    DATA_QUALITY = "data_quality"


class DiscoveredFinding(BaseModel):
    """A noteworthy statistical pattern discovered automatically in the dataset."""

    finding_id: str = Field(default_factory=lambda: f"fnd_{uuid.uuid4().hex[:8]}")
    category: FindingCategory
    title: str
    description: str
    target_metric: str
    target_dimension: Optional[str] = None
    magnitude_score: float = Field(ge=0.0, le=1.0, description="Normalized scale of impact/change")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    composite_rank_score: float = Field(ge=0.0, le=1.0, description="Weighted composite ranking score")
    recommended_question: str = Field(description="Actionable natural-language question for investigation")
    evidence_payload: Dict[str, Any] = Field(default_factory=dict)


class InvestigationTreeNode(BaseModel):
    """Structured hierarchical node in the interactive investigation tree."""

    id: str = Field(default_factory=lambda: f"node_{uuid.uuid4().hex[:8]}")
    label: str
    node_type: str = Field(description="'finding', 'hypothesis', 'evidence', 'test', 'result'")
    status: str = Field(default="verified")
    confidence: float = 1.0
    details: Dict[str, Any] = Field(default_factory=dict)
    children: List["InvestigationTreeNode"] = Field(default_factory=list)


InvestigationTreeNode.model_rebuild()
