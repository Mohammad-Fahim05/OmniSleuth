"""Benchmark suite with categorized test questions and ground truth expectations."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class BenchmarkQuestion(BaseModel):
    """An individual evaluation test case."""

    id: str
    category: str = Field(description="'planted_finding', 'anomaly_detection', 'rag_context', 'ambiguous_invalid'")
    question: str
    expected_tools: List[str]
    expected_ground_truth_keywords: List[str]
    should_succeed: bool = True
    requires_rag: bool = False


BENCHMARK_QUESTIONS: List[BenchmarkQuestion] = [
    # 1. Planted Findings (10 Cases)
    BenchmarkQuestion(
        id="q_pf_01",
        category="planted_finding",
        question="Why did revenue decline in March?",
        expected_tools=["calculate_trend", "compare_segments", "detect_anomalies"],
        expected_ground_truth_keywords=["March", "Electronics", "North", "decline"],
        requires_rag=True,
    ),
    BenchmarkQuestion(
        id="q_pf_02",
        category="planted_finding",
        question="What is the monthly revenue trend across 2024?",
        expected_tools=["calculate_trend"],
        expected_ground_truth_keywords=["Trend", "delta", "overall"],
    ),
    BenchmarkQuestion(
        id="q_pf_03",
        category="planted_finding",
        question="Compare revenue breakdown across categories",
        expected_tools=["compare_segments"],
        expected_ground_truth_keywords=["Electronics", "Furniture", "Clothing"],
    ),
    BenchmarkQuestion(
        id="q_pf_04",
        category="planted_finding",
        question="Compare total sales across regions",
        expected_tools=["compare_segments"],
        expected_ground_truth_keywords=["North", "South", "East", "West"],
    ),
    BenchmarkQuestion(
        id="q_pf_05",
        category="planted_finding",
        question="Which category contributed most to total revenue?",
        expected_tools=["compare_segments"],
        expected_ground_truth_keywords=["Electronics"],
    ),

    # 2. Anomaly Detection (5 Cases)
    BenchmarkQuestion(
        id="q_anom_01",
        category="anomaly_detection",
        question="Are there any extreme outlier anomaly transactions in revenue?",
        expected_tools=["detect_anomalies"],
        expected_ground_truth_keywords=["outlier", "anomaly", "score"],
    ),
    BenchmarkQuestion(
        id="q_anom_02",
        category="anomaly_detection",
        question="Detect unusual spikes or outliers in sales volume",
        expected_tools=["detect_anomalies"],
        expected_ground_truth_keywords=["anomalies", "detected"],
    ),

    # 3. RAG Questions (5 Cases)
    BenchmarkQuestion(
        id="q_rag_01",
        category="rag_context",
        question="What is the business definition and formula for Average Order Value (AOV)?",
        expected_tools=["profile_dataset"],
        expected_ground_truth_keywords=["Revenue", "Order Volume", "AOV"],
        requires_rag=True,
    ),
    BenchmarkQuestion(
        id="q_rag_02",
        category="rag_context",
        question="What operational issues or supplier outages occurred in Q1 2024?",
        expected_tools=["profile_dataset"],
        expected_ground_truth_keywords=["supplier", "North", "microchip", "stockout"],
        requires_rag=True,
    ),

    # 4. Ambiguous / Invalid Questions (5 Cases)
    BenchmarkQuestion(
        id="q_amb_01",
        category="ambiguous_invalid",
        question="What is the quantum orbital velocity of planet Saturn?",
        expected_tools=["profile_dataset"],
        expected_ground_truth_keywords=[],
        should_succeed=False,
    ),
]
