"""Benchmark evaluation queries and ground truth annotations for contextual RAG."""

from typing import List
from pydantic import BaseModel, Field


class RetrievalBenchmarkCase(BaseModel):
    """An individual test case for evaluating retrieval quality."""

    case_id: str
    query: str
    relevant_chunk_ids: List[str] = Field(description="Ground truth chunk IDs containing the necessary facts")
    relevant_doc_ids: List[str] = Field(description="Ground truth document IDs")
    description: str


RETRIEVAL_BENCHMARK_CASES: List[RetrievalBenchmarkCase] = [
    RetrievalBenchmarkCase(
        case_id="rag_eval_01",
        query="What operational issue caused the March Electronics inventory stockout?",
        relevant_chunk_ids=["doc_q1_report_chunk_01"],
        relevant_doc_ids=["doc_q1_report"],
        description="Supplier grid outage in North region during March 2024",
    ),
    RetrievalBenchmarkCase(
        case_id="rag_eval_02",
        query="How is Average Order Value (AOV) calculated in the business metrics?",
        relevant_chunk_ids=["doc_kpi_defs_chunk_01"],
        relevant_doc_ids=["doc_kpi_defs"],
        description="AOV formula: Revenue / Order Volume",
    ),
    RetrievalBenchmarkCase(
        case_id="rag_eval_03",
        query="What are the valid product categories and fields in the sales schema?",
        relevant_chunk_ids=["doc_data_dict_chunk_01"],
        relevant_doc_ids=["doc_data_dict"],
        description="Data dictionary field definitions",
    ),
    RetrievalBenchmarkCase(
        case_id="rag_eval_04",
        query="What is the standard policy regarding attribution vs definite causality?",
        relevant_chunk_ids=["doc_biz_rules_chunk_01"],
        relevant_doc_ids=["doc_biz_rules"],
        description="Business rule 2 on correlation vs causal claims",
    ),
    RetrievalBenchmarkCase(
        case_id="rag_eval_05",
        query="When did operations resume normal capacity after the component supplier outage?",
        relevant_chunk_ids=["doc_q1_report_chunk_01"],
        relevant_doc_ids=["doc_q1_report"],
        description="Resumption in April 2024 following supply line reactivation",
    ),
    RetrievalBenchmarkCase(
        case_id="rag_eval_06",
        query="What statistical threshold is required before flagging an extreme anomaly transaction?",
        relevant_chunk_ids=["doc_biz_rules_chunk_01"],
        relevant_doc_ids=["doc_biz_rules"],
        description="3 standard deviations outlier flagging threshold",
    ),
    RetrievalBenchmarkCase(
        case_id="rag_eval_07",
        query="How is period-over-period segment contribution percentage defined?",
        relevant_chunk_ids=["doc_kpi_defs_chunk_01"],
        relevant_doc_ids=["doc_kpi_defs"],
        description="Segment contribution formula (Delta_segment / Delta_total) * 100%",
    ),
    RetrievalBenchmarkCase(
        case_id="rag_eval_08",
        query="What does the is_returned flag indicate in transaction records?",
        relevant_chunk_ids=["doc_data_dict_chunk_01"],
        relevant_doc_ids=["doc_data_dict"],
        description="Boolean flag indicating order refund or reversal",
    ),
    RetrievalBenchmarkCase(
        case_id="rag_eval_09",
        query="Which geographical sales territory suffered high-margin Electronics stockouts in March?",
        relevant_chunk_ids=["doc_q1_report_chunk_01"],
        relevant_doc_ids=["doc_q1_report"],
        description="North regional distribution center stockouts",
    ),
    RetrievalBenchmarkCase(
        case_id="rag_eval_10",
        query="What is the definition of gross revenue Recognized across completed customer orders?",
        relevant_chunk_ids=["doc_kpi_defs_chunk_01"],
        relevant_doc_ids=["doc_kpi_defs"],
        description="Revenue definition as sum of gross transaction values",
    ),
]
