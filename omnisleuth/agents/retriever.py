"""Retrieval Agent managing contextual domain knowledge retrieval and RAG evidence."""

from typing import Any, Dict, Optional
from omnisleuth.agents.models import AgentTask
from omnisleuth.core.logging import get_logger
from omnisleuth.evidence.models import EvidenceItem, EvidenceType
from omnisleuth.evidence.store import EvidenceStore
from omnisleuth.rag.document_store import DocumentStore

logger = get_logger("agents.retriever")


class RetrievalAgent:
    """Specialized agent retrieving contextual business documents without calculating numbers."""

    def __init__(self, document_store: Optional[DocumentStore] = None):
        self.doc_store = document_store or DocumentStore()

    def execute_task(
        self,
        task: AgentTask,
        evidence_store: EvidenceStore,
    ) -> AgentTask:
        """Query knowledge base, extract contextual passages, and record RAG evidence."""
        query = task.parameters.get("query", "")
        top_k = task.parameters.get("top_k", 3)
        mode = task.parameters.get("mode", "hybrid")

        logger.info(f"RetrievalAgent searching knowledge base for query: '{query}' [mode={mode}]")

        try:
            retrieval_res = self.doc_store.retrieve(query=query, mode=mode, top_k=top_k)
            evidence_ids = []

            if retrieval_res.has_relevant_evidence:
                for chunk in retrieval_res.chunks:
                    ev_item = EvidenceItem(
                        evidence_type=EvidenceType.RAG_DOCUMENT,
                        source_id=chunk.doc_id,
                        location=chunk.citation,
                        content={
                            "doc_title": chunk.doc_title,
                            "doc_type": chunk.doc_type.value,
                            "section": chunk.section_title,
                            "page": chunk.page_number,
                            "excerpt": chunk.content,
                            "similarity_score": chunk.similarity_score,
                            "retrieval_mode": chunk.retrieval_mode.value if chunk.retrieval_mode else str(mode),
                            "rank": chunk.rank,
                            "bm25_score": chunk.bm25_score,
                            "dense_score": chunk.dense_score,
                            "rrf_score": chunk.rrf_score,
                            "rerank_score": chunk.rerank_score,
                        },
                        confidence=min(1.0, max(0.5, chunk.similarity_score)),
                    )
                    evidence_store.add_evidence(ev_item)
                    evidence_ids.append(ev_item.evidence_id)

            task.status = "completed"
            task.result = retrieval_res.model_dump()
            task.evidence_ids = evidence_ids

        except Exception as exc:
            logger.error(f"RetrievalAgent error on query '{query}': {exc}", exc_info=True)
            task.status = "failed"
            task.result = str(exc)

        return task
