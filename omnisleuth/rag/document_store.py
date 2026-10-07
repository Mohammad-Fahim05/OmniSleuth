"""Contextual RAG document store orchestrating BM25, dense vector search, RRF fusion, and semantic reranking."""

from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from omnisleuth.core.config import get_settings
from omnisleuth.core.logging import get_logger
from omnisleuth.rag.bm25 import BM25Retriever
from omnisleuth.rag.dense_embedder import DenseEmbedder
from omnisleuth.rag.faiss_index import FAISSIndex
from omnisleuth.rag.fusion import reciprocal_rank_fusion
from omnisleuth.rag.models import (
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    RetrievalMode,
    RetrievalResult,
)
from omnisleuth.rag.reranker import SemanticReranker
from omnisleuth.rag.seed_docs import SEED_DOCUMENTS

logger = get_logger("rag.document_store")


class DocumentStore:
    """Multi-stage hybrid document store with BM25, FAISS, RRF fusion, and Cross-Encoder reranking."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedder: Optional[DenseEmbedder] = None,
        reranker: Optional[SemanticReranker] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: Dict[str, DocumentMetadata] = {}
        self.chunks: List[DocumentChunk] = []

        self.bm25 = BM25Retriever()
        self.embedder = embedder or DenseEmbedder()
        self.faiss_index = FAISSIndex(dimension=self.embedder.dimension)
        self.reranker = reranker or SemanticReranker()

        self._load_seed_documents()

    def _load_seed_documents(self) -> None:
        """Load default domain documents into the knowledge base and build indices."""
        for doc_id, title, doc_type, text in SEED_DOCUMENTS:
            self.add_document_text(doc_id=doc_id, title=title, doc_type=doc_type, text=text, auto_index=False)
        self._sync_indices()

    def _sync_indices(self) -> None:
        """Synchronize BM25 and FAISS vector indices with current chunk pool."""
        if not self.chunks:
            return

        # 1. Update BM25
        self.bm25.index_chunks(self.chunks)

        # 2. Update Dense Vectors & FAISS
        texts = [c.content for c in self.chunks]
        embeddings = self.embedder.embed_documents(texts)
        self.faiss_index.rebuild(self.chunks, embeddings)
        logger.debug(f"Synced hybrid indices for {len(self.chunks)} document chunks.")

    def split_text(self, text: str) -> List[str]:
        """Split text into logical paragraph-based chunks."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[str] = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) <= self.chunk_size:
                current_chunk += ("\n\n" + p if current_chunk else p)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = p

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def add_document_text(
        self,
        doc_id: str,
        title: str,
        doc_type: DocumentType,
        text: str,
        file_path: Optional[str] = None,
        auto_index: bool = True,
    ) -> DocumentMetadata:
        """Ingest raw text or markdown document into the store."""
        chunks_text = self.split_text(text)
        created_chunks: List[DocumentChunk] = []

        for idx, c_text in enumerate(chunks_text, start=1):
            heading_match = re.search(r"^(?:#+|\*\*)\s*(.+?)(?:\*\*|\n|$)", c_text)
            section = heading_match.group(1).strip() if heading_match else None

            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{idx:02d}",
                doc_id=doc_id,
                doc_title=title,
                doc_type=doc_type,
                content=c_text,
                page_number=1,
                section_title=section,
            )
            created_chunks.append(chunk)

        self.chunks.extend(created_chunks)
        metadata = DocumentMetadata(
            doc_id=doc_id,
            title=title,
            doc_type=doc_type,
            file_path=file_path,
            total_chunks=len(created_chunks),
        )
        self.documents[doc_id] = metadata

        if auto_index:
            self._sync_indices()

        logger.info(f"Ingested document '{title}' ({doc_id}) into {len(created_chunks)} chunks.")
        return metadata

    def add_pdf_file(
        self,
        file_path: Union[str, Path],
        doc_type: DocumentType = DocumentType.QUARTERLY_REPORT,
    ) -> DocumentMetadata:
        """Ingest a PDF file using pypdf."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            title = path.stem.replace("_", " ").title()
            doc_id = f"doc_{path.stem.lower()}"

            created_chunks: List[DocumentChunk] = []
            chunk_seq = 1
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue

                page_chunks = self.split_text(page_text)
                for c_text in page_chunks:
                    chunk = DocumentChunk(
                        chunk_id=f"{doc_id}_p{page_num}_chunk_{chunk_seq:02d}",
                        doc_id=doc_id,
                        doc_title=title,
                        doc_type=doc_type,
                        content=c_text,
                        page_number=page_num,
                    )
                    created_chunks.append(chunk)
                    chunk_seq += 1

            self.chunks.extend(created_chunks)
            meta = DocumentMetadata(
                doc_id=doc_id,
                title=title,
                doc_type=doc_type,
                file_path=str(path.resolve()),
                total_chunks=len(created_chunks),
            )
            self.documents[doc_id] = meta
            self._sync_indices()
            return meta

        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise

    def retrieve(
        self,
        query: str,
        mode: Union[RetrievalMode, str] = RetrievalMode.HYBRID,
        top_k: int = 3,
        doc_type_filter: Optional[DocumentType] = None,
        min_score: float = 0.15,
    ) -> RetrievalResult:
        """Multi-stage retrieval supporting BM25, DENSE, HYBRID (RRF), and RERANKED modes."""
        t_start = time.perf_counter()

        if isinstance(mode, str):
            mode = RetrievalMode(mode.lower())

        settings = get_settings()
        clean_query = query.strip()
        if not clean_query:
            return RetrievalResult(
                query=query,
                mode=mode,
                chunks=[],
                has_relevant_evidence=False,
                top_score=0.0,
                retrieval_notes="Empty or uninformative query.",
                latency_ms=0.0,
            )

        final_chunks: List[DocumentChunk] = []

        if mode == RetrievalMode.BM25:
            # 1. Pure BM25 Lexical Keyword Search
            bm25_scored = self.bm25.search(
                query=clean_query,
                top_k=top_k,
                doc_type_filter=doc_type_filter,
                min_score=min_score,
            )
            for r_idx, (score, chk) in enumerate(bm25_scored, start=1):
                chk.retrieval_mode = RetrievalMode.BM25
                chk.rank = r_idx
                chk.similarity_score = score
                final_chunks.append(chk)

        elif mode == RetrievalMode.DENSE:
            # 2. Pure Dense Neural Vector Search
            query_emb = self.embedder.embed_query(clean_query)
            dense_scored = self.faiss_index.search(
                query_embedding=query_emb,
                top_k=top_k,
                doc_type_filter=doc_type_filter,
                min_score=min_score,
            )
            for r_idx, (score, chk) in enumerate(dense_scored, start=1):
                chk.retrieval_mode = RetrievalMode.DENSE
                chk.rank = r_idx
                chk.similarity_score = score
                final_chunks.append(chk)

        elif mode == RetrievalMode.HYBRID:
            # 3. Hybrid Fusion (BM25 + Dense -> RRF)
            bm25_k = max(top_k * 2, settings.rag_bm25_top_k)
            dense_k = max(top_k * 2, settings.rag_dense_top_k)

            bm25_candidates = self.bm25.search(
                query=clean_query,
                top_k=bm25_k,
                doc_type_filter=doc_type_filter,
            )
            query_emb = self.embedder.embed_query(clean_query)
            dense_candidates = self.faiss_index.search(
                query_embedding=query_emb,
                top_k=dense_k,
                doc_type_filter=doc_type_filter,
            )

            fused = reciprocal_rank_fusion(
                bm25_results=bm25_candidates,
                dense_results=dense_candidates,
                k=settings.rag_rrf_k,
                top_k=top_k,
            )
            for chk in fused:
                chk.retrieval_mode = RetrievalMode.HYBRID
                if chk.similarity_score >= min_score or chk.dense_score is not None:
                    final_chunks.append(chk)

        elif mode == RetrievalMode.RERANKED:
            # 4. Multi-stage: BM25 + Dense -> RRF -> Cross-Encoder Reranker
            bm25_k = max(top_k * 3, settings.rag_bm25_top_k)
            dense_k = max(top_k * 3, settings.rag_dense_top_k)

            bm25_candidates = self.bm25.search(
                query=clean_query,
                top_k=bm25_k,
                doc_type_filter=doc_type_filter,
            )
            query_emb = self.embedder.embed_query(clean_query)
            dense_candidates = self.faiss_index.search(
                query_embedding=query_emb,
                top_k=dense_k,
                doc_type_filter=doc_type_filter,
            )

            hybrid_candidates = reciprocal_rank_fusion(
                bm25_results=bm25_candidates,
                dense_results=dense_candidates,
                k=settings.rag_rrf_k,
                top_k=max(top_k * 2, settings.rag_hybrid_top_k),
            )

            reranked = self.reranker.rerank(
                query=clean_query,
                candidates=hybrid_candidates,
                top_k=top_k,
            )
            for chk in reranked:
                chk.retrieval_mode = RetrievalMode.RERANKED
                if chk.similarity_score >= min_score:
                    final_chunks.append(chk)

        t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        top_score = final_chunks[0].similarity_score if final_chunks else 0.0
        has_relevant = len(final_chunks) > 0 and top_score >= min_score

        notes = (
            f"Retrieved {len(final_chunks)} relevant contextual passages via [{mode.value.upper()}]."
            if has_relevant
            else f"No relevant contextual document evidence found for query via [{mode.value.upper()}]."
        )

        return RetrievalResult(
            query=query,
            mode=mode,
            chunks=final_chunks,
            has_relevant_evidence=has_relevant,
            relevance_threshold=min_score,
            top_score=top_score,
            retrieval_notes=notes,
            latency_ms=t_elapsed_ms,
            total_candidates=len(final_chunks),
        )
