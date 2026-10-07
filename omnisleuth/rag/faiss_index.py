"""FAISS vector index manager preserving full chunk provenance metadata."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from omnisleuth.core.logging import get_logger
from omnisleuth.rag.models import DocumentChunk, DocumentType

logger = get_logger("rag.faiss_index")


class FAISSIndex:
    """FAISS-backed cosine similarity vector index."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._index = None
        self._use_faiss = True
        self._init_faiss_index()

    def _init_faiss_index(self) -> None:
        """Initialize FAISS Flat Inner Product index."""
        try:
            import faiss

            self._index = faiss.IndexFlatIP(self.dimension)
            self._use_faiss = True
            logger.debug(f"Initialized FAISS IndexFlatIP with dimension {self.dimension}.")
        except Exception as exc:
            logger.warning(f"FAISS initialization note: {exc}. Using NumPy vector index fallback.")
            self._use_faiss = False
            self._index = None

    def rebuild(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Rebuild vector index from scratch with new chunks and embeddings."""
        self.chunks = list(chunks)
        if embeddings is None or len(embeddings) == 0:
            self.embeddings = None
            self._init_faiss_index()
            return

        self.dimension = embeddings.shape[1]
        self.embeddings = np.array(embeddings, dtype=np.float32)

        # Ensure unit normalization for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

        self._init_faiss_index()
        if self._use_faiss and self._index is not None:
            self._index.reset()
            self._index.add(self.embeddings)

        logger.debug(f"FAISS index populated with {len(self.chunks)} vectors.")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        doc_type_filter: Optional[DocumentType] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[float, DocumentChunk]]:
        """Search top-k most similar chunks for query vector."""
        if not self.chunks or self.embeddings is None:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scored_results: List[Tuple[float, DocumentChunk]] = []

        if self._use_faiss and self._index is not None and not doc_type_filter:
            # Fast search across entire index
            k_search = min(top_k * 2, len(self.chunks))
            distances, indices = self._index.search(q_vec, k_search)

            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                score = float(dist)
                if score >= min_score:
                    chunk = self.chunks[idx].model_copy()
                    chunk.dense_score = round(score, 4)
                    scored_results.append((score, chunk))
        else:
            # Filtered or NumPy vector search
            sims = np.dot(self.embeddings, q_vec.T).flatten()
            for idx, score_val in enumerate(sims):
                chunk = self.chunks[idx]
                if doc_type_filter and chunk.doc_type != doc_type_filter:
                    continue
                score = float(score_val)
                if score >= min_score:
                    chunk_copy = chunk.model_copy()
                    chunk_copy.dense_score = round(score, 4)
                    scored_results.append((score, chunk_copy))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[:top_k]
