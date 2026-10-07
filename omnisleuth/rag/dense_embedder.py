"""Dense neural embedder for semantic vector representation."""

import hashlib
from typing import Dict, List, Optional
import numpy as np
from omnisleuth.core.config import get_settings
from omnisleuth.core.logging import get_logger

logger = get_logger("rag.dense_embedder")


class DenseEmbedder:
    """Computes dense vector representations for documents and queries."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        dimension: int = 384,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.local_embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.dimension = dimension
        self._model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._init_model()

    def _init_model(self) -> None:
        """Initialize sentence transformer model or prepare deterministic fallback."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading dense embedding model: '{self.model_name}'")
            self._model = SentenceTransformer(self.model_name)
            # Update dimension from model
            if hasattr(self._model, "get_embedding_dimension"):
                self.dimension = self._model.get_embedding_dimension()
            elif hasattr(self._model, "get_sentence_embedding_dimension"):
                self.dimension = self._model.get_sentence_embedding_dimension()
        except Exception as exc:
            logger.warning(
                f"Could not load SentenceTransformer '{self.model_name}': {exc}. "
                f"Using deterministic feature embedding fallback."
            )
            self._model = None

    def _fallback_embed(self, text: str) -> np.ndarray:
        """Deterministic hashing-based vector embedding fallback for offline/test environments."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec

        for idx, word in enumerate(words):
            h_int = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            dim_idx = h_int % self.dimension
            sign = 1.0 if ((h_int >> 4) % 2 == 0) else -1.0
            vec[dim_idx] += sign * (1.0 / (idx + 1.0) ** 0.5)

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate normalized dense embeddings for a list of document strings."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Check cache for existing texts
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        results = [None] * len(texts)

        for idx, text in enumerate(texts):
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in self._embedding_cache:
                results[idx] = self._embedding_cache[text_hash]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            if self._model is not None:
                try:
                    embeddings = self._model.encode(
                        uncached_texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
                    embeddings = np.array(embeddings, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Failed dense model encoding: {e}. Falling back.")
                    embeddings = np.array([self._fallback_embed(t) for t in uncached_texts], dtype=np.float32)
            else:
                embeddings = np.array([self._fallback_embed(t) for t in uncached_texts], dtype=np.float32)

            for u_idx, emb in zip(uncached_indices, embeddings):
                text_hash = hashlib.sha256(texts[u_idx].encode("utf-8")).hexdigest()
                self._embedding_cache[text_hash] = emb
                results[u_idx] = emb

        return np.array(results, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate normalized dense embedding for a single user query string."""
        if self._model is not None:
            try:
                emb = self._model.encode(
                    [query],
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                return np.array(emb[0], dtype=np.float32)
            except Exception as e:
                logger.error(f"Failed query encoding with model: {e}. Falling back.")
                return self._fallback_embed(query)
        return self._fallback_embed(query)
