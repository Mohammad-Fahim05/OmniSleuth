"""Semantic cross-encoder reranker for fine-grained query-document relevance scoring."""

import math
import re
from typing import List, Optional, Tuple
from omnisleuth.core.config import get_settings
from omnisleuth.core.logging import get_logger
from omnisleuth.rag.models import DocumentChunk

logger = get_logger("rag.reranker")


class SemanticReranker:
    """Cross-Encoder neural reranker computing exact joint query-document relevance."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.local_reranker_model
        self.batch_size = batch_size or settings.reranker_batch_size
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize sentence_transformers CrossEncoder or prepare deterministic fallback."""
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading CrossEncoder reranker: '{self.model_name}'")
            self._model = CrossEncoder(self.model_name)
        except Exception as exc:
            logger.warning(
                f"Could not load CrossEncoder '{self.model_name}': {exc}. "
                f"Using deterministic semantic relevance fallback."
            )
            self._model = None

    def _fallback_score(self, query: str, document_text: str) -> float:
        """Deterministic lexical-semantic relevance scorer for offline fallback."""
        q_tokens = [w.lower() for w in re.findall(r"\b\w{2,}\b", query.lower())]
        d_tokens = [w.lower() for w in re.findall(r"\b\w{2,}\b", document_text.lower())]
        if not q_tokens or not d_tokens:
            return 0.0

        d_token_set = set(d_tokens)
        matched_count = sum(1 for t in q_tokens if t in d_token_set)
        jaccard = matched_count / len(set(q_tokens).union(d_token_set))

        # Bigram overlap bonus
        q_bigrams = {f"{q_tokens[i]}_{q_tokens[i+1]}" for i in range(len(q_tokens) - 1)}
        d_bigrams = {f"{d_tokens[i]}_{d_tokens[i+1]}" for i in range(len(d_tokens) - 1)}
        bigram_overlap = len(q_bigrams.intersection(d_bigrams)) / (len(q_bigrams) or 1)

        raw_score = (matched_count / len(q_tokens)) * 0.6 + jaccard * 0.2 + bigram_overlap * 0.2
        return round(raw_score, 4)

    def rerank(
        self,
        query: str,
        candidates: List[DocumentChunk],
        top_k: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """Rerank a list of candidate document chunks against the given query."""
        if not candidates:
            return []

        pairs = [(query, c.content) for c in candidates]

        if self._model is not None:
            try:
                raw_scores = self._model.predict(pairs, batch_size=self.batch_size)
                # Convert logits to probabilities if unbounded
                scores = []
                for s in raw_scores:
                    val = float(s)
                    prob = 1.0 / (1.0 + math.exp(-val)) if (val < -10 or val > 10) else (1.0 / (1.0 + math.exp(-val)))
                    scores.append(round(prob, 4))
            except Exception as e:
                logger.error(f"Failed cross-encoder prediction: {e}. Falling back.")
                scores = [self._fallback_score(query, c.content) for c in candidates]
        else:
            scores = [self._fallback_score(query, c.content) for c in candidates]

        # Associate scores with candidates
        scored_candidates: List[Tuple[float, DocumentChunk]] = []
        for score, chunk in zip(scores, candidates):
            chunk_copy = chunk.model_copy()
            chunk_copy.rerank_score = score
            scored_candidates.append((score, chunk_copy))

        # Sort descending by rerank score, then by previous RRF score/chunk_id
        scored_candidates.sort(
            key=lambda item: (
                item[0],
                item[1].rrf_score or 0.0,
                item[1].similarity_score or 0.0,
                item[1].chunk_id,
            ),
            reverse=True,
        )

        if top_k is not None:
            scored_candidates = scored_candidates[:top_k]

        reranked_chunks: List[DocumentChunk] = []
        for rank_idx, (score, chunk) in enumerate(scored_candidates, start=1):
            chunk.rank = rank_idx
            chunk.similarity_score = score
            reranked_chunks.append(chunk)

        logger.debug(f"Reranked {len(candidates)} candidates down to {len(reranked_chunks)}.")
        return reranked_chunks
