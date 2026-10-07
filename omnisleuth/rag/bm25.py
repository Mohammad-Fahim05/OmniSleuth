"""Deterministic BM25 keyword index and retrieval engine."""

import math
import re
from typing import Dict, List, Optional, Set, Tuple
from omnisleuth.core.logging import get_logger
from omnisleuth.rag.models import DocumentChunk, DocumentType

logger = get_logger("rag.bm25")


class BM25Retriever:
    """Okapi BM25 keyword indexer and scorer."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text into clean alphanumeric terms."""
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())]

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Build BM25 index from a list of document chunks."""
        self.chunks = list(chunks)
        num_docs = len(self.chunks)
        if num_docs == 0:
            self.doc_len = []
            self.avg_doc_len = 0.0
            self.doc_freqs = {}
            self.term_freqs = []
            self.idf = {}
            return

        self.doc_len = []
        self.term_freqs = []
        self.doc_freqs = {}

        for chunk in self.chunks:
            tokens = self._tokenize(chunk.content)
            self.doc_len.append(len(tokens))
            tf_dict: Dict[str, int] = {}
            for t in tokens:
                tf_dict[t] = tf_dict.get(t, 0) + 1
            self.term_freqs.append(tf_dict)

            for unique_token in set(tokens):
                self.doc_freqs[unique_token] = self.doc_freqs.get(unique_token, 0) + 1

        self.avg_doc_len = sum(self.doc_len) / num_docs if num_docs > 0 else 0.0

        # Calculate standard smoothed IDF for all vocabulary terms
        self.idf = {}
        for token, df in self.doc_freqs.items():
            # Standard Okapi BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
            idf_val = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            self.idf[token] = max(0.01, idf_val)

        logger.debug(f"BM25 index built with {num_docs} chunks, {len(self.idf)} vocabulary terms.")

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_type_filter: Optional[DocumentType] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[float, DocumentChunk]]:
        """Search BM25 index for query terms and return ranked (score, chunk) pairs."""
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        scored_results: List[Tuple[float, DocumentChunk]] = []

        for idx, chunk in enumerate(self.chunks):
            if doc_type_filter and chunk.doc_type != doc_type_filter:
                continue

            tf_dict = self.term_freqs[idx]
            d_len = self.doc_len[idx]
            if d_len == 0:
                continue

            score = 0.0
            matched_terms = 0

            for token in query_tokens:
                if token in tf_dict:
                    matched_terms += 1
                    tf = tf_dict[token]
                    idf_val = self.idf.get(token, 0.1)
                    # BM25 term weighting formula
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (d_len / (self.avg_doc_len or 1.0)))
                    score += idf_val * (numerator / (denominator or 1.0))

            if score > min_score and matched_terms > 0:
                # Store normalized score in chunk copy
                chunk_copy = chunk.model_copy()
                chunk_copy.bm25_score = round(score, 4)
                scored_results.append((score, chunk_copy))

        # Sort descending by BM25 score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[:top_k]
