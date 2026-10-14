"""Dataset-aware semantic cache for LLM Gateway responses."""

import hashlib
from typing import Dict, Optional
from omnisleuth.core.config import get_settings
from omnisleuth.core.logging import get_logger
from omnisleuth.llm.models import LLMRequest, LLMResponse

logger = get_logger("llm.cache")


class SemanticCache:
    """Lightweight, dataset-aware in-memory cache for LLM responses."""

    def __init__(self, enabled: Optional[bool] = None, similarity_threshold: Optional[float] = None):
        self.settings = get_settings()
        self.enabled = enabled if enabled is not None else self.settings.llm_cache_enabled
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else self.settings.llm_cache_similarity_threshold
        )
        self._store: Dict[str, LLMResponse] = {}

    def _compute_key(self, request: LLMRequest) -> str:
        """Build deterministic cache key incorporating dataset version, model, and prompt text."""
        dataset_v = request.dataset_version or "no_dataset"
        model_name = request.model or request.provider or "default_model"
        prompt_text = request.get_prompt_text().strip().lower()

        raw_key = f"{dataset_v}::{model_name}::{prompt_text}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Look up response in cache. Returns None on cache miss or if cache is disabled."""
        if not self.enabled:
            return None

        key = self._compute_key(request)
        if key in self._store:
            cached_resp = self._store[key].model_copy(deep=True)
            cached_resp.cached = True
            logger.info(f"SemanticCache HIT for request key: {key[:12]}")
            return cached_resp

        logger.debug(f"SemanticCache MISS for request key: {key[:12]}")
        return None

    def put(self, request: LLMRequest, response: LLMResponse) -> None:
        """Store response in cache if enabled and no error occurred."""
        if not self.enabled or response.error is not None:
            return

        key = self._compute_key(request)
        self._store[key] = response.model_copy(deep=True)
        logger.debug(f"Stored response in SemanticCache under key: {key[:12]}")

    def clear(self) -> None:
        """Clear all stored cache entries."""
        self._store.clear()
        logger.info("SemanticCache cleared.")
