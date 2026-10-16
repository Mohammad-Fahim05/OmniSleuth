"""Configuration management for OmniSleuth using Pydantic Settings."""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    app_env: str = Field(default="development", description="Application runtime environment")
    app_name: str = Field(default="OmniSleuth", description="Application name")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    log_file: Optional[str] = Field(default="logs/omnisleuth.log", description="Path to log file")

    # Bounded Limits (Rule 7)
    max_investigation_depth: int = Field(default=3, description="Maximum hypothesis investigation depth")
    max_tool_calls: int = Field(default=10, description="Maximum allowlisted tool executions per investigation")
    max_retries: int = Field(default=2, description="Maximum retries for failed tool or agent steps")
    max_upload_size_bytes: int = Field(
        default=50 * 1024 * 1024,
        description="Maximum allowed dataset upload size in bytes (default 50 MB)",
    )

    # Storage Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    sqlite_db_path: str = Field(default="data/omnisleuth.db", description="Path to SQLite database")
    vectorstore_dir: str = Field(default="data/vectorstore", description="Path to vector store persistence")
    uploads_dir: str = Field(default="data/uploads", description="Path to uploaded datasets")

    # LLM Settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="Default OpenAI chat model")
    embedding_model: str = Field(default="text-embedding-3-small", description="Default embeddings model")

    # Phase 19 LLM Gateway Settings
    llm_provider: str = Field(default="fallback", description="LLM provider: 'openai', 'azure', 'openrouter', or 'fallback'")
    llm_model: str = Field(default="gpt-4o-mini", description="Primary LLM model name")
    llm_timeout_seconds: float = Field(default=30.0, description="LLM request timeout in seconds")
    llm_max_retries: int = Field(default=3, description="Maximum retries for transient LLM errors")
    llm_retry_base_delay: float = Field(default=1.0, description="Base exponential backoff delay in seconds")
    llm_retry_max_delay: float = Field(default=10.0, description="Maximum exponential backoff delay in seconds")

    # OpenRouter (Optional, OpenAI-compatible API)
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    openrouter_model: str = Field(default="openai/gpt-4o-mini", description="Default OpenRouter model")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter API base URL")

    # Azure OpenAI (Optional)
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint URL")
    azure_openai_deployment: Optional[str] = Field(default=None, description="Azure OpenAI deployment name")
    azure_openai_api_version: str = Field(default="2024-02-15-preview", description="Azure OpenAI API version")

    # Pricing & Metering
    llm_input_price_per_1k: Optional[float] = Field(default=0.00015, description="Input price per 1,000 tokens in USD")
    llm_output_price_per_1k: Optional[float] = Field(default=0.00060, description="Output price per 1,000 tokens in USD")

    # Phase C1/C2 NLQ Planner Selection
    nlq_planner_mode: str = Field(
        default="rule",
        description="NLQ planner selection: 'rule' (deterministic, default) or 'llm' (LLM-assisted)",
    )
    nlq_planner_fallback: str = Field(
        default="rule",
        description="Fallback policy when LLM planning fails: 'rule' (deterministic) or 'fail'",
    )
    nlq_narration_enabled: bool = Field(
        default=False,
        description="Optional LLM rewriting of templated answers (numbers must stay evidence-grounded)",
    )

    # Semantic Cache
    llm_cache_enabled: bool = Field(default=False, description="Enable optional semantic cache")
    llm_cache_similarity_threshold: float = Field(default=0.95, description="Similarity threshold for semantic cache hits")

    # Local Hybrid RAG Settings (Phase 10)
    local_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Local dense embedding model")
    local_reranker_model: str = Field(default="cross-encoder/ms-marco-TinyBERT-L-2-v2", description="Local cross-encoder reranker model")
    embedding_batch_size: int = Field(default=32, description="Batch size for generating document embeddings")
    reranker_batch_size: int = Field(default=16, description="Batch size for cross-encoder reranking")
    rag_bm25_top_k: int = Field(default=10, description="Top candidates retrieved from BM25")
    rag_dense_top_k: int = Field(default=10, description="Top candidates retrieved from dense FAISS index")
    rag_hybrid_top_k: int = Field(default=10, description="Top candidates preserved after Reciprocal Rank Fusion")
    rag_rrf_k: int = Field(default=60, description="RRF constant k (default 60)")
    rag_final_top_k: int = Field(default=5, description="Final number of retrieved chunks returned")


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Singleton getter for application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset cached settings (primarily for testing)."""
    global _settings
    _settings = None
