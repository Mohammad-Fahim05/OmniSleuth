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

    # Storage Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    sqlite_db_path: str = Field(default="data/omnisleuth.db", description="Path to SQLite database")
    vectorstore_dir: str = Field(default="data/vectorstore", description="Path to vector store persistence")
    uploads_dir: str = Field(default="data/uploads", description="Path to uploaded datasets")

    # LLM Settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="Default OpenAI chat model")
    embedding_model: str = Field(default="text-embedding-3-small", description="Default embeddings model")

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
