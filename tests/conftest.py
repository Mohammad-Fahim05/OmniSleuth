"""Global pytest fixtures for OmniSleuth tests."""

import os
import pytest
from pydantic_settings import SettingsConfigDict

from omnisleuth.core.config import Settings, get_settings, reset_settings
from omnisleuth.core.logging import configure_logging


@pytest.fixture(autouse=True)
def clean_environment(tmp_path, monkeypatch):
    """Fixture ensuring a clean environment, temporary paths, and isolation from local .env."""
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "test.log"))
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test_omnisleuth.db"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VECTORSTORE_DIR", str(tmp_path / "vectorstore"))

    # Clear all LLM and provider credential/config environment variables
    for key in list(os.environ.keys()):
        if any(key.startswith(pfx) for pfx in ("OPENROUTER_", "OPENAI_", "AZURE_", "LLM_", "ANTHROPIC_", "COHERE_")):
            monkeypatch.delenv(key, raising=False)

    # Isolate Settings from developer's local .env file
    isolated_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        extra="ignore",
    )
    monkeypatch.setattr(Settings, "model_config", isolated_config)

    reset_settings()
    configure_logging(force_reconfigure=True)

    yield
    reset_settings()
