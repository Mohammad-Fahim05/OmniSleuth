"""Global pytest fixtures for OmniSleuth tests."""

import os
import pytest
from omnisleuth.core.config import get_settings, reset_settings
from omnisleuth.core.logging import configure_logging


@pytest.fixture(autouse=True)
def clean_environment(tmp_path, monkeypatch):
    """Fixture ensuring a clean environment and temporary paths for each test."""
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "test.log"))
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test_omnisleuth.db"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("VECTORSTORE_DIR", str(tmp_path / "vectorstore"))

    reset_settings()
    configure_logging(force_reconfigure=True)

    yield

    reset_settings()
