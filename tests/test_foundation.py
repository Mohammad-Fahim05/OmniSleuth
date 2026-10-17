"""Unit tests for Phase 0 - Foundation (Config, Logging, Exceptions)."""

import logging
from pathlib import Path
from omnisleuth import __version__
from omnisleuth.core.config import Settings, get_settings
from omnisleuth.core.exceptions import (
    BoundedLimitExceededError,
    ConfigurationError,
    IngestionError,
    OmniSleuthError,
    ToolExecutionError,
    ValidationError,
)
from omnisleuth.core.logging import configure_logging, get_logger


def test_package_metadata():
    """Verify package version is defined."""
    assert __version__ == "0.1.0"


def test_settings_defaults_and_env_loading(tmp_path):
    """Test default settings and environment override."""
    settings = get_settings()
    assert settings.app_env == "testing"
    assert settings.log_level == "DEBUG"
    assert settings.max_investigation_depth == 3
    assert settings.max_tool_calls == 10
    assert settings.max_retries == 2
    assert "test_omnisleuth.db" in settings.sqlite_db_path


def test_logging_configuration(tmp_path, caplog):
    """Test logger formatting and file output."""
    log_file = tmp_path / "custom.log"
    logger = configure_logging(log_level="DEBUG", log_file=str(log_file), force_reconfigure=True)

    test_msg = "Foundation logging verification message"
    with caplog.at_level(logging.DEBUG, logger="omnisleuth"):
        logger.info(test_msg)

    assert test_msg in caplog.text

    child_logger = get_logger("analytics")
    assert child_logger.name == "omnisleuth.analytics"
    child_msg = "Child logger message"
    child_logger.debug(child_msg)

    # Check file contents
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert test_msg in content
    assert child_msg in content


def test_exceptions_hierarchy():
    """Verify OmniSleuth domain exceptions hierarchy."""
    assert issubclass(ConfigurationError, OmniSleuthError)
    assert issubclass(IngestionError, OmniSleuthError)
    assert issubclass(ValidationError, OmniSleuthError)
    assert issubclass(ToolExecutionError, OmniSleuthError)
    assert issubclass(BoundedLimitExceededError, OmniSleuthError)


def test_test_isolation_ignores_env_file(tmp_path, monkeypatch):
    """Regression test (T-005S): proves fake keys in .env are ignored by test isolation."""
    from omnisleuth.core.config import reset_settings

    temp_env_dir = tmp_path / "env_isolation_test"
    temp_env_dir.mkdir()
    fake_env_file = temp_env_dir / ".env"
    fake_env_file.write_text(
        "OPENROUTER_API_KEY=sk-or-v1-FAKE\n"
        "LLM_PROVIDER=openrouter\n"
        "OPENAI_API_KEY=sk-FAKE\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(temp_env_dir)
    reset_settings()

    settings = get_settings()
    assert settings.openrouter_api_key is None
    assert settings.openai_api_key is None
    assert settings.llm_provider == "fallback"

    direct_settings = Settings()
    assert direct_settings.openrouter_api_key is None

    explicit_settings = Settings(_env_file=str(fake_env_file))
    assert explicit_settings.openrouter_api_key == "sk-or-v1-FAKE"
    assert explicit_settings.llm_provider == "openrouter"
