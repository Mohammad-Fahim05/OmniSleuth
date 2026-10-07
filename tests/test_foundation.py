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
