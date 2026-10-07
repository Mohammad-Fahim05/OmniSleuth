"""Structured logging configuration for OmniSleuth."""

import logging
import sys
from pathlib import Path
from typing import Optional
from omnisleuth.core.config import get_settings


def configure_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    force_reconfigure: bool = False,
) -> logging.Logger:
    """Configure and return the root OmniSleuth logger.

    Args:
        log_level: Explicit logging level, defaults to settings.log_level.
        log_file: Explicit log file path, defaults to settings.log_file.
        force_reconfigure: Reconfigure existing loggers if True.

    Returns:
        The configured 'omnisleuth' root logger.
    """
    settings = get_settings()
    level_str = (log_level or settings.log_level).upper()
    numeric_level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger("omnisleuth")

    if logger.handlers and not force_reconfigure:
        return logger

    logger.handlers.clear()
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler
    target_log_file = log_file if log_file is not None else settings.log_file
    if target_log_file:
        try:
            log_path = Path(target_log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as exc:
            logger.warning(f"Could not initialize file logger at {target_log_file}: {exc}")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a child logger under the 'omnisleuth' namespace."""
    if name:
        if name.startswith("omnisleuth"):
            return logging.getLogger(name)
        return logging.getLogger(f"omnisleuth.{name}")
    return logging.getLogger("omnisleuth")
