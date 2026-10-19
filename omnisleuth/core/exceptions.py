"""Core exceptions for the OmniSleuth system."""

from typing import Optional


class OmniSleuthError(Exception):
    """Base exception for all OmniSleuth errors."""


class ConfigurationError(OmniSleuthError):
    """Raised when configuration is invalid or missing."""


class IngestionError(OmniSleuthError):
    """Raised when data ingestion or schema validation fails."""


class ValidationError(OmniSleuthError):
    """Raised when programmatic claim validation or schema validation fails."""


class ToolExecutionError(OmniSleuthError):
    """Raised when an allowlisted tool fails during deterministic calculation."""


class BoundedLimitExceededError(OmniSleuthError):
    """Raised when an agentic investigation hits maximum depth or tool call limits."""


class AccessDeniedError(IngestionError):
    """Raised when an operation is attempted without valid session context or violates session isolation."""


class RateLimitExceededError(OmniSleuthError):
    """Raised when an operation exceeds configured session rate limits."""

    def __init__(
        self,
        message: str,
        action: str = "request",
        retry_after_seconds: Optional[float] = None,
    ):
        super().__init__(message)
        self.action = action
        self.retry_after_seconds = retry_after_seconds
