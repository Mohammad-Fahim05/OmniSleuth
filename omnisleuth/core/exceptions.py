"""Core exceptions for the OmniSleuth system."""


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
