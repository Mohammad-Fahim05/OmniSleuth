"""Structured, sanitized LLM error classification for retries and planner failures.

Categories are intentionally coarse and actionable. No category message may ever
contain an API key, Authorization header, or other secret: every message passes
through :func:`omnisleuth.llm.tracing.sanitize_text` plus an explicit literal
secret redaction performed by the caller (see :func:`redact_literals`).
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from omnisleuth.core.exceptions import OmniSleuthError
from omnisleuth.core.logging import get_logger
from omnisleuth.llm.tracing import sanitize_text

logger = get_logger("llm.errors")

MAX_ERROR_MESSAGE_CHARS = 300


class LLMErrorCategory(str, Enum):
    """Actionable failure categories returned to callers such as the LLM planner."""

    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication_error"
    INVALID_REQUEST = "invalid_request_error"
    CONFIGURATION = "configuration_error"
    UNKNOWN = "unknown_error"


TRANSIENT_CATEGORIES = frozenset(
    {
        LLMErrorCategory.RATE_LIMIT,
        LLMErrorCategory.SERVER_ERROR,
        LLMErrorCategory.NETWORK_ERROR,
        LLMErrorCategory.TIMEOUT,
        # Preserves the historical "retry unexpected errors up to the bound" behavior.
        LLMErrorCategory.UNKNOWN,
    }
)


@dataclass(frozen=True)
class LLMErrorInfo:
    """Sanitized, classified description of an LLM provider failure."""

    category: LLMErrorCategory
    message: str
    retryable: bool
    status_code: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "category": self.category.value,
            "message": self.message,
            "retryable": self.retryable,
            "status_code": self.status_code,
        }


class LLMProviderError(OmniSleuthError):
    """Sanitized provider failure carrying an actionable error category.

    The message is sanitized before construction so that raising, tracing, and
    logging this exception can never leak credentials.
    """

    def __init__(
        self,
        message: str,
        category: LLMErrorCategory,
        status_code: Optional[int] = None,
        provider: Optional[str] = None,
    ):
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.provider = provider

    @property
    def info(self) -> LLMErrorInfo:
        return LLMErrorInfo(
            category=self.category,
            message=str(self),
            retryable=self.category in TRANSIENT_CATEGORIES,
            status_code=self.status_code,
        )


def redact_literals(text: str, secrets: Optional[List[str]] = None) -> str:
    """Redact literal secret values (e.g. a configured API key) from text."""
    redacted = sanitize_text(text or "")
    for secret in secrets or []:
        if secret and len(secret) >= 4:
            redacted = redacted.replace(secret, "[REDACTED_SECRET]")
    return redacted


def _clamp(message: str) -> str:
    cleaned = (message or "").strip().replace("\n", " ")
    if len(cleaned) > MAX_ERROR_MESSAGE_CHARS:
        cleaned = cleaned[:MAX_ERROR_MESSAGE_CHARS] + "...[truncated]"
    return cleaned


def _extract_status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    return None


# Permanent failure signatures: checked BEFORE transient signatures so that a
# message which happens to mention connectivity but denies access classifies as
# an authentication failure (and is therefore never retried).
_PERMANENT_SIGNATURES = (
    (
        LLMErrorCategory.AUTHENTICATION,
        (
            "unauthorized",
            "unauthenticated",
            "authentication",
            "invalid_api_key",
            "invalid api key",
            "api key not configured",
            "forbidden",
            "permission_denied",
            "permission denied",
            "invalid authentication",
            "no api key provided",
        ),
    ),
    (
        LLMErrorCategory.INVALID_REQUEST,
        (
            "invalid_request",
            "invalidrequest",
            "bad request",
            "badrequest",
            "400",
            "401",
            "403",
            "404",
            "409",
            "413",
            "422",
            "unprocessable",
            "context_length_exceeded",
            "content_policy_violation",
            "unsupported parameter",
        ),
    ),
    (
        LLMErrorCategory.CONFIGURATION,
        (
            "not configured",
            "not set",
            "missing api key",
            "missing credential",
            "credentials not fully configured",
            "must be set",
            "no such provider",
            "unknown provider",
        ),
    ),
)

_TRANSIENT_SIGNATURES = (
    (
        LLMErrorCategory.RATE_LIMIT,
        (
            "rate limit",
            "rate_limit",
            "ratelimit",
            "429",
            "too many requests",
            "temporarily rate limited",
            "overloaded",
            "quota",
        ),
    ),
    (
        LLMErrorCategory.TIMEOUT,
        (
            "timeout",
            "timed out",
            "timeouterror",
            "deadline exceeded",
            "read timed out",
        ),
    ),
    (
        LLMErrorCategory.SERVER_ERROR,
        (
            "500",
            "502",
            "503",
            "504",
            "internal server error",
            "internal_server",
            "bad gateway",
            "service unavailable",
            "gateway time-out",
            "server error",
            "apistatuserror",
        ),
    ),
    (
        LLMErrorCategory.NETWORK_ERROR,
        (
            "connection",
            "connecterror",
            "connectionerror",
            "network",
            "broken pipe",
            "reset by peer",
            "temporarily unavailable",
            "connection refused",
            "ssl",
            "eof",
        ),
    ),
)


def _match(haystack: str, signatures) -> Optional[LLMErrorCategory]:
    for category, keywords in signatures:
        if any(keyword in haystack for keyword in keywords):
            return category
    return None


def classify_llm_error(exc: Exception, secrets: Optional[List[str]] = None) -> LLMErrorInfo:
    """Classify an exception into a sanitized, actionable LLM error category.

    Permanent failures (authentication, invalid request, configuration) are never
    retried. Transient failures (rate limit, server error, timeout, network) are
    retried by the existing bounded backoff in :func:`execute_with_retry`.
    """
    status_code = _extract_status_code(exc)
    raw_text = f"{type(exc).__name__}: {exc}"
    haystack = redact_literals(raw_text, secrets).lower()
    message = _clamp(redact_literals(str(exc) or type(exc).__name__, secrets))

    category = _match(haystack, _PERMANENT_SIGNATURES)

    if category is None and status_code is not None:
        if status_code in (401, 403):
            category = LLMErrorCategory.AUTHENTICATION
        elif status_code in (400, 404, 409, 413, 422):
            category = LLMErrorCategory.INVALID_REQUEST
        elif status_code == 429:
            category = LLMErrorCategory.RATE_LIMIT
        elif 500 <= status_code <= 599:
            category = LLMErrorCategory.SERVER_ERROR

    if category is None:
        category = _match(haystack, _TRANSIENT_SIGNATURES)

    if category is None:
        category = LLMErrorCategory.UNKNOWN

    return LLMErrorInfo(
        category=category,
        message=message,
        retryable=category in TRANSIENT_CATEGORIES,
        status_code=status_code,
    )


def raise_provider_error(
    provider_name: str,
    exc: Exception,
    secrets: Optional[List[str]] = None,
) -> None:
    """Log a sanitized failure and raise an :class:`LLMProviderError`.

    The raised exception message never contains a raw credential, regardless of
    what the underlying SDK included in its own error text.
    """
    info = classify_llm_error(exc, secrets=secrets)
    logger.warning(
        f"{provider_name} call failed [category={info.category.value} retryable={info.retryable}]: {info.message}"
    )
    raise LLMProviderError(
        message=info.message,
        category=info.category,
        status_code=info.status_code,
        provider=provider_name,
    ) from exc
