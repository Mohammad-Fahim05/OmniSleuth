"""Retry mechanism with exponential backoff and transient error classification."""

import time
from typing import Callable, TypeVar, Tuple
from omnisleuth.core.logging import get_logger

logger = get_logger("llm.retry")

T = TypeVar("T")


class RetryExhaustedError(Exception):
    """Exception raised when maximum retry attempts have been exhausted."""
    def __init__(self, original_exc: Exception, attempts: int):
        super().__init__(str(original_exc))
        self.original_exc = original_exc
        self.attempts = attempts


def is_transient_error(exc: Exception) -> bool:
    """Classify whether an exception is a transient error eligible for retry.

    Transient: Timeouts, rate limits (429), network connection resets, 5xx server errors.
    Permanent: 401/403 Auth errors, 400 Bad Request, missing model errors.
    """
    err_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    # Permanent error keywords
    permanent_keywords = ["unauthorized", "invalid_api_key", "authentication", "forbidden", "401", "403", "400", "invalidrequest"]
    if any(k in err_str for k in permanent_keywords) or any(k in exc_type for k in permanent_keywords):
        return False

    # Transient error keywords
    transient_keywords = ["timeout", "timed out", "rate_limit", "ratelimit", "429", "500", "502", "503", "504", "connection", "connecterror"]
    if any(k in err_str for k in transient_keywords) or any(k in exc_type for k in transient_keywords):
        return True

    # Default: treat unexpected exceptions as transient up to max retries
    return True


def execute_with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
) -> Tuple[T, int]:
    """Execute a callable with exponential backoff on transient errors.

    Returns tuple of (result, retry_count_attempted).
    Raises exception if retries are exhausted or a permanent error occurs.
    """
    for attempt in range(max_retries + 1):
        try:
            result = fn()
            return result, attempt
        except Exception as exc:
            if not is_transient_error(exc):
                logger.warning(f"Permanent error encountered ({type(exc).__name__}: {exc}). Aborting retries.")
                raise exc

            if attempt >= max_retries:
                logger.warning(f"Exhausted maximum retries ({max_retries}) for transient error: {exc}")
                raise RetryExhaustedError(exc, attempts=max_retries) from exc

            delay = min(max_delay, base_delay * (2 ** attempt))
            logger.info(f"Transient error ({type(exc).__name__}: {exc}). Retrying attempt {attempt + 1}/{max_retries} in {delay:.2f}s...")
            time.sleep(delay)

    raise RuntimeError("Retry loop exited unexpectedly")
