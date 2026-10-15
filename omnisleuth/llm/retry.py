"""Retry mechanism with exponential backoff and transient error classification.

Transient errors (rate limits, server errors, timeouts, network failures) are
retried with bounded exponential backoff. Permanent errors (authentication,
invalid request, configuration) abort retries immediately and are surfaced as
sanitized, actionable categories.
"""

import time
from typing import Callable, Optional, TypeVar

from omnisleuth.core.logging import get_logger
from omnisleuth.llm.errors import LLMErrorCategory, classify_llm_error

logger = get_logger("llm.retry")

T = TypeVar("T")


class RetryExhaustedError(Exception):
    """Exception raised when maximum retry attempts have been exhausted.

    ``category`` and ``message`` are sanitized so callers (e.g. the LLM planner)
    never receive raw credentials from the underlying provider exception.
    """

    def __init__(
        self,
        original_exc: Exception,
        attempts: int,
        category: Optional[LLMErrorCategory] = None,
        message: Optional[str] = None,
    ):
        super().__init__(str(message) if message is not None else str(original_exc))
        self.original_exc = original_exc
        self.attempts = attempts
        self.category = category or classify_llm_error(original_exc).category


def is_transient_error(exc: Exception) -> bool:
    """Classify whether an exception is a transient error eligible for retry.

    Transient: Timeouts, rate limits (429), network connection resets, 5xx server errors.
    Permanent: 401/403 auth errors, 400 invalid requests, missing configuration.
    """
    return classify_llm_error(exc).retryable


def describe_error(exc: Exception) -> str:
    """Return a sanitized, actionable description of a provider failure."""
    info = classify_llm_error(exc)
    return f"[{info.category.value}] {info.message}"


def execute_with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
) -> tuple:
    """Execute a callable with exponential backoff on transient errors.

    Returns tuple of (result, retry_count_attempted).
    Raises the original exception for permanent failures, or RetryExhaustedError
    when the retry budget is exhausted on transient failures.
    """
    for attempt in range(max_retries + 1):
        try:
            result = fn()
            return result, attempt
        except Exception as exc:
            info = classify_llm_error(exc)

            if not info.retryable:
                logger.warning(
                    f"Permanent error encountered [category={info.category.value}] "
                    f"({type(exc).__name__}: {info.message}). Aborting retries."
                )
                raise exc

            if attempt >= max_retries:
                logger.warning(
                    f"Exhausted maximum retries ({max_retries}) for transient error "
                    f"[category={info.category.value}]: {info.message}"
                )
                raise RetryExhaustedError(
                    exc, attempts=max_retries, category=info.category, message=info.message
                ) from exc

            delay = min(max_delay, base_delay * (2 ** attempt))
            logger.info(
                f"Transient error [category={info.category.value}] "
                f"({type(exc).__name__}: {info.message}). "
                f"Retrying attempt {attempt + 1}/{max_retries} in {delay:.2f}s..."
            )
            time.sleep(delay)

    raise RuntimeError("Retry loop exited unexpectedly")
