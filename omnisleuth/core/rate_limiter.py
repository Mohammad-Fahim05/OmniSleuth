"""Thread-safe in-memory sliding-window rate limiter for session requests.

Note:
    In-memory rate limits are per application process. They are not
    distributed across multiple application instances, containers, or cluster workers.
    For multi-instance distributed deployments, an external state store (such as Redis)
    would be required as a shared backing engine.
"""

from collections import deque
import threading
import time
from typing import Callable, Dict, Optional, Tuple

from omnisleuth.core.config import get_settings
from omnisleuth.core.exceptions import RateLimitExceededError
from omnisleuth.core.logging import get_logger

logger = get_logger("core.rate_limiter")


class InMemoryRateLimiter:
    """Thread-safe sliding-window rate limiter keyed by trusted session identifier.

    Enforces rate ceilings on natural language questions and dataset uploads
    independently per session context. Thread safety is guaranteed via a re-entrant
    or standard mutex lock around timestamp record maintenance.
    """

    def __init__(
        self,
        questions_per_minute: Optional[int] = None,
        uploads_per_hour: Optional[int] = None,
        time_func: Optional[Callable[[], float]] = None,
    ):
        settings = get_settings()
        self.questions_limit = (
            questions_per_minute
            if questions_per_minute is not None
            else getattr(settings, "rate_limit_questions_per_minute", 30)
        )
        self.uploads_limit = (
            uploads_per_hour
            if uploads_per_hour is not None
            else getattr(settings, "rate_limit_uploads_per_hour", 20)
        )
        self.time_func = time_func or time.time
        self._lock = threading.Lock()

        # Window durations in seconds
        self._windows: Dict[str, float] = {
            "question": 60.0,
            "upload": 3600.0,
        }

        # Records: action -> session_id -> deque of float timestamps
        self._records: Dict[str, Dict[str, deque]] = {
            "question": {},
            "upload": {},
        }
        self._last_prune_time = self.time_func()

    def get_limit_for_action(self, action: str) -> int:
        """Return configured capacity for action type."""
        if action == "question":
            return self.questions_limit
        if action == "upload":
            return self.uploads_limit
        return 30

    def get_window_for_action(self, action: str) -> float:
        """Return sliding window duration in seconds for action type."""
        return self._windows.get(action, 60.0)

    def is_allowed(
        self,
        action: str,
        session_id: str,
    ) -> bool:
        """Check if action is allowed without consuming quota.

        Args:
            action: Action type ('question' or 'upload').
            session_id: Trusted session identifier.

        Returns:
            True if within rate limit, False if limit reached.
        """
        allowed, _, _ = self.check(action, session_id, consume=False)
        return allowed

    def check(
        self,
        action: str,
        session_id: str,
        consume: bool = False,
    ) -> Tuple[bool, int, float]:
        """Check rate limit for session and optionally record consumption atomically.

        Args:
            action: Action type ('question' or 'upload').
            session_id: Trusted session identifier.
            consume: If True, atomically records timestamp if within limits.

        Returns:
            Tuple of (is_allowed: bool, current_count: int, retry_after_seconds: float).
        """
        # Exempt local CLI sessions from interactive rate limits
        if not session_id or session_id == "cli_local" or session_id.startswith("cli_"):
            return True, 0, 0.0

        now = self.time_func()
        window = self.get_window_for_action(action)
        limit = self.get_limit_for_action(action)
        cutoff = now - window

        with self._lock:
            # Periodic background cleanup of dead sessions if map gets large
            if now - self._last_prune_time > 300.0 or len(self._records.get(action, {})) > 500:
                self._prune_all_locked(now)

            action_records = self._records.setdefault(action, {})
            session_deque = action_records.get(session_id)

            if session_deque is None:
                if consume and limit > 0:
                    d = deque()
                    d.append(now)
                    action_records[session_id] = d
                    return True, 1, 0.0
                elif limit <= 0:
                    return False, 0, window
                return True, 0, 0.0

            # Prune timestamps outside sliding window
            while session_deque and session_deque[0] <= cutoff:
                session_deque.popleft()

            current_count = len(session_deque)

            if current_count >= limit:
                oldest = session_deque[0]
                retry_after = max(0.0, (oldest + window) - now)
                return False, current_count, retry_after

            if consume:
                session_deque.append(now)
                current_count += 1

            return True, current_count, 0.0

    def acquire(
        self,
        action: str,
        session_id: str,
    ) -> None:
        """Atomically acquire rate limit token for action and session, or raise error.

        Args:
            action: Action type ('question' or 'upload').
            session_id: Trusted session identifier.

        Raises:
            RateLimitExceededError: If rate limit ceiling has been reached.
        """
        allowed, current_count, retry_after = self.check(action, session_id, consume=True)
        if not allowed:
            limit = self.get_limit_for_action(action)
            window = self.get_window_for_action(action)
            window_desc = "minute" if window <= 60.0 else "hour"
            msg = (
                f"Rate limit exceeded: Session '{session_id}' reached the limit of "
                f"{limit} {action}s per {window_desc}. Please retry in {retry_after:.1f}s."
            )
            logger.warning(msg)
            raise RateLimitExceededError(
                message=msg,
                action=action,
                retry_after_seconds=retry_after,
            )

    def _prune_all_locked(self, now: float) -> None:
        """Internal helper to prune expired timestamps and empty deques across all actions."""
        self._last_prune_time = now
        for act, act_dict in self._records.items():
            window = self.get_window_for_action(act)
            cutoff = now - window
            dead_sessions = []
            for sid, d in act_dict.items():
                while d and d[0] <= cutoff:
                    d.popleft()
                if not d:
                    dead_sessions.append(sid)
            for sid in dead_sessions:
                act_dict.pop(sid, None)

    def prune_expired(self, now: Optional[float] = None) -> int:
        """Explicitly prune all expired timestamps and return number of active sessions retained."""
        current_now = now if now is not None else self.time_func()
        with self._lock:
            self._prune_all_locked(current_now)
            return sum(len(d) for d in self._records.values())

    def reset(self) -> None:
        """Clear all rate limit records (primarily for testing)."""
        with self._lock:
            self._records = {
                "question": {},
                "upload": {},
            }
            self._last_prune_time = self.time_func()


_limiter_instance: Optional[InMemoryRateLimiter] = None


def get_rate_limiter() -> InMemoryRateLimiter:
    """Singleton getter for application in-memory rate limiter."""
    global _limiter_instance
    if _limiter_instance is None:
        _limiter_instance = InMemoryRateLimiter()
    return _limiter_instance


def reset_rate_limiter() -> None:
    """Reset the global rate limiter instance (for testing)."""
    global _limiter_instance
    if _limiter_instance is not None:
        _limiter_instance.reset()
    _limiter_instance = None
