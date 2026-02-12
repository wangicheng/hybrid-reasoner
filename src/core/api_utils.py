"""
API Utilities: Rate Limiting & Retry Logic for Google GenAI calls.

Provides:
- RateLimiter: enforces a minimum interval between API calls (shared across all callers).
- retry_on_rate_limit: decorator that retries on 429/503 errors with exponential backoff.
"""

import time
import threading
import functools
import re


class RateLimiter:
    """
    Thread-safe rate limiter using a simple minimum-interval approach.
    Default: 1 request every 4 seconds ≈ 15 RPM (free tier safe).
    """

    def __init__(self, min_interval: float = 4.0):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call_time = 0.0

    def wait(self):
        """Block until enough time has passed since the last call."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self._last_call_time = time.monotonic()


# Global shared rate limiter instance (all GenAI calls share this)
_global_rate_limiter = RateLimiter(min_interval=4.0)


def get_rate_limiter() -> RateLimiter:
    """Return the global rate limiter instance."""
    return _global_rate_limiter


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a retryable API error (429 or 503)."""
    error_str = str(exc)
    # Match known retryable status codes
    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
        return True
    if "503" in error_str or "UNAVAILABLE" in error_str:
        return True
    return False


def _extract_retry_delay(exc: Exception) -> float | None:
    """Try to extract the server-suggested retry delay from the error message."""
    error_str = str(exc)
    match = re.search(r"retry\s*(?:in|Delay)[:\s]*['\"]?(\d+(?:\.\d+)?)\s*s", error_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def retry_on_rate_limit(max_retries: int = 5, base_delay: float = 5.0, max_delay: float = 120.0):
    """
    Decorator that retries a function on retryable GenAI errors.
    Uses exponential backoff, respecting server-suggested delays when available.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds for exponential backoff.
        max_delay: Maximum delay cap in seconds.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    # Enforce rate limiting before each call
                    get_rate_limiter().wait()
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if not _is_retryable(e):
                        # Non-retryable error, raise immediately
                        raise

                    if attempt == max_retries:
                        # Exhausted all retries
                        print(f"[api_utils] All {max_retries} retries exhausted for {func.__name__}. Giving up.")
                        raise

                    # Calculate delay: prefer server-suggested, otherwise exponential backoff
                    server_delay = _extract_retry_delay(e)
                    if server_delay is not None:
                        delay = min(server_delay + 1.0, max_delay)  # Add 1s buffer
                    else:
                        delay = min(base_delay * (2 ** attempt), max_delay)

                    print(f"[api_utils] {func.__name__} attempt {attempt + 1}/{max_retries} failed "
                          f"(retryable). Retrying in {delay:.1f}s...")
                    time.sleep(delay)

            # Should not reach here, but just in case
            raise last_exception
        return wrapper
    return decorator
