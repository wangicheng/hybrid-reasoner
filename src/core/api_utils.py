"""
API Utilities: Rate Limiting & Retry Logic for Google GenAI calls.

Provides:
- RateLimiter: enforces a minimum interval between API calls per key/bucket.
- retry_on_rate_limit: decorator that retries on 429/503 errors with a fixed retry interval.
"""

import os
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
        self._last_call_time_by_bucket = {}

    def wait(self, bucket: str | None = None):
        """Block until enough time has passed since the last call for this bucket."""
        bucket_key = bucket or "__global__"
        with self._lock:
            now = time.monotonic()
            last_call_time = self._last_call_time_by_bucket.get(bucket_key)
            if last_call_time is not None:
                elapsed = now - last_call_time
            else:
                elapsed = self.min_interval
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self._last_call_time_by_bucket[bucket_key] = time.monotonic()


# Global shared rate limiter instance (callers use per-key buckets when available)
_global_rate_limiter = RateLimiter(min_interval=4.0)


def get_rate_limiter() -> RateLimiter:
    """Return the global rate limiter instance."""
    return _global_rate_limiter


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a retryable API or transient network error."""
    error_str = str(exc)
    # Match known retryable status codes
    if is_rate_limit_error(exc):
        return True
    if "500" in error_str or "INTERNAL" in error_str:
        return True
    if "503" in error_str or "UNAVAILABLE" in error_str:
        return True
    if "502" in error_str or "BAD_GATEWAY" in error_str:
        return True
    if "504" in error_str or "DEADLINE_EXCEEDED" in error_str or "GATEWAY_TIMEOUT" in error_str:
        return True
    # Add network-level errors
    # WinError 10013 is common on Windows when a socket/connect attempt is
    # rejected by local security or transient network policy.
    if "[WinError 10013]" in error_str:
        return True
    if "[WinError 10053]" in error_str:
        return True
    if "[Errno 11001]" in error_str: # getaddrinfo failed
        return True
    if "Connection aborted" in error_str:
        return True
    # Add 403/PERMISSION_DENIED as retryable (especially for leaked keys when rotating)
    if "403" in error_str or "PERMISSION_DENIED" in error_str:
        return True
    return False


def _extract_retry_delay(exc: Exception) -> float | None:
    """Try to extract the server-suggested retry delay from the error message."""
    error_str = str(exc)
    match = re.search(r"retry\s*(?:in|Delay)[:\s]*['\"]?(\d+(?:\.\d+)?)\s*s", error_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception indicates quota exhaustion or provider throttling (or key issues)."""
    error_str = str(exc)
    # Include 403/PERMISSION_DENIED so callers trigger key rotation
    return (
        "429" in error_str or 
        "RESOURCE_EXHAUSTED" in error_str or 
        "403" in error_str or 
        "PERMISSION_DENIED" in error_str
    )


def retry_on_rate_limit(max_retries: int = 5, base_delay: float = 5.0, max_delay: float = 120.0):
    """
    Decorator that retries a function on retryable GenAI errors.
    Uses a fixed retry interval between attempts.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Delay in seconds between retry attempts.
        max_delay: Unused legacy parameter retained for compatibility.
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

                    # Use a fixed retry interval to keep retries predictable.
                    _ = max_delay
                    delay = base_delay

                    print(f"[api_utils] {func.__name__} attempt {attempt + 1}/{max_retries} failed "
                          f"(retryable). Retrying in {delay:.1f}s...")
                    time.sleep(delay)

            # Should not reach here, but just in case
            raise last_exception
        return wrapper
    return decorator


class APIKeyRotator:
    """
    Thread-safe API Key rotator for handling multiple API keys.
    Automatically switches to the next key when one hits rate limits.
    Supports exclusive access via acquire/release.
    """
    
    def __init__(self, api_keys: list, max_concurrent: int = 10):
        """
        Args:
            api_keys: List of API keys to rotate through.
            max_concurrent: Maximum total concurrent requests allowed across all keys.
        """
        self.api_keys = api_keys
        self.current_index = 0
        self._cond = threading.Condition()
        self.sleep_until = {k: 0.0 for k in api_keys}
        self.in_use = {k: False for k in api_keys}
        self.max_concurrent = max_concurrent
        self.current_concurrent = 0
        
        if not api_keys:
            raise ValueError("No API keys provided")
        
        print(f"[APIKeyRotator] Initialized with {len(api_keys)} keys, max_concurrent={max_concurrent}")
    
    def get_current_key(self) -> str:
        """Get a valid API key (legacy method, ignores in_use)."""
        with self._cond:
            now = time.monotonic()
            if self.sleep_until[self.api_keys[self.current_index]] <= now:
                return self.api_keys[self.current_index]
            for i in range(len(self.api_keys)):
                idx = (self.current_index + i) % len(self.api_keys)
                k = self.api_keys[idx]
                if self.sleep_until[k] <= now:
                    self.current_index = idx
                    return k
            return self.api_keys[self.current_index]

    def rotate(self) -> str:
        """Rotate to next key (legacy method)."""
        with self._cond:
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            return self.get_current_key()
            
    def acquire(self) -> str:
        """Acquire an exclusive API key that is not sleeping or in use, respecting global concurrency limit."""
        with self._cond:
            while True:
                now = time.monotonic()
                
                # 1. Check Global Concurrency Limit
                if self.current_concurrent >= self.max_concurrent:
                    self._cond.wait()
                    continue

                # 2. Try to find an available key (not in use and not sleeping)
                start_index = self.current_index
                for offset in range(len(self.api_keys)):
                    idx = (start_index + offset) % len(self.api_keys)
                    k = self.api_keys[idx]
                    if not self.in_use[k] and self.sleep_until[k] <= now:
                        self.current_index = (idx + 1) % len(self.api_keys)
                        self.in_use[k] = True
                        self.current_concurrent += 1
                        return k
                
                # 3. Wait for a key to be released or wake up
                wait_time = None
                for k in self.api_keys:
                    if not self.in_use[k]:
                        t = self.sleep_until[k] - now
                        if wait_time is None or t < wait_time:
                            wait_time = t
                
                if wait_time is None:
                    # All keys physically in use! Wait for release
                    self._cond.wait()
                elif wait_time > 0:
                    # Some keys are sleeping, wait for the earliest one
                    self._cond.wait(timeout=wait_time)
                else:
                    # Should not happen as loop will restart and find the key
                    self._cond.wait(timeout=0.1)

    def release(self, key: str):
        """Release an exclusively acquired API key and decrement global concurrency counter."""
        with self._cond:
            if key in self.in_use and self.in_use[key]:
                self.in_use[key] = False
                self.current_concurrent = max(0, self.current_concurrent - 1)
                self._cond.notify_all()
                
    def sleep_key(self, key: str, sleep_seconds: float):
        """Put a specific key to sleep."""
        with self._cond:
            if key in self.sleep_until:
                self.sleep_until[key] = time.monotonic() + sleep_seconds
                short_k = f"...{key[-4:]}" if len(key)>4 else key
                print(f"[APIKeyRotator] Key {short_k} put to sleep for {sleep_seconds:.1f}s")
                self._cond.notify_all()

    def sleep_current_key(self, sleep_seconds: float):
        """Legacy method to sleep current key."""
        with self._cond:
            k = self.api_keys[self.current_index]
        self.sleep_key(k, sleep_seconds)
    
    def on_rate_limit_error(self, sleep_seconds: float = 0.0):
        """Legacy error handler."""
        if sleep_seconds > 0:
            self.sleep_current_key(sleep_seconds)
        return self.rotate()


# Global API key rotator instance
from src.config import settings

_global_api_key_rotator = None

def get_api_key_rotator() -> APIKeyRotator:
    """Get or initialize the global API key rotator."""
    global _global_api_key_rotator
    if _global_api_key_rotator is None:
        if hasattr(settings, 'GOOGLE_API_KEYS') and settings.GOOGLE_API_KEYS:
            _global_api_key_rotator = APIKeyRotator(settings.GOOGLE_API_KEYS)
        else:
            # Fallback: use single key from environment
            single_key = os.environ.get("GOOGLE_API_KEY", "")
            if single_key:
                _global_api_key_rotator = APIKeyRotator([single_key])
            else:
                raise ValueError("No GOOGLE_API_KEY found in environment")
    return _global_api_key_rotator

def get_current_api_key() -> str:
    """Get the current active API key."""
    return get_api_key_rotator().get_current_key()
