"""
Retry logic with exponential backoff for LLM requests.

Handles rate limits, timeouts, and transient errors.
"""

import time
import functools
from typing import Any, Callable, TypeVar, cast
from loguru import logger

from .errors import LLMRateLimitError, LLMTimeoutError, LLMProviderError


T = TypeVar("T")


def exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential growth

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retries = 0
            delay = base_delay

            while True:
                try:
                    return func(*args, **kwargs)

                except LLMRateLimitError as e:
                    # Use retry_after if provided by API
                    if e.retry_after:
                        delay = min(e.retry_after, max_delay)
                    else:
                        delay = min(delay * exponential_base, max_delay)

                    retries += 1
                    if retries > max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for rate limit"
                        )
                        raise

                    logger.warning(
                        f"Rate limit hit, retrying in {delay}s (attempt {retries}/{max_retries})"
                    )
                    time.sleep(delay)

                except LLMTimeoutError:
                    retries += 1
                    if retries > max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for timeout"
                        )
                        raise

                    delay = min(delay * exponential_base, max_delay)
                    logger.warning(
                        f"Timeout occurred, retrying in {delay}s (attempt {retries}/{max_retries})"
                    )
                    time.sleep(delay)

                except LLMProviderError as e:
                    # Only retry on transient errors (5xx status codes)
                    if e.status_code and 500 <= e.status_code < 600:
                        retries += 1
                        if retries > max_retries:
                            logger.error(
                                f"Max retries ({max_retries}) exceeded for provider error"
                            )
                            raise

                        delay = min(delay * exponential_base, max_delay)
                        logger.warning(
                            f"Provider error (status {e.status_code}), retrying in {delay}s "
                            f"(attempt {retries}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        # Don't retry on client errors (4xx)
                        logger.error(f"Non-retryable provider error: {e}")
                        raise

        return cast(Callable[..., T], wrapper)

    return decorator
