"""
Unit tests for retry logic with exponential backoff.

Tests retry decorator functionality, exponential backoff timing,
error handling, and retry limits.
"""

import pytest
import time
from unittest.mock import patch

from loft.neural.retry import exponential_backoff
from loft.neural.errors import (
    LLMRateLimitError,
    LLMTimeoutError,
    LLMProviderError,
)


class TestExponentialBackoff:
    """Test exponential_backoff decorator."""

    def test_successful_first_try(self):
        """Test that successful calls don't retry."""

        @exponential_backoff()
        def success_func():
            return "success"

        result = success_func()
        assert result == "success"

    def test_function_with_args(self):
        """Test decorated function with arguments."""

        @exponential_backoff()
        def func_with_args(a, b, c=3):
            return a + b + c

        result = func_with_args(1, 2, c=4)
        assert result == 7

    def test_retry_on_rate_limit(self):
        """Test retry on rate limit error."""
        call_count = {"count": 0}

        @exponential_backoff(max_retries=2, base_delay=0.01)
        def rate_limited_func():
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise LLMRateLimitError("Rate limited", provider="test")
            return "success"

        result = rate_limited_func()
        assert result == "success"
        assert call_count["count"] == 3  # Failed twice, succeeded third time

    def test_retry_on_timeout(self):
        """Test retry on timeout error."""
        call_count = {"count": 0}

        @exponential_backoff(max_retries=2, base_delay=0.01)
        def timeout_func():
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise LLMTimeoutError("Timeout", provider="test")
            return "success"

        result = timeout_func()
        assert result == "success"
        assert call_count["count"] == 2

    def test_retry_on_5xx_error(self):
        """Test retry on 5xx provider errors."""
        call_count = {"count": 0}

        @exponential_backoff(max_retries=2, base_delay=0.01)
        def server_error_func():
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise LLMProviderError("Server error", provider="test", status_code=500)
            return "success"

        result = server_error_func()
        assert result == "success"
        assert call_count["count"] == 2

    def test_no_retry_on_4xx_error(self):
        """Test that 4xx errors don't trigger retry."""

        @exponential_backoff(max_retries=3, base_delay=0.01)
        def client_error_func():
            raise LLMProviderError("Client error", provider="test", status_code=400)

        with pytest.raises(LLMProviderError) as exc_info:
            client_error_func()

        assert exc_info.value.status_code == 400

    def test_no_retry_on_provider_error_without_status(self):
        """Test that provider errors without status code don't retry."""

        @exponential_backoff(max_retries=3, base_delay=0.01)
        def error_func():
            raise LLMProviderError("Error", provider="test", status_code=None)

        with pytest.raises(LLMProviderError):
            error_func()

    def test_max_retries_exceeded_rate_limit(self):
        """Test that rate limit errors stop after max retries."""

        @exponential_backoff(max_retries=2, base_delay=0.01)
        def always_rate_limited():
            raise LLMRateLimitError("Rate limited", provider="test")

        with pytest.raises(LLMRateLimitError):
            always_rate_limited()

    def test_max_retries_exceeded_timeout(self):
        """Test that timeout errors stop after max retries."""

        @exponential_backoff(max_retries=2, base_delay=0.01)
        def always_timeout():
            raise LLMTimeoutError("Timeout", provider="test")

        with pytest.raises(LLMTimeoutError):
            always_timeout()

    def test_max_retries_exceeded_server_error(self):
        """Test that server errors stop after max retries."""

        @exponential_backoff(max_retries=2, base_delay=0.01)
        def always_server_error():
            raise LLMProviderError("Server error", provider="test", status_code=500)

        with pytest.raises(LLMProviderError):
            always_server_error()

    def test_exponential_delay_growth(self):
        """Test that delays grow exponentially."""
        delays = []

        @exponential_backoff(max_retries=3, base_delay=0.01, exponential_base=2.0)
        def fail_func():
            if len(delays) < 3:
                delays.append(time.time())
                raise LLMTimeoutError("Timeout", provider="test")
            return "success"

        with patch("loft.neural.retry.time.sleep") as mock_sleep:
            fail_func()

            # Should have called sleep 3 times with increasing delays
            # Note: delay is multiplied BEFORE first sleep, so:
            # Attempt 1 fails: delay = 0.01 * 2 = 0.02
            # Attempt 2 fails: delay = 0.02 * 2 = 0.04
            # Attempt 3 fails: delay = 0.04 * 2 = 0.08
            assert mock_sleep.call_count == 3
            calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert calls[0] == 0.02
            assert calls[1] == 0.04
            assert calls[2] == 0.08

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""

        @exponential_backoff(max_retries=5, base_delay=10.0, max_delay=15.0, exponential_base=2.0)
        def fail_func():
            raise LLMTimeoutError("Timeout", provider="test")

        with patch("loft.neural.retry.time.sleep") as mock_sleep:
            with pytest.raises(LLMTimeoutError):
                fail_func()

            # Delays grow and are capped:
            # First: min(10 * 2, 15) = 15 (capped)
            # Rest: all capped at 15
            calls = [call[0][0] for call in mock_sleep.call_args_list]
            for delay in calls:
                assert delay <= 15.0

    def test_retry_after_header_rate_limit(self):
        """Test that retry_after from API is respected."""

        @exponential_backoff(max_retries=2, base_delay=1.0, max_delay=60.0)
        def rate_limited_with_retry_after():
            raise LLMRateLimitError("Rate limited", provider="test", retry_after=5)

        with patch("loft.neural.retry.time.sleep") as mock_sleep:
            with pytest.raises(LLMRateLimitError):
                rate_limited_with_retry_after()

            # Should use retry_after value (5) instead of exponential backoff
            calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert calls[0] == 5.0

    def test_retry_after_respects_max_delay(self):
        """Test that retry_after is capped by max_delay."""

        @exponential_backoff(max_retries=2, base_delay=1.0, max_delay=3.0)
        def rate_limited_func():
            raise LLMRateLimitError("Rate limited", provider="test", retry_after=10)

        with patch("loft.neural.retry.time.sleep") as mock_sleep:
            with pytest.raises(LLMRateLimitError):
                rate_limited_func()

            # Should be capped at max_delay (3.0)
            calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert calls[0] == 3.0

    def test_custom_max_retries(self):
        """Test custom max_retries parameter."""
        call_count = {"count": 0}

        @exponential_backoff(max_retries=5, base_delay=0.01)
        def custom_retries():
            call_count["count"] += 1
            if call_count["count"] < 4:
                raise LLMTimeoutError("Timeout", provider="test")
            return "success"

        result = custom_retries()
        assert result == "success"
        assert call_count["count"] == 4

    def test_custom_exponential_base(self):
        """Test custom exponential_base parameter."""

        @exponential_backoff(max_retries=3, base_delay=1.0, exponential_base=3.0)
        def fail_func():
            raise LLMTimeoutError("Timeout", provider="test")

        with patch("loft.neural.retry.time.sleep") as mock_sleep:
            with pytest.raises(LLMTimeoutError):
                fail_func()

            # Delays multiply by exponential_base each time:
            # First: 1.0 * 3 = 3.0
            # Second: 3.0 * 3 = 9.0
            # Third: 9.0 * 3 = 27.0
            calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert calls[0] == 3.0
            assert calls[1] == 9.0
            assert calls[2] == 27.0

    def test_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @exponential_backoff()
        def documented_func():
            """This is a documented function."""
            return "result"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a documented function."

    def test_different_error_types(self):
        """Test handling of different error types in sequence."""
        errors = [
            LLMRateLimitError("Rate limit", provider="test"),
            LLMTimeoutError("Timeout", provider="test"),
            LLMProviderError("Server error", provider="test", status_code=503),
        ]
        call_count = {"count": 0}

        @exponential_backoff(max_retries=3, base_delay=0.01)
        def mixed_errors():
            if call_count["count"] < len(errors):
                error = errors[call_count["count"]]
                call_count["count"] += 1
                raise error
            return "success"

        result = mixed_errors()
        assert result == "success"
        assert call_count["count"] == 3

    def test_non_retryable_exception_propagates(self):
        """Test that non-retryable exceptions propagate immediately."""

        @exponential_backoff(max_retries=3, base_delay=0.01)
        def func_with_value_error():
            raise ValueError("Not an LLM error")

        with pytest.raises(ValueError) as exc_info:
            func_with_value_error()

        assert str(exc_info.value) == "Not an LLM error"

    def test_return_value_preserved(self):
        """Test that return values are preserved through decorator."""

        @exponential_backoff()
        def return_complex():
            return {"key": "value", "list": [1, 2, 3]}

        result = return_complex()
        assert result == {"key": "value", "list": [1, 2, 3]}

    def test_zero_retries(self):
        """Test with max_retries=0 (fail immediately)."""

        @exponential_backoff(max_retries=0, base_delay=0.01)
        def fail_immediately():
            raise LLMTimeoutError("Timeout", provider="test")

        with pytest.raises(LLMTimeoutError):
            fail_immediately()

    def test_state_preserved_between_retries(self):
        """Test that function state is reset between retries."""
        call_count = {"count": 0}
        local_vars = []

        @exponential_backoff(max_retries=3, base_delay=0.01)
        def stateful_func():
            local_var = 0
            local_var += 1
            local_vars.append(local_var)
            call_count["count"] += 1

            if call_count["count"] < 3:
                raise LLMTimeoutError("Timeout", provider="test")
            return local_var

        result = stateful_func()
        # Each retry should reset local state
        assert all(var == 1 for var in local_vars)
        assert result == 1

    def test_decorator_with_methods(self):
        """Test that decorator works with class methods."""

        class TestClass:
            def __init__(self):
                self.call_count = 0

            @exponential_backoff(max_retries=2, base_delay=0.01)
            def method(self):
                self.call_count += 1
                if self.call_count < 2:
                    raise LLMTimeoutError("Timeout", provider="test")
                return "success"

        obj = TestClass()
        result = obj.method()
        assert result == "success"
        assert obj.call_count == 2

    def test_logging_on_retry(self):
        """Test that retries are logged."""
        call_count = {"count": 0}

        @exponential_backoff(max_retries=2, base_delay=0.01)
        def fail_once():
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise LLMRateLimitError("Rate limited", provider="test")
            return "success"

        with patch("loft.neural.retry.logger") as mock_logger:
            fail_once()

            # Should have logged the retry
            assert mock_logger.warning.called

    def test_logging_on_max_retries_exceeded(self):
        """Test that max retries exceeded is logged."""

        @exponential_backoff(max_retries=1, base_delay=0.01)
        def always_fail():
            raise LLMTimeoutError("Timeout", provider="test")

        with patch("loft.neural.retry.logger") as mock_logger:
            with pytest.raises(LLMTimeoutError):
                always_fail()

            # Should have logged the error
            assert mock_logger.error.called
