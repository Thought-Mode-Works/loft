"""
Custom exceptions for LLM interface.

Provides specific error types for different failure modes.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    pass


class LLMProviderError(LLMError):
    """Error communicating with LLM provider API."""

    def __init__(self, message: str, provider: str, status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMParsingError(LLMError):
    """Error parsing LLM response into structured format."""

    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


class LLMRateLimitError(LLMProviderError):
    """Rate limit exceeded for LLM provider."""

    def __init__(self, message: str, provider: str, retry_after: int | None = None):
        super().__init__(message, provider, status_code=429)
        self.retry_after = retry_after


class LLMTimeoutError(LLMProviderError):
    """LLM request timed out."""

    pass


class LLMValidationError(LLMError):
    """LLM response failed validation."""

    def __init__(self, message: str, validation_errors: list[str]):
        super().__init__(message)
        self.validation_errors = validation_errors
