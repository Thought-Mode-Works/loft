"""
External legal API integrations.

Provides abstracted, configurable access to legal databases and resources
for validation, case-based learning, and precedent research.
"""

from loft.external.base import (
    LegalAPIProvider,
    CaseLawDocument,
    SearchQuery,
    SearchResult,
    APIResponse,
    APIError,
    RateLimitError,
    AuthenticationError,
)
from loft.external.courtlistener import CourtListenerClient
from loft.external.config import APIConfig, get_configured_clients
from loft.external.manager import LegalAPIManager

__all__ = [
    "LegalAPIProvider",
    "CaseLawDocument",
    "SearchQuery",
    "SearchResult",
    "APIResponse",
    "APIError",
    "RateLimitError",
    "AuthenticationError",
    "CourtListenerClient",
    "APIConfig",
    "get_configured_clients",
    "LegalAPIManager",
]
