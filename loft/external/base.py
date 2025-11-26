"""
Base classes and abstractions for legal API integrations.

Provides a common interface for different legal data providers with
built-in fault tolerance, caching, and rate limiting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from loguru import logger


class APIError(Exception):
    """Base exception for API errors."""

    pass


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""

    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""

    pass


class CourtLevel(str, Enum):
    """Court hierarchy levels."""

    SUPREME = "supreme"
    APPELLATE = "appellate"
    DISTRICT = "district"
    STATE = "state"
    BANKRUPTCY = "bankruptcy"
    TAX = "tax"


class DocumentType(str, Enum):
    """Types of legal documents."""

    OPINION = "opinion"
    DOCKET = "docket"
    ORAL_ARGUMENT = "oral_argument"
    STATUTE = "statute"
    REGULATION = "regulation"
    BRIEF = "brief"


@dataclass
class CaseLawDocument:
    """
    Standardized representation of a legal case document.

    Normalized across different API providers for consistent usage.
    """

    # Identity
    document_id: str
    source_api: str  # "courtlistener", "cap", etc.
    url: str

    # Case information
    case_name: str
    court: str
    court_level: Optional[CourtLevel] = None
    docket_number: Optional[str] = None
    decision_date: Optional[datetime] = None

    # Content
    text: str = ""
    summary: str = ""
    headnotes: List[str] = field(default_factory=list)

    # Citations
    citations: List[str] = field(default_factory=list)
    cited_cases: List[str] = field(default_factory=list)

    # Metadata
    judges: List[str] = field(default_factory=list)
    attorneys: List[str] = field(default_factory=list)
    jurisdiction: Optional[str] = None
    document_type: DocumentType = DocumentType.OPINION

    # Relevance (for search results)
    relevance_score: float = 0.0

    # Raw data from API
    raw_data: Dict[str, Any] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "source_api": self.source_api,
            "url": self.url,
            "case_name": self.case_name,
            "court": self.court,
            "court_level": self.court_level.value if self.court_level else None,
            "docket_number": self.docket_number,
            "decision_date": (self.decision_date.isoformat() if self.decision_date else None),
            "text": self.text,
            "summary": self.summary,
            "headnotes": self.headnotes,
            "citations": self.citations,
            "cited_cases": self.cited_cases,
            "judges": self.judges,
            "attorneys": self.attorneys,
            "jurisdiction": self.jurisdiction,
            "document_type": self.document_type.value,
            "relevance_score": self.relevance_score,
            "fetched_at": self.fetched_at.isoformat(),
        }


@dataclass
class SearchQuery:
    """Search query parameters."""

    query_text: str
    jurisdiction: Optional[str] = None
    court: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    document_type: Optional[DocumentType] = None
    max_results: int = 10
    offset: int = 0

    # Advanced filters
    cited_case: Optional[str] = None
    judge: Optional[str] = None
    docket_number: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_text": self.query_text,
            "jurisdiction": self.jurisdiction,
            "court": self.court,
            "date_from": self.date_from.isoformat() if self.date_from else None,
            "date_to": self.date_to.isoformat() if self.date_to else None,
            "document_type": (self.document_type.value if self.document_type else None),
            "max_results": self.max_results,
            "offset": self.offset,
            "cited_case": self.cited_case,
            "judge": self.judge,
            "docket_number": self.docket_number,
        }


@dataclass
class SearchResult:
    """Search results from API."""

    query: SearchQuery
    documents: List[CaseLawDocument]
    total_results: int
    page: int
    has_more: bool
    search_time_ms: float = 0.0
    source_api: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query.to_dict(),
            "documents": [doc.to_dict() for doc in self.documents],
            "total_results": self.total_results,
            "page": self.page,
            "has_more": self.has_more,
            "search_time_ms": self.search_time_ms,
            "source_api": self.source_api,
        }


@dataclass
class APIResponse:
    """Generic API response wrapper."""

    success: bool
    data: Any
    error_message: str = ""
    status_code: int = 200
    rate_limit_remaining: Optional[int] = None
    response_time_ms: float = 0.0


class LegalAPIProvider(ABC):
    """
    Abstract base class for legal API providers.

    Implementations provide access to specific legal databases with
    consistent interface, error handling, and fault tolerance.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize API provider.

        Args:
            api_key: API authentication key
            base_url: Base URL for API endpoints
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = None  # HTTP session (to be created by implementation)

        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def search(self, query: SearchQuery) -> SearchResult:
        """
        Search for cases matching query.

        Args:
            query: Search parameters

        Returns:
            SearchResult with matching documents

        Raises:
            APIError: If search fails
            RateLimitError: If rate limit exceeded
        """
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> CaseLawDocument:
        """
        Get specific document by ID.

        Args:
            document_id: Document identifier

        Returns:
            CaseLawDocument with full content

        Raises:
            APIError: If retrieval fails
        """
        pass

    @abstractmethod
    def get_citations(self, document_id: str) -> List[str]:
        """
        Get citations from a document.

        Args:
            document_id: Document identifier

        Returns:
            List of citation strings

        Raises:
            APIError: If retrieval fails
        """
        pass

    def is_available(self) -> bool:
        """
        Check if API is available and configured.

        Returns:
            True if API can be used
        """
        return self.api_key is not None and self.base_url is not None

    def get_provider_name(self) -> str:
        """Get human-readable provider name."""
        return self.__class__.__name__.replace("Client", "")

    def _handle_error(self, error: Exception, context: str = "") -> APIResponse:
        """
        Handle API errors with appropriate logging and response.

        Args:
            error: The exception that occurred
            context: Additional context about the operation

        Returns:
            APIResponse indicating failure
        """
        logger.error(f"{self.get_provider_name()} error {context}: {str(error)}")

        if "rate limit" in str(error).lower():
            raise RateLimitError(f"Rate limit exceeded: {error}")
        elif "auth" in str(error).lower() or "401" in str(error):
            raise AuthenticationError(f"Authentication failed: {error}")
        else:
            raise APIError(f"API error: {error}")
