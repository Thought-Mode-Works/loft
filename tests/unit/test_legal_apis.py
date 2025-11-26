"""
Unit tests for legal API integrations.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from loft.external.base import (
    CaseLawDocument,
    SearchQuery,
    SearchResult,
    CourtLevel,
    APIError,
)
from loft.external.config import APIConfig, validate_config
from loft.external.courtlistener import CourtListenerClient
from loft.external.manager import LegalAPIManager


class TestCaseLawDocument:
    """Test CaseLawDocument data structure."""

    def test_document_creation(self):
        """Test creating a case law document."""
        doc = CaseLawDocument(
            document_id="12345",
            source_api="courtlistener",
            url="https://example.com/opinion/12345",
            case_name="Smith v. Jones",
            court="Supreme Court",
            court_level=CourtLevel.SUPREME,
            text="Full opinion text...",
        )

        assert doc.document_id == "12345"
        assert doc.case_name == "Smith v. Jones"
        assert doc.court_level == CourtLevel.SUPREME

    def test_document_serialization(self):
        """Test document to_dict."""
        doc = CaseLawDocument(
            document_id="123",
            source_api="courtlistener",
            url="https://example.com",
            case_name="Test Case",
            court="Test Court",
            decision_date=datetime(2020, 1, 1),
        )

        data = doc.to_dict()

        assert data["document_id"] == "123"
        assert data["case_name"] == "Test Case"
        assert data["decision_date"] == "2020-01-01T00:00:00"


class TestSearchQuery:
    """Test SearchQuery data structure."""

    def test_query_creation(self):
        """Test creating a search query."""
        query = SearchQuery(
            query_text="contract law",
            jurisdiction="california",
            max_results=20,
        )

        assert query.query_text == "contract law"
        assert query.jurisdiction == "california"
        assert query.max_results == 20

    def test_query_serialization(self):
        """Test query to_dict."""
        query = SearchQuery(
            query_text="test query",
            court="supreme",
            date_from=datetime(2020, 1, 1),
        )

        data = query.to_dict()

        assert data["query_text"] == "test query"
        assert data["court"] == "supreme"
        assert data["date_from"] == "2020-01-01T00:00:00"


class TestAPIConfig:
    """Test API configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = APIConfig()

        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.cache_enabled is True

    def test_config_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {
            "courtlistener_api_key": "test-key-123",
            "timeout": 60,
            "cache_enabled": False,
        }

        config = APIConfig.from_dict(config_dict)

        assert config.courtlistener_api_key == "test-key-123"
        assert config.timeout == 60
        assert config.cache_enabled is False

    def test_get_enabled_providers(self):
        """Test getting enabled providers."""
        config = APIConfig(
            courtlistener_enabled=True,
        )

        enabled = config.get_enabled_providers()

        assert "courtlistener" in enabled

    def test_get_configured_providers(self):
        """Test getting configured providers."""
        config = APIConfig(
            courtlistener_api_key="key1",
        )

        configured = config.get_configured_providers()

        assert "courtlistener" in configured

    def test_validate_config(self):
        """Test config validation."""
        config = APIConfig(
            courtlistener_enabled=False,
        )

        warnings = validate_config(config)

        assert len(warnings) > 0
        assert any("No API providers enabled" in w for w in warnings)


class TestCourtListenerClient:
    """Test CourtListener API client."""

    @pytest.fixture
    def client(self):
        """Create CourtListener client."""
        return CourtListenerClient(api_key="test-key")

    def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.api_key == "test-key"
        assert client.base_url == "https://www.courtlistener.com/api/rest/v3"

    def test_is_available(self, client):
        """Test availability check."""
        assert client.is_available() is True

        client_no_key = CourtListenerClient()
        # Without API key but with base_url, not fully available
        assert client_no_key.is_available() is False

    @patch("requests.Session.get")
    def test_search(self, mock_get, client):
        """Test search functionality."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "count": 1,
            "results": [
                {
                    "id": 123,
                    "case_name": "Test v. Case",
                    "court": "scotus",
                    "date_filed": "2020-01-01",
                    "absolute_url": "/opinion/123/",
                    "plain_text": "Opinion text",
                }
            ],
            "next": None,
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        query = SearchQuery(query_text="test", max_results=10)
        result = client.search(query)

        assert len(result.documents) == 1
        assert result.documents[0].case_name == "Test v. Case"
        assert result.total_results == 1

    @patch("requests.Session.get")
    def test_get_document(self, mock_get, client):
        """Test getting a document."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": 456,
            "case_name": "Smith v. Jones",
            "court": "ca9",
            "plain_text": "Full opinion text",
            "absolute_url": "/opinion/456/",
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        doc = client.get_document("456")

        assert doc.document_id == "456"
        assert doc.case_name == "Smith v. Jones"

    def test_determine_court_level(self, client):
        """Test court level determination."""
        assert client._determine_court_level("scotus") == CourtLevel.SUPREME
        assert client._determine_court_level("ca9") == CourtLevel.APPELLATE
        assert client._determine_court_level("district") == CourtLevel.DISTRICT


class TestLegalAPIManager:
    """Test unified API manager."""

    @pytest.fixture
    def mock_clients(self):
        """Create mock clients."""
        cl_client = Mock(spec=CourtListenerClient)

        return {
            "courtlistener": cl_client,
        }

    @pytest.fixture
    def manager(self, mock_clients):
        """Create API manager with mock clients."""
        config = APIConfig()
        return LegalAPIManager(config=config, clients=mock_clients)

    def test_manager_initialization(self, manager, mock_clients):
        """Test manager initialization."""
        assert len(manager.clients) == 1
        assert "courtlistener" in manager.clients

    def test_get_available_providers(self, manager, mock_clients):
        """Test getting available providers."""
        mock_clients["courtlistener"].is_available.return_value = True

        available = manager.get_available_providers()

        assert "courtlistener" in available

    def test_search_error_handling(self, manager, mock_clients):
        """Test search error handling."""
        # Provider fails
        mock_clients["courtlistener"].search.side_effect = APIError("Failed")

        query = SearchQuery(query_text="test")

        # Should raise APIError when all providers fail
        with pytest.raises(APIError):
            manager.search(query)

    def test_cache_functionality(self, manager, mock_clients):
        """Test response caching."""
        mock_result = SearchResult(
            query=SearchQuery(query_text="test"),
            documents=[],
            total_results=0,
            page=1,
            has_more=False,
        )
        mock_clients["courtlistener"].search.return_value = mock_result

        query = SearchQuery(query_text="test")

        # First call hits API
        manager.search(query, providers=["courtlistener"])
        assert mock_clients["courtlistener"].search.call_count == 1

        # Second call uses cache
        manager.search(query, providers=["courtlistener"])
        assert mock_clients["courtlistener"].search.call_count == 1  # No new call

    def test_clear_cache(self, manager):
        """Test clearing cache."""
        manager.cache = {"test": {"data": "value"}}

        manager.clear_cache()

        assert len(manager.cache) == 0

    def test_get_cache_stats(self, manager):
        """Test cache statistics."""
        stats = manager.get_cache_stats()

        assert "total_entries" in stats
        assert "expired_entries" in stats
        assert "active_entries" in stats

    def test_search_results(self, manager, mock_clients):
        """Test basic search functionality."""
        doc1 = CaseLawDocument(
            document_id="1",
            source_api="courtlistener",
            url="url1",
            case_name="Case A",
            court="court1",
        )

        result1 = SearchResult(
            query=SearchQuery(query_text="test"),
            documents=[doc1],
            total_results=1,
            page=1,
            has_more=False,
        )

        mock_clients["courtlistener"].search.return_value = result1

        query = SearchQuery(query_text="test")
        result = manager.search(query)

        assert len(result.documents) == 1
        assert result.documents[0].case_name == "Case A"
