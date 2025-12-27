"""
Unit tests for search schemas.

Issue #275: Rule Retrieval and Search
"""

import pytest

from loft.knowledge.models import LegalRule
from loft.knowledge.search.schemas import SearchQuery, SearchResult, SearchResults


class TestSearchQuery:
    """Test SearchQuery schema."""

    def test_basic_query_creation(self):
        """Test creating a basic search query."""
        query = SearchQuery(query_text="contract formation")

        assert query.query_text == "contract formation"
        assert query.domain is None
        assert query.max_results == 10
        assert query.min_confidence == 0.0

    def test_keyword_extraction(self):
        """Test automatic keyword extraction from query text."""
        query = SearchQuery(query_text="contract offer and acceptance")

        # Should extract keywords, filtering stop words
        assert "contract" in query.keywords
        assert "offer" in query.keywords
        assert "acceptance" in query.keywords
        # Stop words should be filtered
        assert "and" not in query.keywords
        assert "the" not in query.keywords

    def test_custom_keywords(self):
        """Test providing custom keywords."""
        query = SearchQuery(
            query_text="test",
            keywords=["custom", "keywords"],
        )

        # Should use provided keywords
        assert query.keywords == ["custom", "keywords"]

    def test_predicate_filtering(self):
        """Test predicate specification."""
        query = SearchQuery(
            query_text="test",
            predicates=["offer", "acceptance", "consideration"],
        )

        assert len(query.predicates) == 3
        assert "offer" in query.predicates

    def test_domain_filtering(self):
        """Test domain filtering."""
        query = SearchQuery(
            query_text="test",
            domain="contracts",
            jurisdiction="federal",
            doctrine="offer and acceptance",
        )

        assert query.domain == "contracts"
        assert query.jurisdiction == "federal"
        assert query.doctrine == "offer and acceptance"

    def test_performance_boosting(self):
        """Test performance boosting flags."""
        query = SearchQuery(
            query_text="test",
            boost_performance=True,
            boost_confidence=True,
        )

        assert query.boost_performance is True
        assert query.boost_confidence is True

    def test_query_string_representation(self):
        """Test query string representation."""
        query = SearchQuery(
            query_text="contract formation",
            domain="contracts",
            doctrine="formation",
        )

        query_str = str(query)
        assert "contract formation" in query_str
        assert "contracts" in query_str


class TestSearchResult:
    """Test SearchResult schema."""

    @pytest.fixture
    def sample_rule(self):
        """Create sample rule for testing."""
        return LegalRule(
            rule_id="test-rule-123",
            asp_rule="valid_contract(X) :- offer(X), acceptance(X).",
            domain="contracts",
            confidence=0.9,
        )

    def test_result_creation(self, sample_rule):
        """Test creating a search result."""
        result = SearchResult(
            rule=sample_rule,
            relevance_score=0.85,
            text_match_score=0.8,
            domain_match_score=0.9,
            confidence_score=0.9,
            performance_score=0.7,
            matched_keywords=["contract", "offer"],
            explanation="Strong text match",
        )

        assert result.rule == sample_rule
        assert result.relevance_score == 0.85
        assert len(result.matched_keywords) == 2

    def test_rule_id_property(self, sample_rule):
        """Test rule_id property access."""
        result = SearchResult(rule=sample_rule, relevance_score=0.8)

        assert result.rule_id == sample_rule.rule_id

    def test_asp_rule_property(self, sample_rule):
        """Test asp_rule property access."""
        result = SearchResult(rule=sample_rule, relevance_score=0.8)

        assert result.asp_rule == sample_rule.asp_rule

    def test_score_breakdown(self, sample_rule):
        """Test score breakdown method."""
        result = SearchResult(
            rule=sample_rule,
            relevance_score=0.85,
            text_match_score=0.8,
            domain_match_score=0.9,
            confidence_score=0.9,
            performance_score=0.7,
            semantic_score=0.6,
        )

        breakdown = result.get_score_breakdown()

        assert breakdown["overall"] == 0.85
        assert breakdown["text_match"] == 0.8
        assert breakdown["domain_match"] == 0.9
        assert breakdown["confidence"] == 0.9
        assert breakdown["performance"] == 0.7
        assert breakdown["semantic"] == 0.6

    def test_result_string_representation(self, sample_rule):
        """Test result string representation."""
        result = SearchResult(rule=sample_rule, relevance_score=0.85)

        result_str = str(result)
        assert "SearchResult" in result_str
        assert "0.85" in result_str


class TestSearchResults:
    """Test SearchResults collection."""

    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        rules = [
            LegalRule(
                rule_id=f"rule-{i}",
                asp_rule=f"rule_{i}(X) :- condition_{i}(X).",
                domain="test",
                confidence=0.8 + (i * 0.02),
            )
            for i in range(5)
        ]

        search_results = [
            SearchResult(
                rule=rule,
                relevance_score=0.9 - (i * 0.1),  # Descending scores
            )
            for i, rule in enumerate(rules)
        ]

        query = SearchQuery(query_text="test query")

        return SearchResults(
            query=query,
            results=search_results,
            total_searched=10,
            search_time_ms=25.5,
        )

    def test_results_creation(self, sample_results):
        """Test creating search results collection."""
        assert sample_results.count == 5
        assert sample_results.total_searched == 10
        assert sample_results.search_time_ms == 25.5

    def test_count_property(self, sample_results):
        """Test count property."""
        assert sample_results.count == len(sample_results.results)

    def test_top_result_property(self, sample_results):
        """Test top_result property."""
        top = sample_results.top_result

        assert top is not None
        assert top.relevance_score == 0.9  # Highest score

    def test_top_result_empty(self):
        """Test top_result with no results."""
        query = SearchQuery(query_text="test")
        results = SearchResults(query=query, results=[])

        assert results.top_result is None

    def test_average_relevance(self, sample_results):
        """Test average relevance calculation."""
        # Scores are 0.9, 0.8, 0.7, 0.6, 0.5
        expected_avg = (0.9 + 0.8 + 0.7 + 0.6 + 0.5) / 5
        assert abs(sample_results.avg_relevance - expected_avg) < 0.001

    def test_average_relevance_empty(self):
        """Test average relevance with no results."""
        query = SearchQuery(query_text="test")
        results = SearchResults(query=query, results=[])

        assert results.avg_relevance == 0.0

    def test_get_rules(self, sample_results):
        """Test extracting rules from results."""
        rules = sample_results.get_rules()

        assert len(rules) == 5
        assert all(isinstance(rule, LegalRule) for rule in rules)

    def test_filter_by_score(self, sample_results):
        """Test filtering results by score."""
        # Filter for scores >= 0.7
        filtered = sample_results.filter_by_score(min_score=0.7)

        assert filtered.count == 3  # 0.9, 0.8, 0.7
        for result in filtered.results:
            assert result.relevance_score >= 0.7

        # Original should be unchanged
        assert sample_results.count == 5

    def test_to_dict(self, sample_results):
        """Test conversion to dictionary."""
        result_dict = sample_results.to_dict()

        assert "query" in result_dict
        assert "count" in result_dict
        assert "avg_relevance" in result_dict
        assert "search_time_ms" in result_dict
        assert "results" in result_dict

        # Should include top 5 results
        assert len(result_dict["results"]) <= 5

    def test_string_representation(self, sample_results):
        """Test string representation."""
        results_str = str(sample_results)

        assert "SearchResults" in results_str
        assert "5 results" in results_str

    def test_empty_results(self):
        """Test with empty results."""
        query = SearchQuery(query_text="test")
        results = SearchResults(
            query=query,
            results=[],
            total_searched=100,
        )

        assert results.count == 0
        assert results.avg_relevance == 0.0
        assert results.top_result is None
        assert len(results.get_rules()) == 0


class TestSearchQueryEdgeCases:
    """Test edge cases for SearchQuery."""

    def test_empty_query_text(self):
        """Test with empty query text."""
        query = SearchQuery(query_text="")

        assert query.query_text == ""
        assert query.keywords == []

    def test_query_with_only_stopwords(self):
        """Test query containing only stop words."""
        query = SearchQuery(query_text="the and or in of to for")

        # Should filter out all stop words
        assert len(query.keywords) == 0

    def test_query_with_short_words(self):
        """Test query with short words (< 3 characters)."""
        query = SearchQuery(query_text="a I it contract")

        # Short words should be filtered
        assert "a" not in query.keywords
        assert "I" not in query.keywords
        assert "it" not in query.keywords
        # Long word should remain
        assert "contract" in query.keywords

    def test_max_results_validation(self):
        """Test max_results parameter."""
        query = SearchQuery(query_text="test", max_results=100)

        assert query.max_results == 100

    def test_min_confidence_range(self):
        """Test min_confidence parameter."""
        query_low = SearchQuery(query_text="test", min_confidence=0.0)
        query_high = SearchQuery(query_text="test", min_confidence=0.95)

        assert query_low.min_confidence == 0.0
        assert query_high.min_confidence == 0.95
