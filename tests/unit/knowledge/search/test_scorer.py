"""
Unit tests for relevance scoring.

Issue #275: Rule Retrieval and Search
"""

import pytest

from loft.knowledge.models import LegalRule
from loft.knowledge.search.schemas import SearchQuery
from loft.knowledge.search.scorer import RelevanceScorer


@pytest.fixture
def scorer():
    """Create scorer with default weights."""
    return RelevanceScorer()


@pytest.fixture
def sample_rule():
    """Create sample rule for testing."""
    return LegalRule(
        rule_id="test-rule-123",
        asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
        domain="contracts",
        jurisdiction="federal",
        doctrine="contract formation",
        confidence=0.9,
        reasoning="A contract requires offer, acceptance, and consideration",
        validation_count=10,
        success_count=8,
        failure_count=2,
    )


class TestRelevanceScorer:
    """Test relevance scoring functionality."""

    def test_scorer_initialization(self):
        """Test scorer initializes with normalized weights."""
        scorer = RelevanceScorer(
            text_weight=0.4,
            domain_weight=0.3,
            confidence_weight=0.2,
            performance_weight=0.1,
        )

        # Weights should sum to 1.0
        total = (
            scorer.text_weight
            + scorer.domain_weight
            + scorer.confidence_weight
            + scorer.performance_weight
            + scorer.semantic_weight
        )
        assert abs(total - 1.0) < 0.001  # Account for floating point

    def test_text_matching_exact_keyword(self, scorer, sample_rule):
        """Test text matching with exact keyword match."""
        query = SearchQuery(
            query_text="contract offer acceptance",
            keywords=["contract", "offer", "acceptance"],
        )

        result = scorer.score(sample_rule, query)

        # Should have high text match score
        assert result.text_match_score > 0.5
        assert "offer" in result.matched_keywords
        assert "acceptance" in result.matched_keywords

    def test_text_matching_no_keywords(self, scorer, sample_rule):
        """Test text matching with no keywords."""
        query = SearchQuery(query_text="", keywords=[])

        result = scorer.score(sample_rule, query)

        # Should return neutral score
        assert result.text_match_score == 0.5

    def test_domain_matching_exact(self, scorer, sample_rule):
        """Test domain matching with exact domain match."""
        query = SearchQuery(
            query_text="test",
            domain="contracts",
        )

        result = scorer.score(sample_rule, query)

        # Should have perfect domain match
        assert result.domain_match_score == 1.0

    def test_domain_matching_mismatch(self, scorer, sample_rule):
        """Test domain matching with domain mismatch."""
        query = SearchQuery(
            query_text="test",
            domain="torts",
        )

        result = scorer.score(sample_rule, query)

        # Should have low domain match
        assert result.domain_match_score < 0.5

    def test_confidence_scoring(self, scorer, sample_rule):
        """Test confidence-based scoring."""
        query = SearchQuery(
            query_text="test",
            boost_confidence=True,
        )

        result = scorer.score(sample_rule, query)

        # Confidence score should match rule confidence
        assert result.confidence_score == sample_rule.confidence

    def test_performance_scoring(self, scorer, sample_rule):
        """Test performance-based scoring."""
        query = SearchQuery(
            query_text="test",
            boost_performance=True,
        )

        result = scorer.score(sample_rule, query)

        # Should have high performance score (80% success rate)
        assert result.performance_score > 0.7

    def test_overall_relevance_score(self, scorer, sample_rule):
        """Test overall relevance score calculation."""
        query = SearchQuery(
            query_text="contract formation",
            domain="contracts",
            keywords=["contract", "formation"],
        )

        result = scorer.score(sample_rule, query)

        # Overall score should be weighted average
        assert 0.0 <= result.relevance_score <= 1.0

        # Should be combination of component scores
        expected = (
            (result.text_match_score * scorer.text_weight)
            + (result.domain_match_score * scorer.domain_weight)
            + (result.confidence_score * scorer.confidence_weight)
            + (result.performance_score * scorer.performance_weight)
        )
        assert abs(result.relevance_score - expected) < 0.001

    def test_matched_keywords_extraction(self, scorer, sample_rule):
        """Test extraction of matched keywords."""
        query = SearchQuery(
            query_text="offer acceptance consideration",
            keywords=["offer", "acceptance", "consideration", "invalid"],
        )

        result = scorer.score(sample_rule, query)

        # Should match keywords present in rule
        assert "offer" in result.matched_keywords
        assert "acceptance" in result.matched_keywords
        assert "consideration" in result.matched_keywords
        assert "invalid" not in result.matched_keywords

    def test_matched_predicates_extraction(self, scorer, sample_rule):
        """Test extraction of matched predicates."""
        query = SearchQuery(
            query_text="test",
            predicates=["offer", "acceptance", "invalid_predicate"],
        )

        result = scorer.score(sample_rule, query)

        # Should match predicates present in rule
        assert "offer" in result.matched_predicates
        assert "acceptance" in result.matched_predicates
        assert "invalid_predicate" not in result.matched_predicates

    def test_explanation_generation(self, scorer, sample_rule):
        """Test explanation generation."""
        query = SearchQuery(
            query_text="contract offer",
            domain="contracts",
            keywords=["contract", "offer"],
        )

        result = scorer.score(sample_rule, query)

        # Should have explanation
        assert result.explanation
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

    def test_phrase_match_bonus(self, scorer, sample_rule):
        """Test bonus for exact phrase match."""
        # Query with exact phrase from reasoning
        query_exact = SearchQuery(
            query_text="offer, acceptance, and consideration",
            keywords=["offer", "acceptance", "consideration"],
        )

        result_exact = scorer.score(sample_rule, query_exact)

        # Query without exact phrase
        query_no_phrase = SearchQuery(
            query_text="contract elements",
            keywords=["contract", "elements"],
        )

        result_no_phrase = scorer.score(sample_rule, query_no_phrase)

        # Exact phrase should have higher text match
        assert result_exact.text_match_score >= result_no_phrase.text_match_score

    def test_predicate_pattern_matching(self, scorer):
        """Test predicate pattern matching with regex."""
        # Rule with various predicate formats
        rule = LegalRule(
            rule_id="test-rule",
            asp_rule="result(X) :- predicate_one(X), predicate_two(X, Y).",
            domain="test",
        )

        query = SearchQuery(
            query_text="test",
            predicates=["predicate_one", "predicate_two", "result"],
        )

        result = scorer.score(rule, query)

        # Should match all three predicates
        assert len(result.matched_predicates) == 3
        assert "predicate_one" in result.matched_predicates
        assert "predicate_two" in result.matched_predicates
        assert "result" in result.matched_predicates

    def test_custom_weights(self):
        """Test scorer with custom weights."""
        # Scorer heavily favoring text matching
        scorer = RelevanceScorer(
            text_weight=0.8,
            domain_weight=0.1,
            confidence_weight=0.05,
            performance_weight=0.05,
        )

        rule = LegalRule(
            rule_id="test-rule",
            asp_rule="test(X) :- keyword_match(X).",
            domain="test",
            confidence=0.5,
        )

        query = SearchQuery(
            query_text="keyword match",
            domain="test",
            keywords=["keyword", "match"],
        )

        result = scorer.score(rule, query)

        # Text score should dominate
        assert result.text_match_score > 0.5
        # Overall should be heavily influenced by text
        assert abs(result.relevance_score - result.text_match_score) < 0.3
