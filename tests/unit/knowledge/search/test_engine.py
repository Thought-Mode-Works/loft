"""
Unit tests for rule search engine.

Issue #275: Rule Retrieval and Search
"""

import pytest

from loft.knowledge.database import KnowledgeDatabase
from loft.knowledge.models import LegalRule
from loft.knowledge.search.engine import RuleSearchEngine
from loft.knowledge.search.schemas import SearchQuery
from loft.knowledge.search.scorer import RelevanceScorer


@pytest.fixture
def temp_db(tmp_path):
    """Create temporary database for testing."""
    db_path = tmp_path / "test_search.db"
    db = KnowledgeDatabase(f"sqlite:///{db_path}")
    yield db
    db.close()


@pytest.fixture
def populated_db(temp_db):
    """Database populated with sample rules."""
    # Add contract rules
    temp_db.add_rule(
        asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
        domain="contracts",
        jurisdiction="federal",
        confidence=0.95,
        reasoning="Contract formation requires offer, acceptance, and consideration",
    )

    temp_db.add_rule(
        asp_rule="enforceable(X) :- valid_contract(X), capacity(X), legality(X).",
        domain="contracts",
        jurisdiction="federal",
        confidence=0.90,
        reasoning="Contract must be valid and parties must have capacity",
    )

    # Add tort rules
    temp_db.add_rule(
        asp_rule="negligence(X) :- duty(X), breach(X), causation(X), damages(X).",
        domain="torts",
        jurisdiction="federal",
        confidence=0.92,
        reasoning="Negligence requires duty, breach, causation, and damages",
    )

    # Add low confidence rule
    temp_db.add_rule(
        asp_rule="tentative_rule(X) :- some_condition(X).",
        domain="contracts",
        confidence=0.50,
        reasoning="Low confidence experimental rule",
    )

    return temp_db


@pytest.fixture
def search_engine(populated_db):
    """Create search engine with populated database."""
    return RuleSearchEngine(knowledge_db=populated_db)


class TestRuleSearchEngine:
    """Test rule search engine functionality."""

    def test_initialization(self, populated_db):
        """Test search engine initialization."""
        engine = RuleSearchEngine(knowledge_db=populated_db)
        assert engine.db == populated_db
        assert isinstance(engine.scorer, RelevanceScorer)

    def test_search_by_text_basic(self, search_engine):
        """Test basic text search."""
        results = search_engine.search_by_text(
            text="contract formation",
            max_results=10,
        )

        assert results.count > 0
        assert results.search_time_ms > 0
        # Results should be sorted by relevance
        for i in range(len(results.results) - 1):
            assert (
                results.results[i].relevance_score
                >= results.results[i + 1].relevance_score
            )

    def test_search_by_text_with_domain(self, search_engine):
        """Test text search with domain filter."""
        results = search_engine.search_by_text(
            text="valid contract",
            domain="contracts",
            max_results=10,
        )

        assert results.count > 0
        # All results should be from contracts domain
        for result in results.results:
            assert result.rule.domain == "contracts"

    def test_search_by_predicates(self, search_engine):
        """Test search by predicates."""
        results = search_engine.search_by_predicates(
            predicates=["offer", "acceptance"],
            max_results=10,
        )

        assert results.count > 0
        # Results should contain the predicates
        for result in results.results:
            assert any(
                pred in result.matched_predicates for pred in ["offer", "acceptance"]
            )

    def test_search_with_full_query(self, search_engine):
        """Test search with full SearchQuery object."""
        query = SearchQuery(
            query_text="contract formation",
            domain="contracts",
            min_confidence=0.8,
            max_results=5,
            boost_performance=True,
            boost_confidence=True,
        )

        results = search_engine.search(query)

        assert results.count <= 5
        assert results.query == query
        # All results should meet min confidence
        for result in results.results:
            assert result.rule.confidence >= 0.8

    def test_search_result_metadata(self, search_engine):
        """Test search result contains proper metadata."""
        results = search_engine.search_by_text("contract", max_results=5)

        assert hasattr(results, "query")
        assert hasattr(results, "results")
        assert hasattr(results, "total_searched")
        assert hasattr(results, "search_time_ms")
        assert results.total_searched >= results.count

    def test_relevance_scoring(self, search_engine):
        """Test that results are scored and ranked by relevance."""
        results = search_engine.search_by_text(
            text="contract offer acceptance",
            max_results=10,
        )

        # All results should have relevance scores
        for result in results.results:
            assert 0.0 <= result.relevance_score <= 1.0
            assert hasattr(result, "text_match_score")
            assert hasattr(result, "domain_match_score")
            assert hasattr(result, "confidence_score")
            assert hasattr(result, "performance_score")

    def test_find_similar_rules(self, search_engine, populated_db):
        """Test finding similar rules."""
        # Get a rule
        rules = populated_db.search_rules(domain="contracts", limit=1)
        test_rule = rules[0]

        # Find similar rules
        results = search_engine.find_similar_rules(test_rule, max_results=5)

        # Should not include the original rule
        for result in results.results:
            assert result.rule_id != test_rule.rule_id

        # Should have relevance scores
        for result in results.results:
            assert result.relevance_score > 0.0

    def test_search_by_domain(self, search_engine):
        """Test search by domain."""
        results = search_engine.search_by_domain(
            domain="contracts",
            min_confidence=0.8,
            max_results=50,
        )

        assert results.count > 0
        # All should be from contracts domain with high confidence
        for result in results.results:
            assert result.rule.domain == "contracts"
            assert result.rule.confidence >= 0.8

    def test_get_top_rules(self, search_engine):
        """Test getting top-performing rules."""
        top_rules = search_engine.get_top_rules(
            domain="contracts",
            max_results=10,
        )

        assert len(top_rules) > 0
        assert all(isinstance(rule, LegalRule) for rule in top_rules)

    def test_empty_search_results(self, search_engine):
        """Test search with no matching results."""
        results = search_engine.search_by_text(
            text="nonexistent_predicate_xyz",
            max_results=10,
        )

        # Should return empty results, not error
        assert results.count == 0
        assert len(results.results) == 0
        assert results.search_time_ms >= 0

    def test_max_results_limit(self, search_engine):
        """Test that max_results is respected."""
        results = search_engine.search_by_text(
            text="contract",
            max_results=2,
        )

        assert len(results.results) <= 2

    def test_min_confidence_filter(self, search_engine):
        """Test minimum confidence filtering."""
        # Search with high min confidence
        query = SearchQuery(
            query_text="contract",
            min_confidence=0.85,
            max_results=10,
        )

        results = search_engine.search(query)

        # All results should meet minimum
        for result in results.results:
            assert result.rule.confidence >= 0.85

    def test_archived_rules_excluded_by_default(self, populated_db, search_engine):
        """Test that archived rules are excluded by default."""
        # Archive a rule
        rules = populated_db.search_rules(domain="contracts", limit=1)
        if rules:
            populated_db.archive_rule(rules[0].rule_id)

        # Search should not include archived
        query = SearchQuery(
            query_text="contract",
            include_archived=False,
            max_results=10,
        )

        results = search_engine.search(query)

        # No results should be archived
        for result in results.results:
            assert not result.rule.is_archived

    def test_search_with_multiple_predicates(self, search_engine):
        """Test search with multiple predicates."""
        results = search_engine.search_by_predicates(
            predicates=["offer", "acceptance", "consideration"],
            domain="contracts",
            max_results=10,
        )

        # Should find rules with these predicates
        assert results.count > 0

        # Check that matched predicates are tracked
        for result in results.results:
            assert len(result.matched_predicates) > 0

    def test_search_performance(self, search_engine):
        """Test that search completes in reasonable time."""
        import time

        start = time.time()

        results = search_engine.search_by_text(
            text="contract",
            max_results=100,
        )

        elapsed = time.time() - start

        # Should complete quickly (< 1 second for small database)
        assert elapsed < 1.0
        # Reported time should be reasonable
        assert 0 < results.search_time_ms < 1000

    def test_average_relevance_calculation(self, search_engine):
        """Test average relevance calculation."""
        results = search_engine.search_by_text(
            text="contract",
            max_results=5,
        )

        if results.count > 0:
            # Calculate expected average
            expected_avg = sum(r.relevance_score for r in results.results) / len(
                results.results
            )
            assert abs(results.avg_relevance - expected_avg) < 0.001

    def test_custom_scorer(self, populated_db):
        """Test search engine with custom scorer."""
        # Create scorer favoring confidence
        custom_scorer = RelevanceScorer(
            text_weight=0.2,
            domain_weight=0.2,
            confidence_weight=0.5,
            performance_weight=0.1,
        )

        engine = RuleSearchEngine(knowledge_db=populated_db, scorer=custom_scorer)

        results = engine.search_by_text("contract", max_results=5)

        # Should still work with custom scorer
        assert results.count > 0
        for result in results.results:
            assert hasattr(result, "confidence_score")
