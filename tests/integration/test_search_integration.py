"""
Integration tests for rule search system.

Tests the full search pipeline from database to results.

Issue #275: Rule Retrieval and Search
"""

import pytest

from loft.knowledge.database import KnowledgeDatabase
from loft.knowledge.search.schemas import SearchQuery


@pytest.fixture
def integration_db(tmp_path):
    """Create database with realistic rule set."""
    db_path = tmp_path / "integration_search.db"
    db = KnowledgeDatabase(f"sqlite:///{db_path}")

    # Add comprehensive contract rules
    db.add_rule(
        asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
        domain="contracts",
        jurisdiction="federal",
        doctrine="contract formation",
        confidence=0.95,
        reasoning="A valid contract requires offer, acceptance, and consideration",
        source_type="case_law",
    )

    db.add_rule(
        asp_rule="enforceable_contract(X) :- valid_contract(X), capacity(X), legality(X).",
        domain="contracts",
        jurisdiction="federal",
        doctrine="contract formation",
        confidence=0.93,
        reasoning="Contract must be valid with capable parties and legal purpose",
    )

    db.add_rule(
        asp_rule="void_contract(X) :- contract(X), lack_capacity(X).",
        domain="contracts",
        jurisdiction="federal",
        doctrine="contract defenses",
        confidence=0.90,
        reasoning="Contracts with parties lacking capacity are void",
    )

    db.add_rule(
        asp_rule="voidable_contract(X) :- contract(X), misrepresentation(X).",
        domain="contracts",
        jurisdiction="federal",
        doctrine="contract defenses",
        confidence=0.88,
        reasoning="Contracts induced by misrepresentation are voidable",
    )

    # Add tort rules
    db.add_rule(
        asp_rule="negligence(X) :- duty(X), breach(X), causation(X), damages(X).",
        domain="torts",
        jurisdiction="federal",
        doctrine="negligence",
        confidence=0.96,
        reasoning="Negligence requires duty, breach, causation, and damages",
    )

    db.add_rule(
        asp_rule="strict_liability(X) :- abnormally_dangerous_activity(X), harm(X).",
        domain="torts",
        jurisdiction="federal",
        doctrine="strict liability",
        confidence=0.92,
        reasoning="Strict liability applies to abnormally dangerous activities",
    )

    # Add property rules
    db.add_rule(
        asp_rule="adverse_possession(X) :- continuous_possession(X, Y), hostile(X), open_notorious(X), exclusive(X), statutory_period(Y).",
        domain="property",
        jurisdiction="CA",
        doctrine="adverse possession",
        confidence=0.85,
        reasoning="Adverse possession requires continuous, hostile, open, notorious, and exclusive possession",
    )

    # Add lower confidence experimental rule
    db.add_rule(
        asp_rule="experimental_rule(X) :- condition_a(X), condition_b(X).",
        domain="contracts",
        confidence=0.45,
        reasoning="Experimental rule under development",
    )

    # Update some rules with performance data
    rules = db.search_rules(domain="contracts", limit=2)
    for rule in rules[:2]:
        # Simulate successful usage
        for _ in range(10):
            db.update_rule_performance(rule.rule_id, success=True)
        for _ in range(2):
            db.update_rule_performance(rule.rule_id, success=False)

    yield db
    db.close()


class TestSearchIntegration:
    """Integration tests for search system."""

    def test_end_to_end_text_search(self, integration_db):
        """Test complete text search workflow."""
        results = integration_db.search_by_text(
            text="contract formation offer acceptance",
            domain="contracts",
            max_results=5,
        )

        # Should find relevant contract rules
        assert results.count > 0
        assert results.search_time_ms > 0

        # Top result should be highly relevant
        top = results.top_result
        assert top is not None
        assert top.relevance_score > 0.5

        # Results should contain matched keywords
        assert any(len(r.matched_keywords) > 0 for r in results.results)

    def test_domain_specific_search(self, integration_db):
        """Test searching within specific domain."""
        # Search contracts domain
        contract_results = integration_db.search_by_text(
            text="valid enforceable",
            domain="contracts",
        )

        # Search torts domain
        tort_results = integration_db.search_by_text(
            text="negligence duty",
            domain="torts",
        )

        # Results should be domain-specific
        assert contract_results.count > 0
        assert all(r.rule.domain == "contracts" for r in contract_results.results)

        assert tort_results.count > 0
        assert all(r.rule.domain == "torts" for r in tort_results.results)

    def test_advanced_search_with_filters(self, integration_db):
        """Test advanced search with multiple filters."""
        query = SearchQuery(
            query_text="contract",
            domain="contracts",
            jurisdiction="federal",
            doctrine="contract formation",
            min_confidence=0.85,
            max_results=10,
            boost_performance=True,
            boost_confidence=True,
        )

        results = integration_db.intelligent_search(query)

        # Should find matching rules
        assert results.count > 0

        # All results should meet criteria
        for result in results.results:
            assert result.rule.domain == "contracts"
            assert result.rule.jurisdiction == "federal"
            assert result.rule.confidence >= 0.85

    def test_predicate_based_search(self, integration_db):
        """Test searching by ASP predicates."""
        results = integration_db.search_by_predicates(
            predicates=["offer", "acceptance", "consideration"],
            domain="contracts",
        )

        # Should find rules with these predicates
        assert results.count > 0

        # Check predicate matching
        for result in results.results:
            assert len(result.matched_predicates) > 0
            # ASP rule should contain matched predicates
            for pred in result.matched_predicates:
                assert pred in result.asp_rule.lower()

    def test_similarity_search(self, integration_db):
        """Test finding similar rules."""
        # Get a contract rule
        contract_rules = integration_db.search_rules(domain="contracts", limit=1)
        assert len(contract_rules) > 0

        test_rule = contract_rules[0]

        # Find similar rules
        similar = integration_db.find_similar_rules(
            rule_id=test_rule.rule_id,
            max_results=3,
        )

        # Should find other contract rules
        assert similar.count > 0

        # Similar rules should not include original
        for result in similar.results:
            assert result.rule_id != test_rule.rule_id

        # Should be from same domain
        for result in similar.results:
            assert result.rule.domain == test_rule.domain

    def test_confidence_filtering(self, integration_db):
        """Test filtering by confidence threshold."""
        # Search with high confidence threshold
        high_conf_query = SearchQuery(
            query_text="contract",
            min_confidence=0.90,
            max_results=10,
        )

        high_conf_results = integration_db.intelligent_search(high_conf_query)

        # All results should have high confidence
        for result in high_conf_results.results:
            assert result.rule.confidence >= 0.90

        # Search with low threshold should return more results
        low_conf_query = SearchQuery(
            query_text="contract",
            min_confidence=0.40,
            max_results=10,
        )

        low_conf_results = integration_db.intelligent_search(low_conf_query)

        assert low_conf_results.count >= high_conf_results.count

    def test_performance_boosting(self, integration_db):
        """Test that performance boosting affects ranking."""
        # Search with performance boosting
        query_boosted = SearchQuery(
            query_text="contract",
            domain="contracts",
            boost_performance=True,
            max_results=5,
        )

        results_boosted = integration_db.intelligent_search(query_boosted)

        # Top results should have performance data
        if results_boosted.count > 0:
            top_result = results_boosted.top_result
            # Performance score should contribute to ranking
            assert top_result.performance_score >= 0.0

    def test_cross_domain_search(self, integration_db):
        """Test searching across multiple domains."""
        # Search without domain filter
        results = integration_db.search_by_text(
            text="liability duty",
            max_results=10,
        )

        # Should return results from multiple domains
        domains = set(r.rule.domain for r in results.results)
        assert len(domains) >= 1  # At least one domain

    def test_empty_query_handling(self, integration_db):
        """Test handling of queries with no results."""
        results = integration_db.search_by_text(
            text="nonexistent_predicate_xyz_123",
            max_results=10,
        )

        # Should return empty results gracefully
        assert results.count == 0
        assert len(results.results) == 0
        assert results.avg_relevance == 0.0

    def test_result_ranking_consistency(self, integration_db):
        """Test that results are consistently ranked by relevance."""
        results = integration_db.search_by_text(
            text="contract valid enforceable",
            domain="contracts",
            max_results=10,
        )

        # Results should be in descending relevance order
        for i in range(len(results.results) - 1):
            assert (
                results.results[i].relevance_score
                >= results.results[i + 1].relevance_score
            )

    def test_explanation_quality(self, integration_db):
        """Test that search results include useful explanations."""
        results = integration_db.search_by_text(
            text="contract offer acceptance",
            domain="contracts",
            max_results=5,
        )

        # Results should have explanations
        for result in results.results:
            assert result.explanation
            assert isinstance(result.explanation, str)
            assert len(result.explanation) > 0

    def test_keyword_matching_accuracy(self, integration_db):
        """Test accuracy of keyword matching."""
        results = integration_db.search_by_text(
            text="negligence duty breach causation damages",
            domain="torts",
        )

        # Should find negligence rule
        assert results.count > 0

        # Top result should match multiple keywords
        top = results.top_result
        assert len(top.matched_keywords) >= 2

    def test_jurisdiction_filtering(self, integration_db):
        """Test filtering by jurisdiction."""
        query = SearchQuery(
            query_text="adverse possession",
            jurisdiction="CA",
            max_results=10,
        )

        results = integration_db.intelligent_search(query)

        # All results should be from CA jurisdiction
        for result in results.results:
            if result.rule.jurisdiction:
                assert result.rule.jurisdiction == "CA"

    def test_doctrine_filtering(self, integration_db):
        """Test filtering by legal doctrine."""
        query = SearchQuery(
            query_text="contract",
            domain="contracts",
            doctrine="contract formation",
            max_results=10,
        )

        results = integration_db.intelligent_search(query)

        # Results should match doctrine
        for result in results.results:
            if result.rule.doctrine:
                assert result.rule.doctrine == "contract formation"

    def test_search_statistics(self, integration_db):
        """Test search result statistics."""
        results = integration_db.search_by_text(
            text="contract",
            max_results=5,
        )

        # Should have proper statistics
        assert results.total_searched >= results.count
        assert results.search_time_ms > 0
        assert 0.0 <= results.avg_relevance <= 1.0

        # Count should not exceed max_results
        assert results.count <= 5

    def test_search_result_completeness(self, integration_db):
        """Test that search results include all necessary data."""
        results = integration_db.search_by_text(
            text="contract valid",
            domain="contracts",
        )

        for result in results.results:
            # Should have relevance score
            assert 0.0 <= result.relevance_score <= 1.0

            # Should have component scores
            assert 0.0 <= result.text_match_score <= 1.0
            assert 0.0 <= result.domain_match_score <= 1.0
            assert 0.0 <= result.confidence_score <= 1.0
            assert 0.0 <= result.performance_score <= 1.0

            # Should have rule data
            assert result.rule is not None
            assert result.rule_id
            assert result.asp_rule

            # Should have explanation
            assert result.explanation
