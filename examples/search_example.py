"""
Example demonstrating rule search and retrieval functionality.

This example shows how to use the intelligent search system to find
relevant legal rules using text queries, predicates, and filters.

Issue #275: Rule Retrieval and Search
"""

from loft.knowledge.database import KnowledgeDatabase
from loft.knowledge.search.schemas import SearchQuery


def setup_example_database():
    """Create and populate an example database."""
    db = KnowledgeDatabase("sqlite:///search_example.db")

    print("Setting up example database...")

    # Add contract rules
    db.add_rule(
        asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
        domain="contracts",
        jurisdiction="federal",
        doctrine="contract formation",
        confidence=0.95,
        reasoning="A valid contract requires offer, acceptance, and consideration",
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
        asp_rule="breach_of_contract(X) :- contract(X), performance_due(X), non_performance(X).",
        domain="contracts",
        jurisdiction="federal",
        doctrine="breach",
        confidence=0.90,
        reasoning="Breach occurs when performance is due but not rendered",
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
        asp_rule="intentional_tort(X) :- intent(X), volitional_act(X), causation(X), harm(X).",
        domain="torts",
        jurisdiction="federal",
        doctrine="intentional torts",
        confidence=0.94,
        reasoning="Intentional tort requires intent, volitional act, causation, and harm",
    )

    # Simulate some usage for performance scoring
    rules = db.search_rules(domain="contracts", limit=2)
    for rule in rules:
        for _ in range(5):
            db.update_rule_performance(rule.rule_id, success=True)

    print(f"Added {db.get_database_stats().total_rules} rules\n")

    return db


def example_1_basic_text_search(db):
    """Example 1: Basic text search."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Text Search")
    print("=" * 70)

    results = db.search_by_text(
        text="contract formation offer acceptance",
        max_results=3,
    )

    print(f"\nQuery: 'contract formation offer acceptance'")
    print(f"Found {results.count} results in {results.search_time_ms:.1f}ms")
    print(f"Average relevance: {results.avg_relevance:.2f}\n")

    for i, result in enumerate(results.results, 1):
        print(f"{i}. Rule ID: {result.rule_id[:12]}...")
        print(f"   Relevance: {result.relevance_score:.3f}")
        print(f"   Rule: {result.asp_rule}")
        print(f"   Matched Keywords: {', '.join(result.matched_keywords)}")
        print(f"   Explanation: {result.explanation}")
        print()


def example_2_domain_filtered_search(db):
    """Example 2: Search with domain filtering."""
    print("=" * 70)
    print("EXAMPLE 2: Domain-Filtered Search")
    print("=" * 70)

    results = db.search_by_text(
        text="duty breach causation",
        domain="torts",
        max_results=5,
    )

    print(f"\nQuery: 'duty breach causation' in domain 'torts'")
    print(f"Found {results.count} results\n")

    for i, result in enumerate(results.results, 1):
        print(f"{i}. {result.asp_rule}")
        print(f"   Domain: {result.rule.domain}")
        print(f"   Doctrine: {result.rule.doctrine}")
        print(f"   Relevance: {result.relevance_score:.3f}")
        print()


def example_3_predicate_search(db):
    """Example 3: Search by predicates."""
    print("=" * 70)
    print("EXAMPLE 3: Predicate-Based Search")
    print("=" * 70)

    results = db.search_by_predicates(
        predicates=["offer", "acceptance", "consideration"],
        domain="contracts",
    )

    print(f"\nSearching for predicates: offer, acceptance, consideration")
    print(f"Found {results.count} results\n")

    for i, result in enumerate(results.results, 1):
        print(f"{i}. {result.asp_rule}")
        print(f"   Matched Predicates: {', '.join(result.matched_predicates)}")
        print(f"   Relevance: {result.relevance_score:.3f}")
        print()


def example_4_advanced_search(db):
    """Example 4: Advanced search with all filters."""
    print("=" * 70)
    print("EXAMPLE 4: Advanced Search with Filters")
    print("=" * 70)

    query = SearchQuery(
        query_text="contract valid enforceable",
        domain="contracts",
        jurisdiction="federal",
        doctrine="contract formation",
        min_confidence=0.85,
        max_results=5,
        boost_performance=True,
        boost_confidence=True,
    )

    results = db.intelligent_search(query)

    print(f"\nAdvanced Query:")
    print(f"  Text: '{query.query_text}'")
    print(f"  Domain: {query.domain}")
    print(f"  Jurisdiction: {query.jurisdiction}")
    print(f"  Doctrine: {query.doctrine}")
    print(f"  Min Confidence: {query.min_confidence}")
    print(f"\nFound {results.count} results\n")

    for i, result in enumerate(results.results, 1):
        print(f"{i}. {result.asp_rule}")

        # Show score breakdown
        scores = result.get_score_breakdown()
        print(f"   Overall Relevance: {scores['overall']:.3f}")
        print(f"   - Text Match: {scores['text_match']:.3f}")
        print(f"   - Domain Match: {scores['domain_match']:.3f}")
        print(f"   - Confidence: {scores['confidence']:.3f}")
        print(f"   - Performance: {scores['performance']:.3f}")
        print()


def example_5_similar_rules(db):
    """Example 5: Find similar rules."""
    print("=" * 70)
    print("EXAMPLE 5: Finding Similar Rules")
    print("=" * 70)

    # Get a contract rule
    all_rules = db.search_rules(domain="contracts", limit=1)
    if not all_rules:
        print("No rules found to compare against.")
        return

    test_rule = all_rules[0]

    print(f"\nOriginal Rule:")
    print(f"  {test_rule.asp_rule}")
    print(f"  Domain: {test_rule.domain}")
    print()

    # Find similar rules
    similar = db.find_similar_rules(
        rule_id=test_rule.rule_id,
        max_results=3,
    )

    print(f"Found {similar.count} similar rules:\n")

    for i, result in enumerate(similar.results, 1):
        print(f"{i}. {result.asp_rule}")
        print(f"   Similarity: {result.relevance_score:.3f}")
        print(f"   Explanation: {result.explanation}")
        print()


def example_6_score_comparison(db):
    """Example 6: Compare scoring with different boost options."""
    print("=" * 70)
    print("EXAMPLE 6: Score Comparison (Performance Boosting)")
    print("=" * 70)

    # Search without performance boosting
    query_no_boost = SearchQuery(
        query_text="contract",
        domain="contracts",
        boost_performance=False,
        boost_confidence=False,
        max_results=3,
    )

    results_no_boost = db.intelligent_search(query_no_boost)

    # Search with performance boosting
    query_boost = SearchQuery(
        query_text="contract",
        domain="contracts",
        boost_performance=True,
        boost_confidence=True,
        max_results=3,
    )

    results_boost = db.intelligent_search(query_boost)

    print("\nWithout Performance/Confidence Boosting:")
    for i, result in enumerate(results_no_boost.results, 1):
        print(f"{i}. {result.rule_id[:12]}... - Relevance: {result.relevance_score:.3f}")

    print("\nWith Performance/Confidence Boosting:")
    for i, result in enumerate(results_boost.results, 1):
        print(f"{i}. {result.rule_id[:12]}... - Relevance: {result.relevance_score:.3f}")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("RULE SEARCH AND RETRIEVAL EXAMPLES")
    print("Issue #275: Rule Retrieval and Search")
    print("=" * 70 + "\n")

    # Setup database
    db = setup_example_database()

    try:
        # Run examples
        example_1_basic_text_search(db)
        example_2_domain_filtered_search(db)
        example_3_predicate_search(db)
        example_4_advanced_search(db)
        example_5_similar_rules(db)
        example_6_score_comparison(db)

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    finally:
        db.close()
        print("\nDatabase closed.")


if __name__ == "__main__":
    main()
