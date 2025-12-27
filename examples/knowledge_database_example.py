"""
Example usage of the persistent legal knowledge database.

Demonstrates adding rules, searching, tracking performance, and migration.

Issue #271: Persistent Legal Knowledge Database
"""

from loft.knowledge.database import KnowledgeDatabase
from loft.knowledge.migration import migrate_asp_files_to_database


def main():
    # Initialize database
    print("Initializing knowledge database...")
    db = KnowledgeDatabase("sqlite:///legal_knowledge.db")

    # Add some rules
    print("\n1. Adding rules to database...")

    contract_rule_id = db.add_rule(
        asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
        domain="contracts",
        doctrine="offer-acceptance",
        stratification_level="tactical",
        confidence=0.95,
        reasoning="A valid contract requires three essential elements",
        tags=["foundational", "contracts-101"],
    )
    print(f"  Added contract rule: {contract_rule_id}")

    tort_rule_id = db.add_rule(
        asp_rule="negligence(X) :- duty(X), breach(X), causation(X), damages(X).",
        domain="torts",
        doctrine="negligence",
        stratification_level="tactical",
        confidence=0.98,
        reasoning="Negligence requires four elements",
    )
    print(f"  Added tort rule: {tort_rule_id}")

    # Search for rules
    print("\n2. Searching for contract rules...")
    contract_rules = db.search_rules(domain="contracts", min_confidence=0.9)

    for rule in contract_rules:
        print(f"  Rule: {rule.asp_rule}")
        print(f"  Confidence: {rule.confidence}")
        print(f"  Reasoning: {rule.reasoning}")

    # Record a question
    print("\n3. Recording a legal question...")
    question_id = db.add_question(
        question_text="Is a contract valid if there is offer and acceptance but no consideration?",
        asp_query="offer(c1). acceptance(c1). not consideration(c1). ?- valid_contract(c1).",
        answer="no",
        reasoning="The rule requires all three elements; missing consideration makes it invalid",
        rules_used=[contract_rule_id],
        confidence=0.95,
        correct=True,
        domain="contracts",
    )
    print(f"  Recorded question: {question_id}")

    # Update rule performance
    print("\n4. Updating rule performance...")
    db.update_rule_performance(contract_rule_id, success=True)
    print(f"  Marked rule {contract_rule_id[:8]}... as successful")

    # Get coverage statistics
    print("\n5. Coverage statistics:")
    stats = db.get_coverage_stats("contracts")
    print(f"  Domain: {stats.domain}")
    print(f"  Rules: {stats.rule_count}")
    print(f"  Questions: {stats.question_count}")
    print(f"  Accuracy: {stats.accuracy:.0%}" if stats.accuracy else "  Accuracy: N/A")

    # Get overall database stats
    print("\n6. Overall database statistics:")
    overall_stats = db.get_database_stats()
    print(f"  Total rules: {overall_stats.total_rules}")
    print(f"  Active rules: {overall_stats.active_rules}")
    print(f"  Total questions: {overall_stats.total_questions}")
    print(f"  Domains: {', '.join(overall_stats.domains)}")
    print(f"  Average confidence: {overall_stats.avg_confidence:.2%}")

    # Export to ASP files
    print("\n7. Exporting rules to ASP files...")
    export_stats = db.export_to_asp_files("./exported_rules")
    print(f"  Files created: {export_stats['files_created']}")
    print(f"  Rules exported: {export_stats['rules_exported']}")

    # Close database
    db.close()
    print("\nDatabase operations complete!")


def migration_example():
    """Example of migrating existing ASP files to database."""
    print("\nMigration Example")
    print("=" * 50)

    # Migrate existing ASP files
    print("Migrating ASP files from ./asp_rules to database...")

    result = migrate_asp_files_to_database(
        rules_dir="./asp_rules",
        database_url="sqlite:///legal_knowledge.db",
        default_domain="general",
    )

    print("Migration complete:")
    print(f"  Files processed: {result.files_processed}")
    print(f"  Rules imported: {result.rules_imported}")
    print(f"  Rules skipped: {result.rules_skipped}")
    print(f"  Errors: {result.errors}")

    if result.error_messages:
        print("\nErrors encountered:")
        for error in result.error_messages:
            print(f"  - {error}")


if __name__ == "__main__":
    main()

    # Uncomment to run migration example
    # migration_example()
