"""
Example usage of the Legal Question Answering Interface.

Demonstrates asking questions, batch evaluation, and working with the QA system.

Issue #272: Legal Question Answering Interface
"""

from loft.knowledge.database import KnowledgeDatabase
from loft.qa.interface import LegalQAInterface


def main():
    """Run QA interface examples."""
    # Initialize knowledge database
    print("Initializing knowledge database...")
    db = KnowledgeDatabase("sqlite:///legal_knowledge.db")

    # Add some sample rules for demonstration
    print("\n1. Adding sample rules to database...")
    try:
        db.add_rule(
            asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
            domain="contracts",
            doctrine="contract-formation",
            confidence=0.95,
            reasoning="A valid contract requires three essential elements",
        )
        print("  Added: contract validity rule")
    except ValueError:
        print("  Rule already exists (skipping)")

    try:
        db.add_rule(
            asp_rule="not enforceable(X) :- valid_contract(X), signed_under_duress(X).",
            domain="contracts",
            doctrine="defenses",
            confidence=0.90,
            reasoning="Contracts signed under duress are not enforceable",
        )
        print("  Added: duress defense rule")
    except ValueError:
        print("  Rule already exists (skipping)")

    try:
        db.add_rule(
            asp_rule="voidable(X) :- contract(X), party(X, P), minor(P).",
            domain="contracts",
            doctrine="capacity",
            confidence=0.92,
            reasoning="Contracts with minors are voidable",
        )
        print("  Added: minor capacity rule")
    except ValueError:
        print("  Rule already exists (skipping)")

    try:
        db.add_rule(
            asp_rule="negligence(X) :- duty(X), breach(X), causation(X), damages(X).",
            domain="torts",
            doctrine="negligence",
            confidence=0.98,
            reasoning="Negligence requires four elements",
        )
        print("  Added: negligence elements rule")
    except ValueError:
        print("  Rule already exists (skipping)")

    # Initialize QA interface
    print("\n2. Initializing QA interface...")
    qa = LegalQAInterface(knowledge_db=db)

    # Ask individual questions
    print("\n3. Asking individual legal questions...")
    print("=" * 60)

    questions = [
        "Is a contract valid if there is offer and acceptance but no consideration?",
        "Can a minor enter into a binding contract?",
        "Is a contract enforceable if it was signed under duress?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 60)
        answer = qa.ask(question, domain="contracts")
        print(answer.to_natural_language())
        print("=" * 60)

    # Batch evaluation
    print("\n\n4. Batch evaluation on test questions...")
    print("=" * 60)

    test_questions = [
        ("Is a contract valid without consideration?", "no"),
        ("Does negligence require duty, breach, causation, and damages?", "yes"),
        ("Can a minor void a contract?", "yes"),
    ]

    report = qa.batch_eval(test_questions)
    print(report.to_string())

    # Show performance summary
    print("\n5. Performance summary...")
    print("=" * 60)
    stats = qa.get_performance_summary()
    print(f"Total rules: {stats['total_rules']}")
    print(f"Active rules: {stats['active_rules']}")
    print(f"Questions asked: {stats['total_questions']}")
    print(f"Domains: {', '.join(stats['domains'])}")
    if stats.get("avg_confidence"):
        print(f"Average confidence: {stats['avg_confidence']:.1%}")

    # Show coverage by domain
    if stats.get("coverage_by_domain"):
        print("\nCoverage by domain:")
        for domain, count in stats["coverage_by_domain"].items():
            print(f"  {domain}: {count} rules")

    # Close database
    db.close()
    print("\n\nExample complete!")


if __name__ == "__main__":
    main()
