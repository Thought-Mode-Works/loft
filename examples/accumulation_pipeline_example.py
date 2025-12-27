"""
Example usage of the rule accumulation pipeline.

Demonstrates processing legal cases to extract and accumulate rules
in the knowledge database.

Issue #273: Continuous Rule Accumulation Pipeline

Usage:
    python examples/accumulation_pipeline_example.py
"""

import logging
from pathlib import Path

from loft.accumulation import (
    CaseData,
    ConflictDetector,
    RuleAccumulationPipeline,
)
from loft.knowledge.database import KnowledgeDatabase

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def example_1_process_single_case():
    """
    Example 1: Process a single case to extract rules.
    """
    print("\n" + "=" * 70)
    print("Example 1: Processing a Single Case")
    print("=" * 70 + "\n")

    # Initialize database
    db = KnowledgeDatabase("sqlite:///examples/example_knowledge.db")

    # Initialize pipeline
    pipeline = RuleAccumulationPipeline(
        knowledge_db=db,
        min_rule_confidence=0.7,
        auto_resolve_conflicts=True,
    )

    # Create a sample case
    case = CaseData(
        case_id="contract_001",
        description="Basic contract formation case",
        facts=[
            "Alice offered to sell her car to Bob for $5,000",
            "Bob accepted the offer",
            "Bob paid $5,000 to Alice",
            "Alice transferred the car title to Bob",
        ],
        asp_facts="""
offer(contract1).
offeror(contract1, alice).
offeree(contract1, bob).
acceptance(contract1).
consideration(contract1).
consideration_paid(contract1, 5000).
subject_matter(contract1, car).
""",
        question="Is the contract between Alice and Bob valid?",
        ground_truth="valid_contract",
        rationale="""
A valid contract requires three essential elements: offer, acceptance, and consideration.
In this case, Alice made an offer to sell her car, Bob accepted that offer, and
consideration was exchanged (Bob paid $5,000 and received the car). All three elements
are present, therefore the contract is valid.
""",
        legal_citations=["Contract Formation Doctrine", "Essential Elements Rule"],
        difficulty="easy",
        domain="contracts",
    )

    # Process the case
    print(f"Processing case: {case.case_id}")
    print(f"Domain: {case.domain}")
    print(f"Description: {case.description}\n")

    result = pipeline.process_case(case)

    # Display results
    print("Accumulation Result:")
    print(f"  Rules Added: {result.rules_added}")
    print(f"  Rules Skipped: {result.rules_skipped}")
    print(f"  Success Rate: {result.success_rate:.1%}")
    print(f"  Conflicts Found: {len(result.conflicts_found)}")
    print(f"  Processing Time: {result.processing_time_ms:.0f}ms")

    if result.rule_ids:
        print("\n  Added Rule IDs:")
        for rule_id in result.rule_ids:
            print(f"    - {rule_id}")

    if result.conflicts_found:
        print("\n  Conflicts:")
        for conflict in result.conflicts_found:
            print(f"    - {conflict.conflict_type}: {conflict.explanation[:60]}...")

    if result.skipped_reasons:
        print("\n  Skip Reasons:")
        for reason in result.skipped_reasons:
            print(f"    - {reason}")

    return db, pipeline


def example_2_process_dataset(db, pipeline):
    """
    Example 2: Process a batch of cases from a dataset.
    """
    print("\n" + "=" * 70)
    print("Example 2: Processing a Dataset")
    print("=" * 70 + "\n")

    # Check if dataset exists
    dataset_path = Path("datasets/adverse_possession")

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Skipping dataset processing example...")
        return

    print(f"Processing dataset: {dataset_path}")

    # Process dataset
    report = pipeline.process_dataset(
        dataset_path=dataset_path,
        max_cases=3,  # Process only first 3 cases for demo
    )

    # Display report
    print("\n" + report.to_string())


def example_3_check_conflicts():
    """
    Example 3: Check for conflicts before adding rules.
    """
    print("\n" + "=" * 70)
    print("Example 3: Conflict Detection")
    print("=" * 70 + "\n")

    # Initialize database with some existing rules
    db = KnowledgeDatabase("sqlite:///examples/example_knowledge.db")

    # Add an existing rule
    existing_rule_id = db.add_rule(
        asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
        domain="contracts",
        confidence=0.95,
        reasoning="A contract is valid if it has offer, acceptance, and consideration",
    )

    print(f"Added existing rule: {existing_rule_id}")

    # Create conflict detector
    detector = ConflictDetector(knowledge_db=db)

    # Create a new rule candidate that might conflict
    from loft.accumulation.schemas import RuleCandidate

    # Similar rule (subsumption)
    similar_rule = RuleCandidate(
        asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
        domain="contracts",
        confidence=0.90,
        reasoning="Contract requires three elements",
        source_case_id="test_case_1",
    )

    print("\nChecking similar rule for conflicts...")
    conflicts = detector.find_conflicts(similar_rule, domain="contracts")

    if conflicts:
        print(f"Found {len(conflicts)} conflicts:")
        for conflict in conflicts:
            print(f"\n  Type: {conflict.conflict_type}")
            print(f"  Severity: {conflict.severity:.2f}")
            print(f"  Explanation: {conflict.explanation}")

            suggestion = detector.suggest_resolution(conflict)
            print(f"  Suggestion: {suggestion}")
    else:
        print("No conflicts found")

    # Contradicting rule
    contradicting_rule = RuleCandidate(
        asp_rule="not valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
        domain="contracts",
        confidence=0.85,
        reasoning="Contracts are not valid even with all elements",
        source_case_id="test_case_2",
    )

    print("\n\nChecking contradicting rule for conflicts...")
    conflicts = detector.find_conflicts(contradicting_rule, domain="contracts")

    if conflicts:
        print(f"Found {len(conflicts)} conflicts:")
        for conflict in conflicts:
            print(f"\n  Type: {conflict.conflict_type}")
            print(f"  Severity: {conflict.severity:.2f}")
            print(f"  Explanation: {conflict.explanation}")

            suggestion = detector.suggest_resolution(conflict)
            print(f"  Suggestion: {suggestion}")
    else:
        print("No conflicts found")


def example_4_accumulation_stats(pipeline):
    """
    Example 4: View accumulation statistics.
    """
    print("\n" + "=" * 70)
    print("Example 4: Accumulation Statistics")
    print("=" * 70 + "\n")

    stats = pipeline.get_accumulation_stats()

    print("Current Accumulation Stats:")
    print(f"  Total Rules: {stats['total_rules']}")
    print(f"  Active Rules: {stats['active_rules']}")
    print(f"  Archived Rules: {stats['archived_rules']}")
    print(f"  Average Confidence: {stats['avg_confidence']:.2f}")

    if stats["domains"]:
        print("\n  Rules by Domain:")
        for domain, count in sorted(stats["domains"].items()):
            print(f"    {domain}: {count}")


def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 70)
    print("Rule Accumulation Pipeline Examples")
    print("=" * 70)

    # Example 1: Process single case
    db, pipeline = example_1_process_single_case()

    # Example 2: Process dataset (if available)
    example_2_process_dataset(db, pipeline)

    # Example 3: Check conflicts
    example_3_check_conflicts()

    # Example 4: View statistics
    example_4_accumulation_stats(pipeline)

    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70 + "\n")

    print("Database saved to: examples/example_knowledge.db")
    print(
        "\nYou can now use the CLI commands to interact with the accumulated rules:\n"
    )
    print("  # Process a case")
    print(
        "  python -m loft.accumulation.cli process-case datasets/adverse_possession/ap_001.json\n"
    )
    print("  # Process a dataset")
    print(
        "  python -m loft.accumulation.cli process-dataset datasets/adverse_possession\n"
    )
    print("  # Check conflicts")
    print(
        "  python -m loft.accumulation.cli check-conflicts datasets/adverse_possession/ap_001.json\n"
    )
    print("  # View statistics")
    print("  python -m loft.accumulation.cli stats\n")


if __name__ == "__main__":
    main()
