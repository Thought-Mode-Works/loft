#!/usr/bin/env python3
"""
Verify transfer study datasets are properly formatted.

Quick validation script to ensure datasets load correctly before running
expensive transfer learning experiments.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.casework.dataset_loader import DatasetLoader


def verify_dataset(dataset_path: Path, expected_min_scenarios: int = 5) -> bool:
    """Verify a dataset is properly formatted and loadable."""
    print(f"\nVerifying dataset: {dataset_path}")
    print("=" * 60)

    loader = DatasetLoader(dataset_path)

    # Load all scenarios
    scenarios = loader.load_all()

    if not scenarios:
        print(f"❌ ERROR: No scenarios found in {dataset_path}")
        return False

    print(f"✓ Loaded {len(scenarios)} scenarios")

    if len(scenarios) < expected_min_scenarios:
        print(
            f"⚠ WARNING: Only {len(scenarios)} scenarios (expected at least {expected_min_scenarios})"
        )

    # Get statistics
    stats = loader.get_statistics()
    print("\nStatistics:")
    print(f"  Total scenarios: {stats['total_scenarios']}")
    print("  By difficulty:")
    for diff, count in stats["by_difficulty"].items():
        print(f"    {diff}: {count}")
    print("  By ground truth:")
    for truth, count in stats["by_ground_truth"].items():
        print(f"    {truth}: {count}")

    # Verify each scenario has required fields
    print("\nValidating scenario fields...")
    required_fields = [
        "scenario_id",
        "description",
        "facts",
        "question",
        "ground_truth",
        "rationale",
    ]

    for scenario in scenarios:
        missing = []
        for field in required_fields:
            if not getattr(scenario, field, None):
                missing.append(field)

        if missing:
            print(f"❌ Scenario {scenario.scenario_id} missing fields: {missing}")
            return False

    print(f"✓ All {len(scenarios)} scenarios have required fields")

    # Check for ASP facts
    asp_count = sum(1 for s in scenarios if s.asp_facts)
    print(f"✓ {asp_count}/{len(scenarios)} scenarios have ASP facts")

    return True


def main():
    """Main verification routine."""
    print("Transfer Study Dataset Verification")
    print("=" * 60)

    # Paths to datasets
    sof_path = Path("datasets/statute_of_frauds")
    prop_path = Path("datasets/property_law")

    # Verify both datasets
    sof_valid = verify_dataset(sof_path, expected_min_scenarios=10)
    prop_valid = verify_dataset(prop_path, expected_min_scenarios=10)

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"Statute of Frauds dataset: {'✓ PASS' if sof_valid else '❌ FAIL'}")
    print(f"Property Law dataset: {'✓ PASS' if prop_valid else '❌ FAIL'}")

    if sof_valid and prop_valid:
        print("\n✓ All datasets are valid and ready for transfer study experiments!")
        print("\nYou can now run:")
        print("  python3 experiments/casework/transfer_study.py \\")
        print("    --source-domain datasets/statute_of_frauds \\")
        print("    --target-domain datasets/property_law \\")
        print("    --few-shot 10 \\")
        print("    --output reports/transfer_test.json")
        return 0
    else:
        print("\n❌ Some datasets failed validation. Please fix errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
