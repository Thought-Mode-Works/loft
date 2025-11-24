"""
CLI interface for rule modification system.

Provides commands to:
- Run improvement sessions
- Show modification history
- Display policies
- Reset session counters
"""

import sys

from loguru import logger

from loft.core.incorporation import RuleIncorporationEngine
from loft.core.modification_session import ModificationSession
from loft.symbolic.stratification import (
    StratificationLevel,
    get_policy,
    print_all_policies,
)


def run_improvement_session(gaps: int = 5, layer: str = "tactical", candidates_per_gap: int = 3):
    """
    Run an improvement session.

    Args:
        gaps: Number of gaps to address
        layer: Target stratification layer
        candidates_per_gap: Number of candidates per gap
    """
    print("=" * 80)
    print(" LOFT Rule Modification Session")
    print("=" * 80)
    print()

    # Map layer string to enum
    layer_map = {
        "constitutional": StratificationLevel.CONSTITUTIONAL,
        "strategic": StratificationLevel.STRATEGIC,
        "tactical": StratificationLevel.TACTICAL,
        "operational": StratificationLevel.OPERATIONAL,
    }

    target_layer = layer_map.get(layer.lower())
    if target_layer is None:
        print(f"❌ Invalid layer: {layer}")
        print(f"   Valid layers: {', '.join(layer_map.keys())}")
        return

    print("Configuration:")
    print(f"  Target Layer: {target_layer.value}")
    print(f"  Max Gaps: {gaps}")
    print(f"  Candidates per Gap: {candidates_per_gap}")
    print()

    # Show policy
    policy = get_policy(target_layer)
    print("Layer Policy:")
    print(f"  Confidence Threshold: {policy.confidence_threshold:.2f}")
    print(f"  Max Modifications: {policy.max_modifications_per_session}")
    print(f"  Regression Tests: {'Required' if policy.regression_test_required else 'Optional'}")
    print()

    # Initialize components
    print("Initializing modification system...")
    incorporation_engine = RuleIncorporationEngine()
    session = ModificationSession(incorporation_engine)

    print(f"Baseline Accuracy: {incorporation_engine.test_suite.measure_accuracy():.1%}")
    print()

    # Run improvement cycle
    print("Running improvement cycle...")
    print()

    report = session.run_improvement_cycle(
        num_gaps=gaps, target_layer=target_layer, candidates_per_gap=candidates_per_gap
    )

    # Display results
    print(report.summary())

    if report.incorporated_details:
        print("\nIncorporated Rules:")
        print("-" * 80)
        for i, (rule, result) in enumerate(report.incorporated_details, 1):
            print(f"\n{i}. {rule.asp_rule}")
            print(f"   Confidence: {rule.confidence:.2f}")
            print(f"   Source: {rule.source_type}")
            print(f"   Accuracy Impact: {result.accuracy_before:.1%} → {result.accuracy_after:.1%}")


def show_modification_history():
    """Display modification history."""
    print("=" * 80)
    print(" Modification History")
    print("=" * 80)
    print()

    incorporation_engine = RuleIncorporationEngine()

    history = incorporation_engine.get_history()

    if not history:
        print("No modifications yet.")
        return

    print(f"Total Modifications: {len(history)}")
    print()

    for i, entry in enumerate(history[-10:], 1):  # Last 10
        print(f"{i}. {entry['rule'][:60]}...")
        print(f"   Layer: {entry['layer']}")
        print(f"   Confidence: {entry['confidence']:.2f}")
        print(f"   Accuracy: {entry['accuracy_before']:.1%} → {entry['accuracy_after']:.1%}")
        print(f"   Timestamp: {entry['timestamp']}")
        print()


def show_policies():
    """Display all modification policies."""
    print_all_policies()


def reset_session_counters():
    """Reset session modification counters."""
    print("=" * 80)
    print(" Reset Session Counters")
    print("=" * 80)
    print()

    incorporation_engine = RuleIncorporationEngine()
    incorporation_engine.reset_session()

    print("✅ Session counters reset")
    print("   All layers can now accept new modifications")


def show_statistics():
    """Display current statistics."""
    print("=" * 80)
    print(" Modification Statistics")
    print("=" * 80)
    print()

    incorporation_engine = RuleIncorporationEngine()
    stats = incorporation_engine.get_statistics()

    print(f"Total Modifications: {stats['total_modifications']}")
    print()
    print("By Layer:")
    for layer, count in stats["by_layer"].items():
        print(f"  {layer}: {count}")
    print()
    print(f"Current Accuracy: {stats['current_accuracy']:.1%}")


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LOFT Rule Modification Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run improvement session")
    run_parser.add_argument("--gaps", type=int, default=5, help="Number of gaps to address")
    run_parser.add_argument(
        "--layer",
        type=str,
        default="tactical",
        choices=["constitutional", "strategic", "tactical", "operational"],
        help="Target stratification layer",
    )
    run_parser.add_argument(
        "--candidates-per-gap",
        type=int,
        default=3,
        help="Number of candidate rules per gap",
    )

    # History command
    subparsers.add_parser("history", help="Show modification history")

    # Policies command
    subparsers.add_parser("show-policies", help="Show modification policies")

    # Reset command
    subparsers.add_parser("reset-session", help="Reset session counters")

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    # Execute command
    if args.command == "run":
        run_improvement_session(
            gaps=args.gaps,
            layer=args.layer,
            candidates_per_gap=args.candidates_per_gap,
        )

    elif args.command == "history":
        show_modification_history()

    elif args.command == "show-policies":
        show_policies()

    elif args.command == "reset-session":
        reset_session_counters()

    elif args.command == "stats":
        show_statistics()


if __name__ == "__main__":
    main()
