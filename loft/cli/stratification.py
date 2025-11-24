"""
CLI interface for stratification system introspection.

Provides commands to:
- View rules by layer
- Check stratification integrity
- Show modification stats
- Display policies
"""

import sys
from typing import Optional

from loguru import logger

from loft.symbolic.stratification import (
    StratificationLevel,
    get_policy,
    print_all_policies,
)
from loft.symbolic.stratified_core import StratifiedASPCore
from loft.symbolic.stratification_validator import StratificationValidator


def view_rules_by_layer(layer: Optional[str] = None):
    """
    View rules organized by stratification layer.

    Args:
        layer: Specific layer to view (None = all layers)
    """
    print("=" * 80)
    print(" Rules by Stratification Layer")
    print("=" * 80)
    print()

    # Initialize core (would load from actual system)
    core = StratifiedASPCore()

    if layer:
        # Show specific layer
        try:
            level = StratificationLevel(layer.lower())
        except ValueError:
            print(f"❌ Invalid layer: {layer}")
            print(f"   Valid layers: {', '.join(lvl.value for lvl in StratificationLevel)}")
            return

        rules = core.get_rules_by_layer(level)
        policy = get_policy(level)

        print(f"{level.value.upper()} Layer:")
        print(f"  Policy: {policy.description}")
        print(f"  Rules: {len(rules)}")
        print()

        if rules:
            for i, rule in enumerate(rules, 1):
                print(f"{i}. {rule.rule_text}")
                print(f"   Confidence: {rule.confidence:.2f}")
                print()
        else:
            print("  (no rules)")

    else:
        # Show all layers
        for level in StratificationLevel:
            rules = core.get_rules_by_layer(level)
            print(f"{level.value.upper()}: {len(rules)} rules")

        print()
        print("Use --layer <name> to see details for a specific layer")


def check_integrity():
    """Check stratification integrity and display violations."""
    print("=" * 80)
    print(" Stratification Integrity Check")
    print("=" * 80)
    print()

    # Initialize core and validator
    core = StratifiedASPCore()
    validator = StratificationValidator()

    print("Running integrity validation...")
    print()

    report = validator.validate_core(core)

    if report.valid:
        print("✅ All checks passed - stratification integrity maintained")
        print()
    else:
        print(f"❌ Found {len(report.violations)} violations:")
        print()

        # Group by severity
        by_severity = {"critical": [], "high": [], "medium": [], "low": []}
        for violation in report.violations:
            by_severity[violation.severity].append(violation)

        for severity in ["critical", "high", "medium", "low"]:
            violations = by_severity[severity]
            if violations:
                print(f"{severity.upper()}: {len(violations)} violations")
                for v in violations:
                    layer_str = f"[{v.layer.value}]" if v.layer else "[global]"
                    print(f"  {layer_str} {v.violation_type}: {v.description}")
                print()

    if report.cycles_detected:
        print(f"⚠️  Detected {len(report.cycles_detected)} circular dependencies:")
        for i, cycle in enumerate(report.cycles_detected, 1):
            print(f"  {i}. {' → '.join(cycle)}")
        print()

    # Show stats
    print("Modification Statistics:")
    print("-" * 80)
    for layer_key, stats in report.stats.items():
        print(f"{layer_key.upper()}:")
        print(f"  Total modifications: {stats.total_modifications}")
        print(f"  Current rules: {stats.rules_current}")
        if stats.last_modification:
            print(f"  Last modified: {stats.last_modification.strftime('%Y-%m-%d %H:%M:%S')}")
        if stats.cooldown_remaining_hours:
            print(f"  Cooldown remaining: {stats.cooldown_remaining_hours:.1f} hours")
        print()


def show_modification_stats():
    """Display modification statistics by layer."""
    print("=" * 80)
    print(" Modification Statistics")
    print("=" * 80)
    print()

    core = StratifiedASPCore()
    stats = core.get_modification_stats()

    total_mods = sum(s.total_modifications for s in stats.values())
    total_rules = sum(s.rules_current for s in stats.values())

    print(f"Total Modifications: {total_mods}")
    print(f"Total Rules: {total_rules}")
    print()

    print("By Layer:")
    print("-" * 80)

    for level in StratificationLevel:
        layer_stats = stats[level.value]
        policy = get_policy(level)

        print(f"\n{level.value.upper()}:")
        print(f"  Modifications: {layer_stats.total_modifications}")
        print(f"  Current Rules: {layer_stats.rules_current}")
        print(f"  Confidence Threshold: {policy.confidence_threshold:.2f}")
        print(f"  Cooldown: {policy.modification_cooldown_hours}h")

        if layer_stats.last_modification:
            print(f"  Last Modified: {layer_stats.last_modification.strftime('%Y-%m-%d %H:%M:%S')}")

        if layer_stats.cooldown_remaining_hours:
            print(f"  ⏳ Cooldown Remaining: {layer_stats.cooldown_remaining_hours:.1f}h")


def show_policies():
    """Display all modification policies."""
    print_all_policies()


def show_dependency_graph():
    """Display predicate dependency graph."""
    print("=" * 80)
    print(" Predicate Dependency Graph")
    print("=" * 80)
    print()

    core = StratifiedASPCore()

    # Build dependency info
    predicate_info = {}

    for level in StratificationLevel:
        rules = core.get_rules_by_layer(level)

        for rule in rules:
            for pred in rule.new_predicates:
                deps = [p for p in rule.predicates_used if p != pred]
                predicate_info[pred] = {"layer": level.value, "depends_on": deps}

    if not predicate_info:
        print("No predicates defined yet.")
        return

    # Display by layer
    for level in StratificationLevel:
        layer_preds = [p for p, info in predicate_info.items() if info["layer"] == level.value]

        if layer_preds:
            print(f"{level.value.upper()}:")
            for pred in sorted(layer_preds):
                deps = predicate_info[pred]["depends_on"]
                if deps:
                    print(f"  {pred} ← {', '.join(deps)}")
                else:
                    print(f"  {pred} (no dependencies)")
            print()


def main():
    """Main entry point for stratification CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LOFT Stratification System Introspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # View rules command
    view_parser = subparsers.add_parser("view-rules", help="View rules by layer")
    view_parser.add_argument(
        "--layer",
        type=str,
        help="Specific layer to view (constitutional, strategic, tactical, operational)",
    )

    # Integrity check command
    subparsers.add_parser("check-integrity", help="Check stratification integrity")

    # Stats command
    subparsers.add_parser("stats", help="Show modification statistics")

    # Policies command
    subparsers.add_parser("show-policies", help="Display modification policies")

    # Dependency graph command
    subparsers.add_parser("show-dependencies", help="Display predicate dependency graph")

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
    if args.command == "view-rules":
        view_rules_by_layer(args.layer)

    elif args.command == "check-integrity":
        check_integrity()

    elif args.command == "stats":
        show_modification_stats()

    elif args.command == "show-policies":
        show_policies()

    elif args.command == "show-dependencies":
        show_dependency_graph()


if __name__ == "__main__":
    main()
