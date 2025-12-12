#!/usr/bin/env python3
"""
Corpus snapshot management CLI.

Provides commands to create, list, restore, compare, and delete
snapshots of the rule evolution corpus.

Usage:
    # Create a snapshot
    python scripts/corpus_snapshot.py --create my_snapshot --description "Before experiment"

    # List all snapshots
    python scripts/corpus_snapshot.py --list

    # Show snapshot details
    python scripts/corpus_snapshot.py --show my_snapshot

    # Restore a snapshot
    python scripts/corpus_snapshot.py --restore my_snapshot

    # Compare two snapshots
    python scripts/corpus_snapshot.py --compare snapshot1 snapshot2

    # Delete a snapshot
    python scripts/corpus_snapshot.py --delete my_snapshot

    # Show storage statistics
    python scripts/corpus_snapshot.py --stats
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loft.evolution import RuleEvolutionStorage, StorageConfig


def create_snapshot(storage: RuleEvolutionStorage, name: str, description: str) -> None:
    """Create a new corpus snapshot."""
    try:
        snapshot = storage.create_snapshot(name=name, description=description)
        print(f"Created snapshot: {snapshot.name}")
        print(f"  ID: {snapshot.snapshot_id}")
        print(f"  Created: {snapshot.created_at.isoformat()}")
        print(f"  Rules: {snapshot.rule_count}")
        print(f"  A/B Tests: {snapshot.ab_test_count}")
        if description:
            print(f"  Description: {description}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def list_snapshots(storage: RuleEvolutionStorage) -> None:
    """List all available snapshots."""
    snapshots = storage.load_all_snapshots()

    if not snapshots:
        print("No snapshots found.")
        return

    print(f"Found {len(snapshots)} snapshot(s):\n")
    print(f"{'Name':<25} {'Created':<20} {'Rules':>8} {'A/B Tests':>10}")
    print("-" * 70)

    for snapshot in sorted(snapshots, key=lambda s: s.created_at, reverse=True):
        created_str = snapshot.created_at.strftime("%Y-%m-%d %H:%M")
        print(
            f"{snapshot.name:<25} {created_str:<20} "
            f"{snapshot.rule_count:>8} {snapshot.ab_test_count:>10}"
        )


def show_snapshot(storage: RuleEvolutionStorage, name: str) -> None:
    """Show detailed information about a snapshot."""
    snapshot = storage.get_snapshot(name)

    if not snapshot:
        print(f"Error: Snapshot '{name}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Snapshot: {snapshot.name}")
    print(f"  ID: {snapshot.snapshot_id}")
    print(f"  Created: {snapshot.created_at.isoformat()}")
    print(f"  Description: {snapshot.description or '(none)'}")
    print(f"  Rules: {snapshot.rule_count}")
    print(f"  A/B Tests: {snapshot.ab_test_count}")

    if snapshot.metadata:
        print(f"  Metadata: {json.dumps(snapshot.metadata, indent=4)}")


def restore_snapshot(storage: RuleEvolutionStorage, name: str, no_clear: bool = False) -> None:
    """Restore corpus from a snapshot."""
    try:
        snapshot = storage.restore_snapshot(name=name, clear_existing=not no_clear)
        print(f"Restored snapshot: {snapshot.name}")
        print(f"  Rules restored: {snapshot.rule_count}")
        print(f"  A/B Tests restored: {snapshot.ab_test_count}")
        if not no_clear:
            print("  (Existing rules were cleared)")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def compare_snapshots(storage: RuleEvolutionStorage, name1: str, name2: str) -> None:
    """Compare two snapshots."""
    try:
        comparison = storage.compare_snapshots(name1, name2)

        print(f"Comparing snapshots: {name1} -> {name2}\n")

        print("Snapshot 1:")
        s1 = comparison["snapshot1"]
        print(f"  Name: {s1['name']}")
        print(f"  Created: {s1['created_at']}")
        print(f"  Rules: {s1['rule_count']}")
        print(f"  A/B Tests: {s1['ab_test_count']}")

        print("\nSnapshot 2:")
        s2 = comparison["snapshot2"]
        print(f"  Name: {s2['name']}")
        print(f"  Created: {s2['created_at']}")
        print(f"  Rules: {s2['rule_count']}")
        print(f"  A/B Tests: {s2['ab_test_count']}")

        print("\nChanges:")
        print(f"  Rule count delta: {comparison['rule_count_delta']:+d}")
        print(f"  A/B test count delta: {comparison['ab_test_count_delta']:+d}")
        print(f"  Common rules: {comparison['common_rules']}")

        if comparison["added_rules"]:
            print(f"\n  Added rules ({len(comparison['added_rules'])}):")
            for rule_id in comparison["added_rules"][:10]:
                print(f"    + {rule_id}")
            if len(comparison["added_rules"]) > 10:
                print(f"    ... and {len(comparison['added_rules']) - 10} more")

        if comparison["removed_rules"]:
            print(f"\n  Removed rules ({len(comparison['removed_rules'])}):")
            for rule_id in comparison["removed_rules"][:10]:
                print(f"    - {rule_id}")
            if len(comparison["removed_rules"]) > 10:
                print(f"    ... and {len(comparison['removed_rules']) - 10} more")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def delete_snapshot(storage: RuleEvolutionStorage, name: str) -> None:
    """Delete a snapshot."""
    if storage.delete_snapshot(name):
        print(f"Deleted snapshot: {name}")
    else:
        print(f"Error: Snapshot '{name}' not found.", file=sys.stderr)
        sys.exit(1)


def show_stats(storage: RuleEvolutionStorage) -> None:
    """Show storage statistics."""
    stats = storage.get_storage_stats()

    print("Storage Statistics:")
    print(f"  Total rules: {stats['total_rules']}")
    print(f"  Total A/B tests: {stats['total_ab_tests']}")
    print(f"  Active A/B tests: {stats['active_ab_tests']}")
    print(f"  Total snapshots: {stats['total_snapshots']}")

    if stats["status_distribution"]:
        print("\n  Status distribution:")
        for status, count in sorted(stats["status_distribution"].items()):
            print(f"    {status}: {count}")

    if stats["layer_distribution"]:
        print("\n  Layer distribution:")
        for layer, count in sorted(stats["layer_distribution"].items()):
            print(f"    {layer}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Corpus snapshot management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("data/rule_evolution"),
        help="Base path for rule evolution storage (default: data/rule_evolution)",
    )

    # Mutually exclusive operations
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--create",
        metavar="NAME",
        help="Create a new snapshot with the given name",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List all available snapshots",
    )
    group.add_argument(
        "--show",
        metavar="NAME",
        help="Show details of a snapshot",
    )
    group.add_argument(
        "--restore",
        metavar="NAME",
        help="Restore corpus from a snapshot",
    )
    group.add_argument(
        "--compare",
        nargs=2,
        metavar=("NAME1", "NAME2"),
        help="Compare two snapshots",
    )
    group.add_argument(
        "--delete",
        metavar="NAME",
        help="Delete a snapshot",
    )
    group.add_argument(
        "--stats",
        action="store_true",
        help="Show storage statistics",
    )

    # Additional options
    parser.add_argument(
        "--description",
        "-d",
        default="",
        help="Description for the snapshot (used with --create)",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing rules when restoring (merge mode)",
    )

    args = parser.parse_args()

    # Initialize storage
    config = StorageConfig(base_path=args.base_path)
    storage = RuleEvolutionStorage(config)

    # Execute command
    if args.create:
        create_snapshot(storage, args.create, args.description)
    elif args.list:
        list_snapshots(storage)
    elif args.show:
        show_snapshot(storage, args.show)
    elif args.restore:
        restore_snapshot(storage, args.restore, args.no_clear)
    elif args.compare:
        compare_snapshots(storage, args.compare[0], args.compare[1])
    elif args.delete:
        delete_snapshot(storage, args.delete)
    elif args.stats:
        show_stats(storage)


if __name__ == "__main__":
    main()
