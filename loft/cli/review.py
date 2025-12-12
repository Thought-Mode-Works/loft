"""
CLI interface for human-in-the-loop rule review.

Provides interactive command-line interface for reviewing rules,
making decisions, and managing the review queue.
"""

import sys
from pathlib import Path
from typing import Optional


from loft.validation.review_queue import ReviewQueue
from loft.validation.review_schemas import ReviewItem


class ReviewCLI:
    """
    Command-line interface for rule review.

    Provides interactive review session with clear presentation of rules,
    validation results, and decision collection.
    """

    def __init__(self, review_queue: Optional[ReviewQueue] = None):
        """
        Initialize review CLI.

        Args:
            review_queue: ReviewQueue instance (creates default if None)
        """
        self.review_queue = review_queue or ReviewQueue()

    def run(self, reviewer_id: str) -> None:
        """
        Run interactive review session.

        Args:
            reviewer_id: ID of the person reviewing

        Example:
            >>> cli = ReviewCLI()
            >>> cli.run(reviewer_id="alice")
        """
        print(f"\n{'=' * 80}")
        print("  LOFT Rule Review Session")
        print(f"  Reviewer: {reviewer_id}")
        print(f"{'=' * 80}\n")

        stats = self.review_queue.get_statistics()
        print(stats.summary())
        print()

        if stats.pending == 0:
            print("✓ No items in review queue!")
            return

        reviewed_count = 0
        while True:
            item = self.review_queue.get_next(reviewer_id)

            if not item:
                print("\n✓ No more pending items in queue.")
                break

            decision = self._review_item(item)

            if decision == "skip":
                # Return item to pending state
                item.status = "pending"
                item.reviewer_id = None
                item.review_started_at = None
                self.review_queue.storage.store(item.id, item.model_dump(mode="json"))
                print("\n⏭  Skipped, returned to queue\n")
                continue

            elif decision == "quit":
                print("\n👋 Exiting review session\n")
                break

            reviewed_count += 1

        if reviewed_count > 0:
            print(f"\n✓ Reviewed {reviewed_count} item(s) this session")
            updated_stats = self.review_queue.get_statistics()
            print(f"  Remaining pending: {updated_stats.pending}")

    def _review_item(self, item: ReviewItem) -> str:
        """
        Present item for review and collect decision.

        Args:
            item: ReviewItem to review

        Returns:
            Decision action ("accept", "reject", "revise", "skip", "quit")
        """
        print("\n" + "=" * 80)
        print(f"REVIEW ITEM: {item.id}")
        print("=" * 80)
        print(f"\n📋 Priority: {item.priority.upper()}")
        print(f"🔍 Reason: {item.reason}")
        print(f"⏰ Queued: {item.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n" + "-" * 80)
        print("GENERATED RULE")
        print("-" * 80)
        print("\nASP Rule:")
        print(f"  {item.rule.asp_rule}")
        print(f"\nSource: {item.rule.source_type}")
        print(f"Confidence: {item.rule.confidence:.2f}")
        print("\nReasoning:")
        print(f"  {item.rule.reasoning}")

        if item.rule.predicates_used:
            print(f"\nPredicates Used: {', '.join(item.rule.predicates_used)}")
        if item.rule.new_predicates:
            print(f"New Predicates: {', '.join(item.rule.new_predicates)}")

        print("\n" + "-" * 80)
        print("VALIDATION REPORT")
        print("-" * 80)
        print(f"\nFinal Decision: {item.validation_report.final_decision}")

        # Show validation stage results
        for stage_name, result in item.validation_report.stage_results.items():
            print(f"\n{stage_name.title()}:")
            if hasattr(result, "is_valid"):
                print(f"  Valid: {result.is_valid}")
            if hasattr(result, "accuracy"):
                print(f"  Accuracy: {result.accuracy:.2f}")
            if hasattr(result, "error_messages") and result.error_messages:
                print(f"  Errors: {'; '.join(result.error_messages[:3])}")

        # Show empirical failures if available
        if "empirical" in item.validation_report.stage_results:
            empirical = item.validation_report.stage_results["empirical"]
            if hasattr(empirical, "failures") and empirical.failures:
                print(f"\n⚠️  Failed Test Cases ({len(empirical.failures)}):")
                for i, failure in enumerate(empirical.failures[:3], 1):
                    print(f"\n  {i}. {failure.test_case.description}")
                    print(f"     Expected: {failure.expected}")
                    print(f"     Got: {failure.actual}")
                if len(empirical.failures) > 3:
                    print(f"\n  ... and {len(empirical.failures) - 3} more")

        # Get decision
        print("\n" + "=" * 80)
        print("YOUR DECISION")
        print("=" * 80)
        print("\n  1️⃣  Accept    - Add rule to knowledge base")
        print("  2️⃣  Reject    - Discard rule")
        print("  3️⃣  Revise    - Suggest an improvement")
        print("  4️⃣  Skip      - Review later")
        print("  5️⃣  Quit      - Exit review session")

        while True:
            choice = input("\nEnter choice (1-5): ").strip()

            if choice == "1":
                notes = input("Reviewer notes (optional): ").strip() or "Approved"
                self.review_queue.submit_review(item.id, "accept", notes)
                print("\n✅ Rule ACCEPTED")
                return "accept"

            elif choice == "2":
                notes = input("Rejection reason: ").strip()
                if not notes:
                    print("❌ Rejection reason required")
                    continue
                self.review_queue.submit_review(item.id, "reject", notes)
                print("\n❌ Rule REJECTED")
                return "reject"

            elif choice == "3":
                revision = input("Suggested revision (ASP rule): ").strip()
                if not revision:
                    print("❌ Revision required")
                    continue
                notes = input("Revision notes: ").strip() or "See suggested revision"
                self.review_queue.submit_review(
                    item.id, "revise", notes, suggested_revision=revision
                )
                print("\n🔄 Revision SUGGESTED")
                return "revise"

            elif choice == "4":
                return "skip"

            elif choice == "5":
                return "quit"

            else:
                print("❌ Invalid choice, please enter 1-5")

    def show_stats(self) -> None:
        """Display current queue statistics."""
        stats = self.review_queue.get_statistics()
        print("\n" + stats.summary())

    def show_pending(self, priority: Optional[str] = None) -> None:
        """
        Show pending items.

        Args:
            priority: Filter by priority (critical/high/medium/low)
        """
        if priority:
            items = self.review_queue.get_by_priority(priority)
            items = [i for i in items if i.status == "pending"]
            print(f"\nPending {priority.upper()} priority items:")
        else:
            items = self.review_queue.get_pending()
            print("\nAll pending items:")

        if not items:
            print("  None")
            return

        print(f"\n{'ID':<20} {'Priority':<12} {'Reason':<40} {'Age'}")
        print("-" * 80)

        from datetime import datetime

        for item in items:
            age_hours = (datetime.now() - item.created_at).total_seconds() / 3600
            age_str = (
                f"{age_hours:.1f}h" if age_hours < 24 else f"{age_hours / 24:.1f}d"
            )

            reason_short = (
                item.reason[:37] + "..." if len(item.reason) > 40 else item.reason
            )

            print(f"{item.id:<20} {item.priority:<12} {reason_short:<40} {age_str}")

    def export_reviews(self, output_file: Path) -> None:
        """
        Export all reviews to JSON file.

        Args:
            output_file: Path to output file
        """
        self.review_queue.export_reviews(output_file)
        print(f"\n✓ Exported reviews to {output_file}")


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LOFT Rule Review Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start review session")
    start_parser.add_argument(
        "--reviewer", required=True, help="Reviewer ID (e.g., alice)"
    )

    # Stats command
    subparsers.add_parser("stats", help="Show queue statistics")

    # Pending command
    pending_parser = subparsers.add_parser("pending", help="Show pending items")
    pending_parser.add_argument(
        "--priority",
        choices=["critical", "high", "medium", "low"],
        help="Filter by priority",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export reviews")
    export_parser.add_argument("--output", required=True, help="Output file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cli = ReviewCLI()

    if args.command == "start":
        cli.run(reviewer_id=args.reviewer)

    elif args.command == "stats":
        cli.show_stats()

    elif args.command == "pending":
        cli.show_pending(priority=args.priority)

    elif args.command == "export":
        cli.export_reviews(Path(args.output))


if __name__ == "__main__":
    main()
