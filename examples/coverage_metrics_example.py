"""
Example usage of knowledge coverage metrics.

Demonstrates tracking and analyzing knowledge base coverage.

Issue #274: Knowledge Coverage Metrics

Usage:
    python examples/coverage_metrics_example.py
"""

import logging
from pathlib import Path

from loft.knowledge.coverage_calculator import CoverageCalculator
from loft.knowledge.coverage_dashboard import CoverageDashboard
from loft.knowledge.coverage_tracker import CoverageTracker
from loft.knowledge.database import KnowledgeDatabase

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def example_1_calculate_metrics():
    """
    Example 1: Calculate coverage metrics.
    """
    print("\n" + "=" * 70)
    print("Example 1: Calculate Coverage Metrics")
    print("=" * 70 + "\n")

    # Initialize database
    db = KnowledgeDatabase("sqlite:///examples/example_knowledge.db")

    # Add some sample rules if database is empty
    stats = db.get_database_stats()
    if stats.total_rules == 0:
        print("Adding sample rules...")

        db.add_rule(
            asp_rule="valid_contract(X) :- offer(X), acceptance(X), consideration(X).",
            domain="contracts",
            doctrine="contract-formation",
            confidence=0.95,
            reasoning="Basic contract requires three essential elements",
        )

        db.add_rule(
            asp_rule="enforceable(X) :- valid_contract(X), not signed_under_duress(X).",
            domain="contracts",
            confidence=0.90,
            reasoning="Contracts signed under duress are not enforceable",
        )

        db.add_rule(
            asp_rule="negligence(X) :- duty(X), breach(X), causation(X), damages(X).",
            domain="torts",
            doctrine="negligence",
            confidence=0.98,
            reasoning="Negligence requires four elements",
        )

        print("Sample rules added.\n")

    # Calculate metrics
    calculator = CoverageCalculator(db)
    metrics = calculator.calculate_metrics()

    # Display results
    print("Coverage Metrics:")
    print(f"  Total Rules:        {metrics.total_rules}")
    print(f"  Active Rules:       {metrics.active_rules}")
    print(f"  Domains Covered:    {metrics.domain_count}")
    print(f"  Average Confidence: {metrics.avg_confidence:.2%}")
    print(f"  Quality Score:      {metrics.quality.quality_score:.2%}")

    print("\nBy Domain:")
    for domain_name, domain in metrics.domains.items():
        print(f"  {domain_name}:")
        print(f"    Rules:          {domain.rule_count}")
        print(f"    Confidence:     {domain.avg_confidence:.2%}")
        print(f"    Coverage Score: {domain.coverage_score:.2%}")

    return db, calculator


def example_2_identify_gaps(calculator):
    """
    Example 2: Identify coverage gaps.
    """
    print("\n" + "=" * 70)
    print("Example 2: Identify Coverage Gaps")
    print("=" * 70 + "\n")

    metrics = calculator.calculate_metrics()
    gaps = calculator.identify_gaps(metrics)

    if gaps:
        print(f"Found {len(gaps)} coverage gaps:\n")

        for i, gap in enumerate(gaps, 1):
            print(f"Gap #{i}: {gap.area} - {gap.gap_type}")
            print(f"  Severity:    {gap.severity:.2%}")
            print(f"  Description: {gap.description}")
            print(f"  Action:      {gap.suggested_action}")
            print()
    else:
        print("No significant coverage gaps identified!")


def example_3_generate_report(calculator):
    """
    Example 3: Generate comprehensive report.
    """
    print("\n" + "=" * 70)
    print("Example 3: Generate Comprehensive Report")
    print("=" * 70 + "\n")

    report = calculator.generate_report()

    print("Report Summary:")
    print(f"  Domains:         {report.metrics.domain_count}")
    print(f"  Total Rules:     {report.metrics.total_rules}")
    print(f"  Gaps Identified: {len(report.gaps)}")
    print(f"  Recommendations: {len(report.recommendations)}")

    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")

    # Save markdown report
    report_path = Path("examples/coverage_report.md")
    with open(report_path, "w") as f:
        f.write(report.to_markdown())

    print(f"\nFull report saved to: {report_path}")


def example_4_track_trends(db):
    """
    Example 4: Track metrics over time.
    """
    print("\n" + "=" * 70)
    print("Example 4: Track Metrics Over Time")
    print("=" * 70 + "\n")

    # Initialize tracker
    tracker = CoverageTracker(db, storage_path=Path("examples/metrics_history.json"))

    print(f"Existing Snapshots: {tracker.get_snapshot_count()}")

    # Take a snapshot
    print("\nTaking snapshot...")
    metrics = tracker.take_snapshot()

    print("Snapshot taken!")
    print(f"  Total Rules: {metrics.total_rules}")
    print(f"  Domains:     {metrics.domain_count}")
    print(f"  Total Snapshots: {tracker.get_snapshot_count()}")

    # Get trends (if we have historical data)
    if tracker.get_snapshot_count() > 1:
        print("\nTrend Analysis:")

        trend = tracker.get_trend("total_rules", days=30)
        if trend.values:
            print("  Total Rules Trend:")
            print(f"    Samples:   {len(trend.values)}")
            print(f"    Latest:    {trend.latest_value}")
            print(f"    Direction: {trend.trend_direction}")


def example_5_dashboard(db):
    """
    Example 5: Display dashboard.
    """
    print("\n" + "=" * 70)
    print("Example 5: Dashboard Display")
    print("=" * 70 + "\n")

    dashboard = CoverageDashboard(db)

    # Display summary
    summary = dashboard.display_summary()
    print(summary)

    # Display quality report
    print("\n")
    quality = dashboard.display_quality_report()
    print(quality)


def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 70)
    print("Knowledge Coverage Metrics Examples")
    print("=" * 70)

    # Example 1: Calculate metrics
    db, calculator = example_1_calculate_metrics()

    # Example 2: Identify gaps
    example_2_identify_gaps(calculator)

    # Example 3: Generate report
    example_3_generate_report(calculator)

    # Example 4: Track trends
    example_4_track_trends(db)

    # Example 5: Dashboard
    example_5_dashboard(db)

    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70 + "\n")

    print("Database saved to: examples/example_knowledge.db")
    print("Metrics history saved to: examples/metrics_history.json")
    print("Report saved to: examples/coverage_report.md")
    print("\nYou can now use the CLI commands:\n")
    print("  # Display metrics summary")
    print("  python -m loft.knowledge.cli metrics-summary\n")
    print("  # Show domain details")
    print("  python -m loft.knowledge.cli metrics-domain contracts\n")
    print("  # Display quality metrics")
    print("  python -m loft.knowledge.cli metrics-quality\n")
    print("  # Show coverage gaps")
    print("  python -m loft.knowledge.cli metrics-gaps\n")
    print("  # Generate full report")
    print("  python -m loft.knowledge.cli metrics-report --output report.md\n")
    print("  # Take a snapshot")
    print("  python -m loft.knowledge.cli metrics-snapshot\n")
    print("  # View trends")
    print(
        "  python -m loft.knowledge.cli metrics-trends --metric total_rules --days 30\n"
    )


if __name__ == "__main__":
    main()
