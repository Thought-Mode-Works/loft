"""
Dashboard and visualization for knowledge coverage metrics.

Provides formatted reports and visualizations.

Issue #274: Knowledge Coverage Metrics
"""

import logging
from datetime import datetime, timedelta

from loft.knowledge.coverage_calculator import CoverageCalculator
from loft.knowledge.coverage_tracker import CoverageTracker
from loft.knowledge.database import KnowledgeDatabase

logger = logging.getLogger(__name__)


class CoverageDashboard:
    """
    Dashboard for displaying coverage metrics.

    Formats metrics for terminal display and generates reports.
    """

    def __init__(self, knowledge_db: KnowledgeDatabase):
        """
        Initialize dashboard.

        Args:
            knowledge_db: Knowledge database instance
        """
        self.db = knowledge_db
        self.calculator = CoverageCalculator(knowledge_db)
        self.tracker = CoverageTracker(knowledge_db)

    def display_summary(self) -> str:
        """
        Display summary of current coverage.

        Returns:
            Formatted summary string
        """
        metrics = self.calculator.calculate_metrics()

        lines = []
        lines.append("=" * 70)
        lines.append("KNOWLEDGE COVERAGE SUMMARY")
        lines.append("=" * 70)
        lines.append("")

        # Overall stats
        lines.append("Overall Statistics")
        lines.append("-" * 70)
        lines.append(f"  Total Rules:        {metrics.total_rules}")
        lines.append(f"  Active Rules:       {metrics.active_rules}")
        lines.append(f"  Archived Rules:     {metrics.archived_rules}")
        lines.append(f"  Domains Covered:    {metrics.domain_count}")
        lines.append(f"  Average Confidence: {metrics.avg_confidence:.2%}")

        if metrics.overall_accuracy is not None:
            lines.append(f"  Overall Accuracy:   {metrics.overall_accuracy:.2%}")

        lines.append("")

        # Quality metrics
        lines.append("Quality Metrics")
        lines.append("-" * 70)
        lines.append(f"  Quality Score:           {metrics.quality.quality_score:.2%}")
        lines.append(
            f"  High Confidence Rules:   {metrics.quality.high_confidence_rules} "
            f"({metrics.quality.high_confidence_rules / max(metrics.total_rules, 1):.1%})"
        )
        lines.append(
            f"  Medium Confidence Rules: {metrics.quality.medium_confidence_rules} "
            f"({metrics.quality.medium_confidence_rules / max(metrics.total_rules, 1):.1%})"
        )
        lines.append(
            f"  Low Confidence Rules:    {metrics.quality.low_confidence_rules} "
            f"({metrics.quality.low_confidence_rules / max(metrics.total_rules, 1):.1%})"
        )
        lines.append(
            f"  Rules with Reasoning:    {metrics.quality.rules_with_reasoning} "
            f"({metrics.quality.rules_with_reasoning / max(metrics.total_rules, 1):.1%})"
        )
        lines.append("")

        # Top domains
        if metrics.domains:
            lines.append("Top Domains by Coverage")
            lines.append("-" * 70)

            sorted_domains = sorted(
                metrics.domains.items(),
                key=lambda x: x[1].coverage_score,
                reverse=True,
            )

            for domain_name, domain in sorted_domains[:5]:
                acc_str = (
                    f"{domain.accuracy:.1%}" if domain.accuracy is not None else "N/A"
                )
                lines.append(
                    f"  {domain_name:20s}  "
                    f"Rules: {domain.rule_count:4d}  "
                    f"Accuracy: {acc_str:6s}  "
                    f"Coverage: {domain.coverage_score:.1%}"
                )

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def display_domain_details(self, domain: str) -> str:
        """
        Display detailed metrics for a specific domain.

        Args:
            domain: Domain name

        Returns:
            Formatted details string
        """
        metrics = self.calculator.calculate_metrics()

        if domain not in metrics.domains:
            return f"Domain '{domain}' not found in knowledge base."

        domain_metrics = metrics.domains[domain]

        lines = []
        lines.append("=" * 70)
        lines.append(f"DOMAIN: {domain.upper()}")
        lines.append("=" * 70)
        lines.append("")

        # Basic stats
        lines.append("Statistics")
        lines.append("-" * 70)
        lines.append(f"  Total Rules:       {domain_metrics.rule_count}")
        lines.append(f"  Active Rules:      {domain_metrics.active_rule_count}")
        lines.append(f"  Archived Rules:    {domain_metrics.archived_rule_count}")
        lines.append(f"  Average Confidence: {domain_metrics.avg_confidence:.2%}")

        if domain_metrics.accuracy is not None:
            lines.append(f"  Accuracy:          {domain_metrics.accuracy:.2%}")

        lines.append(f"  Coverage Score:    {domain_metrics.coverage_score:.2%}")
        lines.append("")

        # Questions
        lines.append("Questions")
        lines.append("-" * 70)
        lines.append(f"  Total Questions:   {domain_metrics.question_count}")
        lines.append(f"  Answered:          {domain_metrics.answered_question_count}")
        lines.append("")

        # Doctrines
        if domain_metrics.doctrines:
            lines.append("Doctrines Covered")
            lines.append("-" * 70)
            for doctrine in sorted(domain_metrics.doctrines):
                # Get doctrine metrics if available
                key = f"{domain}:{doctrine}"
                if key in metrics.doctrines:
                    doc_metrics = metrics.doctrines[key]
                    lines.append(f"  {doctrine:30s}  {doc_metrics.rule_count} rules")
                else:
                    lines.append(f"  {doctrine}")
            lines.append("")

        # Jurisdictions
        if domain_metrics.jurisdictions:
            lines.append("Jurisdictions")
            lines.append("-" * 70)
            for jurisdiction in sorted(domain_metrics.jurisdictions):
                lines.append(f"  {jurisdiction}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def display_trends(self, metric_name: str = "total_rules", days: int = 30) -> str:
        """
        Display trend visualization for a metric.

        Args:
            metric_name: Name of metric
            days: Number of days to display

        Returns:
            Formatted trend string
        """
        trend = self.tracker.get_trend(metric_name, days=days)

        if not trend.values:
            return f"No trend data available for '{metric_name}' in last {days} days."

        lines = []
        lines.append("=" * 70)
        lines.append(f"TREND: {metric_name.upper()} (Last {days} days)")
        lines.append("=" * 70)
        lines.append("")

        # Summary stats
        lines.append(f"  Samples:        {len(trend.values)}")
        lines.append(f"  Latest Value:   {trend.latest_value:.2f}")
        lines.append(f"  Trend:          {trend.trend_direction.upper()}")
        lines.append("")

        # Simple ASCII visualization
        if len(trend.values) >= 2:
            lines.append("Timeline")
            lines.append("-" * 70)

            # Normalize values to 0-50 range for display
            min_val = min(trend.values)
            max_val = max(trend.values)
            value_range = max_val - min_val if max_val > min_val else 1

            for i, (timestamp, value) in enumerate(
                zip(trend.timestamps[-10:], trend.values[-10:])
            ):
                # Scale to 50 chars
                scaled = int(((value - min_val) / value_range) * 50)

                bar = "#" * scaled
                date_str = timestamp.strftime("%Y-%m-%d")

                lines.append(f"  {date_str}  {bar} {value:.2f}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def display_quality_report(self) -> str:
        """
        Display comprehensive quality report.

        Returns:
            Formatted quality report string
        """
        metrics = self.calculator.calculate_metrics()

        lines = []
        lines.append("=" * 70)
        lines.append("QUALITY REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Overall quality
        lines.append("Overall Quality")
        lines.append("-" * 70)
        lines.append(f"  Quality Score:     {metrics.quality.quality_score:.2%}")
        lines.append("")

        # Confidence distribution
        lines.append("Confidence Distribution")
        lines.append("-" * 70)

        total = metrics.quality.total_rules
        if total > 0:
            high_pct = metrics.quality.high_confidence_rules / total
            medium_pct = metrics.quality.medium_confidence_rules / total
            low_pct = metrics.quality.low_confidence_rules / total

            lines.append(
                f"  High (≥0.9):    {metrics.quality.high_confidence_rules:4d}  "
                f"[{'#' * int(high_pct * 30):30s}] {high_pct:.1%}"
            )
            lines.append(
                f"  Medium (0.7-0.9): {metrics.quality.medium_confidence_rules:4d}  "
                f"[{'#' * int(medium_pct * 30):30s}] {medium_pct:.1%}"
            )
            lines.append(
                f"  Low (<0.7):     {metrics.quality.low_confidence_rules:4d}  "
                f"[{'#' * int(low_pct * 30):30s}] {low_pct:.1%}"
            )
        lines.append("")

        # Metadata completeness
        lines.append("Metadata Completeness")
        lines.append("-" * 70)

        if total > 0:
            reasoning_pct = metrics.quality.rules_with_reasoning / total
            sources_pct = metrics.quality.rules_with_sources / total
            validated_pct = metrics.quality.validated_rules / total

            lines.append(
                f"  With Reasoning: {metrics.quality.rules_with_reasoning:4d}  "
                f"[{'#' * int(reasoning_pct * 30):30s}] {reasoning_pct:.1%}"
            )
            lines.append(
                f"  With Sources:   {metrics.quality.rules_with_sources:4d}  "
                f"[{'#' * int(sources_pct * 30):30s}] {sources_pct:.1%}"
            )
            lines.append(
                f"  Validated:      {metrics.quality.validated_rules:4d}  "
                f"[{'#' * int(validated_pct * 30):30s}] {validated_pct:.1%}"
            )

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def display_gaps(self) -> str:
        """
        Display identified coverage gaps.

        Returns:
            Formatted gaps report string
        """
        metrics = self.calculator.calculate_metrics()
        gaps = self.calculator.identify_gaps(metrics)

        if not gaps:
            return "No significant coverage gaps identified."

        # Sort by severity
        gaps = sorted(gaps, key=lambda g: g.severity, reverse=True)

        lines = []
        lines.append("=" * 70)
        lines.append("COVERAGE GAPS")
        lines.append("=" * 70)
        lines.append("")

        for i, gap in enumerate(gaps, 1):
            lines.append(f"Gap #{i}: {gap.area} - {gap.gap_type}")
            lines.append("-" * 70)
            lines.append(f"  Severity:    {gap.severity:.2%}")
            lines.append(f"  Description: {gap.description}")
            lines.append(f"  Action:      {gap.suggested_action}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def generate_full_report(self) -> str:
        """
        Generate comprehensive coverage report.

        Returns:
            Full formatted report string
        """
        report = self.calculator.generate_report()

        return report.to_markdown()

    def compare_periods(self, days_ago: int = 7, comparison_days: int = 7) -> str:
        """
        Compare metrics between two time periods.

        Args:
            days_ago: How many days ago to compare from
            comparison_days: Duration of comparison period

        Returns:
            Formatted comparison string
        """
        now = datetime.utcnow()
        before = now - timedelta(days=days_ago + comparison_days)
        after = now - timedelta(days=days_ago)

        comparison = self.tracker.compare_snapshots(before, after)

        if not comparison:
            return "Insufficient historical data for comparison."

        lines = []
        lines.append("=" * 70)
        lines.append("PERIOD COMPARISON")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Before: {comparison.get('before_timestamp', 'N/A')}")
        lines.append(f"After:  {comparison.get('after_timestamp', 'N/A')}")
        lines.append("")

        changes = comparison.get("changes", {})

        for metric, change in changes.items():
            lines.append(f"{metric}:")
            lines.append(f"  Before: {change.get('before', 0)}")
            lines.append(f"  After:  {change.get('after', 0)}")
            lines.append(f"  Change: {change.get('delta', 0):+}")

            if "percent_change" in change:
                lines.append(f"  % Change: {change['percent_change']:+.1f}%")

            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)
