"""
Baseline metrics comparison framework.

Issue #257: Baseline Validation Benchmarks
"""

from dataclasses import dataclass, field
from typing import List

from loft.benchmarks.baseline_metrics import BaselineMetrics


@dataclass
class MetricChange:
    """Represents a change in a metric."""

    name: str
    previous: float
    current: float
    change_pct: float
    is_regression: bool


@dataclass
class ComparisonReport:
    """Report comparing two baselines."""

    previous_timestamp: str
    current_timestamp: str
    previous_commit: str
    current_commit: str

    # Detected changes
    regressions: List[MetricChange] = field(default_factory=list)
    improvements: List[MetricChange] = field(default_factory=list)
    stable_metrics: List[str] = field(default_factory=list)

    # Summary
    total_metrics: int = 0
    regression_count: int = 0
    improvement_count: int = 0

    def add_change(
        self, name: str, previous: float, current: float, threshold: float = 0.10
    ):
        """
        Add a metric change and classify it.

        Args:
            name: Metric name
            previous: Previous value
            current: Current value
            threshold: Regression threshold (default 10%)
        """
        self.total_metrics += 1

        if previous == 0:
            # Handle division by zero
            if current > 0:
                self.improvements.append(
                    MetricChange(name, previous, current, float("inf"), False)
                )
                self.improvement_count += 1
            else:
                self.stable_metrics.append(name)
            return

        change_pct = (current - previous) / previous

        if abs(change_pct) < 0.01:  # < 1% change considered stable
            self.stable_metrics.append(name)
        elif change_pct < -threshold:  # Significant degradation
            self.regressions.append(
                MetricChange(name, previous, current, change_pct, True)
            )
            self.regression_count += 1
        elif change_pct > threshold:  # Significant improvement
            self.improvements.append(
                MetricChange(name, previous, current, change_pct, False)
            )
            self.improvement_count += 1
        else:
            self.stable_metrics.append(name)

    def to_markdown(self) -> str:
        """Generate markdown comparison report."""
        report = f"""# Baseline Comparison Report

**Previous**: {self.previous_timestamp} (commit {self.previous_commit[:7]})
**Current**: {self.current_timestamp} (commit {self.current_commit[:7]})

## Summary

- **Total Metrics**: {self.total_metrics}
- **Regressions**: {self.regression_count} (>{10}% degradation)
- **Improvements**: {self.improvement_count} (>{10}% improvement)
- **Stable**: {len(self.stable_metrics)} (<1% change)

"""

        if self.regressions:
            report += "## Regressions Detected ⚠️\n\n"
            report += "| Metric | Previous | Current | Change |\n"
            report += "|--------|----------|---------|--------|\n"
            for reg in self.regressions:
                report += f"| {reg.name} | {reg.previous:.2f} | {reg.current:.2f} | {reg.change_pct:+.1%} |\n"
            report += "\n"

        if self.improvements:
            report += "## Improvements ✅\n\n"
            report += "| Metric | Previous | Current | Change |\n"
            report += "|--------|----------|---------|--------|\n"
            for imp in self.improvements:
                report += f"| {imp.name} | {imp.previous:.2f} | {imp.current:.2f} | {imp.change_pct:+.1%} |\n"
            report += "\n"

        return report


def compare_baselines(
    previous: BaselineMetrics, current: BaselineMetrics, threshold: float = 0.10
) -> ComparisonReport:
    """
    Compare two baseline metrics.

    Args:
        previous: Previous baseline
        current: Current baseline
        threshold: Regression detection threshold (default 10%)

    Returns:
        Comparison report
    """
    report = ComparisonReport(
        previous_timestamp=previous.timestamp,
        current_timestamp=current.timestamp,
        previous_commit=previous.commit_hash,
        current_commit=current.commit_hash,
    )

    # Rule generation comparisons
    report.add_change(
        "Rule generation - rules/hour",
        previous.rule_generation.rules_per_hour,
        current.rule_generation.rules_per_hour,
        threshold,
    )
    report.add_change(
        "Rule generation - avg time (ms)",
        previous.rule_generation.avg_generation_time_ms,
        current.rule_generation.avg_generation_time_ms,
        threshold,
    )
    report.add_change(
        "Rule generation - success rate",
        previous.rule_generation.success_rate,
        current.rule_generation.success_rate,
        threshold,
    )

    # Validation comparisons
    report.add_change(
        "Validation - overall pass rate",
        previous.validation.overall_pass_rate,
        current.validation.overall_pass_rate,
        threshold,
    )
    report.add_change(
        "Validation - avg time (ms)",
        previous.validation.avg_validation_time_ms,
        current.validation.avg_validation_time_ms,
        threshold,
    )

    # Translation comparisons
    report.add_change(
        "Translation - roundtrip fidelity",
        previous.translation.roundtrip_fidelity,
        current.translation.roundtrip_fidelity,
        threshold,
    )

    # End-to-end comparisons
    report.add_change(
        "End-to-end - cases/hour",
        previous.end_to_end.cases_per_hour,
        current.end_to_end.cases_per_hour,
        threshold,
    )
    report.add_change(
        "End-to-end - avg time (ms)",
        previous.end_to_end.avg_case_time_ms,
        current.end_to_end.avg_case_time_ms,
        threshold,
    )

    # Persistence comparisons
    report.add_change(
        "Persistence - avg save time (ms)",
        previous.persistence.avg_save_time_ms,
        current.persistence.avg_save_time_ms,
        threshold,
    )
    report.add_change(
        "Persistence - avg load time (ms)",
        previous.persistence.avg_load_time_ms,
        current.persistence.avg_load_time_ms,
        threshold,
    )

    return report
