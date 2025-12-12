"""
Reporting and analysis for consistency checking.

Provides tools for tracking consistency over time, visualizing conflicts,
and detecting regressions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path
from .checker import ConsistencyReport


@dataclass
class ConsistencyMetrics:
    """Metrics for consistency analysis."""

    timestamp: str
    total_rules: int
    passed: bool
    error_count: int
    warning_count: int
    info_count: int
    consistency_score: float  # 0.0 to 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_rules": self.total_rules,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "consistency_score": self.consistency_score,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConsistencyMetrics":
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            total_rules=data["total_rules"],
            passed=data["passed"],
            error_count=data["error_count"],
            warning_count=data["warning_count"],
            info_count=data["info_count"],
            consistency_score=data["consistency_score"],
        )


@dataclass
class ConsistencyHistory:
    """Track consistency metrics over time."""

    metrics: List[ConsistencyMetrics] = field(default_factory=list)

    def add_report(self, report: ConsistencyReport, total_rules: int) -> None:
        """
        Add a consistency report to history.

        Args:
            report: Consistency report to add
            total_rules: Total number of rules in the state
        """
        # Calculate consistency score
        # Score = 1.0 - (weighted_errors / total_rules)
        # Errors weighted more heavily than warnings
        if total_rules == 0:
            score = 1.0
        else:
            weighted_issues = (
                (report.errors * 2.0) + (report.warnings * 1.0) + (report.info * 0.5)
            )
            score = max(0.0, 1.0 - (weighted_issues / total_rules))

        metrics = ConsistencyMetrics(
            timestamp=datetime.utcnow().isoformat(),
            total_rules=total_rules,
            passed=report.passed,
            error_count=report.errors,
            warning_count=report.warnings,
            info_count=report.info,
            consistency_score=score,
        )

        self.metrics.append(metrics)

    def get_latest(self) -> Optional[ConsistencyMetrics]:
        """Get most recent metrics."""
        if not self.metrics:
            return None
        return self.metrics[-1]

    def get_trend(self, window: int = 10) -> str:
        """
        Get consistency trend over recent checks.

        Args:
            window: Number of recent checks to analyze

        Returns:
            Trend description: "improving", "stable", "declining", or "unknown"
        """
        if len(self.metrics) < 2:
            return "unknown"

        recent = self.metrics[-window:]
        if len(recent) < 2:
            return "unknown"

        # Compare average of first half vs second half
        mid = len(recent) // 2
        first_half_avg = sum(m.consistency_score for m in recent[:mid]) / mid
        second_half_avg = sum(m.consistency_score for m in recent[mid:]) / (
            len(recent) - mid
        )

        diff = second_half_avg - first_half_avg

        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"

    def detect_regression(self, threshold: float = 0.1) -> bool:
        """
        Detect if there's been a recent regression in consistency.

        Args:
            threshold: Minimum score drop to consider a regression

        Returns:
            True if regression detected
        """
        if len(self.metrics) < 2:
            return False

        latest = self.metrics[-1]
        previous = self.metrics[-2]

        drop = previous.consistency_score - latest.consistency_score
        return drop > threshold

    def save(self, path: Path) -> None:
        """
        Save history to file.

        Args:
            path: File path to save to
        """
        data = {
            "metrics": [m.to_dict() for m in self.metrics],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ConsistencyHistory":
        """
        Load history from file.

        Args:
            path: File path to load from

        Returns:
            ConsistencyHistory instance
        """
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = json.load(f)

        history = cls()
        history.metrics = [
            ConsistencyMetrics.from_dict(m) for m in data.get("metrics", [])
        ]

        return history


class ConsistencyReporter:
    """Enhanced reporting for consistency checks."""

    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize reporter.

        Args:
            history_file: Optional file to track consistency history
        """
        self.history_file = history_file
        self.history = (
            ConsistencyHistory.load(history_file)
            if history_file
            else ConsistencyHistory()
        )

    def report(
        self, report: ConsistencyReport, total_rules: int, save_history: bool = True
    ) -> str:
        """
        Generate enhanced report with historical context.

        Args:
            report: Consistency report
            total_rules: Total number of rules
            save_history: Whether to save to history file

        Returns:
            Formatted report string
        """
        lines = []

        # Add to history
        self.history.add_report(report, total_rules)

        # Save history if requested
        if save_history and self.history_file:
            self.history.save(self.history_file)

        # Basic report
        lines.append(report.format())

        # Historical context
        latest = self.history.get_latest()
        if latest:
            lines.append("Historical Context:")
            lines.append("-" * 60)
            lines.append(f"Consistency Score: {latest.consistency_score:.3f}")
            lines.append(f"Trend: {self.history.get_trend()}")

            if self.history.detect_regression():
                lines.append("⚠️  REGRESSION DETECTED!")

            lines.append("")

        return "\n".join(lines)

    def summary_by_type(self, report: ConsistencyReport) -> Dict[str, int]:
        """
        Get summary of inconsistencies by type.

        Args:
            report: Consistency report

        Returns:
            Dictionary mapping inconsistency type to count
        """
        counts: Dict[str, int] = {}

        for inconsistency in report.inconsistencies:
            inc_type = inconsistency.type.value
            counts[inc_type] = counts.get(inc_type, 0) + 1

        return counts

    def format_conflict_graph(self, report: ConsistencyReport) -> str:
        """
        Format a simple text-based visualization of rule conflicts.

        Args:
            report: Consistency report

        Returns:
            Text visualization of conflicts
        """
        lines = ["Rule Conflict Graph:", "-" * 60]

        # Group inconsistencies by involved rules
        conflict_edges: List[tuple[str, str, str]] = []

        for inc in report.inconsistencies:
            if len(inc.rule_ids) >= 2:
                # Create edges between conflicting rules
                for i in range(len(inc.rule_ids) - 1):
                    conflict_edges.append(
                        (inc.rule_ids[i], inc.rule_ids[i + 1], inc.type.value)
                    )

        if not conflict_edges:
            lines.append("No conflicts detected")
        else:
            for rule1, rule2, conflict_type in conflict_edges:
                lines.append(f"{rule1} <--[{conflict_type}]--> {rule2}")

        lines.append("")
        return "\n".join(lines)
