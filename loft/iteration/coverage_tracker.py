"""
Coverage tracking for iterative rule building.

Tracks coverage expansion over iterations including:
- Predicate coverage (% of domain predicates covered)
- Case coverage (% of test cases with predictions)
- Scenario coverage (% of scenarios with applicable rules)
- Monotonicity verification
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import re


@dataclass
class CoverageMetrics:
    """Metrics for domain coverage at a point in time."""

    timestamp: datetime
    predicates_total: int  # Total predicates in domain ontology
    predicates_covered: int  # Predicates appearing in rule heads
    covered_predicates: List[str]  # List of covered predicate names

    cases_total: int  # Total test cases
    cases_with_predictions: int  # Cases where system makes prediction

    scenarios_total: int  # Total scenarios in dataset
    scenarios_covered: int  # Scenarios with applicable rules

    total_rules: int  # Total rules in system
    rules_by_layer: Dict[str, int] = field(
        default_factory=dict
    )  # Rule counts per layer

    @property
    def predicate_coverage(self) -> float:
        """Get predicate coverage percentage."""
        if self.predicates_total == 0:
            return 0.0
        return self.predicates_covered / self.predicates_total

    @property
    def case_coverage(self) -> float:
        """Get case coverage percentage."""
        if self.cases_total == 0:
            return 0.0
        return self.cases_with_predictions / self.cases_total

    @property
    def scenario_coverage(self) -> float:
        """Get scenario coverage percentage."""
        if self.scenarios_total == 0:
            return 0.0
        return self.scenarios_covered / self.scenarios_total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "predicates_total": self.predicates_total,
            "predicates_covered": self.predicates_covered,
            "covered_predicates": self.covered_predicates,
            "cases_total": self.cases_total,
            "cases_with_predictions": self.cases_with_predictions,
            "scenarios_total": self.scenarios_total,
            "scenarios_covered": self.scenarios_covered,
            "total_rules": self.total_rules,
            "rules_by_layer": self.rules_by_layer,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoverageMetrics":
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class CoverageTracker:
    """
    Tracks coverage expansion over iterations.

    Maintains history of coverage metrics and provides analysis
    of coverage progression and monotonicity.
    """

    def __init__(self, domain_predicates: List[str], test_cases: List[Dict[str, Any]]):
        """
        Initialize coverage tracker.

        Args:
            domain_predicates: Complete list of predicates in domain ontology
            test_cases: Test cases for coverage measurement
        """
        self.domain_predicates = set(domain_predicates)
        self.test_cases = test_cases

        # Coverage history over time
        self.history: List[CoverageMetrics] = []

        # Track newly covered predicates since last update
        self.newly_covered_predicates: List[str] = []

    def record_metrics(
        self,
        covered_predicates: Set[str],
        cases_with_predictions: int,
        scenarios_covered: int,
        total_rules: int,
        rules_by_layer: Optional[Dict[str, int]] = None,
    ) -> CoverageMetrics:
        """
        Record coverage metrics at current point in time.

        Args:
            covered_predicates: Set of predicates currently covered
            cases_with_predictions: Number of cases with predictions
            scenarios_covered: Number of scenarios covered
            total_rules: Total number of rules
            rules_by_layer: Optional rule counts per layer

        Returns:
            CoverageMetrics object for this snapshot
        """
        # Calculate newly covered predicates
        if self.history:
            previous_covered = set(self.history[-1].covered_predicates)
            self.newly_covered_predicates = list(covered_predicates - previous_covered)
        else:
            self.newly_covered_predicates = list(covered_predicates)

        # Create metrics
        metrics = CoverageMetrics(
            timestamp=datetime.utcnow(),
            predicates_total=len(self.domain_predicates),
            predicates_covered=len(covered_predicates),
            covered_predicates=sorted(list(covered_predicates)),
            cases_total=len(self.test_cases),
            cases_with_predictions=cases_with_predictions,
            scenarios_total=self._count_unique_scenarios(),
            scenarios_covered=scenarios_covered,
            total_rules=total_rules,
            rules_by_layer=rules_by_layer or {},
        )

        self.history.append(metrics)
        return metrics

    def _count_unique_scenarios(self) -> int:
        """Count unique scenarios in test cases."""
        scenarios = set()
        for case in self.test_cases:
            scenario_id = case.get("scenario_id") or case.get("id", "default")
            scenarios.add(scenario_id)
        return len(scenarios)

    @property
    def current_metrics(self) -> Optional[CoverageMetrics]:
        """Get most recent coverage metrics."""
        if not self.history:
            return None
        return self.history[-1]

    @property
    def current_coverage(self) -> float:
        """Get current predicate coverage percentage."""
        if not self.history:
            return 0.0
        return self.history[-1].predicate_coverage

    def is_monotonic(self, tolerance: float = 0.0) -> bool:
        """
        Check if coverage is monotonically increasing.

        Args:
            tolerance: Allow small decreases within tolerance

        Returns:
            True if coverage never decreases beyond tolerance
        """
        if len(self.history) < 2:
            return True

        for i in range(1, len(self.history)):
            prev_coverage = self.history[i - 1].predicate_coverage
            curr_coverage = self.history[i].predicate_coverage

            # Check if current is less than previous (allowing tolerance)
            if curr_coverage < prev_coverage - tolerance:
                return False

        return True

    def get_coverage_trend(self, window: int = 5) -> str:
        """
        Get coverage trend over recent history.

        Args:
            window: Number of recent snapshots to analyze

        Returns:
            "increasing", "decreasing", "stable", or "insufficient_data"
        """
        if len(self.history) < 2:
            return "insufficient_data"

        # Get recent metrics
        recent = self.history[-min(window, len(self.history)) :]

        if len(recent) < 2:
            return "insufficient_data"

        # Calculate average change
        total_change = 0.0
        for i in range(1, len(recent)):
            change = recent[i].predicate_coverage - recent[i - 1].predicate_coverage
            total_change += change

        avg_change = total_change / (len(recent) - 1)

        if avg_change > 0.01:  # Significant increase
            return "increasing"
        elif avg_change < -0.01:  # Significant decrease
            return "decreasing"
        else:
            return "stable"

    def get_uncovered_predicates(self) -> List[str]:
        """Get list of predicates not yet covered by any rules."""
        if not self.history:
            return sorted(list(self.domain_predicates))

        covered = set(self.history[-1].covered_predicates)
        uncovered = self.domain_predicates - covered
        return sorted(list(uncovered))

    def extract_predicates_from_rules(self, rules: List[Any]) -> Set[str]:
        """
        Extract predicates from rule heads.

        Args:
            rules: List of ASPRule objects

        Returns:
            Set of predicate names found in rule heads
        """
        predicates = set()

        for rule in rules:
            # Extract from rule head (new_predicates field)
            if hasattr(rule, "new_predicates"):
                predicates.update(rule.new_predicates)
            elif hasattr(rule, "asp_text"):
                # Fallback: extract from ASP text
                head_predicates = self._extract_head_predicates(rule.asp_text)
                predicates.update(head_predicates)

        return predicates

    def _extract_head_predicates(self, asp_text: str) -> Set[str]:
        """
        Extract predicate names from ASP rule head.

        Args:
            asp_text: ASP rule text

        Returns:
            Set of predicate names
        """
        # Split on :- to get head
        if ":-" in asp_text:
            head = asp_text.split(":-", 1)[0].strip()
        else:
            # Fact (entire text is head)
            head = asp_text.strip()

        # Extract predicate names (lowercase identifiers)
        pattern = r"\b([a-z][a-z0-9_]*)(?:\(|$)"
        matches = re.findall(pattern, head)

        # Filter out keywords
        keywords = {"not", "and", "or"}
        predicates = {m for m in matches if m not in keywords}

        return predicates

    def generate_coverage_report(self) -> str:
        """
        Generate human-readable coverage report.

        Returns:
            Markdown-formatted coverage report
        """
        if not self.history:
            return "# Coverage Report\n\nNo coverage data available.\n"

        current = self.history[-1]

        report = f"""# Coverage Report

## Current Coverage (as of {current.timestamp.strftime('%Y-%m-%d %H:%M:%S')})

| Metric | Coverage | Count |
|--------|----------|-------|
| Predicates | {current.predicate_coverage:.1%} | {current.predicates_covered}/{current.predicates_total} |
| Cases | {current.case_coverage:.1%} | {current.cases_with_predictions}/{current.cases_total} |
| Scenarios | {current.scenario_coverage:.1%} | {current.scenarios_covered}/{current.scenarios_total} |
| Total Rules | - | {current.total_rules} |

## Rules by Layer

"""

        for layer, count in sorted(current.rules_by_layer.items()):
            report += f"- **{layer}**: {count} rules\n"

        report += "\n## Coverage Trend\n\n"
        trend = self.get_coverage_trend()
        report += f"Recent trend: **{trend}**\n\n"

        report += f"Monotonicity: **{'✓ Maintained' if self.is_monotonic() else '✗ Violated'}**\n\n"

        # Uncovered predicates
        uncovered = self.get_uncovered_predicates()
        report += f"## Uncovered Predicates ({len(uncovered)})\n\n"

        if uncovered:
            for pred in uncovered[:20]:  # Show first 20
                report += f"- `{pred}`\n"

            if len(uncovered) > 20:
                report += f"\n... and {len(uncovered) - 20} more\n"
        else:
            report += "All domain predicates are covered!\n"

        # Coverage history
        if len(self.history) > 1:
            report += f"\n## Coverage History ({len(self.history)} snapshots)\n\n"
            report += "| Timestamp | Predicates | Cases | Rules |\n"
            report += "|-----------|------------|-------|-------|\n"

            for metrics in self.history[-10:]:  # Show last 10
                timestamp = metrics.timestamp.strftime("%m-%d %H:%M")
                report += f"| {timestamp} | {metrics.predicate_coverage:.1%} | {metrics.case_coverage:.1%} | {metrics.total_rules} |\n"

        return report

    def save_history(self, filepath: str) -> None:
        """
        Save coverage history to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        import json
        from pathlib import Path

        data = {
            "domain_predicates": sorted(list(self.domain_predicates)),
            "test_cases_count": len(self.test_cases),
            "history": [m.to_dict() for m in self.history],
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_history(
        cls, filepath: str, test_cases: List[Dict[str, Any]]
    ) -> "CoverageTracker":
        """
        Load coverage tracker from JSON file.

        Args:
            filepath: Path to JSON file
            test_cases: Test cases for new tracker

        Returns:
            CoverageTracker with loaded history
        """
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        tracker = cls(
            domain_predicates=data["domain_predicates"], test_cases=test_cases
        )

        tracker.history = [CoverageMetrics.from_dict(m) for m in data["history"]]

        return tracker
