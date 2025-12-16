"""
Fidelity tracking for ASP ↔ NL translations.

Tracks translation fidelity metrics over time, detects regressions,
and generates trend reports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from statistics import mean


@dataclass
class TranslationResult:
    """Result of a single translation."""

    original: str
    translated: str
    back_translated: Optional[str] = None
    fidelity: float = 0.0
    rule_type: Optional[str] = None
    predicates: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FidelitySnapshot:
    """Snapshot of translation fidelity at a point in time."""

    timestamp: str
    total_translations: int
    avg_fidelity: float
    perfect_rate: float  # Percentage with fidelity == 1.0
    min_fidelity: float
    max_fidelity: float
    fidelity_distribution: Dict[
        str, int
    ]  # Buckets: <0.5, 0.5-0.7, 0.7-0.9, 0.9-1.0, 1.0
    fidelity_by_type: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "total_translations": self.total_translations,
            "avg_fidelity": self.avg_fidelity,
            "perfect_rate": self.perfect_rate,
            "min_fidelity": self.min_fidelity,
            "max_fidelity": self.max_fidelity,
            "fidelity_distribution": self.fidelity_distribution,
            "fidelity_by_type": self.fidelity_by_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FidelitySnapshot":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Regression:
    """Detected fidelity regression."""

    baseline_fidelity: float
    current_fidelity: float
    degradation: float
    timestamp: str
    affected_types: List[str] = field(default_factory=list)


class FidelityTracker:
    """
    Tracks translation fidelity metrics over time.

    Maintains history of fidelity snapshots, detects regressions,
    and generates trend reports.
    """

    def __init__(self):
        """Initialize fidelity tracker."""
        self.history: List[FidelitySnapshot] = []

    def record_snapshot(
        self,
        translations: List[TranslationResult],
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FidelitySnapshot:
        """
        Record a fidelity snapshot.

        Args:
            translations: List of translation results
            timestamp: Optional timestamp (defaults to now)
            metadata: Optional metadata for this snapshot

        Returns:
            FidelitySnapshot object
        """
        if not translations:
            raise ValueError("Cannot record snapshot with no translations")

        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        fidelities = [t.fidelity for t in translations]

        snapshot = FidelitySnapshot(
            timestamp=timestamp,
            total_translations=len(translations),
            avg_fidelity=mean(fidelities),
            perfect_rate=sum(1 for f in fidelities if f == 1.0) / len(fidelities),
            min_fidelity=min(fidelities),
            max_fidelity=max(fidelities),
            fidelity_distribution=self._compute_distribution(fidelities),
            fidelity_by_type=self._compute_by_type(translations),
            metadata=metadata or {},
        )

        self.history.append(snapshot)
        return snapshot

    def _compute_distribution(self, fidelities: List[float]) -> Dict[str, int]:
        """Compute fidelity distribution buckets."""
        buckets = {
            "<0.5": 0,
            "0.5-0.7": 0,
            "0.7-0.9": 0,
            "0.9-1.0": 0,
            "1.0": 0,
        }

        for f in fidelities:
            if f == 1.0:
                buckets["1.0"] += 1
            elif f >= 0.9:
                buckets["0.9-1.0"] += 1
            elif f >= 0.7:
                buckets["0.7-0.9"] += 1
            elif f >= 0.5:
                buckets["0.5-0.7"] += 1
            else:
                buckets["<0.5"] += 1

        return buckets

    def _compute_by_type(
        self, translations: List[TranslationResult]
    ) -> Dict[str, float]:
        """Compute average fidelity by rule type."""
        by_type: Dict[str, List[float]] = {}

        for t in translations:
            if t.rule_type:
                if t.rule_type not in by_type:
                    by_type[t.rule_type] = []
                by_type[t.rule_type].append(t.fidelity)

        return {rtype: mean(fidelities) for rtype, fidelities in by_type.items()}

    @property
    def current_snapshot(self) -> Optional[FidelitySnapshot]:
        """Get most recent snapshot."""
        if not self.history:
            return None
        return self.history[-1]

    @property
    def current_fidelity(self) -> float:
        """Get current average fidelity."""
        if not self.history:
            return 0.0
        return self.history[-1].avg_fidelity

    def detect_regression(
        self, threshold: float = 0.05, window: int = 5
    ) -> Optional[Regression]:
        """
        Detect fidelity regression compared to baseline.

        Args:
            threshold: Minimum degradation to consider a regression
            window: Number of recent snapshots to use for baseline

        Returns:
            Regression object if detected, None otherwise
        """
        if len(self.history) < 2:
            return None

        # Use average of first `window` snapshots as baseline
        baseline_snapshots = self.history[: min(window, len(self.history) - 1)]
        baseline_fidelity = mean([s.avg_fidelity for s in baseline_snapshots])

        current = self.history[-1]

        if current.avg_fidelity < baseline_fidelity - threshold:
            # Identify affected rule types
            affected_types = []
            if baseline_snapshots[0].fidelity_by_type and current.fidelity_by_type:
                for rtype in current.fidelity_by_type:
                    baseline_type_fidelity = mean(
                        [
                            s.fidelity_by_type.get(rtype, 0.0)
                            for s in baseline_snapshots
                            if rtype in s.fidelity_by_type
                        ]
                    )
                    if (
                        current.fidelity_by_type[rtype]
                        < baseline_type_fidelity - threshold
                    ):
                        affected_types.append(rtype)

            return Regression(
                baseline_fidelity=baseline_fidelity,
                current_fidelity=current.avg_fidelity,
                degradation=baseline_fidelity - current.avg_fidelity,
                timestamp=current.timestamp,
                affected_types=affected_types,
            )

        return None

    def get_trend(self, window: int = 5) -> str:
        """
        Get fidelity trend over recent history.

        Args:
            window: Number of recent snapshots to analyze

        Returns:
            "improving", "stable", "degrading", or "insufficient_data"
        """
        if len(self.history) < 2:
            return "insufficient_data"

        recent = self.history[-min(window, len(self.history)) :]

        if len(recent) < 2:
            return "insufficient_data"

        # Calculate trend using simple linear regression
        fidelities = [s.avg_fidelity for s in recent]

        # Simple slope calculation
        n = len(fidelities)
        x_mean = (n - 1) / 2
        y_mean = mean(fidelities)

        numerator = sum((i - x_mean) * (f - y_mean) for i, f in enumerate(fidelities))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"

    def generate_trend_report(self) -> str:
        """
        Generate markdown trend report.

        Returns:
            Markdown-formatted trend report
        """
        if not self.history:
            return "# Translation Fidelity Trend Report\n\nNo data available.\n"

        current = self.history[-1]
        trend = self.get_trend()
        regression = self.detect_regression()

        report = f"""# Translation Fidelity Trend Report

## Current Status (as of {current.timestamp})

| Metric | Value |
|--------|-------|
| Average Fidelity | {current.avg_fidelity:.2%} |
| Perfect Roundtrips | {current.perfect_rate:.2%} |
| Min Fidelity | {current.min_fidelity:.2%} |
| Max Fidelity | {current.max_fidelity:.2%} |
| Total Translations | {current.total_translations} |

## Trend

**Recent Trend**: {trend.upper()}

"""

        if regression:
            report += f"""## ⚠️ Regression Detected

- **Baseline Fidelity**: {regression.baseline_fidelity:.2%}
- **Current Fidelity**: {regression.current_fidelity:.2%}
- **Degradation**: {regression.degradation:.2%}
- **Detected**: {regression.timestamp}

"""
            if regression.affected_types:
                report += "**Affected Rule Types**:\n"
                for rtype in regression.affected_types:
                    report += f"- {rtype}\n"
                report += "\n"

        # Fidelity distribution
        report += "## Fidelity Distribution\n\n"
        for bucket, count in current.fidelity_distribution.items():
            percentage = count / current.total_translations * 100
            bar = "█" * int(percentage / 2)
            report += f"- **{bucket}**: {count} ({percentage:.1f}%) {bar}\n"

        # Fidelity by type
        if current.fidelity_by_type:
            report += "\n## Fidelity by Rule Type\n\n"
            for rtype, fidelity in sorted(
                current.fidelity_by_type.items(), key=lambda x: x[1], reverse=True
            ):
                report += f"- **{rtype}**: {fidelity:.2%}\n"

        # History table
        if len(self.history) > 1:
            report += "\n## History\n\n"
            report += "| Timestamp | Avg Fidelity | Perfect Rate | Total |\n"
            report += "|-----------|--------------|--------------|-------|\n"

            for snapshot in self.history[-10:]:  # Last 10 snapshots
                ts = snapshot.timestamp[:19]  # Truncate to YYYY-MM-DDTHH:MM:SS
                report += f"| {ts} | {snapshot.avg_fidelity:.2%} | {snapshot.perfect_rate:.2%} | {snapshot.total_translations} |\n"

        return report

    def save_history(self, filepath: str) -> None:
        """
        Save fidelity history to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "snapshots": [s.to_dict() for s in self.history],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_history(cls, filepath: str) -> "FidelityTracker":
        """
        Load fidelity tracker from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            FidelityTracker with loaded history
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        tracker = cls()
        tracker.history = [
            FidelitySnapshot.from_dict(s) for s in data.get("snapshots", [])
        ]

        return tracker
