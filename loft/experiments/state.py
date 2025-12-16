"""
Experiment state management for persistence and resumption.

Issue #256: Long-Running Experiment Runner
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CumulativeMetrics:
    """Cumulative metrics across all cycles."""

    # Overall performance
    total_cases_processed: int = 0
    total_predictions_correct: int = 0
    total_predictions_incorrect: int = 0
    total_predictions_unknown: int = 0

    # Rule generation
    total_rules_generated: int = 0
    total_rules_incorporated: int = 0
    total_rules_rejected: int = 0

    # Accuracy over time
    accuracy_by_cycle: List[float] = field(default_factory=list)
    coverage_by_cycle: List[float] = field(default_factory=list)

    # Timing
    avg_case_processing_time_ms: float = 0.0
    total_llm_calls: int = 0
    total_llm_cost_usd: float = 0.0

    def update_from_cycle(self, cycle_metrics: Dict[str, Any]):
        """
        Update cumulative metrics from a cycle result.

        Args:
            cycle_metrics: Metrics from a single improvement cycle
        """
        # Cases
        self.total_cases_processed += cycle_metrics.get("cases_processed", 0)
        self.total_predictions_correct += cycle_metrics.get("predictions_correct", 0)
        self.total_predictions_incorrect += cycle_metrics.get(
            "predictions_incorrect", 0
        )
        self.total_predictions_unknown += cycle_metrics.get("predictions_unknown", 0)

        # Rules
        self.total_rules_generated += cycle_metrics.get("rules_generated", 0)
        self.total_rules_incorporated += cycle_metrics.get("rules_incorporated", 0)
        self.total_rules_rejected += cycle_metrics.get("rules_rejected", 0)

        # Per-cycle tracking
        accuracy = cycle_metrics.get("accuracy", 0.0)
        coverage = cycle_metrics.get("coverage", 0.0)
        self.accuracy_by_cycle.append(accuracy)
        self.coverage_by_cycle.append(coverage)

        # Timing
        proc_time = cycle_metrics.get("avg_processing_time_ms", 0.0)
        if proc_time > 0:
            # Running average
            total_cases = self.total_cases_processed
            if total_cases > 0:
                self.avg_case_processing_time_ms = (
                    self.avg_case_processing_time_ms * (total_cases - 1) + proc_time
                ) / total_cases

        # LLM usage
        self.total_llm_calls += cycle_metrics.get("llm_calls", 0)
        self.total_llm_cost_usd += cycle_metrics.get("llm_cost_usd", 0.0)

    @property
    def current_accuracy(self) -> float:
        """Get current overall accuracy."""
        total_predictions = (
            self.total_predictions_correct + self.total_predictions_incorrect
        )
        if total_predictions == 0:
            return 0.0
        return self.total_predictions_correct / total_predictions

    @property
    def current_coverage(self) -> float:
        """Get current coverage (non-unknown predictions)."""
        total = (
            self.total_predictions_correct
            + self.total_predictions_incorrect
            + self.total_predictions_unknown
        )
        if total == 0:
            return 0.0
        return (
            self.total_predictions_correct + self.total_predictions_incorrect
        ) / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CumulativeMetrics":
        """Load from dictionary."""
        return cls(**data)


@dataclass
class ExperimentState:
    """Persistent state for experiment resumption."""

    # Identity
    experiment_id: str
    started_at: str
    last_updated: str

    # Progress
    cycles_completed: int = 0
    cases_processed: int = 0
    rules_generated: int = 0
    rules_incorporated: int = 0

    # Cumulative metrics
    cumulative_metrics: CumulativeMetrics = field(default_factory=CumulativeMetrics)

    # Dataset cursor
    dataset_cursor: int = 0

    # Meta-state reference
    meta_state_path: Optional[str] = None

    # Goals tracking
    goals_achieved: Dict[str, bool] = field(default_factory=dict)

    def save(self, path: Path):
        """
        Save state to disk.

        Args:
            path: Path to state file
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        self.last_updated = datetime.now().isoformat()

        # Convert to dict
        state_dict = {
            "experiment_id": self.experiment_id,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "cycles_completed": self.cycles_completed,
            "cases_processed": self.cases_processed,
            "rules_generated": self.rules_generated,
            "rules_incorporated": self.rules_incorporated,
            "cumulative_metrics": self.cumulative_metrics.to_dict(),
            "dataset_cursor": self.dataset_cursor,
            "meta_state_path": self.meta_state_path,
            "goals_achieved": self.goals_achieved,
        }

        # Save to JSON
        with open(path, "w") as f:
            json.dump(state_dict, f, indent=2)

        logger.info(f"Saved experiment state to {path}")

    @classmethod
    def load(cls, path: Path) -> "ExperimentState":
        """
        Load state from disk.

        Args:
            path: Path to state file

        Returns:
            Loaded ExperimentState
        """
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        # Convert cumulative_metrics from dict
        metrics_dict = data.pop("cumulative_metrics")
        data["cumulative_metrics"] = CumulativeMetrics.from_dict(metrics_dict)

        logger.info(f"Loaded experiment state from {path}")
        return cls(**data)

    @classmethod
    def load_or_create(
        cls, path: Path, experiment_id: str, description: str = ""
    ) -> "ExperimentState":
        """
        Load existing state or create new.

        Args:
            path: Path to state file
            experiment_id: Experiment ID for new state
            description: Description for new state

        Returns:
            Loaded or new ExperimentState
        """
        if path.exists():
            return cls.load(path)
        else:
            now = datetime.now().isoformat()
            return cls(
                experiment_id=experiment_id,
                started_at=now,
                last_updated=now,
            )

    def all_goals_achieved(
        self, target_accuracy: float, target_coverage: float, target_rule_count: int
    ) -> bool:
        """
        Check if all experiment goals are met.

        Args:
            target_accuracy: Target accuracy threshold
            target_coverage: Target coverage threshold
            target_rule_count: Target rule count

        Returns:
            True if all goals achieved
        """
        accuracy_met = self.cumulative_metrics.current_accuracy >= target_accuracy
        coverage_met = self.cumulative_metrics.current_coverage >= target_coverage
        rules_met = self.rules_incorporated >= target_rule_count

        # Update goals tracking
        self.goals_achieved = {
            "accuracy": accuracy_met,
            "coverage": coverage_met,
            "rule_count": rules_met,
        }

        return accuracy_met and coverage_met and rules_met

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current state."""
        return {
            "experiment_id": self.experiment_id,
            "cycles_completed": self.cycles_completed,
            "cases_processed": self.cases_processed,
            "rules_incorporated": self.rules_incorporated,
            "current_accuracy": self.cumulative_metrics.current_accuracy,
            "current_coverage": self.cumulative_metrics.current_coverage,
            "goals_achieved": self.goals_achieved,
            "elapsed_time": (
                datetime.fromisoformat(self.last_updated)
                - datetime.fromisoformat(self.started_at)
            ).total_seconds(),
        }
