"""
Learning metrics for casework exploration.

Tracks how the system improves over multiple cases.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class CaseResult:
    """Result for a single case."""

    scenario_id: str
    prediction: str
    ground_truth: str
    correct: bool
    confidence: float
    gaps_identified: int
    rules_generated: int
    rules_accepted: int
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LearningMetrics:
    """Metrics tracking learning across multiple cases."""

    total_cases: int = 0
    cases_correct: int = 0
    cumulative_accuracy: List[float] = field(default_factory=list)
    total_gaps_identified: int = 0
    total_rules_generated: int = 0
    total_rules_accepted: int = 0
    total_rules_incorporated: int = 0
    case_results: List[CaseResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    final_kb_accuracy: float = 0.0  # Accuracy with complete knowledge base

    def add_case_result(self, result: CaseResult) -> None:
        """Add a case result and update metrics."""
        self.case_results.append(result)
        self.total_cases += 1

        if result.correct:
            self.cases_correct += 1

        # Update cumulative accuracy
        current_accuracy = self.cases_correct / self.total_cases
        self.cumulative_accuracy.append(current_accuracy)

        # Update totals
        self.total_gaps_identified += result.gaps_identified
        self.total_rules_generated += result.rules_generated
        self.total_rules_accepted += result.rules_accepted

    def get_current_accuracy(self) -> float:
        """Get current accuracy."""
        return self.cases_correct / self.total_cases if self.total_cases > 0 else 0.0

    def get_learning_curve(self) -> List[Dict[str, Any]]:
        """Get learning curve data points."""
        return [
            {
                "case_number": i + 1,
                "accuracy": acc,
                "timestamp": self.case_results[i].timestamp.isoformat(),
            }
            for i, acc in enumerate(self.cumulative_accuracy)
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_time = ((self.end_time or datetime.now()) - self.start_time).total_seconds()

        return {
            "total_cases": self.total_cases,
            "cases_correct": self.cases_correct,
            "final_accuracy": self.get_current_accuracy(),
            "final_kb_accuracy": self.final_kb_accuracy,
            "total_gaps_identified": self.total_gaps_identified,
            "total_rules_generated": self.total_rules_generated,
            "total_rules_accepted": self.total_rules_accepted,
            "acceptance_rate": (
                self.total_rules_accepted / self.total_rules_generated
                if self.total_rules_generated > 0
                else 0.0
            ),
            "total_time_seconds": total_time,
            "avg_time_per_case": total_time / self.total_cases if self.total_cases > 0 else 0.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": self.get_summary(),
            "learning_curve": self.get_learning_curve(),
            "case_results": [
                {
                    "scenario_id": r.scenario_id,
                    "prediction": r.prediction,
                    "ground_truth": r.ground_truth,
                    "correct": r.correct,
                    "confidence": r.confidence,
                    "gaps_identified": r.gaps_identified,
                    "rules_generated": r.rules_generated,
                    "rules_accepted": r.rules_accepted,
                    "processing_time": r.processing_time,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self.case_results
            ],
        }
