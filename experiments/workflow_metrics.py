"""
Workflow Metrics for LOFT Integration Testing

This module defines metrics for tracking the complete LOFT workflow:
gap identification -> rule generation -> validation -> incorporation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class WorkflowMetrics:
    """
    Comprehensive metrics for the full LOFT workflow.

    Tracks performance across:
    - Gap identification
    - Rule generation (with LLM costs)
    - Validation (precision/recall)
    - Incorporation
    - Overall system improvement
    """

    # Gap Identification Metrics
    gap_identification_time: float
    gap_identification_accuracy: float  # What % of identified gaps are real gaps
    gaps_identified: int = 0

    # Rule Generation Metrics
    rule_generation_time: float = 0.0
    rule_generation_cost: float = 0.0  # LLM cost in USD
    rules_generated: int = 0

    # Validation Metrics
    validation_time: float = 0.0
    validation_precision: float = 0.0  # TP / (TP + FP) - % of accepted rules that are good
    validation_recall: float = 0.0  # TP / (TP + FN) - % of good rules that are accepted
    rules_validated: int = 0
    rules_accepted: int = 0
    rules_rejected: int = 0

    # Incorporation Metrics
    incorporation_time: float = 0.0
    incorporation_success_rate: float = 0.0  # % of attempted incorporations that succeed
    rules_incorporated: int = 0
    incorporation_failures: int = 0

    # Overall System Improvement
    baseline_accuracy: float = 0.0
    final_accuracy: float = 0.0
    overall_accuracy_improvement: float = 0.0  # final - baseline

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    test_suite_name: str = ""
    test_cases_evaluated: int = 0

    # Detailed breakdowns
    per_gap_metrics: List[Dict] = field(default_factory=list)
    per_rule_metrics: List[Dict] = field(default_factory=list)

    def compute_derived_metrics(self) -> None:
        """Compute derived metrics from raw data."""
        self.overall_accuracy_improvement = self.final_accuracy - self.baseline_accuracy

        total_validated = self.rules_accepted + self.rules_rejected
        if total_validated > 0:
            self.acceptance_rate = self.rules_accepted / total_validated
        else:
            self.acceptance_rate = 0.0

        total_incorporation_attempts = self.rules_incorporated + self.incorporation_failures
        if total_incorporation_attempts > 0:
            self.incorporation_success_rate = self.rules_incorporated / total_incorporation_attempts

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "gap_identification": {
                "time_seconds": self.gap_identification_time,
                "accuracy": self.gap_identification_accuracy,
                "gaps_identified": self.gaps_identified,
            },
            "rule_generation": {
                "time_seconds": self.rule_generation_time,
                "cost_usd": self.rule_generation_cost,
                "rules_generated": self.rules_generated,
            },
            "validation": {
                "time_seconds": self.validation_time,
                "precision": self.validation_precision,
                "recall": self.validation_recall,
                "rules_validated": self.rules_validated,
                "rules_accepted": self.rules_accepted,
                "rules_rejected": self.rules_rejected,
            },
            "incorporation": {
                "time_seconds": self.incorporation_time,
                "success_rate": self.incorporation_success_rate,
                "rules_incorporated": self.rules_incorporated,
                "failures": self.incorporation_failures,
            },
            "overall": {
                "baseline_accuracy": self.baseline_accuracy,
                "final_accuracy": self.final_accuracy,
                "improvement": self.overall_accuracy_improvement,
                "test_cases_evaluated": self.test_cases_evaluated,
            },
            "metadata": {
                "timestamp": self.timestamp.isoformat(),
                "test_suite": self.test_suite_name,
            },
            "detailed_metrics": {
                "per_gap": self.per_gap_metrics,
                "per_rule": self.per_rule_metrics,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "WorkflowMetrics":
        """Create WorkflowMetrics from dictionary."""
        gap = data.get("gap_identification", {})
        gen = data.get("rule_generation", {})
        val = data.get("validation", {})
        inc = data.get("incorporation", {})
        overall = data.get("overall", {})
        meta = data.get("metadata", {})
        detailed = data.get("detailed_metrics", {})

        return cls(
            gap_identification_time=gap.get("time_seconds", 0.0),
            gap_identification_accuracy=gap.get("accuracy", 0.0),
            gaps_identified=gap.get("gaps_identified", 0),
            rule_generation_time=gen.get("time_seconds", 0.0),
            rule_generation_cost=gen.get("cost_usd", 0.0),
            rules_generated=gen.get("rules_generated", 0),
            validation_time=val.get("time_seconds", 0.0),
            validation_precision=val.get("precision", 0.0),
            validation_recall=val.get("recall", 0.0),
            rules_validated=val.get("rules_validated", 0),
            rules_accepted=val.get("rules_accepted", 0),
            rules_rejected=val.get("rules_rejected", 0),
            incorporation_time=inc.get("time_seconds", 0.0),
            incorporation_success_rate=inc.get("success_rate", 0.0),
            rules_incorporated=inc.get("rules_incorporated", 0),
            incorporation_failures=inc.get("failures", 0),
            baseline_accuracy=overall.get("baseline_accuracy", 0.0),
            final_accuracy=overall.get("final_accuracy", 0.0),
            overall_accuracy_improvement=overall.get("improvement", 0.0),
            test_cases_evaluated=overall.get("test_cases_evaluated", 0),
            timestamp=datetime.fromisoformat(meta.get("timestamp", datetime.now().isoformat())),
            test_suite_name=meta.get("test_suite", ""),
            per_gap_metrics=detailed.get("per_gap", []),
            per_rule_metrics=detailed.get("per_rule", []),
        )


@dataclass
class GapMetrics:
    """Metrics for a single identified gap."""

    gap_id: str
    description: str
    identification_time: float
    candidates_generated: int
    candidates_accepted: int
    best_rule_confidence: float
    addressed: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "gap_id": self.gap_id,
            "description": self.description,
            "identification_time": self.identification_time,
            "candidates_generated": self.candidates_generated,
            "candidates_accepted": self.candidates_accepted,
            "best_rule_confidence": self.best_rule_confidence,
            "addressed": self.addressed,
        }


@dataclass
class RuleMetrics:
    """Metrics for a single generated rule."""

    rule_id: str
    gap_id: str
    generation_time: float
    generation_cost: float
    validation_time: float
    validation_result: str  # "accepted", "rejected", "review"
    confidence_score: float
    accuracy_impact: Optional[float] = None
    incorporated: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "gap_id": self.gap_id,
            "generation_time": self.generation_time,
            "generation_cost": self.generation_cost,
            "validation_time": self.validation_time,
            "validation_result": self.validation_result,
            "confidence_score": self.confidence_score,
            "accuracy_impact": self.accuracy_impact,
            "incorporated": self.incorporated,
        }
