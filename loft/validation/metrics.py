"""
Performance and quality metrics tracking for LOFT.

This module tracks all validation metrics specified in ROADMAP.md:
- Prediction accuracy
- Logical consistency
- Rule stability
- Translation fidelity
- Confidence calibration
- Coverage
- Performance metrics
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger


@dataclass
class ValidationMetrics:
    """
    Comprehensive validation metrics for a single point in time.

    All metrics from ROADMAP.md Phase 0-1 validation criteria.
    """

    timestamp: datetime = field(default_factory=datetime.now)

    # Core metrics from ROADMAP.md
    prediction_accuracy: float = 0.0  # % correct on test cases (0-1)
    logical_consistency: float = 1.0  # 1.0 if consistent, 0.0 if not
    rule_stability: float = 0.0  # modifications per 100 queries
    translation_fidelity: float = 0.0  # roundtrip similarity (0-1)
    confidence_calibration_error: float = 0.0  # |predicted - actual| accuracy
    coverage: float = 0.0  # % of queries system can handle (0-1)

    # ASP-specific metrics
    answer_set_count: int = 0  # number of answer sets
    grounding_time_ms: float = 0.0  # time to ground program
    solving_time_ms: float = 0.0  # time to find answer sets
    program_size: int = 0  # number of rules

    # Performance metrics
    query_latency_ms: float = 0.0  # average query time

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "prediction_accuracy": self.prediction_accuracy,
            "logical_consistency": self.logical_consistency,
            "rule_stability": self.rule_stability,
            "translation_fidelity": self.translation_fidelity,
            "confidence_calibration_error": self.confidence_calibration_error,
            "coverage": self.coverage,
            "answer_set_count": self.answer_set_count,
            "grounding_time_ms": self.grounding_time_ms,
            "solving_time_ms": self.solving_time_ms,
            "program_size": self.program_size,
            "query_latency_ms": self.query_latency_ms,
        }

    def meets_phase_0_criteria(self) -> bool:
        """
        Check if metrics meet Phase 0 MVP criteria from ROADMAP.md.

        Returns:
            True if all Phase 0 validation criteria are met
        """
        return self.logical_consistency == 1.0  # Main requirement for Phase 0

    def meets_phase_1_criteria(self) -> bool:
        """
        Check if metrics meet Phase 1 MVP criteria from ROADMAP.md.

        Phase 1 criteria:
        - Prediction accuracy > 85%
        - Logical consistency = 100%
        - Translation fidelity > 90%

        Returns:
            True if all Phase 1 validation criteria are met
        """
        return (
            self.prediction_accuracy > 0.85
            and self.logical_consistency == 1.0
            and self.translation_fidelity > 0.90
        )


@dataclass
class MetricsDelta:
    """Represents change in metrics between two time points."""

    metric_name: str
    previous_value: float
    current_value: float
    delta: float
    percent_change: float

    def is_regression(self, threshold: float = 0.05) -> bool:
        """Check if this represents a regression (decline > threshold)."""
        return self.delta < -threshold

    def is_improvement(self, threshold: float = 0.05) -> bool:
        """Check if this represents an improvement (gain > threshold)."""
        return self.delta > threshold


class MetricsTracker:
    """
    Track and compute validation metrics over time.

    Maintains history of metrics and detects regressions.
    """

    def __init__(self) -> None:
        """Initialize metrics tracker with empty history."""
        self.metrics_history: List[ValidationMetrics] = []
        logger.info("MetricsTracker initialized")

    def record_metrics(self, metrics: ValidationMetrics) -> None:
        """
        Record a new set of metrics.

        Args:
            metrics: ValidationMetrics instance to record
        """
        self.metrics_history.append(metrics)
        logger.info(
            f"Recorded metrics: accuracy={metrics.prediction_accuracy:.2%}, "
            f"consistency={metrics.logical_consistency:.2f}"
        )

    def get_latest_metrics(self) -> Optional[ValidationMetrics]:
        """
        Get the most recent metrics.

        Returns:
            Latest ValidationMetrics or None if no history
        """
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]

    def get_metrics_at(self, timestamp: datetime) -> Optional[ValidationMetrics]:
        """
        Get metrics closest to a specific timestamp.

        Args:
            timestamp: Target timestamp

        Returns:
            ValidationMetrics closest to timestamp, or None
        """
        if not self.metrics_history:
            return None

        # Find closest timestamp
        closest = min(
            self.metrics_history,
            key=lambda m: abs((m.timestamp - timestamp).total_seconds()),
        )
        return closest

    def compute_deltas(
        self, current: ValidationMetrics, previous: ValidationMetrics
    ) -> List[MetricsDelta]:
        """
        Compute changes between two metric snapshots.

        Args:
            current: Current metrics
            previous: Previous metrics to compare against

        Returns:
            List of MetricsDelta showing changes
        """
        deltas = []

        metric_pairs = [
            (
                "prediction_accuracy",
                current.prediction_accuracy,
                previous.prediction_accuracy,
            ),
            (
                "logical_consistency",
                current.logical_consistency,
                previous.logical_consistency,
            ),
            ("rule_stability", current.rule_stability, previous.rule_stability),
            (
                "translation_fidelity",
                current.translation_fidelity,
                previous.translation_fidelity,
            ),
            ("coverage", current.coverage, previous.coverage),
        ]

        for name, curr_val, prev_val in metric_pairs:
            delta = curr_val - prev_val
            percent_change = ((delta / prev_val) * 100) if prev_val != 0 else 0

            deltas.append(
                MetricsDelta(
                    metric_name=name,
                    previous_value=prev_val,
                    current_value=curr_val,
                    delta=delta,
                    percent_change=percent_change,
                )
            )

        return deltas

    def check_regression(self, current: Optional[ValidationMetrics] = None) -> List[str]:
        """
        Check if metrics have regressed from previous measurement.

        Args:
            current: Current metrics (if None, uses latest from history)

        Returns:
            List of regression warning messages

        Example:
            >>> tracker = MetricsTracker()
            >>> m1 = ValidationMetrics(prediction_accuracy=0.90)
            >>> tracker.record_metrics(m1)
            >>> m2 = ValidationMetrics(prediction_accuracy=0.80)
            >>> warnings = tracker.check_regression(m2)
            >>> assert len(warnings) > 0  # Accuracy dropped
        """
        if current is None:
            current = self.get_latest_metrics()

        if current is None or len(self.metrics_history) < 1:
            logger.debug("Insufficient history for regression check")
            return []

        # Get previous metrics (second to last if current is in history)
        if current in self.metrics_history and len(self.metrics_history) > 1:
            previous = self.metrics_history[-2]
        elif len(self.metrics_history) > 0:
            previous = self.metrics_history[-1]
        else:
            return []

        regressions = []

        # Check accuracy regression (>5% drop is significant)
        if current.prediction_accuracy < previous.prediction_accuracy - 0.05:
            regressions.append(
                f"Accuracy dropped: {previous.prediction_accuracy:.2%} → "
                f"{current.prediction_accuracy:.2%}"
            )

        # Check consistency regression (any drop is critical)
        if current.logical_consistency < previous.logical_consistency:
            regressions.append(
                f"Logical consistency violated! "
                f"{previous.logical_consistency:.2f} → {current.logical_consistency:.2f}"
            )

        # Check fidelity regression (>5% drop)
        if current.translation_fidelity < previous.translation_fidelity - 0.05:
            regressions.append(
                f"Translation fidelity dropped: {previous.translation_fidelity:.2%} → "
                f"{current.translation_fidelity:.2%}"
            )

        # Check coverage regression (>5% drop)
        if current.coverage < previous.coverage - 0.05:
            regressions.append(
                f"Coverage dropped: {previous.coverage:.2%} → {current.coverage:.2%}"
            )

        if regressions:
            logger.warning(f"Detected {len(regressions)} regression(s)")
            for regression in regressions:
                logger.warning(f"  - {regression}")

        return regressions

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of metrics history.

        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {"count": 0, "message": "No metrics recorded"}

        latest = self.metrics_history[-1]
        earliest = self.metrics_history[0]

        return {
            "count": len(self.metrics_history),
            "earliest_timestamp": earliest.timestamp.isoformat(),
            "latest_timestamp": latest.timestamp.isoformat(),
            "latest_accuracy": latest.prediction_accuracy,
            "latest_consistency": latest.logical_consistency,
            "latest_fidelity": latest.translation_fidelity,
            "meets_phase_0_criteria": latest.meets_phase_0_criteria(),
            "meets_phase_1_criteria": latest.meets_phase_1_criteria(),
        }

    def export_history(self) -> List[Dict[str, Any]]:
        """
        Export all metrics history as list of dictionaries.

        Returns:
            List of metrics dictionaries suitable for JSON serialization
        """
        return [m.to_dict() for m in self.metrics_history]


def compute_accuracy(correct_count: int, total_count: int) -> float:
    """
    Compute prediction accuracy metric.

    Args:
        correct_count: Number of correct predictions
        total_count: Total number of predictions

    Returns:
        Accuracy as a float between 0 and 1

    Example:
        >>> accuracy = compute_accuracy(85, 100)
        >>> assert accuracy == 0.85
    """
    if total_count == 0:
        return 0.0
    return correct_count / total_count


def compute_confidence_calibration_error(
    predicted_confidence: List[float], actual_correctness: List[bool]
) -> float:
    """
    Compute calibration error between predicted confidence and actual accuracy.

    Measures how well confidence scores align with actual performance.

    Args:
        predicted_confidence: List of confidence scores (0-1)
        actual_correctness: List of whether predictions were correct

    Returns:
        Calibration error (0 = perfect calibration)

    Example:
        >>> error = compute_confidence_calibration_error(
        ...     [0.9, 0.8, 0.7],
        ...     [True, True, False]
        ... )
        >>> # Error should be low since 2/3 correct matches ~0.8 avg confidence
    """
    if len(predicted_confidence) != len(actual_correctness):
        raise ValueError("Confidence and correctness lists must have same length")

    if len(predicted_confidence) == 0:
        return 0.0

    avg_confidence = sum(predicted_confidence) / len(predicted_confidence)
    actual_accuracy = sum(1 for c in actual_correctness if c) / len(actual_correctness)

    calibration_error = abs(avg_confidence - actual_accuracy)
    return calibration_error
