"""
Unit tests for metrics tracking.

Tests ValidationMetrics, MetricsTracker, and metric computation functions.
"""

import pytest
from datetime import datetime, timedelta
from loft.validation import (
    ValidationMetrics,
    MetricsTracker,
    compute_accuracy,
    compute_confidence_calibration_error,
)


class TestValidationMetrics:
    """Tests for ValidationMetrics dataclass."""

    def test_default_metrics(self) -> None:
        """Test that metrics initialize with sensible defaults."""
        metrics = ValidationMetrics()

        assert metrics.prediction_accuracy == 0.0
        assert metrics.logical_consistency == 1.0  # Default to consistent
        assert metrics.rule_stability == 0.0
        assert metrics.translation_fidelity == 0.0
        assert metrics.confidence_calibration_error == 0.0
        assert metrics.coverage == 0.0
        assert isinstance(metrics.timestamp, datetime)

    def test_custom_metrics(self) -> None:
        """Test creating metrics with custom values."""
        metrics = ValidationMetrics(
            prediction_accuracy=0.95,
            logical_consistency=1.0,
            translation_fidelity=0.92,
        )

        assert metrics.prediction_accuracy == 0.95
        assert metrics.logical_consistency == 1.0
        assert metrics.translation_fidelity == 0.92

    def test_meets_phase_0_criteria(self) -> None:
        """Test Phase 0 MVP criteria check."""
        # Phase 0 main requirement: logical consistency
        metrics_pass = ValidationMetrics(logical_consistency=1.0)
        assert metrics_pass.meets_phase_0_criteria()

        metrics_fail = ValidationMetrics(logical_consistency=0.8)
        assert not metrics_fail.meets_phase_0_criteria()

    def test_meets_phase_1_criteria(self) -> None:
        """Test Phase 1 MVP criteria check."""
        # Phase 1: accuracy > 85%, consistency = 100%, fidelity > 90%
        metrics_pass = ValidationMetrics(
            prediction_accuracy=0.90,
            logical_consistency=1.0,
            translation_fidelity=0.95,
        )
        assert metrics_pass.meets_phase_1_criteria()

        # Fail on accuracy
        metrics_fail_accuracy = ValidationMetrics(
            prediction_accuracy=0.80,
            logical_consistency=1.0,
            translation_fidelity=0.95,
        )
        assert not metrics_fail_accuracy.meets_phase_1_criteria()

        # Fail on fidelity
        metrics_fail_fidelity = ValidationMetrics(
            prediction_accuracy=0.90,
            logical_consistency=1.0,
            translation_fidelity=0.85,
        )
        assert not metrics_fail_fidelity.meets_phase_1_criteria()

    def test_to_dict(self) -> None:
        """Test metrics conversion to dictionary."""
        metrics = ValidationMetrics(
            prediction_accuracy=0.85, logical_consistency=1.0
        )
        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d["prediction_accuracy"] == 0.85
        assert d["logical_consistency"] == 1.0
        assert "timestamp" in d


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    def test_initialization(self) -> None:
        """Test tracker initializes empty."""
        tracker = MetricsTracker()
        assert len(tracker.metrics_history) == 0
        assert tracker.get_latest_metrics() is None

    def test_record_metrics(self) -> None:
        """Test recording metrics."""
        tracker = MetricsTracker()
        metrics = ValidationMetrics(prediction_accuracy=0.85)

        tracker.record_metrics(metrics)

        assert len(tracker.metrics_history) == 1
        assert tracker.get_latest_metrics() == metrics

    def test_get_latest_metrics(self) -> None:
        """Test retrieving latest metrics."""
        tracker = MetricsTracker()

        metrics1 = ValidationMetrics(prediction_accuracy=0.80)
        metrics2 = ValidationMetrics(prediction_accuracy=0.90)

        tracker.record_metrics(metrics1)
        tracker.record_metrics(metrics2)

        latest = tracker.get_latest_metrics()
        assert latest == metrics2
        assert latest.prediction_accuracy == 0.90

    def test_check_regression_no_history(self) -> None:
        """Test regression check with no history."""
        tracker = MetricsTracker()
        regressions = tracker.check_regression()

        assert len(regressions) == 0

    def test_check_regression_accuracy_drop(self) -> None:
        """Test detection of accuracy regression."""
        tracker = MetricsTracker()

        metrics1 = ValidationMetrics(prediction_accuracy=0.90)
        metrics2 = ValidationMetrics(prediction_accuracy=0.80)

        tracker.record_metrics(metrics1)
        regressions = tracker.check_regression(metrics2)

        assert len(regressions) > 0
        assert any("accuracy" in r.lower() for r in regressions)

    def test_check_regression_consistency_drop(self) -> None:
        """Test detection of consistency regression (critical)."""
        tracker = MetricsTracker()

        metrics1 = ValidationMetrics(logical_consistency=1.0)
        metrics2 = ValidationMetrics(logical_consistency=0.9)

        tracker.record_metrics(metrics1)
        regressions = tracker.check_regression(metrics2)

        assert len(regressions) > 0
        assert any("consistency" in r.lower() for r in regressions)

    def test_check_regression_improvement(self) -> None:
        """Test that improvements don't trigger regressions."""
        tracker = MetricsTracker()

        metrics1 = ValidationMetrics(prediction_accuracy=0.80)
        metrics2 = ValidationMetrics(prediction_accuracy=0.90)

        tracker.record_metrics(metrics1)
        regressions = tracker.check_regression(metrics2)

        assert len(regressions) == 0

    def test_get_summary(self) -> None:
        """Test getting summary of metrics history."""
        tracker = MetricsTracker()

        # Empty tracker
        summary_empty = tracker.get_summary()
        assert summary_empty["count"] == 0

        # Add some metrics
        tracker.record_metrics(ValidationMetrics(prediction_accuracy=0.85))
        tracker.record_metrics(ValidationMetrics(prediction_accuracy=0.90))

        summary = tracker.get_summary()
        assert summary["count"] == 2
        assert summary["latest_accuracy"] == 0.90
        assert "earliest_timestamp" in summary
        assert "latest_timestamp" in summary

    def test_export_history(self) -> None:
        """Test exporting metrics history."""
        tracker = MetricsTracker()

        metrics1 = ValidationMetrics(prediction_accuracy=0.85)
        metrics2 = ValidationMetrics(prediction_accuracy=0.90)

        tracker.record_metrics(metrics1)
        tracker.record_metrics(metrics2)

        history = tracker.export_history()

        assert len(history) == 2
        assert all(isinstance(m, dict) for m in history)
        assert history[0]["prediction_accuracy"] == 0.85
        assert history[1]["prediction_accuracy"] == 0.90


class TestMetricComputationFunctions:
    """Tests for metric computation utility functions."""

    def test_compute_accuracy(self) -> None:
        """Test accuracy computation."""
        accuracy = compute_accuracy(85, 100)
        assert accuracy == 0.85

        accuracy_perfect = compute_accuracy(100, 100)
        assert accuracy_perfect == 1.0

        accuracy_zero = compute_accuracy(0, 100)
        assert accuracy_zero == 0.0

    def test_compute_accuracy_zero_total(self) -> None:
        """Test accuracy with zero total count."""
        accuracy = compute_accuracy(0, 0)
        assert accuracy == 0.0

    def test_compute_confidence_calibration_error_perfect(self) -> None:
        """Test calibration error with perfect calibration."""
        # All confident predictions are correct
        error = compute_confidence_calibration_error(
            [0.9, 0.9, 0.9], [True, True, True]
        )
        # Avg confidence = 0.9, accuracy = 1.0, error = 0.1
        assert 0.0 <= error <= 0.2

    def test_compute_confidence_calibration_error_poor(self) -> None:
        """Test calibration error with poor calibration."""
        # Confident predictions but all wrong
        error = compute_confidence_calibration_error(
            [0.9, 0.9, 0.9], [False, False, False]
        )
        # Avg confidence = 0.9, accuracy = 0.0, error = 0.9
        assert error > 0.8

    def test_compute_confidence_calibration_error_mismatch(self) -> None:
        """Test that mismatched lists raise error."""
        with pytest.raises(ValueError):
            compute_confidence_calibration_error([0.9, 0.8], [True, True, False])

    def test_compute_confidence_calibration_error_empty(self) -> None:
        """Test calibration error with empty lists."""
        error = compute_confidence_calibration_error([], [])
        assert error == 0.0


class TestMetricsTrackerAdvanced:
    """Advanced tests for MetricsTracker."""

    def test_compute_deltas(self) -> None:
        """Test computing deltas between metrics."""
        tracker = MetricsTracker()

        metrics1 = ValidationMetrics(prediction_accuracy=0.80, coverage=0.60)
        metrics2 = ValidationMetrics(prediction_accuracy=0.90, coverage=0.70)

        deltas = tracker.compute_deltas(metrics2, metrics1)

        # Find accuracy delta
        accuracy_delta = next(
            d for d in deltas if d.metric_name == "prediction_accuracy"
        )
        assert accuracy_delta.previous_value == 0.80
        assert accuracy_delta.current_value == 0.90
        assert accuracy_delta.delta == 0.10
        assert accuracy_delta.is_improvement()

        # Find coverage delta
        coverage_delta = next(d for d in deltas if d.metric_name == "coverage")
        assert coverage_delta.delta == 0.10

    def test_multiple_recordings(self) -> None:
        """Test tracker with multiple metric recordings."""
        tracker = MetricsTracker()

        for i in range(5):
            metrics = ValidationMetrics(prediction_accuracy=0.80 + i * 0.02)
            tracker.record_metrics(metrics)

        assert len(tracker.metrics_history) == 5

        latest = tracker.get_latest_metrics()
        assert latest.prediction_accuracy == 0.88  # 0.80 + 4*0.02
