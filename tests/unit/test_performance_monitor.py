"""
Unit tests for performance monitoring.

Tests snapshot capture, regression detection, trend analysis, and reporting.
"""

from datetime import datetime, timedelta

import pytest

from loft.monitoring.performance_monitor import PerformanceMonitor
from loft.monitoring.performance_schemas import PerformanceSnapshot


class TestPerformanceMonitor:
    """Test performance monitor functionality."""

    def test_monitor_initialization(self):
        """Test monitor initializes with correct parameters."""
        monitor = PerformanceMonitor(
            regression_threshold=0.03, trend_lookback_days=14, max_snapshots=500
        )

        assert monitor.regression_threshold == 0.03
        assert monitor.trend_lookback_days == 14
        assert monitor.max_snapshots == 500
        assert len(monitor.snapshot_history) == 0
        assert len(monitor.active_alerts) == 0

    def test_capture_snapshot(self):
        """Test capturing performance snapshot."""
        monitor = PerformanceMonitor()

        snapshot = monitor.capture_snapshot(
            core_version_id="v1.0.0",
            overall_accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=100,
            rules_by_layer={"constitutional": 10, "strategic": 30},
            avg_confidence=0.85,
            query_latency_ms=50.0,
            memory_usage_mb=128.0,
            test_cases_passing=95,
            test_cases_total=100,
        )

        assert snapshot.core_version_id == "v1.0.0"
        assert snapshot.overall_accuracy == 0.92
        assert snapshot.precision == 0.90
        assert snapshot.total_rules == 100
        assert len(monitor.snapshot_history) == 1

    def test_snapshot_history_limit(self):
        """Test snapshot history is limited to max_snapshots."""
        monitor = PerformanceMonitor(max_snapshots=5)

        # Capture 10 snapshots
        for i in range(10):
            monitor.capture_snapshot(
                core_version_id=f"v{i}",
                overall_accuracy=0.90,
                precision=0.85,
                recall=0.88,
                f1_score=0.865,
                total_rules=100,
                rules_by_layer={},
                avg_confidence=0.85,
            )

        # Should only keep last 5
        assert len(monitor.snapshot_history) == 5
        assert monitor.snapshot_history[0].core_version_id == "v5"
        assert monitor.snapshot_history[-1].core_version_id == "v9"

    def test_detect_regression_no_data(self):
        """Test regression detection with insufficient data."""
        monitor = PerformanceMonitor()

        snapshot = monitor.capture_snapshot(
            core_version_id="v1",
            overall_accuracy=0.90,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
        )

        regressions = monitor.detect_regressions(snapshot)
        assert len(regressions) == 0

    def test_detect_regression_accuracy_drop(self):
        """Test detecting accuracy regression."""
        monitor = PerformanceMonitor(regression_threshold=0.02)

        # Baseline snapshot
        baseline = monitor.capture_snapshot(
            core_version_id="v1",
            overall_accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
        )

        # Current snapshot with accuracy drop > 2%
        current = monitor.capture_snapshot(
            core_version_id="v2",
            overall_accuracy=0.88,  # 4% drop
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
        )

        regressions = monitor.detect_regressions(current, baseline)

        assert len(regressions) == 1
        assert regressions[0].metric == "Overall Accuracy"
        assert regressions[0].baseline_value == 0.92
        assert regressions[0].current_value == 0.88
        assert regressions[0].degradation == pytest.approx(0.04, abs=0.001)
        assert regressions[0].severity == "critical"  # > 2x threshold

    def test_detect_regression_multiple_metrics(self):
        """Test detecting regressions in multiple metrics."""
        monitor = PerformanceMonitor(regression_threshold=0.02)

        baseline = monitor.capture_snapshot(
            core_version_id="v1",
            overall_accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
            logical_consistency_score=0.95,
        )

        current = monitor.capture_snapshot(
            core_version_id="v2",
            overall_accuracy=0.88,  # 4% drop
            precision=0.86,  # 4% drop
            recall=0.88,  # No drop
            f1_score=0.87,  # 2% drop - at threshold
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
            logical_consistency_score=0.95,
        )

        regressions = monitor.detect_regressions(current, baseline)

        # Should detect accuracy and precision (f1_score is exactly at threshold)
        assert len(regressions) >= 2
        metrics = {r.metric for r in regressions}
        assert "Overall Accuracy" in metrics
        assert "Precision" in metrics

    def test_detect_no_regression_within_threshold(self):
        """Test no regression detected when within threshold."""
        monitor = PerformanceMonitor(regression_threshold=0.02)

        baseline = monitor.capture_snapshot(
            core_version_id="v1",
            overall_accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
        )

        # Small drop within threshold
        current = monitor.capture_snapshot(
            core_version_id="v2",
            overall_accuracy=0.91,  # 1% drop - within threshold
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
        )

        regressions = monitor.detect_regressions(current, baseline)
        assert len(regressions) == 0

    def test_analyze_trends_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        monitor = PerformanceMonitor()

        trend = monitor.analyze_trends("Overall Accuracy")
        assert trend is None

        # Single snapshot
        monitor.capture_snapshot(
            core_version_id="v1",
            overall_accuracy=0.90,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
        )

        trend = monitor.analyze_trends("Overall Accuracy")
        assert trend is None

    def test_analyze_trends_improving(self):
        """Test detecting improving trend."""
        monitor = PerformanceMonitor(trend_lookback_days=7)

        # Create improving trend
        for i in range(5):
            monitor.capture_snapshot(
                core_version_id=f"v{i}",
                overall_accuracy=0.85 + (i * 0.02),  # Improving
                precision=0.85,
                recall=0.88,
                f1_score=0.865,
                total_rules=100,
                rules_by_layer={},
                avg_confidence=0.85,
            )

        trend = monitor.analyze_trends("Overall Accuracy")

        assert trend is not None
        assert trend.trend_direction == "improving"
        assert trend.change_rate > 0
        assert trend.alert_level == "none"

    def test_analyze_trends_degrading(self):
        """Test detecting degrading trend."""
        monitor = PerformanceMonitor(trend_lookback_days=7)

        # Create degrading trend
        for i in range(5):
            monitor.capture_snapshot(
                core_version_id=f"v{i}",
                overall_accuracy=0.95 - (i * 0.02),  # Degrading
                precision=0.85,
                recall=0.88,
                f1_score=0.865,
                total_rules=100,
                rules_by_layer={},
                avg_confidence=0.85,
            )

        trend = monitor.analyze_trends("Overall Accuracy")

        assert trend is not None
        assert trend.trend_direction == "degrading"
        assert trend.change_rate < 0
        assert trend.alert_level in ["watch", "warning", "critical"]

    def test_analyze_trends_stable(self):
        """Test detecting stable trend."""
        monitor = PerformanceMonitor(trend_lookback_days=7)

        # Create stable trend
        for i in range(5):
            monitor.capture_snapshot(
                core_version_id=f"v{i}",
                overall_accuracy=0.90,  # Stable
                precision=0.85,
                recall=0.88,
                f1_score=0.865,
                total_rules=100,
                rules_by_layer={},
                avg_confidence=0.85,
            )

        trend = monitor.analyze_trends("Overall Accuracy")

        assert trend is not None
        assert trend.trend_direction == "stable"
        assert abs(trend.change_rate) < 0.001
        assert trend.alert_level == "none"

    def test_add_alert(self):
        """Test adding performance alert."""
        monitor = PerformanceMonitor()

        alert = monitor.add_alert(
            severity="high",
            category="regression",
            message="Accuracy dropped below threshold",
            metric="Overall Accuracy",
            details={"baseline": 0.92, "current": 0.88},
        )

        assert alert.severity == "high"
        assert alert.category == "regression"
        assert alert.resolved is False
        assert len(monitor.active_alerts) == 1

    def test_resolve_alert(self):
        """Test resolving performance alert."""
        monitor = PerformanceMonitor()

        alert = monitor.add_alert(
            severity="medium", category="performance", message="Test alert"
        )

        assert monitor.resolve_alert(alert.alert_id)
        assert alert.resolved is True

    def test_resolve_nonexistent_alert(self):
        """Test resolving nonexistent alert."""
        monitor = PerformanceMonitor()

        assert not monitor.resolve_alert("nonexistent-id")

    def test_get_snapshot_history_no_filters(self):
        """Test getting all snapshot history."""
        monitor = PerformanceMonitor()

        for i in range(5):
            monitor.capture_snapshot(
                core_version_id=f"v{i}",
                overall_accuracy=0.90,
                precision=0.85,
                recall=0.88,
                f1_score=0.865,
                total_rules=100,
                rules_by_layer={},
                avg_confidence=0.85,
            )

        history = monitor.get_snapshot_history()
        assert len(history) == 5

    def test_get_snapshot_history_with_limit(self):
        """Test getting limited snapshot history."""
        monitor = PerformanceMonitor()

        for i in range(10):
            monitor.capture_snapshot(
                core_version_id=f"v{i}",
                overall_accuracy=0.90,
                precision=0.85,
                recall=0.88,
                f1_score=0.865,
                total_rules=100,
                rules_by_layer={},
                avg_confidence=0.85,
            )

        history = monitor.get_snapshot_history(limit=3)
        assert len(history) == 3
        assert history[0].core_version_id == "v7"

    def test_get_snapshot_history_with_since(self):
        """Test getting snapshot history since specific date."""
        monitor = PerformanceMonitor()

        base_time = datetime.now()

        for i in range(5):
            snapshot = PerformanceSnapshot(
                timestamp=base_time - timedelta(days=5 - i),
                core_version_id=f"v{i}",
                overall_accuracy=0.90,
                precision=0.85,
                recall=0.88,
                f1_score=0.865,
                total_rules=100,
                rules_by_layer={},
                avg_confidence=0.85,
                query_latency_ms=50.0,
                memory_usage_mb=128.0,
                logical_consistency_score=1.0,
                stratification_violations=0,
                rules_incorporated_today=0,
                rollbacks_today=0,
                test_cases_passing=100,
                test_cases_total=100,
            )
            monitor.snapshot_history.append(snapshot)

        cutoff = base_time - timedelta(days=2)
        history = monitor.get_snapshot_history(since=cutoff)

        # Snapshots at days -2, -1 should match (v3, v4)
        assert len(history) == 2

    def test_generate_report_empty(self):
        """Test generating report with no data."""
        monitor = PerformanceMonitor()

        report = monitor.generate_report()

        assert report.current_snapshot is None
        assert len(report.trends) == 0
        assert len(report.regressions) == 0
        assert len(report.recommendations) > 0

    def test_generate_report_with_data(self):
        """Test generating comprehensive report."""
        monitor = PerformanceMonitor()

        # Add some snapshots
        for i in range(5):
            monitor.capture_snapshot(
                core_version_id=f"v{i}",
                overall_accuracy=0.90 - (i * 0.01),  # Slight degradation
                precision=0.85,
                recall=0.88,
                f1_score=0.865,
                total_rules=100 + i,
                rules_by_layer={"constitutional": 10, "strategic": 20 + i},
                avg_confidence=0.85,
            )

        report = monitor.generate_report()

        assert report.current_snapshot is not None
        assert report.current_snapshot.core_version_id == "v4"
        assert len(report.trends) > 0
        assert len(report.recommendations) > 0

    def test_generate_report_with_regressions(self):
        """Test report includes detected regressions."""
        monitor = PerformanceMonitor(regression_threshold=0.02)

        # Baseline
        monitor.capture_snapshot(
            core_version_id="v1",
            overall_accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
        )

        # Regression
        monitor.capture_snapshot(
            core_version_id="v2",
            overall_accuracy=0.85,  # 7% drop
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
        )

        report = monitor.generate_report()

        assert len(report.regressions) > 0
        assert any("regression" in rec.lower() for rec in report.recommendations)

    def test_recommendations_stratification_violations(self):
        """Test recommendations include stratification violation alerts."""
        monitor = PerformanceMonitor()

        monitor.capture_snapshot(
            core_version_id="v1",
            overall_accuracy=0.90,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
            stratification_violations=5,
        )

        report = monitor.generate_report()

        assert any("stratification" in rec.lower() for rec in report.recommendations)

    def test_recommendations_high_rollback_rate(self):
        """Test recommendations for high rollback rate."""
        monitor = PerformanceMonitor()

        monitor.capture_snapshot(
            core_version_id="v1",
            overall_accuracy=0.90,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
            rules_incorporated_today=10,
            rollbacks_today=3,  # 30% rollback rate
        )

        report = monitor.generate_report()

        assert any("rollback" in rec.lower() for rec in report.recommendations)

    def test_recommendations_low_test_pass_rate(self):
        """Test recommendations for low test pass rate."""
        monitor = PerformanceMonitor()

        monitor.capture_snapshot(
            core_version_id="v1",
            overall_accuracy=0.90,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
            test_cases_passing=85,
            test_cases_total=100,  # 85% pass rate
        )

        report = monitor.generate_report()

        assert any("test" in rec.lower() for rec in report.recommendations)

    def test_report_to_markdown(self):
        """Test generating markdown report."""
        monitor = PerformanceMonitor()

        monitor.capture_snapshot(
            core_version_id="v1",
            overall_accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=100,
            rules_by_layer={"constitutional": 10, "strategic": 30},
            avg_confidence=0.85,
            test_cases_passing=95,
            test_cases_total=100,
        )

        report = monitor.generate_report()
        markdown = report.to_markdown()

        assert "# Performance Report" in markdown
        assert "Current Status" in markdown
        assert "Accuracy:" in markdown
        assert "Recommendations" in markdown
