"""
Integration tests for performance monitoring system.

Tests end-to-end monitoring workflows with realistic scenarios.
"""

from datetime import datetime, timedelta

import pytest

from loft.monitoring.performance_monitor import PerformanceMonitor
from loft.monitoring.performance_schemas import PerformanceSnapshot


class TestPerformanceMonitoringIntegration:
    """Integration tests for performance monitoring."""

    def test_complete_monitoring_workflow(self):
        """Test complete monitoring workflow from capture to report."""
        monitor = PerformanceMonitor(regression_threshold=0.02, trend_lookback_days=7)

        # Simulate a week of performance data
        base_accuracy = 0.92
        for day in range(7):
            # Slight degradation over time
            accuracy = base_accuracy - (day * 0.005)

            monitor.capture_snapshot(
                core_version_id=f"v1.{day}",
                overall_accuracy=accuracy,
                precision=0.90,
                recall=0.88,
                f1_score=0.89,
                total_rules=100 + day,
                rules_by_layer={
                    "constitutional": 10,
                    "strategic": 30 + day,
                    "tactical": 40,
                    "operational": 20,
                },
                avg_confidence=0.85,
                query_latency_ms=50.0 + day,
                memory_usage_mb=128.0 + (day * 2),
                test_cases_passing=95,
                test_cases_total=100,
            )

        # Generate comprehensive report
        report = monitor.generate_report()

        # Verify report structure
        assert report.current_snapshot is not None
        assert report.current_snapshot.core_version_id == "v1.6"
        assert len(report.trends) > 0
        assert len(report.recommendations) > 0

        # Verify trends are detected
        accuracy_trend = report.trends.get("Overall Accuracy")
        assert accuracy_trend is not None
        assert accuracy_trend.trend_direction == "degrading"

        # Verify markdown generation works
        markdown = report.to_markdown()
        assert "# Performance Report" in markdown
        assert "Accuracy:" in markdown

    def test_regression_detection_and_alerting(self):
        """Test regression detection triggers appropriate alerts."""
        monitor = PerformanceMonitor(regression_threshold=0.02)

        # Baseline performance
        monitor.capture_snapshot(
            core_version_id="v1.0",
            overall_accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=100,
            rules_by_layer={"constitutional": 10, "strategic": 30},
            avg_confidence=0.85,
        )

        # Introduce regression
        monitor.capture_snapshot(
            core_version_id="v1.1",
            overall_accuracy=0.87,  # 5% drop - triggers critical
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=101,
            rules_by_layer={"constitutional": 10, "strategic": 31},
            avg_confidence=0.85,
        )

        # Generate report
        report = monitor.generate_report()

        # Verify regression detected
        assert len(report.regressions) > 0
        accuracy_regression = next(
            (r for r in report.regressions if r.metric == "Overall Accuracy"), None
        )
        assert accuracy_regression is not None
        assert accuracy_regression.severity == "critical"
        assert accuracy_regression.degradation == pytest.approx(0.05, abs=0.001)

        # Verify recommendations include regression alert
        assert any("regression" in rec.lower() for rec in report.recommendations)

    def test_trend_analysis_across_multiple_metrics(self):
        """Test trend analysis tracks multiple metrics correctly."""
        monitor = PerformanceMonitor(trend_lookback_days=5)

        # Create different trends for different metrics
        for i in range(6):
            monitor.capture_snapshot(
                core_version_id=f"v{i}",
                overall_accuracy=0.90 + (i * 0.01),  # Improving
                precision=0.85 - (i * 0.005),  # Degrading
                recall=0.88,  # Stable
                f1_score=0.865,
                total_rules=100,
                rules_by_layer={},
                avg_confidence=0.85,
            )

        report = monitor.generate_report()

        # Check accuracy trend
        accuracy_trend = report.trends.get("Overall Accuracy")
        assert accuracy_trend is not None
        assert accuracy_trend.trend_direction == "improving"
        assert accuracy_trend.alert_level == "none"

        # Check precision trend
        precision_trend = report.trends.get("Precision")
        assert precision_trend is not None
        assert precision_trend.trend_direction == "degrading"

        # Check recall trend
        recall_trend = report.trends.get("Recall")
        assert recall_trend is not None
        assert recall_trend.trend_direction == "stable"

    def test_alert_lifecycle(self):
        """Test complete alert lifecycle from creation to resolution."""
        monitor = PerformanceMonitor()

        # Create alerts
        alert1 = monitor.add_alert(
            severity="high",
            category="regression",
            message="Accuracy dropped significantly",
            metric="Overall Accuracy",
        )

        alert2 = monitor.add_alert(
            severity="medium",
            category="performance",
            message="Query latency increased",
            metric="Query Latency",
        )

        # Verify alerts are active
        assert len(monitor.active_alerts) == 2
        assert not alert1.resolved
        assert not alert2.resolved

        # Resolve first alert
        assert monitor.resolve_alert(alert1.alert_id)
        assert alert1.resolved

        # Generate report with remaining alert
        report = monitor.generate_report()
        active_unresolved = [a for a in report.active_alerts if not a.resolved]
        assert len(active_unresolved) == 1

    def test_snapshot_history_management(self):
        """Test snapshot history is managed correctly over time."""
        monitor = PerformanceMonitor(max_snapshots=10)

        # Capture more snapshots than limit
        for i in range(15):
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

        # Verify history is trimmed
        assert len(monitor.snapshot_history) == 10
        assert monitor.snapshot_history[0].core_version_id == "v5"
        assert monitor.snapshot_history[-1].core_version_id == "v14"

        # Verify recent history can be retrieved
        recent = monitor.get_snapshot_history(limit=3)
        assert len(recent) == 3
        assert recent[-1].core_version_id == "v14"

    def test_stratification_violation_monitoring(self):
        """Test monitoring detects and reports stratification violations."""
        monitor = PerformanceMonitor()

        # Capture snapshot with violations
        monitor.capture_snapshot(
            core_version_id="v1.0",
            overall_accuracy=0.90,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            total_rules=100,
            rules_by_layer={"constitutional": 10, "strategic": 30},
            avg_confidence=0.85,
            stratification_violations=3,
        )

        report = monitor.generate_report()

        # Verify violation appears in recommendations
        violation_recs = [
            r for r in report.recommendations if "stratification" in r.lower()
        ]
        assert len(violation_recs) > 0
        assert "3" in violation_recs[0]

    def test_rollback_rate_monitoring(self):
        """Test monitoring tracks and alerts on high rollback rates."""
        monitor = PerformanceMonitor()

        # High rollback rate scenario
        monitor.capture_snapshot(
            core_version_id="v1.0",
            overall_accuracy=0.90,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
            rules_incorporated_today=10,
            rollbacks_today=4,  # 40% rollback rate
        )

        report = monitor.generate_report()

        # Verify rollback rate appears in recommendations
        rollback_recs = [r for r in report.recommendations if "rollback" in r.lower()]
        assert len(rollback_recs) > 0

    def test_test_coverage_monitoring(self):
        """Test monitoring tracks test coverage and pass rates."""
        monitor = PerformanceMonitor()

        # Low pass rate scenario
        monitor.capture_snapshot(
            core_version_id="v1.0",
            overall_accuracy=0.90,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
            test_cases_passing=80,
            test_cases_total=100,  # 80% pass rate
        )

        report = monitor.generate_report()

        # Verify test pass rate appears in recommendations
        test_recs = [r for r in report.recommendations if "test" in r.lower()]
        assert len(test_recs) > 0

    def test_performance_degradation_recovery(self):
        """Test monitoring detects both degradation and recovery."""
        monitor = PerformanceMonitor(regression_threshold=0.02)

        # Initial good performance
        monitor.capture_snapshot(
            core_version_id="v1.0",
            overall_accuracy=0.92,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
        )

        # Degradation
        monitor.capture_snapshot(
            core_version_id="v1.1",
            overall_accuracy=0.85,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            total_rules=100,
            rules_by_layer={},
            avg_confidence=0.85,
        )

        # Recovery
        for i in range(4):
            monitor.capture_snapshot(
                core_version_id=f"v1.{i + 2}",
                overall_accuracy=0.85 + (i * 0.02),  # Improving
                precision=0.90,
                recall=0.88,
                f1_score=0.89,
                total_rules=100,
                rules_by_layer={},
                avg_confidence=0.85,
            )

        report = monitor.generate_report()

        # Verify recovery is reflected in trends
        accuracy_trend = report.trends.get("Overall Accuracy")
        assert accuracy_trend is not None
        assert accuracy_trend.trend_direction == "improving"

    def test_multi_day_trend_analysis(self):
        """Test trend analysis over extended period with realistic timestamps."""
        monitor = PerformanceMonitor(trend_lookback_days=7)

        base_time = datetime.now()

        # Create snapshots over 10 days
        for day in range(10):
            timestamp = base_time - timedelta(days=10 - day)

            snapshot = PerformanceSnapshot(
                timestamp=timestamp,
                core_version_id=f"v1.{day}",
                overall_accuracy=0.90 - (day * 0.005),  # Gradual decline
                precision=0.85,
                recall=0.88,
                f1_score=0.865,
                total_rules=100 + day,
                rules_by_layer={"constitutional": 10, "strategic": 30 + day},
                avg_confidence=0.85,
                query_latency_ms=50.0,
                memory_usage_mb=128.0,
                logical_consistency_score=1.0,
                stratification_violations=0,
                rules_incorporated_today=0,
                rollbacks_today=0,
                test_cases_passing=95,
                test_cases_total=100,
            )
            monitor.snapshot_history.append(snapshot)

        # Analyze trends with lookback
        trend = monitor.analyze_trends("Overall Accuracy", lookback_days=7)

        assert trend is not None
        assert trend.trend_direction == "degrading"
        assert trend.change_rate < 0

    def test_comprehensive_report_generation(self):
        """Test comprehensive report with all features enabled."""
        monitor = PerformanceMonitor(regression_threshold=0.02, trend_lookback_days=5)

        # Build realistic history
        base_accuracy = 0.92

        for i in range(6):
            monitor.capture_snapshot(
                core_version_id=f"v1.{i}",
                overall_accuracy=base_accuracy - (i * 0.003),
                precision=0.90,
                recall=0.88,
                f1_score=0.89,
                total_rules=100 + i,
                rules_by_layer={
                    "constitutional": 10,
                    "strategic": 30 + i,
                    "tactical": 40,
                    "operational": 20,
                },
                avg_confidence=0.85,
                query_latency_ms=50.0 + i,
                memory_usage_mb=128.0,
                logical_consistency_score=0.98,
                stratification_violations=0,
                rules_incorporated_today=2,
                rollbacks_today=0,
                test_cases_passing=98,
                test_cases_total=100,
            )

        # Add an alert
        monitor.add_alert(
            severity="medium",
            category="performance",
            message="Query latency increasing",
            metric="Query Latency",
        )

        # Generate comprehensive report
        report = monitor.generate_report()

        # Verify all components present
        assert report.current_snapshot is not None
        assert len(report.trends) >= 3
        assert len(report.active_alerts) == 1
        assert len(report.recommendations) > 0

        # Verify serialization works
        report_dict = report.to_dict()
        assert "generated_at" in report_dict
        assert "current_snapshot" in report_dict
        assert "trends" in report_dict
        assert "recommendations" in report_dict

        # Verify markdown generation works
        markdown = report.to_markdown()
        assert "# Performance Report" in markdown
        assert "Key Metrics" in markdown
        assert "Trends" in markdown
        assert "Active Alerts" in markdown
        assert "Recommendations" in markdown
