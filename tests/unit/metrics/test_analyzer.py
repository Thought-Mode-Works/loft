"""Tests for MetricsAnalyzer."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from loft.metrics.analyzer import (
    Anomaly,
    AnomalyReport,
    MetricsAnalyzer,
    ScaleReport,
    TrendAnalysis,
    TrendPoint,
)


class TestTrendPoint:
    """Tests for TrendPoint dataclass."""

    def test_create_trend_point(self):
        """Test creating a trend point."""
        point = TrendPoint(
            timestamp=datetime.now(),
            rules_count=50,
            accuracy=0.85,
            cases_processed=100,
            avg_latency_ms=150.0,
        )

        assert point.rules_count == 50
        assert point.accuracy == 0.85

    def test_trend_point_to_dict(self):
        """Test serialization."""
        point = TrendPoint(
            timestamp=datetime.now(),
            rules_count=25,
            accuracy=0.75,
            cases_processed=50,
            avg_latency_ms=100.0,
        )

        data = point.to_dict()

        assert data["rules_count"] == 25
        assert data["accuracy"] == 0.75


class TestAnomaly:
    """Tests for Anomaly dataclass."""

    def test_create_anomaly(self):
        """Test creating an anomaly."""
        anomaly = Anomaly(
            timestamp=datetime.now(),
            metric_name="latency",
            expected_value=100.0,
            actual_value=500.0,
            deviation_factor=4.0,
            severity="high",
            description="Latency spike detected",
        )

        assert anomaly.severity == "high"
        assert anomaly.deviation_factor == 4.0

    def test_anomaly_to_dict(self):
        """Test serialization."""
        anomaly = Anomaly(
            timestamp=datetime.now(),
            metric_name="accuracy",
            expected_value=0.85,
            actual_value=0.65,
            deviation_factor=2.0,
            severity="medium",
            description="Accuracy drop",
        )

        data = anomaly.to_dict()

        assert data["metric_name"] == "accuracy"
        assert data["severity"] == "medium"


class TestMetricsAnalyzer:
    """Tests for MetricsAnalyzer."""

    def test_create_analyzer(self):
        """Test creating an analyzer."""
        analyzer = MetricsAnalyzer()

        assert analyzer.anomaly_threshold_std == 2.0
        assert analyzer.min_data_points == 5

    def test_analyze_trends_empty(self):
        """Test trend analysis with no data."""
        analyzer = MetricsAnalyzer()

        trends = analyzer.analyze_trends([])

        assert trends.trend_direction == "insufficient_data"

    def test_analyze_trends_improving(self):
        """Test trend analysis with improving accuracy."""
        analyzer = MetricsAnalyzer()

        base_time = datetime.now()
        milestones = [
            {
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
                "rules_count": i * 10,
                "accuracy": 0.5 + i * 0.05,
                "cases_processed": i * 20,
                "timing_aggregates": {"case_processing": {"mean_ms": 100 + i * 10}},
            }
            for i in range(10)
        ]

        trends = analyzer.analyze_trends(milestones)

        assert trends.accuracy_trend == "improving"
        assert len(trends.data_points) == 10

    def test_analyze_trends_degrading(self):
        """Test trend analysis with degrading accuracy."""
        analyzer = MetricsAnalyzer()

        base_time = datetime.now()
        milestones = [
            {
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
                "rules_count": i * 10,
                "accuracy": 0.9 - i * 0.05,
                "cases_processed": i * 20,
                "timing_aggregates": {},
            }
            for i in range(10)
        ]

        trends = analyzer.analyze_trends(milestones)

        assert trends.accuracy_trend == "degrading"

    def test_detect_anomalies_none(self):
        """Test anomaly detection with no anomalies."""
        analyzer = MetricsAnalyzer()

        timing_stats = {
            "operation": {
                "mean": 100.0,
                "std_dev": 10.0,
                "max": 120.0,  # Within 2 std devs
            }
        }

        report = analyzer.detect_anomalies(timing_stats, [])

        assert report.total_anomalies == 0

    def test_detect_anomalies_latency_spike(self):
        """Test anomaly detection with latency spike."""
        analyzer = MetricsAnalyzer()

        timing_stats = {
            "operation": {
                "mean": 100.0,
                "std_dev": 10.0,
                "max": 500.0,  # Way outside normal range
            }
        }

        report = analyzer.detect_anomalies(timing_stats, [])

        assert report.total_anomalies >= 1
        assert any(a.metric_name == "operation" for a in report.anomalies)

    def test_detect_anomalies_accuracy_drop(self):
        """Test anomaly detection with accuracy drop."""
        analyzer = MetricsAnalyzer()

        milestones = [
            {"timestamp": datetime.now().isoformat(), "accuracy": 0.85},
            {
                "timestamp": (datetime.now() + timedelta(minutes=1)).isoformat(),
                "accuracy": 0.60,  # >10% drop
            },
        ]

        report = analyzer.detect_anomalies({}, milestones)

        assert report.total_anomalies >= 1
        assert any(a.metric_name == "accuracy" for a in report.anomalies)

    def test_generate_scale_report(self):
        """Test generating a scale report."""
        analyzer = MetricsAnalyzer()

        base_time = datetime.now()
        metrics_data = {
            "batch_id": "test_batch",
            "started_at": base_time.isoformat(),
            "elapsed_seconds": 300.0,
            "counters": {
                "cases_processed": 100,
                "rules_accepted": 50,
            },
            "gauges": {},
            "timing_stats": {
                "case_processing": {
                    "mean": 150.0,
                    "median": 140.0,
                    "min": 50.0,
                    "max": 500.0,
                    "std_dev": 50.0,
                }
            },
            "milestones": [
                {
                    "timestamp": base_time.isoformat(),
                    "rules_count": 10,
                    "accuracy": 0.7,
                    "cases_processed": 20,
                    "timing_aggregates": {},
                },
                {
                    "timestamp": (base_time + timedelta(minutes=5)).isoformat(),
                    "rules_count": 50,
                    "accuracy": 0.85,
                    "cases_processed": 100,
                    "timing_aggregates": {},
                },
            ],
        }

        report = analyzer.generate_scale_report("test_batch", metrics_data)

        assert report.batch_id == "test_batch"
        assert report.total_cases == 100
        assert report.total_rules == 50
        assert report.initial_accuracy == 0.7
        assert report.final_accuracy == 0.85
        assert abs(report.accuracy_change - 0.15) < 0.001

    def test_generate_scale_report_with_profiler(self):
        """Test scale report generation with profiler data."""
        analyzer = MetricsAnalyzer()

        metrics_data = {
            "batch_id": "test",
            "started_at": datetime.now().isoformat(),
            "elapsed_seconds": 100.0,
            "counters": {},
            "gauges": {},
            "timing_stats": {},
            "milestones": [],
        }

        profiler_data = {
            "summary": {
                "peak_memory_mb": 1024.0,
                "by_operation": {
                    "slow_op": {
                        "avg_time_ms": 2000.0,
                        "count": 10,
                        "total_time_ms": 20000.0,
                    }
                },
            }
        }

        report = analyzer.generate_scale_report("test", metrics_data, profiler_data)

        assert report.peak_memory_mb == 1024.0
        assert len(report.bottlenecks) == 1
        assert report.bottlenecks[0]["operation"] == "slow_op"

    def test_correlation_calculation(self):
        """Test correlation calculation."""
        analyzer = MetricsAnalyzer(min_data_points=3)

        # Perfect positive correlation
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        correlation = analyzer._calculate_correlation(x, y)

        assert abs(correlation - 1.0) < 0.001  # Should be ~1.0

    def test_correlation_insufficient_data(self):
        """Test correlation with insufficient data."""
        analyzer = MetricsAnalyzer(min_data_points=5)

        x = [1.0, 2.0, 3.0]
        y = [2.0, 4.0, 6.0]

        correlation = analyzer._calculate_correlation(x, y)

        assert correlation == 0.0  # Not enough data

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        analyzer = MetricsAnalyzer()

        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

        p50 = analyzer._percentile(values, 50)
        p95 = analyzer._percentile(values, 95)

        assert abs(p50 - 55.0) < 1.0  # Median
        assert p95 >= 90.0

    def test_severity_classification(self):
        """Test anomaly severity classification."""
        analyzer = MetricsAnalyzer()

        assert analyzer._classify_severity(5.0) == "high"
        assert analyzer._classify_severity(3.5) == "medium"
        assert analyzer._classify_severity(2.5) == "low"


class TestScaleReport:
    """Tests for ScaleReport."""

    def test_to_dict(self):
        """Test converting report to dictionary."""
        trends = TrendAnalysis(
            metric_name="test",
            data_points=[],
            trend_direction="stable",
            trend_slope=0.0,
            accuracy_trend="stable",
            latency_trend="stable",
            rules_growth_rate=0.0,
            accuracy_vs_rules_correlation=0.0,
            latency_vs_rules_correlation=0.0,
        )

        anomaly_report = AnomalyReport(
            analyzed_at=datetime.now(),
            total_anomalies=0,
            anomalies=[],
            high_severity_count=0,
            medium_severity_count=0,
            low_severity_count=0,
            recommendations=[],
        )

        report = ScaleReport(
            batch_id="test",
            generated_at=datetime.now(),
            total_cases=100,
            total_rules=50,
            total_runtime_seconds=300.0,
            avg_case_latency_ms=150.0,
            p50_latency_ms=140.0,
            p95_latency_ms=300.0,
            p99_latency_ms=500.0,
            peak_memory_mb=512.0,
            initial_accuracy=0.7,
            final_accuracy=0.85,
            accuracy_change=0.15,
            consistency_score=0.95,
            trends=trends,
            anomaly_report=anomaly_report,
        )

        data = report.to_dict()

        assert data["batch_id"] == "test"
        assert data["total_cases"] == 100
        assert data["accuracy_change"] == 0.15

    def test_save_report(self):
        """Test saving report to file."""
        trends = TrendAnalysis(
            metric_name="test",
            data_points=[],
            trend_direction="stable",
            trend_slope=0.0,
            accuracy_trend="stable",
            latency_trend="stable",
            rules_growth_rate=0.0,
            accuracy_vs_rules_correlation=0.0,
            latency_vs_rules_correlation=0.0,
        )

        anomaly_report = AnomalyReport(
            analyzed_at=datetime.now(),
            total_anomalies=0,
            anomalies=[],
            high_severity_count=0,
            medium_severity_count=0,
            low_severity_count=0,
            recommendations=[],
        )

        report = ScaleReport(
            batch_id="save_test",
            generated_at=datetime.now(),
            total_cases=50,
            total_rules=25,
            total_runtime_seconds=150.0,
            avg_case_latency_ms=100.0,
            p50_latency_ms=90.0,
            p95_latency_ms=200.0,
            p99_latency_ms=300.0,
            peak_memory_mb=256.0,
            initial_accuracy=0.6,
            final_accuracy=0.8,
            accuracy_change=0.2,
            consistency_score=0.9,
            trends=trends,
            anomaly_report=anomaly_report,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            report.save(filepath)

            assert filepath.exists()


class TestAnomalyReport:
    """Tests for AnomalyReport."""

    def test_to_dict(self):
        """Test serialization."""
        report = AnomalyReport(
            analyzed_at=datetime.now(),
            total_anomalies=3,
            anomalies=[
                Anomaly(
                    timestamp=datetime.now(),
                    metric_name="test",
                    expected_value=100.0,
                    actual_value=200.0,
                    deviation_factor=2.0,
                    severity="medium",
                    description="Test anomaly",
                )
            ],
            high_severity_count=0,
            medium_severity_count=1,
            low_severity_count=2,
            recommendations=["Fix the issue"],
        )

        data = report.to_dict()

        assert data["total_anomalies"] == 3
        assert len(data["anomalies"]) == 1
        assert data["medium_severity_count"] == 1
