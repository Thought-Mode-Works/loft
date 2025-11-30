"""Tests for MetricsCollector."""

import tempfile
import time
from datetime import datetime
from pathlib import Path

from loft.metrics.collector import (
    AggregatedMetrics,
    MetricsCollector,
    MetricsSample,
    PerformanceTimer,
)


class TestMetricsSample:
    """Tests for MetricsSample dataclass."""

    def test_create_sample(self):
        """Test creating a metrics sample."""
        sample = MetricsSample(
            timestamp=datetime.now(),
            metric_name="test_metric",
            value=42.5,
            unit="ms",
            tags={"type": "timing"},
        )

        assert sample.metric_name == "test_metric"
        assert sample.value == 42.5
        assert sample.unit == "ms"
        assert sample.tags["type"] == "timing"

    def test_sample_roundtrip(self):
        """Test serialization roundtrip."""
        original = MetricsSample(
            timestamp=datetime.now(),
            metric_name="roundtrip_test",
            value=100.0,
            unit="count",
            tags={"domain": "contracts"},
        )

        data = original.to_dict()
        restored = MetricsSample.from_dict(data)

        assert restored.metric_name == original.metric_name
        assert restored.value == original.value
        assert restored.unit == original.unit
        assert restored.tags == original.tags


class TestPerformanceTimer:
    """Tests for PerformanceTimer."""

    def test_timer_basic(self):
        """Test basic timer functionality."""
        timer = PerformanceTimer("test_op")

        with timer:
            time.sleep(0.01)  # 10ms

        assert timer.elapsed_ms >= 10
        assert timer.elapsed_ms < 100  # Should be much less than 100ms

    def test_timer_with_collector(self):
        """Test timer with collector integration."""
        collector = MetricsCollector(batch_id="test")

        with PerformanceTimer("timed_op", collector) as timer:
            time.sleep(0.005)

        assert timer.elapsed_ms >= 5
        assert "timed_op" in collector.timings
        assert len(collector.timings["timed_op"]) == 1


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_create_collector(self):
        """Test creating a collector."""
        collector = MetricsCollector(batch_id="test_batch")

        assert collector.batch_id == "test_batch"
        assert len(collector.samples) == 0
        assert len(collector.timings) == 0

    def test_record_sample(self):
        """Test recording a sample."""
        collector = MetricsCollector(batch_id="test")

        sample = collector.record_sample("accuracy", 0.85, tags={"domain": "contracts"})

        assert sample.metric_name == "accuracy"
        assert sample.value == 0.85
        assert len(collector.samples) == 1

    def test_record_timing(self):
        """Test recording timing measurements."""
        collector = MetricsCollector(batch_id="test")

        collector.record_timing("rule_generation", 150.5)
        collector.record_timing("rule_generation", 200.0)
        collector.record_timing("validation", 50.0)

        assert len(collector.timings["rule_generation"]) == 2
        assert len(collector.timings["validation"]) == 1

    def test_increment_counter(self):
        """Test counter increment."""
        collector = MetricsCollector(batch_id="test")

        value1 = collector.increment_counter("cases_processed")
        value2 = collector.increment_counter("cases_processed")
        value3 = collector.increment_counter("cases_processed", 5)

        assert value1 == 1
        assert value2 == 2
        assert value3 == 7
        assert collector.counters["cases_processed"] == 7

    def test_set_gauge(self):
        """Test gauge setting."""
        collector = MetricsCollector(batch_id="test")

        collector.set_gauge("memory_mb", 512.0)
        collector.set_gauge("memory_mb", 600.0)

        assert collector.gauges["memory_mb"] == 600.0

    def test_time_operation(self):
        """Test context manager for timing."""
        collector = MetricsCollector(batch_id="test")

        with collector.time_operation("slow_operation") as timer:
            time.sleep(0.01)

        assert timer.elapsed_ms >= 10
        assert "slow_operation" in collector.timings

    def test_record_milestone(self):
        """Test milestone recording."""
        collector = MetricsCollector(batch_id="test")

        # Add some timings first
        collector.record_timing("case_processing", 100.0)
        collector.record_timing("case_processing", 150.0)

        milestone = collector.record_milestone(
            milestone_name="10_rules",
            rules_count=10,
            cases_processed=20,
            accuracy=0.75,
            additional_metrics={"domain": "contracts"},
        )

        assert milestone["milestone_name"] == "10_rules"
        assert milestone["rules_count"] == 10
        assert milestone["accuracy"] == 0.75
        assert "timing_aggregates" in milestone
        assert len(collector.milestones) == 1

    def test_get_timing_stats(self):
        """Test getting timing statistics."""
        collector = MetricsCollector(batch_id="test")

        collector.record_timing("operation", 100.0)
        collector.record_timing("operation", 200.0)
        collector.record_timing("operation", 150.0)

        stats = collector.get_timing_stats("operation")

        assert stats is not None
        assert stats.count == 3
        assert stats.total == 450.0
        assert stats.mean == 150.0
        assert stats.min_value == 100.0
        assert stats.max_value == 200.0

    def test_get_timing_stats_empty(self):
        """Test getting stats for non-existent timing."""
        collector = MetricsCollector(batch_id="test")

        stats = collector.get_timing_stats("nonexistent")

        assert stats is None

    def test_to_dict(self):
        """Test converting collector to dictionary."""
        collector = MetricsCollector(batch_id="test")
        collector.record_timing("op1", 100.0)
        collector.increment_counter("cases")
        collector.set_gauge("memory", 512.0)

        data = collector.to_dict()

        assert data["batch_id"] == "test"
        assert "elapsed_seconds" in data
        assert "counters" in data
        assert "gauges" in data
        assert "timing_stats" in data

    def test_save_and_load(self):
        """Test saving and loading metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create and populate collector
            collector = MetricsCollector(
                batch_id="save_test",
                output_dir=output_dir,
            )
            collector.record_timing("op", 100.0)
            collector.increment_counter("count")
            collector.record_milestone(
                milestone_name="test",
                rules_count=5,
                cases_processed=10,
                accuracy=0.8,
            )

            # Save
            filepath = collector.save()

            # Load
            loaded = MetricsCollector.load(filepath)

            assert loaded.batch_id == "save_test"
            assert loaded.counters["count"] == 1
            assert len(loaded.milestones) == 1

    def test_callbacks(self):
        """Test sample and milestone callbacks."""
        collector = MetricsCollector(batch_id="test")

        samples_received = []
        milestones_received = []

        collector.set_callbacks(
            on_sample=lambda s: samples_received.append(s),
            on_milestone=lambda m: milestones_received.append(m),
        )

        collector.record_sample("test", 1.0)
        collector.record_milestone("m1", 10, 20, 0.8)

        assert len(samples_received) == 1
        assert len(milestones_received) == 1


class TestAggregatedMetrics:
    """Tests for AggregatedMetrics."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = AggregatedMetrics(
            metric_name="test",
            count=10,
            total=1000.0,
            mean=100.0,
            median=95.0,
            min_value=50.0,
            max_value=200.0,
            std_dev=25.0,
            unit="ms",
        )

        data = metrics.to_dict()

        assert data["metric_name"] == "test"
        assert data["count"] == 10
        assert data["mean"] == 100.0
        assert data["unit"] == "ms"
