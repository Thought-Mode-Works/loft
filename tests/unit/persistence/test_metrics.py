"""
Unit tests for persistence metrics module.

Tests PersistenceMetrics, PersistenceMetricsCollector, and BaselineReport.

Issue #254: ASP Rule Persistence Validation & Baseline Metrics.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

from loft.persistence.metrics import (
    BaselineReport,
    PersistenceMetrics,
    PersistenceMetricsCollector,
    create_metrics_collector,
)


class TestPersistenceMetrics:
    """Tests for PersistenceMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = PersistenceMetrics()

        assert metrics.save_time_ms == 0.0
        assert metrics.load_time_ms == 0.0
        assert metrics.total_rules == 0
        assert metrics.rules_by_layer == {}
        assert metrics.save_errors == 0
        assert metrics.metadata_fields_preserved == 1.0

    def test_custom_values(self):
        """Test metrics with custom values."""
        metrics = PersistenceMetrics(
            save_time_ms=150.5,
            load_time_ms=75.2,
            total_rules=25,
            rules_by_layer={"tactical": 15, "operational": 10},
            save_errors=1,
        )

        assert metrics.save_time_ms == 150.5
        assert metrics.load_time_ms == 75.2
        assert metrics.total_rules == 25
        assert metrics.rules_by_layer["tactical"] == 15

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = PersistenceMetrics(
            save_time_ms=100.0,
            total_rules=10,
            rules_by_layer={"tactical": 10},
        )

        data = metrics.to_dict()

        assert data["save_time_ms"] == 100.0
        assert data["total_rules"] == 10
        assert "timestamp" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "save_time_ms": 200.0,
            "load_time_ms": 100.0,
            "snapshot_time_ms": 50.0,
            "total_rules": 20,
            "rules_by_layer": {"constitutional": 5},
            "file_sizes_bytes": {},
            "total_size_bytes": 1024,
            "snapshot_count": 2,
            "snapshot_total_size_bytes": 2048,
            "rules_with_metadata": 15,
            "metadata_fields_preserved": 0.95,
            "save_errors": 0,
            "load_errors": 0,
            "recovery_attempts": 0,
            "recovery_successes": 0,
            "timestamp": datetime.now().isoformat(),
        }

        metrics = PersistenceMetrics.from_dict(data)

        assert metrics.save_time_ms == 200.0
        assert metrics.total_rules == 20
        assert metrics.rules_by_layer["constitutional"] == 5


class TestBaselineReport:
    """Tests for BaselineReport dataclass."""

    def test_default_values(self):
        """Test default report values."""
        report = BaselineReport()

        assert report.total_cycles == 0
        assert report.avg_save_time_ms == 0.0
        assert report.scalability_data == []

    def test_to_markdown(self):
        """Test markdown report generation."""
        report = BaselineReport(
            total_cycles=100,
            total_rules_processed=500,
            avg_save_time_ms=50.0,
            avg_load_time_ms=25.0,
            save_error_rate=0.01,
            metadata_preservation_rate=0.98,
        )

        markdown = report.to_markdown()

        assert "# ASP Persistence Baseline Metrics" in markdown
        assert "Total Cycles: 100" in markdown
        assert "50.00 ms" in markdown

    def test_to_markdown_with_scalability(self):
        """Test markdown with scalability data."""
        report = BaselineReport(
            total_cycles=10,
            scalability_data=[
                {"rules": 100, "save_ms": 50.0, "load_ms": 25.0},
                {"rules": 500, "save_ms": 200.0, "load_ms": 100.0},
            ],
        )

        markdown = report.to_markdown()

        assert "## Scalability" in markdown
        assert "| 100 |" in markdown
        assert "| 500 |" in markdown

    def test_to_dict(self):
        """Test serialization to dictionary."""
        report = BaselineReport(total_cycles=50)
        data = report.to_dict()

        assert data["total_cycles"] == 50


class TestPersistenceMetricsCollector:
    """Tests for PersistenceMetricsCollector."""

    def test_initialization(self):
        """Test collector initialization."""
        collector = PersistenceMetricsCollector()

        assert collector.collected_metrics == []
        assert collector.error_log == []

    def test_factory_function(self):
        """Test factory function creates collector."""
        collector = create_metrics_collector()

        assert isinstance(collector, PersistenceMetricsCollector)

    def test_measure_save_cycle(self):
        """Test save cycle measurement."""
        collector = PersistenceMetricsCollector()

        # Create mock manager
        mock_manager = MagicMock()
        mock_manager.base_dir = "/tmp/test_rules"
        mock_manager.get_stats.return_value = {"snapshots_count": 2}

        # Create mock rules with layer enum simulation
        class MockLayer:
            value = "tactical"

        mock_rule = MagicMock()
        mock_rule.metadata = {"provenance": "test"}

        rules_by_layer = {MockLayer(): [mock_rule, mock_rule]}

        with patch("pathlib.Path.glob", return_value=[]):
            metrics = collector.measure_save_cycle(mock_manager, rules_by_layer)

        assert metrics.total_rules == 2
        assert metrics.rules_with_metadata == 2
        assert "tactical" in metrics.rules_by_layer
        assert len(collector.collected_metrics) == 1

    def test_measure_load_cycle(self):
        """Test load cycle measurement."""
        collector = PersistenceMetricsCollector()

        # Create mock manager
        mock_manager = MagicMock()

        class MockLayer:
            value = "operational"

        mock_rule = MagicMock()
        mock_rule.metadata = None

        mock_manager.load_all_rules.return_value = {MockLayer(): [mock_rule]}

        metrics = collector.measure_load_cycle(mock_manager)

        assert metrics.total_rules == 1
        assert metrics.rules_with_metadata == 0
        assert metrics.load_time_ms >= 0

    def test_measure_save_cycle_with_error(self):
        """Test save cycle handles errors."""
        collector = PersistenceMetricsCollector()

        mock_manager = MagicMock()
        mock_manager.base_dir = "/tmp/test"
        mock_manager.save_all_rules.side_effect = Exception("Save failed")

        with patch("pathlib.Path.glob", return_value=[]):
            metrics = collector.measure_save_cycle(mock_manager, {})

        assert metrics.save_errors == 1
        assert len(collector.error_log) == 1
        assert "save_error" in collector.error_log[0]["type"]

    def test_verify_roundtrip_integrity(self):
        """Test roundtrip integrity verification."""
        collector = PersistenceMetricsCollector()

        class MockLayer:
            value = "tactical"

        mock_rule = MagicMock()
        mock_rule.asp_text = "valid(X) :- test(X)."

        original = {MockLayer(): [mock_rule]}
        loaded = {MockLayer(): [mock_rule]}

        results = collector.verify_roundtrip_integrity(original, loaded)

        assert results["rules_match"] is True

    def test_verify_roundtrip_integrity_missing_rules(self):
        """Test roundtrip detects missing rules."""
        collector = PersistenceMetricsCollector()

        class MockLayer:
            value = "tactical"

        mock_rule1 = MagicMock()
        mock_rule1.asp_text = "rule1(X)."
        mock_rule2 = MagicMock()
        mock_rule2.asp_text = "rule2(X)."

        original = {MockLayer(): [mock_rule1, mock_rule2]}
        loaded = {MockLayer(): [mock_rule1]}  # Missing rule2

        results = collector.verify_roundtrip_integrity(original, loaded)

        assert results["rules_match"] is False
        assert len(results["missing_rules"]) > 0

    def test_generate_baseline_report_empty(self):
        """Test report generation with no metrics."""
        collector = PersistenceMetricsCollector()
        report = collector.generate_baseline_report()

        assert report.total_cycles == 0

    def test_generate_baseline_report(self):
        """Test baseline report generation."""
        collector = PersistenceMetricsCollector()

        # Add some metrics
        collector.collected_metrics = [
            PersistenceMetrics(
                save_time_ms=100.0,
                load_time_ms=50.0,
                total_rules=10,
                total_size_bytes=1000,
                rules_with_metadata=8,
                metadata_fields_preserved=0.8,
            ),
            PersistenceMetrics(
                save_time_ms=150.0,
                load_time_ms=75.0,
                total_rules=20,
                total_size_bytes=2000,
                rules_with_metadata=18,
                metadata_fields_preserved=0.9,
            ),
        ]

        report = collector.generate_baseline_report()

        assert report.total_cycles == 2
        assert report.avg_save_time_ms == 125.0
        assert report.avg_load_time_ms == 62.5
        assert report.total_rules_processed == 30

    def test_reset(self):
        """Test collector reset."""
        collector = PersistenceMetricsCollector()
        collector.collected_metrics = [PersistenceMetrics()]
        collector.error_log = [{"error": "test"}]

        collector.reset()

        assert collector.collected_metrics == []
        assert collector.error_log == []


class TestBaselineReportScalability:
    """Tests for scalability data in baseline reports."""

    def test_scalability_data_grouping(self):
        """Test that scalability data groups by rule count."""
        collector = PersistenceMetricsCollector()

        # Add metrics with same rule count
        collector.collected_metrics = [
            PersistenceMetrics(total_rules=100, save_time_ms=50.0, load_time_ms=25.0),
            PersistenceMetrics(total_rules=100, save_time_ms=60.0, load_time_ms=30.0),
            PersistenceMetrics(total_rules=200, save_time_ms=100.0, load_time_ms=50.0),
        ]

        report = collector.generate_baseline_report()

        # Should have 2 scalability points (100 and 200 rules)
        assert len(report.scalability_data) == 2

        # Find the 100-rule entry
        entry_100 = next(
            (e for e in report.scalability_data if e["rules"] == 100), None
        )
        assert entry_100 is not None
        assert entry_100["save_ms"] == 55.0  # Average of 50 and 60
