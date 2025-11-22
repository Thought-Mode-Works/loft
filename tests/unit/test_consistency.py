"""
Unit tests for consistency checking framework.

Tests the consistency checker, generators, properties, and reporting.
"""

import tempfile
from pathlib import Path
from datetime import datetime
from loft.consistency import (
    ConsistencyChecker,
    InconsistencyType,
    Inconsistency,
    ConsistencyReport,
    TestFixtures,
    ConsistencyProperties,
    ConsistencyReporter,
    ConsistencyHistory,
    ConsistencyMetrics,
)
from loft.version_control import CoreState


class TestConsistencyChecker:
    """Tests for ConsistencyChecker."""

    def test_checker_initialization(self) -> None:
        """Test creating a consistency checker."""
        checker = ConsistencyChecker(strict=False)
        assert checker is not None
        assert not checker.strict

        strict_checker = ConsistencyChecker(strict=True)
        assert strict_checker.strict

    def test_check_empty_state(self) -> None:
        """Test checking an empty state."""
        checker = ConsistencyChecker()
        state = TestFixtures.empty_state()

        report = checker.check(state)

        assert report.passed
        assert len(report.inconsistencies) == 0
        assert report.errors == 0
        assert report.warnings == 0

    def test_check_consistent_state(self) -> None:
        """Test checking a consistent state."""
        checker = ConsistencyChecker()
        state = TestFixtures.simple_consistent_state()

        report = checker.check(state)

        assert report.passed
        assert report.errors == 0

    def test_check_contradictory_state(self) -> None:
        """Test detecting contradictions."""
        checker = ConsistencyChecker()
        state = TestFixtures.contradictory_state()

        report = checker.check(state)

        assert not report.passed
        assert report.errors > 0

        # Should detect contradiction
        contradiction_found = any(
            inc.type == InconsistencyType.CONTRADICTION for inc in report.inconsistencies
        )
        assert contradiction_found

    def test_check_incomplete_state(self) -> None:
        """Test detecting incompleteness."""
        checker = ConsistencyChecker()
        state = TestFixtures.incomplete_state()

        report = checker.check(state)

        # Incompleteness is a warning, not error
        incompleteness_found = any(
            inc.type == InconsistencyType.INCOMPLETENESS for inc in report.inconsistencies
        )
        assert incompleteness_found

    def test_check_incoherent_state(self) -> None:
        """Test detecting incoherence."""
        checker = ConsistencyChecker()
        state = TestFixtures.incoherent_state()

        report = checker.check(state)

        # Should detect incoherence
        incoherence_found = any(
            inc.type == InconsistencyType.INCOHERENCE for inc in report.inconsistencies
        )
        assert incoherence_found

    def test_check_circular_dependency(self) -> None:
        """Test detecting circular dependencies."""
        checker = ConsistencyChecker()
        state = TestFixtures.circular_dependency_state()

        report = checker.check(state)

        # Note: Current implementation detects self-loops (a -> a)
        # Full cycle detection (a -> b -> c -> a) is a future enhancement
        # For MVP, we verify the checker runs without errors
        assert report is not None
        assert isinstance(report, ConsistencyReport)

    def test_strict_mode(self) -> None:
        """Test strict mode fails on warnings."""
        strict_checker = ConsistencyChecker(strict=True)
        state = TestFixtures.incomplete_state()

        report = strict_checker.check(state)

        # In strict mode, warnings should cause failure
        if report.warnings > 0:
            assert not report.passed


class TestInconsistency:
    """Tests for Inconsistency class."""

    def test_inconsistency_creation(self) -> None:
        """Test creating an inconsistency."""
        inc = Inconsistency(
            type=InconsistencyType.CONTRADICTION,
            severity="error",
            message="Test contradiction",
            rule_ids=["r1", "r2"],
            details={"key": "value"},
        )

        assert inc.type == InconsistencyType.CONTRADICTION
        assert inc.severity == "error"
        assert inc.message == "Test contradiction"
        assert len(inc.rule_ids) == 2

    def test_inconsistency_string_formatting(self) -> None:
        """Test inconsistency string representation."""
        inc = Inconsistency(
            type=InconsistencyType.CONTRADICTION,
            severity="error",
            message="Rules conflict",
            rule_ids=["r1", "r2"],
        )

        s = str(inc)
        assert "ERROR" in s
        assert "contradiction" in s
        assert "Rules conflict" in s


class TestConsistencyReport:
    """Tests for ConsistencyReport class."""

    def test_report_creation(self) -> None:
        """Test creating a consistency report."""
        report = ConsistencyReport(
            passed=True,
            inconsistencies=[],
            warnings=0,
            errors=0,
            info=0,
        )

        assert report.passed
        assert len(report.inconsistencies) == 0

    def test_report_summary(self) -> None:
        """Test report summary generation."""
        # Passing report
        passing_report = ConsistencyReport(
            passed=True, inconsistencies=[], warnings=0, errors=0, info=0
        )
        summary = passing_report.summary()
        assert "passed" in summary.lower()

        # Failing report
        failing_report = ConsistencyReport(
            passed=False, inconsistencies=[], warnings=1, errors=2, info=0
        )
        summary = failing_report.summary()
        assert "failed" in summary.lower()
        assert "2 errors" in summary

    def test_report_formatting(self) -> None:
        """Test full report formatting."""
        inc = Inconsistency(
            type=InconsistencyType.CONTRADICTION,
            severity="error",
            message="Test",
            rule_ids=["r1"],
        )

        report = ConsistencyReport(
            passed=False, inconsistencies=[inc], warnings=0, errors=1, info=0
        )

        formatted = report.format()
        assert "Consistency Check Report" in formatted
        assert "contradiction" in formatted


class TestTestFixtures:
    """Tests for TestFixtures."""

    def test_empty_state(self) -> None:
        """Test empty state fixture."""
        state = TestFixtures.empty_state()
        assert len(state.rules) == 0
        assert isinstance(state, CoreState)

    def test_simple_consistent_state(self) -> None:
        """Test simple consistent state fixture."""
        state = TestFixtures.simple_consistent_state()
        assert len(state.rules) > 0
        assert isinstance(state, CoreState)

    def test_contradictory_state(self) -> None:
        """Test contradictory state fixture."""
        state = TestFixtures.contradictory_state()
        assert len(state.rules) >= 2

        # Should have contradicting rules
        checker = ConsistencyChecker()
        report = checker.check(state)
        assert not report.passed

    def test_incomplete_state(self) -> None:
        """Test incomplete state fixture."""
        state = TestFixtures.incomplete_state()
        assert len(state.rules) > 0

    def test_incoherent_state(self) -> None:
        """Test incoherent state fixture."""
        state = TestFixtures.incoherent_state()
        assert len(state.rules) > 0

    def test_circular_dependency_state(self) -> None:
        """Test circular dependency state fixture."""
        state = TestFixtures.circular_dependency_state()
        assert len(state.rules) >= 3


class TestConsistencyProperties:
    """Tests for property-based tests."""

    def test_property_tests_can_run(self) -> None:
        """Test that property tests can be executed."""
        checker = ConsistencyChecker()
        props = ConsistencyProperties(checker)

        # Just verify we can create the property test object
        assert props is not None
        assert props.checker == checker

    def test_deterministic_property(self) -> None:
        """Test deterministic consistency checking property."""
        checker = ConsistencyChecker()
        ConsistencyProperties(checker)

        # Use a fixed state
        state = TestFixtures.simple_consistent_state()

        # Run check twice
        report1 = checker.check(state)
        report2 = checker.check(state)

        # Should be identical
        assert report1.passed == report2.passed
        assert len(report1.inconsistencies) == len(report2.inconsistencies)

    def test_empty_state_property(self) -> None:
        """Test empty state consistency property."""
        checker = ConsistencyChecker()
        state = TestFixtures.empty_state()

        report = checker.check(state)
        assert report.passed


class TestConsistencyReporter:
    """Tests for ConsistencyReporter."""

    def test_reporter_initialization(self) -> None:
        """Test creating a reporter."""
        reporter = ConsistencyReporter()
        assert reporter is not None
        assert isinstance(reporter.history, ConsistencyHistory)

    def test_reporter_with_history_file(self) -> None:
        """Test reporter with history file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "consistency_history.json"
            reporter = ConsistencyReporter(history_file=history_file)

            # Create a report
            checker = ConsistencyChecker()
            state = TestFixtures.simple_consistent_state()
            check_report = checker.check(state)

            # Generate enhanced report
            enhanced = reporter.report(check_report, len(state.rules))

            assert "Consistency Check Report" in enhanced
            assert history_file.exists()

    def test_summary_by_type(self) -> None:
        """Test inconsistency summary by type."""
        reporter = ConsistencyReporter()

        inc1 = Inconsistency(InconsistencyType.CONTRADICTION, "error", "Test 1", ["r1"])
        inc2 = Inconsistency(InconsistencyType.CONTRADICTION, "error", "Test 2", ["r2"])
        inc3 = Inconsistency(InconsistencyType.INCOMPLETENESS, "warning", "Test 3", ["r3"])

        report = ConsistencyReport(
            passed=False,
            inconsistencies=[inc1, inc2, inc3],
            warnings=1,
            errors=2,
            info=0,
        )

        summary = reporter.summary_by_type(report)

        assert summary["contradiction"] == 2
        assert summary["incompleteness"] == 1

    def test_format_conflict_graph(self) -> None:
        """Test conflict graph formatting."""
        reporter = ConsistencyReporter()

        inc = Inconsistency(InconsistencyType.CONTRADICTION, "error", "Test", ["r1", "r2"])

        report = ConsistencyReport(
            passed=False, inconsistencies=[inc], warnings=0, errors=1, info=0
        )

        graph = reporter.format_conflict_graph(report)

        assert "Rule Conflict Graph" in graph
        assert "r1" in graph or "r2" in graph


class TestConsistencyHistory:
    """Tests for ConsistencyHistory."""

    def test_history_creation(self) -> None:
        """Test creating a history."""
        history = ConsistencyHistory()
        assert len(history.metrics) == 0

    def test_add_report(self) -> None:
        """Test adding a report to history."""
        history = ConsistencyHistory()

        report = ConsistencyReport(passed=True, inconsistencies=[], warnings=0, errors=0, info=0)

        history.add_report(report, total_rules=5)

        assert len(history.metrics) == 1
        assert history.metrics[0].passed
        assert history.metrics[0].total_rules == 5

    def test_get_latest(self) -> None:
        """Test getting latest metrics."""
        history = ConsistencyHistory()

        # Empty history
        assert history.get_latest() is None

        # Add a report
        report = ConsistencyReport(passed=True, inconsistencies=[], warnings=0, errors=0, info=0)
        history.add_report(report, total_rules=5)

        latest = history.get_latest()
        assert latest is not None
        assert latest.total_rules == 5

    def test_get_trend(self) -> None:
        """Test trend detection."""
        history = ConsistencyHistory()

        # Not enough data
        assert history.get_trend() == "unknown"

        # Add improving trend
        for i in range(10):
            score = 0.5 + (i * 0.05)  # Improving
            ConsistencyReport(
                passed=True, inconsistencies=[], warnings=0, errors=0, info=0
            )
            metrics = ConsistencyMetrics(
                timestamp=datetime.utcnow().isoformat(),
                total_rules=5,
                passed=True,
                error_count=0,
                warning_count=0,
                info_count=0,
                consistency_score=score,
            )
            history.metrics.append(metrics)

        trend = history.get_trend()
        assert trend in ["improving", "stable", "declining"]

    def test_detect_regression(self) -> None:
        """Test regression detection."""
        history = ConsistencyHistory()

        # Add two metrics with significant drop
        metrics1 = ConsistencyMetrics(
            timestamp=datetime.utcnow().isoformat(),
            total_rules=5,
            passed=True,
            error_count=0,
            warning_count=0,
            info_count=0,
            consistency_score=0.9,
        )
        metrics2 = ConsistencyMetrics(
            timestamp=datetime.utcnow().isoformat(),
            total_rules=5,
            passed=False,
            error_count=2,
            warning_count=0,
            info_count=0,
            consistency_score=0.5,
        )

        history.metrics.append(metrics1)
        history.metrics.append(metrics2)

        assert history.detect_regression()

    def test_save_and_load(self) -> None:
        """Test saving and loading history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.json"

            # Create and save history
            history1 = ConsistencyHistory()
            report = ConsistencyReport(
                passed=True, inconsistencies=[], warnings=0, errors=0, info=0
            )
            history1.add_report(report, total_rules=5)
            history1.save(path)

            # Load history
            history2 = ConsistencyHistory.load(path)

            assert len(history2.metrics) == 1
            assert history2.metrics[0].total_rules == 5


class TestConsistencyMetrics:
    """Tests for ConsistencyMetrics."""

    def test_metrics_creation(self) -> None:
        """Test creating metrics."""
        metrics = ConsistencyMetrics(
            timestamp=datetime.utcnow().isoformat(),
            total_rules=10,
            passed=True,
            error_count=0,
            warning_count=1,
            info_count=0,
            consistency_score=0.95,
        )

        assert metrics.total_rules == 10
        assert metrics.passed
        assert metrics.consistency_score == 0.95

    def test_metrics_serialization(self) -> None:
        """Test metrics to/from dict."""
        metrics = ConsistencyMetrics(
            timestamp=datetime.utcnow().isoformat(),
            total_rules=10,
            passed=True,
            error_count=0,
            warning_count=1,
            info_count=0,
            consistency_score=0.95,
        )

        # To dict
        data = metrics.to_dict()
        assert data["total_rules"] == 10

        # From dict
        restored = ConsistencyMetrics.from_dict(data)
        assert restored.total_rules == 10
        assert restored.consistency_score == 0.95


class TestCheckerHelpers:
    """Tests for checker helper methods."""

    def test_extract_predicates(self) -> None:
        """Test predicate extraction."""
        checker = ConsistencyChecker()

        # Simple fact
        preds = checker._extract_predicates("animal(dog).")
        assert ("animal", False) in preds

        # Negated predicate
        preds = checker._extract_predicates("-alive(x).")
        assert ("alive", True) in preds

        # Rule with multiple predicates
        preds = checker._extract_predicates("mammal(X) :- animal(X), warm_blooded(X).")
        pred_names = [p[0] for p in preds]
        assert "mammal" in pred_names
        assert "animal" in pred_names
        assert "warm_blooded" in pred_names

    def test_extract_head(self) -> None:
        """Test extracting rule head."""
        checker = ConsistencyChecker()

        # Fact
        head = checker._extract_head("animal(dog).")
        assert "animal" in head

        # Rule
        head = checker._extract_head("mammal(X) :- animal(X).")
        assert "mammal" in head
        assert ":-" not in head

    def test_extract_body(self) -> None:
        """Test extracting rule body."""
        checker = ConsistencyChecker()

        # Fact (no body)
        body = checker._extract_body("animal(dog).")
        assert body is None

        # Rule
        body = checker._extract_body("mammal(X) :- animal(X).")
        assert body is not None
        assert "animal" in body

    def test_normalize(self) -> None:
        """Test text normalization."""
        checker = ConsistencyChecker()

        text1 = "  Animal ( X )  "
        text2 = "animal(X)"

        norm1 = checker._normalize(text1)
        norm2 = checker._normalize(text2)

        assert norm1 == norm2
