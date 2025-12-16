"""Tests for coverage tracker."""

from datetime import datetime
import pytest

from loft.iteration.coverage_tracker import CoverageTracker, CoverageMetrics
from loft.symbolic.asp_rule import ASPRule, RuleMetadata
from loft.symbolic.stratification import StratificationLevel


@pytest.fixture
def domain_predicates():
    """Sample domain predicates."""
    return ["contract_valid", "breach_occurred", "damages_owed", "liable_party"]


@pytest.fixture
def test_cases():
    """Sample test cases."""
    return [
        {"id": "case_001", "scenario_id": "breach"},
        {"id": "case_002", "scenario_id": "breach"},
        {"id": "case_003", "scenario_id": "formation"},
    ]


@pytest.fixture
def tracker(domain_predicates, test_cases):
    """Coverage tracker fixture."""
    return CoverageTracker(domain_predicates, test_cases)


@pytest.fixture
def sample_rule():
    """Sample ASP rule."""
    return ASPRule(
        rule_id="rule_001",
        asp_text="contract_valid(X) :- offer(X), acceptance(X).",
        stratification_level=StratificationLevel.OPERATIONAL,
        confidence=0.85,
        metadata=RuleMetadata(
            provenance="test", timestamp=datetime.utcnow().isoformat()
        ),
    )


def test_coverage_metrics_properties():
    """Test coverage metrics property calculations."""
    metrics = CoverageMetrics(
        timestamp=datetime.utcnow(),
        predicates_total=10,
        predicates_covered=5,
        covered_predicates=["p1", "p2", "p3", "p4", "p5"],
        cases_total=20,
        cases_with_predictions=15,
        scenarios_total=5,
        scenarios_covered=3,
        total_rules=10,
    )

    assert metrics.predicate_coverage == 0.5
    assert metrics.case_coverage == 0.75
    assert metrics.scenario_coverage == 0.6


def test_coverage_metrics_to_dict():
    """Test metrics serialization."""
    timestamp = datetime.utcnow()
    metrics = CoverageMetrics(
        timestamp=timestamp,
        predicates_total=10,
        predicates_covered=5,
        covered_predicates=["p1"],
        cases_total=20,
        cases_with_predictions=15,
        scenarios_total=5,
        scenarios_covered=3,
        total_rules=10,
    )

    data = metrics.to_dict()

    assert data["timestamp"] == timestamp.isoformat()
    assert data["predicates_total"] == 10
    assert data["predicates_covered"] == 5


def test_coverage_metrics_from_dict():
    """Test metrics deserialization."""
    timestamp = datetime.utcnow()
    data = {
        "timestamp": timestamp.isoformat(),
        "predicates_total": 10,
        "predicates_covered": 5,
        "covered_predicates": ["p1"],
        "cases_total": 20,
        "cases_with_predictions": 15,
        "scenarios_total": 5,
        "scenarios_covered": 3,
        "total_rules": 10,
        "rules_by_layer": {},
    }

    metrics = CoverageMetrics.from_dict(data)

    assert metrics.predicates_total == 10
    assert metrics.predicates_covered == 5


def test_tracker_initialization(tracker, domain_predicates, test_cases):
    """Test tracker initialization."""
    assert tracker.domain_predicates == set(domain_predicates)
    assert tracker.test_cases == test_cases
    assert tracker.history == []
    assert tracker.newly_covered_predicates == []


def test_record_metrics(tracker):
    """Test recording coverage metrics."""
    covered = {"contract_valid", "breach_occurred"}

    metrics = tracker.record_metrics(
        covered_predicates=covered,
        cases_with_predictions=2,
        scenarios_covered=1,
        total_rules=3,
    )

    assert len(tracker.history) == 1
    assert metrics.predicates_covered == 2
    assert metrics.cases_with_predictions == 2
    assert sorted(tracker.newly_covered_predicates) == sorted(list(covered))


def test_record_metrics_tracks_new_predicates(tracker):
    """Test that newly covered predicates are tracked."""
    # First recording
    covered1 = {"contract_valid"}
    tracker.record_metrics(
        covered_predicates=covered1,
        cases_with_predictions=1,
        scenarios_covered=1,
        total_rules=1,
    )

    assert set(tracker.newly_covered_predicates) == covered1

    # Second recording with additional predicate
    covered2 = {"contract_valid", "breach_occurred"}
    tracker.record_metrics(
        covered_predicates=covered2,
        cases_with_predictions=2,
        scenarios_covered=1,
        total_rules=2,
    )

    assert set(tracker.newly_covered_predicates) == {"breach_occurred"}


def test_current_metrics(tracker):
    """Test getting current metrics."""
    assert tracker.current_metrics is None

    covered = {"contract_valid"}
    tracker.record_metrics(
        covered_predicates=covered,
        cases_with_predictions=1,
        scenarios_covered=1,
        total_rules=1,
    )

    assert tracker.current_metrics is not None
    assert tracker.current_metrics.predicates_covered == 1


def test_current_coverage(tracker):
    """Test current coverage calculation."""
    assert tracker.current_coverage == 0.0

    # 2 out of 4 predicates covered
    covered = {"contract_valid", "breach_occurred"}
    tracker.record_metrics(
        covered_predicates=covered,
        cases_with_predictions=1,
        scenarios_covered=1,
        total_rules=1,
    )

    assert tracker.current_coverage == 0.5


def test_is_monotonic_empty_history(tracker):
    """Test monotonicity check with empty history."""
    assert tracker.is_monotonic() is True


def test_is_monotonic_increasing(tracker):
    """Test monotonicity with increasing coverage."""
    tracker.record_metrics(
        covered_predicates={"contract_valid"},
        cases_with_predictions=1,
        scenarios_covered=1,
        total_rules=1,
    )

    tracker.record_metrics(
        covered_predicates={"contract_valid", "breach_occurred"},
        cases_with_predictions=2,
        scenarios_covered=1,
        total_rules=2,
    )

    assert tracker.is_monotonic() is True


def test_is_monotonic_stable(tracker):
    """Test monotonicity with stable coverage."""
    covered = {"contract_valid"}

    tracker.record_metrics(
        covered_predicates=covered,
        cases_with_predictions=1,
        scenarios_covered=1,
        total_rules=1,
    )

    tracker.record_metrics(
        covered_predicates=covered,
        cases_with_predictions=1,
        scenarios_covered=1,
        total_rules=1,
    )

    assert tracker.is_monotonic() is True


def test_is_monotonic_violation(tracker):
    """Test monotonicity violation detection."""
    tracker.record_metrics(
        covered_predicates={"contract_valid", "breach_occurred"},
        cases_with_predictions=2,
        scenarios_covered=1,
        total_rules=2,
    )

    # Decrease coverage
    tracker.record_metrics(
        covered_predicates={"contract_valid"},
        cases_with_predictions=1,
        scenarios_covered=1,
        total_rules=1,
    )

    assert tracker.is_monotonic() is False


def test_get_coverage_trend_insufficient_data(tracker):
    """Test trend with insufficient data."""
    assert tracker.get_coverage_trend() == "insufficient_data"

    tracker.record_metrics(
        covered_predicates={"contract_valid"},
        cases_with_predictions=1,
        scenarios_covered=1,
        total_rules=1,
    )

    assert tracker.get_coverage_trend() == "insufficient_data"


def test_get_coverage_trend_increasing(tracker):
    """Test increasing coverage trend."""
    for i in range(5):
        covered = set(list(tracker.domain_predicates)[: i + 1])
        tracker.record_metrics(
            covered_predicates=covered,
            cases_with_predictions=i + 1,
            scenarios_covered=1,
            total_rules=i + 1,
        )

    assert tracker.get_coverage_trend() == "increasing"


def test_get_coverage_trend_stable(tracker):
    """Test stable coverage trend."""
    covered = {"contract_valid"}

    for _ in range(5):
        tracker.record_metrics(
            covered_predicates=covered,
            cases_with_predictions=1,
            scenarios_covered=1,
            total_rules=1,
        )

    assert tracker.get_coverage_trend() == "stable"


def test_get_uncovered_predicates(tracker):
    """Test getting uncovered predicates."""
    # Initially all uncovered
    assert set(tracker.get_uncovered_predicates()) == set(tracker.domain_predicates)

    # Cover some
    covered = {"contract_valid", "breach_occurred"}
    tracker.record_metrics(
        covered_predicates=covered,
        cases_with_predictions=1,
        scenarios_covered=1,
        total_rules=1,
    )

    uncovered = set(tracker.get_uncovered_predicates())
    expected = {"damages_owed", "liable_party"}

    assert uncovered == expected


def test_extract_predicates_from_rules(tracker, sample_rule):
    """Test extracting predicates from rules."""
    rules = [sample_rule]

    predicates = tracker.extract_predicates_from_rules(rules)

    assert "contract_valid" in predicates


def test_generate_coverage_report_empty(tracker):
    """Test report generation with no data."""
    report = tracker.generate_coverage_report()

    assert "No coverage data available" in report


def test_generate_coverage_report(tracker):
    """Test report generation with data."""
    covered = {"contract_valid", "breach_occurred"}
    tracker.record_metrics(
        covered_predicates=covered,
        cases_with_predictions=2,
        scenarios_covered=1,
        total_rules=3,
        rules_by_layer={"operational": 3},
    )

    report = tracker.generate_coverage_report()

    assert "Coverage Report" in report
    assert "50.0%" in report  # 2/4 predicates
    assert "operational" in report


def test_save_and_load_history(tracker, tmp_path):
    """Test saving and loading history."""
    covered = {"contract_valid"}
    tracker.record_metrics(
        covered_predicates=covered,
        cases_with_predictions=1,
        scenarios_covered=1,
        total_rules=1,
    )

    # Save
    filepath = tmp_path / "coverage.json"
    tracker.save_history(str(filepath))

    assert filepath.exists()

    # Load
    test_cases = [{"id": "case_001"}]
    loaded_tracker = CoverageTracker.load_history(str(filepath), test_cases)

    assert len(loaded_tracker.history) == 1
    assert loaded_tracker.history[0].predicates_covered == 1
