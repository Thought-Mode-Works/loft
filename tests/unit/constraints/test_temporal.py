"""
Unit tests for Temporal Consistency Invariance (Phase 7).
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from loft.constraints.temporal import TemporalConsistencyTester, TemporalField


def test_detect_temporal_fields() -> None:
    """Auto-detect date fields in case."""
    case = {
        "contract": {
            "formation_date": "2020-01-15",
            "performance_deadline": "2021-01-15",
        },
        "parties": ["alice", "bob"],
        "amount": 500,
    }
    tester = TemporalConsistencyTester([])
    fields = tester.detect_temporal_fields(case)
    assert len(fields) == 2
    paths = [f.path for f in fields]
    assert "contract.formation_date" in paths
    assert "contract.performance_deadline" in paths


def test_uniform_shift_application() -> None:
    """Verify uniform shift correctly transforms dates."""
    case = {"date": "2020-01-15", "other": "value"}
    tester = TemporalConsistencyTester([TemporalField(path="date", field_type="date")])
    shifted = tester.apply_uniform_shift(case, timedelta(days=365))

    expected_date = (
        (datetime.fromisoformat("2020-01-15") + timedelta(days=365)).date().isoformat()
    )
    assert shifted["date"] == expected_date
    assert shifted["other"] == "value"


def test_shift_invariant_rule_passes() -> None:
    """Shift-invariant rule passes consistency test."""

    # Rule depends only on duration, not absolute dates
    def duration_rule(case: Dict[str, Any]) -> Dict[str, bool]:
        start = datetime.fromisoformat(case["start"])
        end = datetime.fromisoformat(case["end"])
        return {"valid": (end - start).days <= 365}

    test_cases = [{"start": "2020-01-01", "end": "2020-06-01"}]

    tester = TemporalConsistencyTester(
        [
            TemporalField(path="start", field_type="date"),
            TemporalField(path="end", field_type="date"),
        ]
    )
    report = tester.test_shift_invariance(duration_rule, test_cases)
    assert report.is_consistent
    assert report.shift_invariant


def test_absolute_date_rule_fails() -> None:
    """Rule depending on absolute dates fails shift invariance."""

    def absolute_rule(case: Dict[str, Any]) -> Dict[str, bool]:
        dt = datetime.fromisoformat(case["date"])
        return {"valid": dt.year >= 2020}  # Depends on absolute year

    test_cases = [{"date": "2020-06-01"}]  # Valid initially
    # Shift backwards by 365 days -> 2019 -> Invalid

    tester = TemporalConsistencyTester([TemporalField(path="date", field_type="date")])
    report = tester.test_shift_invariance(absolute_rule, test_cases)
    assert not report.is_consistent
    assert not report.shift_invariant
    assert len(report.violations) > 0
    assert report.violations[0].violation_type == "shift_invariance"


def test_nested_temporal_fields() -> None:
    """Test detection and shifting of nested temporal fields."""
    case = {
        "meta": {
            "dates": [
                {"event": "start", "timestamp": "2023-01-01T10:00:00"},
                {"event": "end", "timestamp": "2023-01-02T10:00:00"},
            ]
        }
    }
    tester = TemporalConsistencyTester([])
    fields = tester.detect_temporal_fields(case)
    # detected: meta.dates[0].timestamp, meta.dates[1].timestamp
    assert len(fields) == 2

    # Update tester with detected fields for shift
    tester.temporal_fields = fields

    shifted = tester.apply_uniform_shift(case, timedelta(hours=1))
    assert shifted["meta"]["dates"][0]["timestamp"] == "2023-01-01T11:00:00"
    assert shifted["meta"]["dates"][1]["timestamp"] == "2023-01-02T11:00:00"


def test_ignore_missing_fields() -> None:
    """Test that applying shift ignores missing fields gracefully."""
    tester = TemporalConsistencyTester(
        [TemporalField(path="missing.field", field_type="date")]
    )
    case = {"other": "value"}
    shifted = tester.apply_uniform_shift(case, timedelta(days=1))
    assert shifted == case


def test_order_preservation_passes() -> None:
    """Test that purely order-dependent rule passes order preservation."""

    # First to file wins (independent of exact timing)
    def priority_rule(case: Dict[str, Any]) -> Dict[str, str]:
        d1 = datetime.fromisoformat(case["filing1"])
        d2 = datetime.fromisoformat(case["filing2"])
        return {"winner": "party1" if d1 < d2 else "party2"}

    test_cases = [{"filing1": "2020-01-01", "filing2": "2020-01-02"}]

    tester = TemporalConsistencyTester(
        [
            TemporalField(path="filing1", field_type="date"),
            TemporalField(path="filing2", field_type="date"),
        ]
    )

    # Scale x2: 2020-01-01 -> 2020-01-01, 2020-01-02 -> 2020-01-03
    # Order preserved.
    report = tester.test_order_preservation(priority_rule, test_cases)
    assert report.order_invariant


def test_order_preservation_fails() -> None:
    """Test that duration-dependent rule fails order preservation."""

    # Must be within 10 days
    def duration_rule(case: Dict[str, Any]) -> Dict[str, bool]:
        start = datetime.fromisoformat(case["start"])
        end = datetime.fromisoformat(case["end"])
        return {"valid": (end - start).days <= 10}

    test_cases = [{"start": "2020-01-01", "end": "2020-01-10"}]  # 9 days diff, Valid

    tester = TemporalConsistencyTester(
        [
            TemporalField(path="start", field_type="date"),
            TemporalField(path="end", field_type="date"),
        ]
    )

    # Scale x2: start->2020-01-01, end->2020-01-19 (18 days diff) -> Invalid
    report = tester.test_order_preservation(duration_rule, test_cases)
    assert not report.order_invariant
    assert len(report.violations) > 0
