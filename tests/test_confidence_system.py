"""
Unit tests for confidence scoring and calibration system.

Tests all components: schemas, aggregator, calibrator, gate, and tracker.
"""

import pytest
from datetime import datetime

from loft.validation.confidence_schemas import (
    AggregatedConfidence,
    CalibrationPoint,
    CalibrationReport,
    GateDecision,
    ConfidenceTrends,
)
from loft.validation.confidence_aggregator import ConfidenceAggregator
from loft.validation.confidence_calibrator import ConfidenceCalibrator
from loft.validation.confidence_gate import ConfidenceGate
from loft.validation.confidence_tracker import ConfidenceTracker


# ===== Confidence Schemas Tests =====


def test_aggregated_confidence_creation():
    """Test creating AggregatedConfidence."""
    conf = AggregatedConfidence(
        score=0.85,
        components={"generation": 0.8, "syntax": 1.0, "empirical": 0.8},
        weights={"generation": 0.2, "syntax": 0.2, "empirical": 0.6},
        variance=0.05,
        is_reliable=True,
    )

    assert conf.score == 0.85
    assert conf.is_reliable is True
    assert len(conf.components) == 3
    assert "generation" in conf.components


def test_aggregated_confidence_explanation():
    """Test explanation generation."""
    conf = AggregatedConfidence(
        score=0.85,
        components={"generation": 0.8, "empirical": 0.9},
        weights={"generation": 0.3, "empirical": 0.7},
        variance=0.05,
        is_reliable=True,
    )

    explanation = conf.explanation()
    assert "Overall Confidence: 0.85" in explanation
    assert "Generation:" in explanation
    assert "Empirical:" in explanation
    assert "reliable" in explanation


def test_calibration_point_creation():
    """Test creating CalibrationPoint."""
    point = CalibrationPoint(predicted=0.9, actual=0.85, rule_id="rule_1")

    assert point.predicted == 0.9
    assert point.actual == 0.85
    assert point.rule_id == "rule_1"
    assert isinstance(point.timestamp, datetime)


def test_calibration_report_improvement():
    """Test CalibrationReport improvement calculation."""
    report = CalibrationReport(
        method="isotonic",
        num_points=50,
        before_mse=0.05,
        after_mse=0.02,
        before_ece=0.10,
        after_ece=0.05,
    )

    assert report.improvement == 0.5  # 50% improvement
    assert "50.0%" in report.summary()


def test_gate_decision_creation():
    """Test creating GateDecision."""
    decision = GateDecision(
        action="accept",
        reasoning="High confidence",
        raw_confidence=0.88,
        calibrated_confidence=0.85,
        threshold=0.8,
        target_layer="tactical",
        rule_impact="medium",
    )

    assert decision.action == "accept"
    assert decision.calibrated_confidence == 0.85


def test_confidence_trends_summary():
    """Test ConfidenceTrends summary."""
    trends = ConfidenceTrends(
        mean_confidence=0.82,
        median_confidence=0.85,
        std_confidence=0.10,
        mean_variance=0.05,
        num_reliable=80,
        num_total=100,
        reliability_rate=0.8,
    )

    summary = trends.summary()
    assert "Mean: 0.82" in summary
    assert "80.0%" in summary


# ===== Confidence Aggregator Tests =====


def test_aggregator_initialization():
    """Test ConfidenceAggregator initialization."""
    aggregator = ConfidenceAggregator()

    assert aggregator.weights["empirical"] == 0.40  # Highest weight
    assert aggregator.variance_threshold == 0.1


def test_aggregator_basic_aggregation():
    """Test basic confidence aggregation."""
    aggregator = ConfidenceAggregator()

    result = aggregator.aggregate(
        generation_confidence=0.8,
        syntax_valid=True,
        semantic_valid=True,
        empirical_accuracy=0.85,
        consensus_strength=0.9,
    )

    assert isinstance(result, AggregatedConfidence)
    assert 0.0 <= result.score <= 1.0
    assert len(result.components) == 5
    assert result.components["syntax"] == 1.0
    assert result.components["semantic"] == 1.0


def test_aggregator_missing_components():
    """Test aggregation with missing optional components."""
    aggregator = ConfidenceAggregator()

    result = aggregator.aggregate(
        generation_confidence=0.7,
        syntax_valid=True,
        semantic_valid=False,
        # No empirical or consensus
    )

    assert isinstance(result, AggregatedConfidence)
    assert len(result.components) == 3  # Only generation, syntax, semantic
    assert result.components["semantic"] == 0.0  # Invalid


def test_aggregator_variance_calculation():
    """Test variance calculation."""
    aggregator = ConfidenceAggregator()

    # High agreement
    result_low_var = aggregator.aggregate(
        generation_confidence=0.85,
        syntax_valid=True,
        semantic_valid=True,
    )
    assert result_low_var.variance < 0.1

    # Low agreement
    result_high_var = aggregator.aggregate(
        generation_confidence=0.3,
        syntax_valid=True,
        semantic_valid=False,
    )
    assert result_high_var.variance > 0.1


def test_aggregator_custom_weights():
    """Test aggregator with custom weights."""
    custom_weights = {
        "generation": 0.5,
        "syntax": 0.1,
        "semantic": 0.1,
        "empirical": 0.2,
        "consensus": 0.1,
    }

    aggregator = ConfidenceAggregator(weights=custom_weights)

    assert aggregator.weights["generation"] == 0.5


# ===== Confidence Calibrator Tests =====


def test_calibrator_initialization():
    """Test ConfidenceCalibrator initialization."""
    calibrator = ConfidenceCalibrator(min_calibration_points=10)

    assert calibrator.min_calibration_points == 10
    assert len(calibrator.calibration_data) == 0
    assert calibrator.calibration_function is None


def test_calibrator_record():
    """Test recording calibration points."""
    calibrator = ConfidenceCalibrator()

    calibrator.record(0.9, 0.85, "rule_1")
    calibrator.record(0.7, 0.72, "rule_2")

    assert len(calibrator.calibration_data) == 2
    assert calibrator.calibration_data[0].predicted == 0.9
    assert calibrator.calibration_data[1].actual == 0.72


def test_calibrator_insufficient_data():
    """Test calibration with insufficient data."""
    calibrator = ConfidenceCalibrator(min_calibration_points=10)

    # Record only 5 points
    for i in range(5):
        calibrator.record(0.8, 0.75, f"rule_{i}")

    with pytest.raises(ValueError, match="Need at least 10 calibration points"):
        calibrator.calibrate()


def test_calibrator_linear_calibration():
    """Test linear calibration."""
    calibrator = ConfidenceCalibrator(min_calibration_points=10)

    # Record points with systematic overconfidence
    for i in range(15):
        predicted = 0.5 + i * 0.03
        actual = predicted - 0.1  # Systematically 0.1 lower
        calibrator.record(predicted, actual, f"rule_{i}")

    report = calibrator.calibrate(method="linear")

    assert isinstance(report, CalibrationReport)
    assert report.method == "linear"
    assert report.num_points == 15
    assert report.after_ece < report.before_ece  # Calibration should improve


def test_calibrator_identity_calibration():
    """Test identity calibration (no adjustment)."""
    calibrator = ConfidenceCalibrator(min_calibration_points=10)

    for i in range(15):
        predicted = 0.5 + i * 0.03
        calibrator.record(predicted, predicted, f"rule_{i}")  # Perfect calibration

    report = calibrator.calibrate(method="identity")

    assert report.method == "identity"
    # MSE and ECE should be very low since data is already well-calibrated
    assert report.before_mse < 0.01


def test_calibrator_score_clamping():
    """Test that calibrated scores are clamped to [0, 1]."""
    calibrator = ConfidenceCalibrator(min_calibration_points=10)

    for i in range(15):
        calibrator.record(0.8, 0.75, f"rule_{i}")

    calibrator.calibrate(method="linear")

    # Test clamping
    assert calibrator.calibrate_score(-0.5) == 0.0
    assert calibrator.calibrate_score(1.5) == 1.0
    assert 0.0 <= calibrator.calibrate_score(0.5) <= 1.0


def test_calibrator_no_function_warning():
    """Test calibrate_score without calibration function."""
    calibrator = ConfidenceCalibrator()

    # Should return raw score with warning
    assert calibrator.calibrate_score(0.75) == 0.75


# ===== Confidence Gate Tests =====


def test_gate_initialization():
    """Test ConfidenceGate initialization."""
    gate = ConfidenceGate()

    assert gate.thresholds["operational"] == 0.6
    assert gate.thresholds["tactical"] == 0.8
    assert gate.thresholds["strategic"] == 0.9
    assert gate.flag_margin == 0.05


def test_gate_accept_decision():
    """Test gate accepts high-confidence rule."""
    gate = ConfidenceGate()

    confidence = AggregatedConfidence(
        score=0.88,
        components={"generation": 0.85, "empirical": 0.90},
        weights={"generation": 0.4, "empirical": 0.6},
        variance=0.02,
        is_reliable=True,
    )

    decision = gate.should_accept(
        confidence, target_layer="tactical", rule_impact="medium"
    )

    assert decision.action == "accept"
    assert decision.threshold == 0.8
    assert "meets threshold" in decision.reasoning.lower()


def test_gate_reject_decision():
    """Test gate rejects low-confidence rule."""
    gate = ConfidenceGate()

    confidence = AggregatedConfidence(
        score=0.55,
        components={"generation": 0.50, "empirical": 0.60},
        weights={"generation": 0.4, "empirical": 0.6},
        variance=0.05,
        is_reliable=True,
    )

    decision = gate.should_accept(
        confidence, target_layer="tactical", rule_impact="medium"
    )

    assert decision.action == "reject"
    assert "below threshold" in decision.reasoning.lower()


def test_gate_flag_decision():
    """Test gate flags borderline rule."""
    gate = ConfidenceGate()

    confidence = AggregatedConfidence(
        score=0.77,  # Just below 0.8 threshold but within flag_margin
        components={"generation": 0.75, "empirical": 0.79},
        weights={"generation": 0.4, "empirical": 0.6},
        variance=0.02,
        is_reliable=True,
    )

    decision = gate.should_accept(
        confidence, target_layer="tactical", rule_impact="medium"
    )

    assert decision.action == "flag_for_review"
    assert "flagging for human review" in decision.reasoning.lower()


def test_gate_impact_adjustment():
    """Test threshold adjustment based on rule impact."""
    gate = ConfidenceGate()

    confidence = AggregatedConfidence(
        score=0.75,
        components={"generation": 0.75, "empirical": 0.75},
        weights={"generation": 0.5, "empirical": 0.5},
        variance=0.0,
        is_reliable=True,
    )

    # Low impact: threshold reduced by 0.1
    decision_low = gate.should_accept(
        confidence, target_layer="tactical", rule_impact="low"
    )
    assert decision_low.action == "accept"  # 0.75 >= 0.7

    # High impact: threshold increased by 0.1
    decision_high = gate.should_accept(
        confidence, target_layer="tactical", rule_impact="high"
    )
    assert decision_high.action == "reject"  # 0.75 < 0.9


def test_gate_constitutional_layer():
    """Test constitutional layer always requires review."""
    gate = ConfidenceGate()

    confidence = AggregatedConfidence(
        score=0.99,  # Very high confidence
        components={"generation": 0.99},
        weights={"generation": 1.0},
        variance=0.0,
        is_reliable=True,
    )

    decision = gate.should_accept(
        confidence, target_layer="constitutional", rule_impact="low"
    )

    assert decision.action == "flag_for_review"
    assert "constitutional" in decision.reasoning.lower()


def test_gate_batch_evaluate():
    """Test batch evaluation."""
    gate = ConfidenceGate()

    confidences = [
        AggregatedConfidence(
            score=0.85,
            components={"gen": 0.85},
            weights={"gen": 1.0},
            variance=0.0,
            is_reliable=True,
        ),
        AggregatedConfidence(
            score=0.55,
            components={"gen": 0.55},
            weights={"gen": 1.0},
            variance=0.0,
            is_reliable=True,
        ),
    ]

    decisions = gate.batch_evaluate(confidences, target_layer="tactical")

    assert len(decisions) == 2
    assert decisions[0].action == "accept"
    assert decisions[1].action == "reject"


def test_gate_statistics():
    """Test gate decision statistics."""
    gate = ConfidenceGate()

    decisions = [
        GateDecision(
            action="accept",
            reasoning="",
            raw_confidence=0.85,
            calibrated_confidence=0.85,
            threshold=0.8,
            target_layer="tactical",
            rule_impact="medium",
        ),
        GateDecision(
            action="reject",
            reasoning="",
            raw_confidence=0.55,
            calibrated_confidence=0.55,
            threshold=0.8,
            target_layer="tactical",
            rule_impact="medium",
        ),
        GateDecision(
            action="flag_for_review",
            reasoning="",
            raw_confidence=0.77,
            calibrated_confidence=0.77,
            threshold=0.8,
            target_layer="tactical",
            rule_impact="medium",
        ),
    ]

    stats = gate.get_statistics(decisions)

    assert stats["total"] == 3
    assert stats["accepted"] == 1
    assert stats["rejected"] == 1
    assert stats["flagged"] == 1
    assert stats["acceptance_rate"] == 1 / 3


# ===== Confidence Tracker Tests =====


def test_tracker_initialization():
    """Test ConfidenceTracker initialization."""
    tracker = ConfidenceTracker(history_limit=100)

    assert tracker.history_limit == 100
    assert len(tracker.history) == 0


def test_tracker_record():
    """Test recording confidence scores."""
    tracker = ConfidenceTracker()

    confidence = AggregatedConfidence(
        score=0.85,
        components={"generation": 0.85},
        weights={"generation": 1.0},
        variance=0.0,
        is_reliable=True,
    )

    tracker.record(confidence, category="contract_rules")

    assert len(tracker.history) == 1
    assert "contract_rules" in tracker.by_category
    assert len(tracker.by_category["contract_rules"]) == 1


def test_tracker_history_limit():
    """Test history limit enforcement."""
    tracker = ConfidenceTracker(history_limit=10)

    # Record 20 scores
    for i in range(20):
        confidence = AggregatedConfidence(
            score=0.5 + i * 0.01,
            components={"gen": 0.5},
            weights={"gen": 1.0},
            variance=0.0,
            is_reliable=True,
        )
        tracker.record(confidence)

    assert len(tracker.history) == 10  # Only last 10 kept


def test_tracker_get_trends():
    """Test getting confidence trends."""
    tracker = ConfidenceTracker()

    # Record various scores
    scores = [0.7, 0.75, 0.8, 0.85, 0.9]
    for score in scores:
        confidence = AggregatedConfidence(
            score=score,
            components={"gen": score},
            weights={"gen": 1.0},
            variance=0.02,
            is_reliable=True,
        )
        tracker.record(confidence)

    trends = tracker.get_trends()

    assert trends.num_total == 5
    assert trends.mean_confidence == pytest.approx(0.8, abs=0.01)
    assert trends.num_reliable == 5
    assert trends.reliability_rate == 1.0


def test_tracker_category_comparison():
    """Test category comparison."""
    tracker = ConfidenceTracker()

    # Record scores in different categories
    for i in range(5):
        conf1 = AggregatedConfidence(
            score=0.9,
            components={"gen": 0.9},
            weights={"gen": 1.0},
            variance=0.01,
            is_reliable=True,
        )
        tracker.record(conf1, category="high_quality")

        conf2 = AggregatedConfidence(
            score=0.6,
            components={"gen": 0.6},
            weights={"gen": 1.0},
            variance=0.05,
            is_reliable=False,
        )
        tracker.record(conf2, category="low_quality")

    comparison = tracker.get_category_comparison()

    assert "high_quality" in comparison
    assert "low_quality" in comparison
    assert comparison["high_quality"]["mean"] > comparison["low_quality"]["mean"]


def test_tracker_low_confidence_cases():
    """Test finding low-confidence cases."""
    tracker = ConfidenceTracker()

    # Record mix of high and low confidence
    for score in [0.5, 0.6, 0.85, 0.9]:
        confidence = AggregatedConfidence(
            score=score,
            components={"gen": score},
            weights={"gen": 1.0},
            variance=0.0,
            is_reliable=True,
        )
        tracker.record(confidence)

    low_conf = tracker.get_low_confidence_cases(threshold=0.7)

    assert len(low_conf) == 2  # 0.5 and 0.6


def test_tracker_high_variance_cases():
    """Test finding high-variance cases."""
    tracker = ConfidenceTracker()

    # Record cases with different variances
    conf1 = AggregatedConfidence(
        score=0.8,
        components={"gen": 0.8},
        weights={"gen": 1.0},
        variance=0.01,
        is_reliable=True,
    )
    conf2 = AggregatedConfidence(
        score=0.8,
        components={"gen": 0.8},
        weights={"gen": 1.0},
        variance=0.20,
        is_reliable=False,
    )

    tracker.record(conf1)
    tracker.record(conf2)

    high_var = tracker.get_high_variance_cases(variance_threshold=0.15)

    assert len(high_var) == 1
    assert high_var[0].variance == 0.20


def test_tracker_summary():
    """Test tracker summary statistics."""
    tracker = ConfidenceTracker()

    # Record some scores
    for i in range(10):
        confidence = AggregatedConfidence(
            score=0.7 + i * 0.02,
            components={"gen": 0.7},
            weights={"gen": 1.0},
            variance=0.02,
            is_reliable=True,
        )
        tracker.record(confidence, category=f"category_{i % 3}")

    summary = tracker.get_summary()

    assert summary["total_tracked"] == 10
    assert summary["categories"] == 3
    assert summary["mean_confidence"] > 0.7


def test_tracker_clear_history():
    """Test clearing tracker history."""
    tracker = ConfidenceTracker()

    # Record some scores
    for i in range(5):
        confidence = AggregatedConfidence(
            score=0.8,
            components={"gen": 0.8},
            weights={"gen": 1.0},
            variance=0.0,
            is_reliable=True,
        )
        tracker.record(confidence, category="test")

    tracker.clear_history(category="test")
    assert len(tracker.by_category.get("test", [])) == 0
    assert len(tracker.history) == 5  # Overall history still intact

    tracker.clear_history()
    assert len(tracker.history) == 0
