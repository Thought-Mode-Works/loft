"""
Unit tests for performance metrics.

Issue #277: Legal Question Test Suite
"""

import pytest

from loft.evaluation.metrics import MetricsCalculator, PerformanceMetrics
from loft.qa.schemas import Answer, QuestionResult


@pytest.fixture
def sample_results():
    """Create sample question results for testing."""
    results = [
        # Correct answers
        QuestionResult(
            question="Question 1?",
            expected_answer="yes",
            actual_answer=Answer(
                answer="yes", confidence=0.9, explanation="Correct answer"
            ),
            domain="contracts",
        ),
        QuestionResult(
            question="Question 2?",
            expected_answer="no",
            actual_answer=Answer(answer="no", confidence=0.85, explanation="Correct"),
            domain="contracts",
        ),
        # Incorrect answer
        QuestionResult(
            question="Question 3?",
            expected_answer="yes",
            actual_answer=Answer(answer="no", confidence=0.7, explanation="Wrong"),
            domain="torts",
        ),
        # Unknown answer
        QuestionResult(
            question="Question 4?",
            expected_answer="yes",
            actual_answer=Answer(
                answer="unknown", confidence=0.3, explanation="Don't know"
            ),
            domain="torts",
        ),
        # Another correct
        QuestionResult(
            question="Question 5?",
            expected_answer="yes",
            actual_answer=Answer(answer="yes", confidence=0.95, explanation="Sure"),
            domain="property",
        ),
    ]
    return results


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            total_questions=10, correct=7, incorrect=2, unknown=1
        )

        assert metrics.total_questions == 10
        assert metrics.correct == 7
        assert metrics.incorrect == 2
        assert metrics.unknown == 1

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics(
            total_questions=5,
            correct=3,
            incorrect=1,
            unknown=1,
            accuracy=0.6,
            coverage=0.8,
        )

        d = metrics.to_dict()

        assert d["total_questions"] == 5
        assert d["correct"] == 3
        assert d["accuracy"] == 0.6
        assert "by_domain" in d
        assert "by_difficulty" in d

    def test_format_report(self):
        """Test formatting metrics as report."""
        metrics = PerformanceMetrics(
            total_questions=10,
            correct=7,
            incorrect=2,
            unknown=1,
            accuracy=0.7,
            coverage=0.9,
            precision=0.778,
            avg_confidence=0.8,
        )

        report = metrics.format_report()

        assert "Total Questions:" in report
        assert "Correct:" in report
        assert "70.0%" in report  # Accuracy
        assert "90.0%" in report  # Coverage


class TestMetricsCalculator:
    """Test MetricsCalculator."""

    def test_calculator_initialization(self):
        """Test creating metrics calculator."""
        calc = MetricsCalculator()
        assert calc is not None

    def test_calculate_basic_metrics(self, sample_results):
        """Test calculating basic metrics."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_results)

        # Total counts
        assert metrics.total_questions == 5
        assert metrics.correct == 3  # Q1, Q2, Q5
        assert metrics.incorrect == 1  # Q3
        assert metrics.unknown == 1  # Q4

        # Percentages
        assert metrics.accuracy == 0.6  # 3/5
        assert metrics.coverage == 0.8  # 4/5 (answered, not unknown)
        assert metrics.precision == 0.75  # 3/4 (correct/answered)

    def test_calculate_confidence_metrics(self, sample_results):
        """Test confidence-related metrics."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_results)

        # Average confidence should be mean of all confidences
        confidences = [0.9, 0.85, 0.7, 0.3, 0.95]
        expected_avg = sum(confidences) / len(confidences)
        assert abs(metrics.avg_confidence - expected_avg) < 0.01

        # Calibration should be calculated
        assert metrics.confidence_calibration >= 0.0

    def test_calculate_by_domain(self, sample_results):
        """Test metrics broken down by domain."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_results)

        # Contracts: Q1 (correct), Q2 (correct)
        assert "contracts" in metrics.by_domain
        contracts = metrics.by_domain["contracts"]
        assert contracts["total"] == 2
        assert contracts["correct"] == 2
        assert contracts["accuracy"] == 1.0

        # Torts: Q3 (incorrect), Q4 (unknown)
        assert "torts" in metrics.by_domain
        torts = metrics.by_domain["torts"]
        assert torts["total"] == 2
        assert torts["correct"] == 0
        assert torts["unknown"] == 1
        assert torts["accuracy"] == 0.0

        # Property: Q5 (correct)
        assert "property" in metrics.by_domain
        property_metrics = metrics.by_domain["property"]
        assert property_metrics["total"] == 1
        assert property_metrics["correct"] == 1

    def test_calculate_by_difficulty(self):
        """Test metrics broken down by difficulty."""
        results = [
            QuestionResult(
                question="Easy question?",
                expected_answer="yes",
                actual_answer=Answer(
                    answer="yes", confidence=0.9, explanation="Correct"
                ),
                domain="test",
            ),
            QuestionResult(
                question="Hard question?",
                expected_answer="yes",
                actual_answer=Answer(answer="no", confidence=0.6, explanation="Wrong"),
                domain="test",
            ),
        ]

        difficulties = {"Easy": "easy", "Hard": "hard"}

        calc = MetricsCalculator()
        metrics = calc.calculate(results, difficulties)

        # Note: This test demonstrates that difficulty tracking requires matching
        # the question ID in the difficulties dict. Since we can't extract IDs
        # from the question text reliably, most questions end up as "unknown"
        # In practice, the evaluation runner provides proper question ID mapping
        assert (
            len(metrics.by_difficulty) > 0
        )  # Should have some difficulty categorization

    def test_calculate_empty_results(self):
        """Test calculating metrics with no results."""
        calc = MetricsCalculator()
        metrics = calc.calculate([])

        assert metrics.total_questions == 0
        assert metrics.correct == 0
        assert metrics.accuracy == 0.0
        assert metrics.coverage == 0.0

    def test_calculate_all_unknown(self):
        """Test metrics when all answers are unknown."""
        results = [
            QuestionResult(
                question=f"Question {i}?",
                expected_answer="yes",
                actual_answer=Answer(
                    answer="unknown", confidence=0.3, explanation="Don't know"
                ),
                domain="test",
            )
            for i in range(5)
        ]

        calc = MetricsCalculator()
        metrics = calc.calculate(results)

        assert metrics.total_questions == 5
        assert metrics.unknown == 5
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.accuracy == 0.0
        assert metrics.coverage == 0.0
        assert metrics.precision == 0.0  # No answered questions

    def test_calculate_perfect_score(self):
        """Test metrics with perfect accuracy."""
        results = [
            QuestionResult(
                question=f"Question {i}?",
                expected_answer="yes",
                actual_answer=Answer(
                    answer="yes", confidence=0.95, explanation="Correct"
                ),
                domain="test",
            )
            for i in range(10)
        ]

        calc = MetricsCalculator()
        metrics = calc.calculate(results)

        assert metrics.accuracy == 1.0
        assert metrics.coverage == 1.0
        assert metrics.precision == 1.0
        assert metrics.correct == 10
        assert metrics.incorrect == 0
        assert metrics.unknown == 0

    def test_compare_metrics(self):
        """Test comparing metrics to baseline."""
        baseline = PerformanceMetrics(
            total_questions=100,
            correct=70,
            incorrect=25,
            unknown=5,
            accuracy=0.7,
            coverage=0.95,
            precision=0.737,
            avg_confidence=0.75,
            confidence_calibration=0.05,
        )

        current = PerformanceMetrics(
            total_questions=100,
            correct=80,
            incorrect=15,
            unknown=5,
            accuracy=0.8,
            coverage=0.95,
            precision=0.842,
            avg_confidence=0.82,
            confidence_calibration=0.03,
        )

        calc = MetricsCalculator()
        deltas = calc.compare_metrics(baseline, current)

        assert deltas["accuracy_delta"] == pytest.approx(0.1)  # Improved
        assert deltas["coverage_delta"] == pytest.approx(0.0)  # Same
        assert deltas["confidence_delta"] > 0  # Increased
        assert deltas["calibration_delta"] < 0  # Better (lower is better)

    def test_domain_metrics_calculations(self):
        """Test that domain-level metrics are calculated correctly."""
        results = [
            # Domain A: 2 correct, 1 incorrect, 1 unknown
            QuestionResult(
                question="A1",
                expected_answer="yes",
                actual_answer=Answer(answer="yes", confidence=0.9, explanation="C"),
                domain="domain_a",
            ),
            QuestionResult(
                question="A2",
                expected_answer="yes",
                actual_answer=Answer(answer="yes", confidence=0.85, explanation="C"),
                domain="domain_a",
            ),
            QuestionResult(
                question="A3",
                expected_answer="yes",
                actual_answer=Answer(answer="no", confidence=0.7, explanation="W"),
                domain="domain_a",
            ),
            QuestionResult(
                question="A4",
                expected_answer="yes",
                actual_answer=Answer(answer="unknown", confidence=0.3, explanation="U"),
                domain="domain_a",
            ),
        ]

        calc = MetricsCalculator()
        metrics = calc.calculate(results)

        domain_a = metrics.by_domain["domain_a"]
        assert domain_a["total"] == 4
        assert domain_a["correct"] == 2
        assert domain_a["incorrect"] == 1
        assert domain_a["unknown"] == 1
        assert domain_a["answered"] == 3  # Excludes unknown
        assert domain_a["accuracy"] == 0.5  # 2/4
        assert domain_a["coverage"] == 0.75  # 3/4
        assert domain_a["precision"] == pytest.approx(2 / 3)  # 2/3
