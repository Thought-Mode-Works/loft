"""
Performance metrics and analysis for legal QA evaluation.

Calculates accuracy, coverage, confidence calibration, and other
metrics for evaluating question answering performance.

Issue #277: Legal Question Test Suite
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loft.qa.schemas import EvaluationReport, QuestionResult

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for QA evaluation.

    Attributes:
        total_questions: Total questions evaluated
        correct: Number of correct answers
        incorrect: Number of incorrect answers
        unknown: Number of unknown answers
        accuracy: Overall accuracy (correct / total)
        coverage: Coverage (answered / total)
        precision: Precision (correct / answered)
        avg_confidence: Average confidence score
        confidence_calibration: How well confidence aligns with accuracy
        by_domain: Metrics broken down by domain
        by_difficulty: Metrics broken down by difficulty
    """

    total_questions: int = 0
    correct: int = 0
    incorrect: int = 0
    unknown: int = 0
    accuracy: float = 0.0
    coverage: float = 0.0
    precision: float = 0.0
    avg_confidence: float = 0.0
    confidence_calibration: float = 0.0
    by_domain: Dict[str, Dict] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "total_questions": self.total_questions,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "unknown": self.unknown,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "precision": self.precision,
            "avg_confidence": self.avg_confidence,
            "confidence_calibration": self.confidence_calibration,
            "by_domain": self.by_domain,
            "by_difficulty": self.by_difficulty,
        }

    def format_report(self) -> str:
        """Format metrics as human-readable report."""
        report = "=" * 60 + "\n"
        report += "PERFORMANCE METRICS\n"
        report += "=" * 60 + "\n\n"

        report += "Overall Performance:\n"
        report += f"  Total Questions:    {self.total_questions}\n"
        report += f"  Correct:            {self.correct} ({self.accuracy:.1%})\n"
        report += f"  Incorrect:          {self.incorrect} ({self.incorrect/self.total_questions:.1%})\n"
        report += f"  Unknown:            {self.unknown} ({self.unknown/self.total_questions:.1%})\n"
        report += f"  Coverage:           {self.coverage:.1%}\n"
        report += f"  Precision:          {self.precision:.1%}\n"
        report += f"  Avg Confidence:     {self.avg_confidence:.1%}\n"
        report += f"  Calibration:        {self.confidence_calibration:.3f}\n\n"

        if self.by_domain:
            report += "By Domain:\n"
            for domain, stats in sorted(self.by_domain.items()):
                acc = stats.get("accuracy", 0)
                cov = stats.get("coverage", 0)
                report += f"  {domain:20s}: {stats['correct']:2d}/{stats['total']:2d} "
                report += f"(Acc: {acc:.1%}, Cov: {cov:.1%})\n"
            report += "\n"

        if self.by_difficulty:
            report += "By Difficulty:\n"
            for diff, stats in sorted(self.by_difficulty.items()):
                acc = stats.get("accuracy", 0)
                report += f"  {diff:10s}: {stats['correct']:2d}/{stats['total']:2d} ({acc:.1%})\n"
            report += "\n"

        report += "=" * 60 + "\n"
        return report


class MetricsCalculator:
    """
    Calculates performance metrics from evaluation results.

    Analyzes question results to compute accuracy, coverage,
    confidence calibration, and breakdowns by domain/difficulty.
    """

    def calculate(
        self,
        results: List[QuestionResult],
        difficulties: Optional[Dict[str, str]] = None,
    ) -> PerformanceMetrics:
        """
        Calculate metrics from question results.

        Args:
            results: List of QuestionResult objects
            difficulties: Optional mapping of question IDs to difficulty levels

        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        metrics = PerformanceMetrics()
        metrics.total_questions = len(results)

        if metrics.total_questions == 0:
            return metrics

        # Count answers
        confidences = []
        correct_confidences = []

        for result in results:
            answer = result.actual_answer.answer
            correct = result.correct

            if answer == "unknown":
                metrics.unknown += 1
            elif correct is True:
                metrics.correct += 1
                correct_confidences.append(result.actual_answer.confidence)
            elif correct is False:
                metrics.incorrect += 1

            confidences.append(result.actual_answer.confidence)

            # Track by domain
            domain = result.domain or "unknown"
            if domain not in metrics.by_domain:
                metrics.by_domain[domain] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "unknown": 0,
                    "answered": 0,
                }

            metrics.by_domain[domain]["total"] += 1
            if answer == "unknown":
                metrics.by_domain[domain]["unknown"] += 1
            else:
                metrics.by_domain[domain]["answered"] += 1
                if correct:
                    metrics.by_domain[domain]["correct"] += 1
                else:
                    metrics.by_domain[domain]["incorrect"] += 1

            # Track by difficulty if provided
            if difficulties:
                # Extract question ID from result
                q_id = result.question.split()[0] if result.question else None
                difficulty = difficulties.get(q_id, "unknown")

                if difficulty not in metrics.by_difficulty:
                    metrics.by_difficulty[difficulty] = {
                        "total": 0,
                        "correct": 0,
                        "answered": 0,
                    }

                metrics.by_difficulty[difficulty]["total"] += 1
                if answer != "unknown":
                    metrics.by_difficulty[difficulty]["answered"] += 1
                if correct:
                    metrics.by_difficulty[difficulty]["correct"] += 1

        # Calculate overall metrics
        answered = metrics.correct + metrics.incorrect
        metrics.accuracy = (
            metrics.correct / metrics.total_questions
            if metrics.total_questions > 0
            else 0.0
        )
        metrics.coverage = (
            answered / metrics.total_questions if metrics.total_questions > 0 else 0.0
        )
        metrics.precision = metrics.correct / answered if answered > 0 else 0.0

        # Average confidence
        metrics.avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # Confidence calibration (how well confidence matches accuracy)
        # Perfect calibration = 0, higher values = worse calibration
        if correct_confidences:
            avg_correct_conf = sum(correct_confidences) / len(correct_confidences)
            actual_accuracy = metrics.precision  # Accuracy among answered questions
            metrics.confidence_calibration = abs(avg_correct_conf - actual_accuracy)
        else:
            metrics.confidence_calibration = 1.0  # Worst case

        # Calculate domain-level metrics
        for domain_stats in metrics.by_domain.values():
            total = domain_stats["total"]
            correct = domain_stats["correct"]
            answered = domain_stats["answered"]

            domain_stats["accuracy"] = correct / total if total > 0 else 0.0
            domain_stats["coverage"] = answered / total if total > 0 else 0.0
            domain_stats["precision"] = correct / answered if answered > 0 else 0.0

        # Calculate difficulty-level metrics
        for diff_stats in metrics.by_difficulty.values():
            total = diff_stats["total"]
            correct = diff_stats["correct"]
            answered = diff_stats["answered"]

            diff_stats["accuracy"] = correct / total if total > 0 else 0.0
            diff_stats["coverage"] = answered / total if total > 0 else 0.0
            diff_stats["precision"] = correct / answered if answered > 0 else 0.0

        return metrics

    def calculate_from_report(self, report: EvaluationReport) -> PerformanceMetrics:
        """
        Calculate metrics from an EvaluationReport.

        Args:
            report: EvaluationReport from QA interface

        Returns:
            PerformanceMetrics calculated from report
        """
        return self.calculate(report.results)

    def compare_metrics(
        self, baseline: PerformanceMetrics, current: PerformanceMetrics
    ) -> Dict[str, float]:
        """
        Compare current metrics to baseline.

        Args:
            baseline: Baseline metrics
            current: Current metrics to compare

        Returns:
            Dictionary of deltas (current - baseline)
        """
        return {
            "accuracy_delta": current.accuracy - baseline.accuracy,
            "coverage_delta": current.coverage - baseline.coverage,
            "precision_delta": current.precision - baseline.precision,
            "confidence_delta": current.avg_confidence - baseline.avg_confidence,
            "calibration_delta": current.confidence_calibration
            - baseline.confidence_calibration,
        }
