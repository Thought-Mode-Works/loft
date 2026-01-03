"""
Rule performance tracker for feedback loop.

Collects and tracks how individual rules perform on questions,
enabling analysis and refinement.

Issue #278: Rule Refinement and Feedback Loop
"""

from typing import Dict, List

from loft.feedback.schemas import (
    RuleFeedbackEntry,
    RuleOutcome,
    RulePerformanceMetrics,
)
from loft.qa.schemas import QuestionResult


class RulePerformanceTracker:
    """
    Tracks rule performance across multiple question evaluations.

    Maintains performance metrics for each rule, tracking when rules
    are used, when they lead to correct/incorrect answers, and
    aggregating statistics for analysis.
    """

    def __init__(self):
        """Initialize empty performance tracker."""
        self.rule_metrics: Dict[str, RulePerformanceMetrics] = {}

    def record_question_result(
        self,
        result: QuestionResult,
        difficulty: str = None,
    ) -> None:
        """
        Record feedback from a question result.

        Extracts which rules were used and whether the answer was correct,
        then updates performance metrics for all involved rules.

        Args:
            result: QuestionResult from evaluation
            difficulty: Question difficulty level (optional)
        """
        # Determine outcome
        if result.actual_answer.answer == "unknown":
            base_outcome = RuleOutcome.UNKNOWN
        elif result.correct is True:
            base_outcome = RuleOutcome.CORRECT
        elif result.correct is False:
            base_outcome = RuleOutcome.INCORRECT
        else:
            base_outcome = RuleOutcome.UNKNOWN

        # Get rules used in this question
        rules_used = set(result.actual_answer.rules_used)

        # Track all rules we know about from this result
        # For now, we only know about rules that were actually used
        # In a full system, we'd track all loaded rules
        for rule_id in rules_used:
            entry = RuleFeedbackEntry(
                rule_id=rule_id,
                question=result.question,
                expected_answer=result.expected_answer or "unknown",
                actual_answer=result.actual_answer.answer,
                outcome=base_outcome,
                rule_used=True,
                confidence=result.actual_answer.confidence,
                domain=result.domain,
                difficulty=difficulty,
            )

            self._record_entry(entry)

    def _record_entry(self, entry: RuleFeedbackEntry) -> None:
        """Record a single feedback entry."""
        if entry.rule_id not in self.rule_metrics:
            self.rule_metrics[entry.rule_id] = RulePerformanceMetrics(
                rule_id=entry.rule_id
            )

        self.rule_metrics[entry.rule_id].update_from_entry(entry)

    def get_rule_performance(self, rule_id: str) -> RulePerformanceMetrics:
        """
        Get performance metrics for a specific rule.

        Args:
            rule_id: Rule identifier

        Returns:
            RulePerformanceMetrics for the rule

        Raises:
            KeyError: If rule_id not found
        """
        if rule_id not in self.rule_metrics:
            raise KeyError(f"No performance data for rule: {rule_id}")

        return self.rule_metrics[rule_id]

    def get_all_metrics(self) -> Dict[str, RulePerformanceMetrics]:
        """Get performance metrics for all tracked rules."""
        return self.rule_metrics.copy()

    def get_underperforming_rules(
        self, accuracy_threshold: float = 0.7, min_usage: int = 3
    ) -> List[RulePerformanceMetrics]:
        """
        Identify rules with poor accuracy.

        Args:
            accuracy_threshold: Rules below this accuracy are underperforming
            min_usage: Minimum times rule must be used to evaluate

        Returns:
            List of underperforming rule metrics
        """
        underperforming = []

        for metrics in self.rule_metrics.values():
            if metrics.times_used < min_usage:
                continue

            if metrics.accuracy_when_used < accuracy_threshold:
                underperforming.append(metrics)

        return sorted(underperforming, key=lambda m: m.accuracy_when_used)

    def get_rarely_used_rules(
        self, usage_threshold: float = 0.1, min_questions: int = 10
    ) -> List[RulePerformanceMetrics]:
        """
        Identify rules that are rarely used.

        Args:
            usage_threshold: Rules used less than this fraction are rare
            min_questions: Minimum questions seen to evaluate

        Returns:
            List of rarely used rule metrics
        """
        rarely_used = []

        for metrics in self.rule_metrics.values():
            if metrics.total_questions < min_questions:
                continue

            if metrics.usage_rate < usage_threshold:
                rarely_used.append(metrics)

        return sorted(rarely_used, key=lambda m: m.usage_rate)

    def get_high_performing_rules(
        self, accuracy_threshold: float = 0.9, min_usage: int = 5
    ) -> List[RulePerformanceMetrics]:
        """
        Identify rules with excellent accuracy.

        Args:
            accuracy_threshold: Rules above this accuracy are high-performing
            min_usage: Minimum times rule must be used to evaluate

        Returns:
            List of high-performing rule metrics
        """
        high_performing = []

        for metrics in self.rule_metrics.values():
            if metrics.times_used < min_usage:
                continue

            if metrics.accuracy_when_used >= accuracy_threshold:
                high_performing.append(metrics)

        return sorted(high_performing, key=lambda m: m.accuracy_when_used, reverse=True)

    def clear(self) -> None:
        """Clear all tracked performance data."""
        self.rule_metrics.clear()

    def export_metrics(self) -> Dict:
        """
        Export all metrics as dictionary.

        Returns:
            Dictionary mapping rule_id to metrics dict
        """
        return {
            rule_id: metrics.to_dict() for rule_id, metrics in self.rule_metrics.items()
        }
