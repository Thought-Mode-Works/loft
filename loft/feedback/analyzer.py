"""
Feedback analyzer for identifying rule performance issues.

Analyzes rule performance metrics to identify patterns, issues,
and opportunities for improvement.

Issue #278: Rule Refinement and Feedback Loop
"""

from typing import Dict, List

from loft.feedback.schemas import (
    FeedbackAnalysisReport,
    PerformanceIssue,
    RulePerformanceMetrics,
)


class FeedbackAnalyzer:
    """
    Analyzes rule performance feedback to identify improvement opportunities.

    Examines performance metrics to detect:
    - Rules with low accuracy
    - Rules that are rarely used
    - Domain-specific performance issues
    - Rules that may be too broad or too narrow
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.7,
        usage_threshold: float = 0.2,
        min_usage: int = 3,
    ):
        """
        Initialize feedback analyzer.

        Args:
            accuracy_threshold: Rules below this accuracy are underperforming
            usage_threshold: Rules used less than this fraction are rare
            min_usage: Minimum times rule must be used to evaluate
        """
        self.accuracy_threshold = accuracy_threshold
        self.usage_threshold = usage_threshold
        self.min_usage = min_usage

    def analyze(
        self, metrics_dict: Dict[str, RulePerformanceMetrics]
    ) -> FeedbackAnalysisReport:
        """
        Analyze all rule performance metrics.

        Args:
            metrics_dict: Dictionary mapping rule_id to metrics

        Returns:
            FeedbackAnalysisReport with identified issues
        """
        issues: List[PerformanceIssue] = []
        underperforming_rules: List[str] = []
        overperforming_rules: List[str] = []
        rarely_used_rules: List[str] = []

        for rule_id, metrics in metrics_dict.items():
            # Skip rules with insufficient data
            if metrics.times_used < self.min_usage:
                # Check if rarely used
                if (
                    metrics.total_questions >= 10
                    and metrics.usage_rate < self.usage_threshold
                ):
                    rarely_used_rules.append(rule_id)
                    issues.append(self._create_rarely_used_issue(metrics))
                continue

            # Check accuracy
            if metrics.accuracy_when_used < self.accuracy_threshold:
                underperforming_rules.append(rule_id)
                issues.extend(self._analyze_underperformance(metrics))
            elif metrics.accuracy_when_used >= 0.9:
                overperforming_rules.append(rule_id)

            # Check domain-specific issues
            domain_issues = self._analyze_domain_performance(metrics)
            issues.extend(domain_issues)

            # Check difficulty-specific issues
            difficulty_issues = self._analyze_difficulty_performance(metrics)
            issues.extend(difficulty_issues)

        return FeedbackAnalysisReport(
            total_rules_analyzed=len(metrics_dict),
            underperforming_rules=underperforming_rules,
            overperforming_rules=overperforming_rules,
            rarely_used_rules=rarely_used_rules,
            issues_found=issues,
            refinement_proposals=[],  # Will be filled by refiner
        )

    def _create_rarely_used_issue(
        self, metrics: RulePerformanceMetrics
    ) -> PerformanceIssue:
        """Create issue for a rarely used rule."""
        return PerformanceIssue(
            issue_type="rarely_used",
            severity=0.6,
            description=f"Rule {metrics.rule_id[:16]}... used only {metrics.times_used}/{metrics.total_questions} times ({metrics.usage_rate:.1%})",
            suggested_action="Consider broadening rule conditions or removing if redundant",
        )

    def _analyze_underperformance(
        self, metrics: RulePerformanceMetrics
    ) -> List[PerformanceIssue]:
        """Analyze why a rule is underperforming."""
        issues = []

        # Low overall accuracy
        severity = 1.0 - metrics.accuracy_when_used
        issues.append(
            PerformanceIssue(
                issue_type="low_accuracy",
                severity=severity,
                description=f"Rule {metrics.rule_id[:16]}... has {metrics.accuracy_when_used:.1%} accuracy ({metrics.correct_when_used}/{metrics.times_used})",
                example_failures=self._get_example_failures(metrics),
                suggested_action="Consider strengthening rule conditions or adding exceptions",
            )
        )

        return issues

    def _analyze_domain_performance(
        self, metrics: RulePerformanceMetrics
    ) -> List[PerformanceIssue]:
        """Analyze domain-specific performance issues."""
        issues = []

        # Check for domains where rule performs poorly
        for domain, stats in metrics.by_domain.items():
            if stats["used"] < 3:
                continue

            domain_accuracy = (
                stats["correct"] / stats["used"] if stats["used"] > 0 else 0
            )
            if domain_accuracy < self.accuracy_threshold:
                issues.append(
                    PerformanceIssue(
                        issue_type="domain_specific_failure",
                        severity=0.7,
                        description=f"Rule {metrics.rule_id[:16]}... performs poorly in {domain} domain ({domain_accuracy:.1%})",
                        affected_domains=[domain],
                        suggested_action=f"Consider adding domain-specific conditions for {domain}",
                    )
                )

        return issues

    def _analyze_difficulty_performance(
        self, metrics: RulePerformanceMetrics
    ) -> List[PerformanceIssue]:
        """Analyze difficulty-specific performance issues."""
        issues = []

        # Check for difficulty levels where rule performs poorly
        for difficulty, stats in metrics.by_difficulty.items():
            if stats["used"] < 3:
                continue

            diff_accuracy = stats["correct"] / stats["used"] if stats["used"] > 0 else 0
            if diff_accuracy < self.accuracy_threshold:
                issues.append(
                    PerformanceIssue(
                        issue_type="difficulty_specific_failure",
                        severity=0.6,
                        description=f"Rule {metrics.rule_id[:16]}... struggles with {difficulty} questions ({diff_accuracy:.1%})",
                        suggested_action=f"Consider refining rule to handle {difficulty} cases better",
                    )
                )

        return issues

    def _get_example_failures(
        self, metrics: RulePerformanceMetrics, limit: int = 3
    ) -> List[str]:
        """Get example questions where rule led to incorrect answer."""
        failures = []

        for entry in metrics.feedback_entries:
            if entry.outcome.value == "incorrect" and entry.rule_used:
                failures.append(entry.question)
                if len(failures) >= limit:
                    break

        return failures

    def identify_refinement_candidates(
        self, metrics_dict: Dict[str, RulePerformanceMetrics]
    ) -> List[str]:
        """
        Identify rules that are good candidates for refinement.

        Args:
            metrics_dict: Dictionary mapping rule_id to metrics

        Returns:
            List of rule IDs that should be refined
        """
        candidates = []

        for rule_id, metrics in metrics_dict.items():
            # Skip rules with insufficient data
            if metrics.times_used < self.min_usage:
                continue

            # Underperforming but used enough to matter
            if (
                metrics.accuracy_when_used < self.accuracy_threshold
                and metrics.times_used >= 5
            ):
                candidates.append(rule_id)

        # Sort by impact: rules used more frequently are higher priority
        candidates.sort(key=lambda rid: metrics_dict[rid].times_used, reverse=True)

        return candidates

    def compare_metrics(
        self, baseline: RulePerformanceMetrics, current: RulePerformanceMetrics
    ) -> Dict:
        """
        Compare two sets of metrics for the same rule.

        Useful for evaluating whether a refinement improved performance.

        Args:
            baseline: Original metrics before refinement
            current: Current metrics after refinement

        Returns:
            Dictionary with delta metrics
        """
        return {
            "accuracy_delta": current.accuracy_when_used - baseline.accuracy_when_used,
            "usage_delta": current.usage_rate - baseline.usage_rate,
            "confidence_delta": current.avg_confidence - baseline.avg_confidence,
            "questions_delta": current.total_questions - baseline.total_questions,
            "improvement": current.accuracy_when_used > baseline.accuracy_when_used,
        }
