"""
Unit tests for feedback analyzer.

Issue #278: Rule Refinement and Feedback Loop
"""

import pytest

from loft.feedback.analyzer import FeedbackAnalyzer
from loft.feedback.schemas import RuleFeedbackEntry, RuleOutcome, RulePerformanceMetrics


@pytest.fixture
def analyzer():
    """Create feedback analyzer with default thresholds."""
    return FeedbackAnalyzer(accuracy_threshold=0.7, usage_threshold=0.2, min_usage=3)


@pytest.fixture
def sample_metrics():
    """Create sample performance metrics for testing."""
    # High performing rule
    high_perf = RulePerformanceMetrics(rule_id="high_perf_rule")
    for i in range(10):
        high_perf.update_from_entry(
            RuleFeedbackEntry(
                rule_id="high_perf_rule",
                question=f"Q{i}",
                expected_answer="yes",
                actual_answer="yes",
                outcome=RuleOutcome.CORRECT,
                rule_used=True,
                confidence=0.95,
                domain="contracts",
            )
        )

    # Low performing rule
    low_perf = RulePerformanceMetrics(rule_id="low_perf_rule")
    for i in range(10):
        outcome = RuleOutcome.CORRECT if i < 4 else RuleOutcome.INCORRECT
        actual = "yes" if i < 4 else "no"
        low_perf.update_from_entry(
            RuleFeedbackEntry(
                rule_id="low_perf_rule",
                question=f"Q{i}",
                expected_answer="yes",
                actual_answer=actual,
                outcome=outcome,
                rule_used=True,
                confidence=0.6,
                domain="torts",
            )
        )

    # Rarely used rule - need to track when NOT used as well
    rarely_used = RulePerformanceMetrics(rule_id="rarely_used_rule")
    for i in range(20):
        # Only used 2 times out of 20
        # We need to create entries for all questions to track total_questions
        rarely_used.update_from_entry(
            RuleFeedbackEntry(
                rule_id="rarely_used_rule",
                question=f"Q{i}",
                expected_answer="yes",
                actual_answer="yes" if i < 2 else "unknown",
                outcome=RuleOutcome.CORRECT if i < 2 else RuleOutcome.UNUSED,
                rule_used=True if i < 2 else False,
                confidence=0.8 if i < 2 else 0.0,
                domain="property",
            )
        )

    return {
        "high_perf_rule": high_perf,
        "low_perf_rule": low_perf,
        "rarely_used_rule": rarely_used,
    }


class TestFeedbackAnalyzer:
    """Test FeedbackAnalyzer."""

    def test_analyzer_initialization(self):
        """Test creating analyzer with custom thresholds."""
        analyzer = FeedbackAnalyzer(
            accuracy_threshold=0.8, usage_threshold=0.3, min_usage=5
        )

        assert analyzer.accuracy_threshold == 0.8
        assert analyzer.usage_threshold == 0.3
        assert analyzer.min_usage == 5

    def test_analyze_basic_report(self, analyzer, sample_metrics):
        """Test generating basic analysis report."""
        report = analyzer.analyze(sample_metrics)

        assert report.total_rules_analyzed == 3
        assert len(report.underperforming_rules) >= 1
        assert len(report.overperforming_rules) >= 1
        assert len(report.issues_found) > 0

    def test_identify_underperforming_rules(self, analyzer, sample_metrics):
        """Test identifying underperforming rules."""
        report = analyzer.analyze(sample_metrics)

        # low_perf_rule has 40% accuracy (below 70% threshold)
        assert "low_perf_rule" in report.underperforming_rules
        # high_perf_rule has 100% accuracy
        assert "high_perf_rule" not in report.underperforming_rules

    def test_identify_overperforming_rules(self, analyzer, sample_metrics):
        """Test identifying high-performing rules."""
        report = analyzer.analyze(sample_metrics)

        # high_perf_rule has 100% accuracy (above 90%)
        assert "high_perf_rule" in report.overperforming_rules
        # low_perf_rule has 40% accuracy
        assert "low_perf_rule" not in report.overperforming_rules

    def test_identify_rarely_used_rules(self, analyzer, sample_metrics):
        """Test identifying rarely used rules."""
        report = analyzer.analyze(sample_metrics)

        # rarely_used_rule used 2/20 times (10%, below 20% threshold)
        # But it has insufficient usage (< 3) so won't be in underperforming
        # It should be in rarely_used
        assert "rarely_used_rule" in report.rarely_used_rules

    def test_low_accuracy_issue_detection(self, analyzer, sample_metrics):
        """Test detecting low accuracy issues."""
        report = analyzer.analyze(sample_metrics)

        # Should have at least one low_accuracy issue for low_perf_rule
        low_accuracy_issues = [
            issue for issue in report.issues_found if issue.issue_type == "low_accuracy"
        ]
        assert len(low_accuracy_issues) > 0

        # Check severity is calculated
        for issue in low_accuracy_issues:
            assert 0.0 <= issue.severity <= 1.0

    def test_rarely_used_issue_detection(self, analyzer, sample_metrics):
        """Test detecting rarely used issues."""
        report = analyzer.analyze(sample_metrics)

        # Should have rarely_used issue
        rarely_used_issues = [
            issue for issue in report.issues_found if issue.issue_type == "rarely_used"
        ]
        assert len(rarely_used_issues) > 0

    def test_domain_specific_issue_detection(self, analyzer):
        """Test detecting domain-specific performance issues."""
        # Create a rule that performs well in contracts but poorly in torts
        metrics = RulePerformanceMetrics(rule_id="domain_issue_rule")

        # Good performance in contracts
        for i in range(5):
            metrics.update_from_entry(
                RuleFeedbackEntry(
                    rule_id="domain_issue_rule",
                    question=f"Contract Q{i}",
                    expected_answer="yes",
                    actual_answer="yes",
                    outcome=RuleOutcome.CORRECT,
                    rule_used=True,
                    confidence=0.9,
                    domain="contracts",
                )
            )

        # Poor performance in torts
        for i in range(5):
            metrics.update_from_entry(
                RuleFeedbackEntry(
                    rule_id="domain_issue_rule",
                    question=f"Tort Q{i}",
                    expected_answer="yes",
                    actual_answer="no",
                    outcome=RuleOutcome.INCORRECT,
                    rule_used=True,
                    confidence=0.5,
                    domain="torts",
                )
            )

        report = analyzer.analyze({"domain_issue_rule": metrics})

        # Should detect domain-specific failure in torts
        domain_issues = [
            issue
            for issue in report.issues_found
            if issue.issue_type == "domain_specific_failure"
        ]
        assert len(domain_issues) > 0
        assert any("torts" in issue.affected_domains for issue in domain_issues)

    def test_difficulty_specific_issue_detection(self, analyzer):
        """Test detecting difficulty-specific performance issues."""
        # Create a rule that works on easy but fails on hard questions
        metrics = RulePerformanceMetrics(rule_id="difficulty_issue_rule")

        # Good on easy
        for i in range(5):
            metrics.update_from_entry(
                RuleFeedbackEntry(
                    rule_id="difficulty_issue_rule",
                    question=f"Easy Q{i}",
                    expected_answer="yes",
                    actual_answer="yes",
                    outcome=RuleOutcome.CORRECT,
                    rule_used=True,
                    confidence=0.9,
                    domain="test",
                    difficulty="easy",
                )
            )

        # Poor on hard
        for i in range(5):
            metrics.update_from_entry(
                RuleFeedbackEntry(
                    rule_id="difficulty_issue_rule",
                    question=f"Hard Q{i}",
                    expected_answer="yes",
                    actual_answer="no",
                    outcome=RuleOutcome.INCORRECT,
                    rule_used=True,
                    confidence=0.5,
                    domain="test",
                    difficulty="hard",
                )
            )

        report = analyzer.analyze({"difficulty_issue_rule": metrics})

        # Should detect difficulty-specific failure
        difficulty_issues = [
            issue
            for issue in report.issues_found
            if issue.issue_type == "difficulty_specific_failure"
        ]
        assert len(difficulty_issues) > 0

    def test_identify_refinement_candidates(self, analyzer, sample_metrics):
        """Test identifying rules for refinement."""
        candidates = analyzer.identify_refinement_candidates(sample_metrics)

        # low_perf_rule should be a candidate (low accuracy, used enough)
        assert "low_perf_rule" in candidates
        # high_perf_rule should not be (high accuracy)
        assert "high_perf_rule" not in candidates

    def test_refinement_candidates_priority(self, analyzer):
        """Test that refinement candidates are prioritized by usage."""
        # Create two underperforming rules with different usage
        high_usage = RulePerformanceMetrics(rule_id="high_usage_bad")
        for i in range(20):
            high_usage.update_from_entry(
                RuleFeedbackEntry(
                    rule_id="high_usage_bad",
                    question=f"Q{i}",
                    expected_answer="yes",
                    actual_answer="no" if i < 15 else "yes",
                    outcome=RuleOutcome.INCORRECT if i < 15 else RuleOutcome.CORRECT,
                    rule_used=True,
                    confidence=0.5,
                )
            )

        low_usage = RulePerformanceMetrics(rule_id="low_usage_bad")
        for i in range(5):
            low_usage.update_from_entry(
                RuleFeedbackEntry(
                    rule_id="low_usage_bad",
                    question=f"Q{i}",
                    expected_answer="yes",
                    actual_answer="no",
                    outcome=RuleOutcome.INCORRECT,
                    rule_used=True,
                    confidence=0.5,
                )
            )

        metrics = {"high_usage_bad": high_usage, "low_usage_bad": low_usage}
        candidates = analyzer.identify_refinement_candidates(metrics)

        # high_usage_bad should come first (higher priority)
        assert candidates[0] == "high_usage_bad"
        assert candidates[1] == "low_usage_bad"

    def test_compare_metrics(self, analyzer):
        """Test comparing baseline and current metrics."""
        baseline = RulePerformanceMetrics(rule_id="test_rule")
        for i in range(10):
            baseline.update_from_entry(
                RuleFeedbackEntry(
                    rule_id="test_rule",
                    question=f"Q{i}",
                    expected_answer="yes",
                    actual_answer="yes" if i < 6 else "no",
                    outcome=RuleOutcome.CORRECT if i < 6 else RuleOutcome.INCORRECT,
                    rule_used=True,
                    confidence=0.7,
                )
            )

        # Improved version
        current = RulePerformanceMetrics(rule_id="test_rule")
        for i in range(10):
            current.update_from_entry(
                RuleFeedbackEntry(
                    rule_id="test_rule",
                    question=f"Q{i}",
                    expected_answer="yes",
                    actual_answer="yes" if i < 8 else "no",
                    outcome=RuleOutcome.CORRECT if i < 8 else RuleOutcome.INCORRECT,
                    rule_used=True,
                    confidence=0.85,
                )
            )

        deltas = analyzer.compare_metrics(baseline, current)

        assert deltas["accuracy_delta"] == pytest.approx(0.2)  # 60% -> 80%
        assert deltas["confidence_delta"] == pytest.approx(0.15)  # 0.7 -> 0.85
        assert deltas["improvement"] is True

    def test_report_summary(self, analyzer, sample_metrics):
        """Test generating report summary."""
        report = analyzer.analyze(sample_metrics)
        summary = report.summary()

        assert "Feedback Analysis Report" in summary
        assert "Total rules analyzed:" in summary
        assert "Underperforming rules:" in summary
        assert str(report.total_rules_analyzed) in summary

    def test_empty_metrics_analysis(self, analyzer):
        """Test analyzing empty metrics."""
        report = analyzer.analyze({})

        assert report.total_rules_analyzed == 0
        assert len(report.underperforming_rules) == 0
        assert len(report.overperforming_rules) == 0
        assert len(report.issues_found) == 0

    def test_skip_insufficient_data(self, analyzer):
        """Test that rules with insufficient data are skipped."""
        # Create a rule with only 2 uses (below min_usage=3)
        metrics = RulePerformanceMetrics(rule_id="insufficient_data")
        for i in range(2):
            metrics.update_from_entry(
                RuleFeedbackEntry(
                    rule_id="insufficient_data",
                    question=f"Q{i}",
                    expected_answer="yes",
                    actual_answer="no",  # 0% accuracy
                    outcome=RuleOutcome.INCORRECT,
                    rule_used=True,
                    confidence=0.5,
                )
            )

        report = analyzer.analyze({"insufficient_data": metrics})

        # Should not be in underperforming despite 0% accuracy
        assert "insufficient_data" not in report.underperforming_rules
