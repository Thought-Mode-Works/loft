"""
Feedback loop module for rule refinement.

Tracks rule performance, analyzes feedback, and proposes refinements
to improve underperforming rules.

Issue #278: Rule Refinement and Feedback Loop

Usage:
    from loft.feedback import RulePerformanceTracker, FeedbackAnalyzer, RuleRefiner

    # Track performance
    tracker = RulePerformanceTracker()
    for result in evaluation_results:
        tracker.record_question_result(result)

    # Analyze feedback
    analyzer = FeedbackAnalyzer()
    report = analyzer.analyze(tracker.get_all_metrics())

    # Generate refinements
    refiner = RuleRefiner(llm_interface)
    for rule_id in analyzer.identify_refinement_candidates(tracker.get_all_metrics()):
        metrics = tracker.get_rule_performance(rule_id)
        issues = [issue for issue in report.issues_found if rule_id in issue.description]
        proposal = refiner.propose_refinement(rule_text, rule_id, metrics, issues)
"""

from loft.feedback.analyzer import FeedbackAnalyzer
from loft.feedback.refiner import RuleRefiner
from loft.feedback.schemas import (
    FeedbackAnalysisReport,
    PerformanceIssue,
    RefinementProposal,
    RuleFeedbackEntry,
    RuleOutcome,
    RulePerformanceMetrics,
)
from loft.feedback.tracker import RulePerformanceTracker

__all__ = [
    "RulePerformanceTracker",
    "FeedbackAnalyzer",
    "RuleRefiner",
    "RuleFeedbackEntry",
    "RuleOutcome",
    "RulePerformanceMetrics",
    "PerformanceIssue",
    "RefinementProposal",
    "FeedbackAnalysisReport",
]
