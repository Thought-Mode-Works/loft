"""
Data schemas for rule refinement and feedback loop.

Defines data structures for tracking rule performance, analyzing feedback,
and proposing rule refinements.

Issue #278: Rule Refinement and Feedback Loop
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class RuleOutcome(str, Enum):
    """Outcome of applying a rule to a question."""

    CORRECT = "correct"  # Rule fired and led to correct answer
    INCORRECT = "incorrect"  # Rule fired and led to incorrect answer
    UNUSED = "unused"  # Rule didn't fire
    UNKNOWN = "unknown"  # Answer was unknown, unclear if rule helped


@dataclass
class RuleFeedbackEntry:
    """
    Single feedback entry tracking a rule's performance on one question.

    Attributes:
        rule_id: ID of the rule being tracked
        question: The question text
        expected_answer: Expected answer (yes/no)
        actual_answer: Actual answer given
        outcome: Whether rule helped/hindered
        rule_used: Whether the rule was actually applied
        confidence: Confidence of the answer
        timestamp: When feedback was recorded
        domain: Legal domain
        difficulty: Question difficulty level
    """

    rule_id: str
    question: str
    expected_answer: str
    actual_answer: str
    outcome: RuleOutcome
    rule_used: bool
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    domain: Optional[str] = None
    difficulty: Optional[str] = None


@dataclass
class RulePerformanceMetrics:
    """
    Aggregated performance metrics for a single rule.

    Tracks how well a rule performs across multiple questions.
    """

    rule_id: str
    total_questions: int = 0
    times_used: int = 0
    correct_when_used: int = 0
    incorrect_when_used: int = 0
    unused_on_correct: int = 0  # Correct answers where rule didn't fire
    unused_on_incorrect: int = 0  # Incorrect answers where rule didn't fire
    accuracy_when_used: float = 0.0
    usage_rate: float = 0.0
    avg_confidence: float = 0.0
    feedback_entries: List[RuleFeedbackEntry] = field(default_factory=list)
    by_domain: Dict[str, Dict] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict] = field(default_factory=dict)

    def update_from_entry(self, entry: RuleFeedbackEntry) -> None:
        """Add a feedback entry and update metrics."""
        self.feedback_entries.append(entry)
        self.total_questions += 1

        if entry.rule_used:
            self.times_used += 1
            if entry.outcome == RuleOutcome.CORRECT:
                self.correct_when_used += 1
            elif entry.outcome == RuleOutcome.INCORRECT:
                self.incorrect_when_used += 1
        else:
            if entry.outcome == RuleOutcome.CORRECT:
                self.unused_on_correct += 1
            elif entry.outcome == RuleOutcome.INCORRECT:
                self.unused_on_incorrect += 1

        # Recalculate aggregates
        if self.times_used > 0:
            self.accuracy_when_used = self.correct_when_used / self.times_used
        if self.total_questions > 0:
            self.usage_rate = self.times_used / self.total_questions

        # Update confidence average
        confidences = [e.confidence for e in self.feedback_entries]
        self.avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # Update domain breakdown
        if entry.domain:
            if entry.domain not in self.by_domain:
                self.by_domain[entry.domain] = {
                    "total": 0,
                    "used": 0,
                    "correct": 0,
                }
            self.by_domain[entry.domain]["total"] += 1
            if entry.rule_used:
                self.by_domain[entry.domain]["used"] += 1
                if entry.outcome == RuleOutcome.CORRECT:
                    self.by_domain[entry.domain]["correct"] += 1

        # Update difficulty breakdown
        if entry.difficulty:
            if entry.difficulty not in self.by_difficulty:
                self.by_difficulty[entry.difficulty] = {
                    "total": 0,
                    "used": 0,
                    "correct": 0,
                }
            self.by_difficulty[entry.difficulty]["total"] += 1
            if entry.rule_used:
                self.by_difficulty[entry.difficulty]["used"] += 1
                if entry.outcome == RuleOutcome.CORRECT:
                    self.by_difficulty[entry.difficulty]["correct"] += 1

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "rule_id": self.rule_id,
            "total_questions": self.total_questions,
            "times_used": self.times_used,
            "correct_when_used": self.correct_when_used,
            "incorrect_when_used": self.incorrect_when_used,
            "unused_on_correct": self.unused_on_correct,
            "unused_on_incorrect": self.unused_on_incorrect,
            "accuracy_when_used": self.accuracy_when_used,
            "usage_rate": self.usage_rate,
            "avg_confidence": self.avg_confidence,
            "by_domain": self.by_domain,
            "by_difficulty": self.by_difficulty,
        }


@dataclass
class PerformanceIssue:
    """
    Identified issue with a rule's performance.

    Describes a specific problem pattern detected in rule behavior.
    """

    issue_type: str  # "low_accuracy", "rarely_used", "domain_specific", etc.
    severity: float  # 0.0-1.0
    description: str
    affected_domains: List[str] = field(default_factory=list)
    example_failures: List[str] = field(default_factory=list)
    suggested_action: str = ""


@dataclass
class RefinementProposal:
    """
    Proposed refinement to improve a rule.

    Attributes:
        original_rule_id: ID of rule being refined
        proposed_asp_rule: Refined ASP rule
        refinement_type: Type of refinement (strengthen, weaken, generalize, etc.)
        rationale: Explanation of why this refinement should help
        expected_impact: Expected improvement in metrics
        confidence: Confidence in the proposal
        test_cases: Test cases to validate the refinement
    """

    original_rule_id: str
    proposed_asp_rule: str
    refinement_type: str
    rationale: str
    expected_impact: str
    confidence: float
    test_cases: List[str] = field(default_factory=list)
    issues_addressed: List[PerformanceIssue] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeedbackAnalysisReport:
    """
    Comprehensive analysis of rule performance feedback.

    Summarizes performance issues and proposed refinements across all rules.
    """

    total_rules_analyzed: int
    underperforming_rules: List[str]
    overperforming_rules: List[str]
    rarely_used_rules: List[str]
    issues_found: List[PerformanceIssue]
    refinement_proposals: List[RefinementProposal]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Feedback Analysis Report",
            "=" * 60,
            f"Total rules analyzed: {self.total_rules_analyzed}",
            f"Underperforming rules: {len(self.underperforming_rules)}",
            f"Rarely used rules: {len(self.rarely_used_rules)}",
            f"Issues identified: {len(self.issues_found)}",
            f"Refinement proposals: {len(self.refinement_proposals)}",
            "",
        ]

        if self.underperforming_rules:
            lines.append("Underperforming Rules:")
            for rule_id in self.underperforming_rules[:5]:
                lines.append(f"  - {rule_id[:16]}...")
            if len(self.underperforming_rules) > 5:
                lines.append(f"  ... and {len(self.underperforming_rules) - 5} more")
            lines.append("")

        if self.issues_found:
            lines.append("Top Issues:")
            sorted_issues = sorted(
                self.issues_found, key=lambda i: i.severity, reverse=True
            )
            for issue in sorted_issues[:3]:
                lines.append(f"  - [{issue.issue_type}] {issue.description}")
                lines.append(f"    Severity: {issue.severity:.2f}")
            lines.append("")

        if self.refinement_proposals:
            lines.append("Top Refinement Proposals:")
            sorted_proposals = sorted(
                self.refinement_proposals, key=lambda p: p.confidence, reverse=True
            )
            for proposal in sorted_proposals[:3]:
                lines.append(
                    f"  - {proposal.refinement_type} for {proposal.original_rule_id[:16]}..."
                )
                lines.append(f"    Confidence: {proposal.confidence:.0%}")
                lines.append(f"    Rationale: {proposal.rationale[:80]}...")
            lines.append("")

        return "\n".join(lines)
