"""
Pydantic schemas for human-in-the-loop review system.

Defines structured outputs for review queue items, decisions,
triggers, and statistics.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from loft.neural.rule_schemas import GeneratedRule
from loft.validation.validation_schemas import ValidationReport


class ReviewItem(BaseModel):
    """Item in review queue awaiting human decision."""

    id: str = Field(description="Unique review item ID")
    rule: GeneratedRule = Field(description="Generated rule needing review")
    validation_report: ValidationReport = Field(description="Full validation results")
    priority: Literal["critical", "high", "medium", "low"] = Field(
        description="Urgency level for review"
    )
    reason: str = Field(description="Why human review is needed")
    status: Literal["pending", "in_review", "reviewed"] = Field(
        default="pending", description="Current review status"
    )
    reviewer_id: Optional[str] = Field(default=None, description="ID of person reviewing")
    review_started_at: Optional[datetime] = Field(default=None, description="When review began")
    review_decision: Optional["ReviewDecision"] = Field(
        default=None, description="Human decision (if reviewed)"
    )
    metadata: Dict = Field(default_factory=dict, description="Additional context")
    created_at: datetime = Field(default_factory=datetime.now, description="When item was queued")

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Review Item {self.id}",
            f"  Priority: {self.priority}",
            f"  Status: {self.status}",
            f"  Reason: {self.reason}",
            f"  Rule: {self.rule.asp_rule[:80]}...",
            f"  Created: {self.created_at}",
        ]
        if self.reviewer_id:
            lines.append(f"  Reviewer: {self.reviewer_id}")
        return "\n".join(lines)


class ReviewDecision(BaseModel):
    """Human review decision on a rule."""

    item_id: str = Field(description="Review item ID")
    decision: Literal["accept", "reject", "revise"] = Field(description="Accept/reject/revise")
    reviewer_notes: str = Field(description="Human explanation")
    suggested_revision: Optional[str] = Field(
        default=None, description="If revise, suggested improvement"
    )
    reviewed_at: datetime = Field(
        default_factory=datetime.now, description="When review was completed"
    )
    review_time_seconds: float = Field(default=0.0, description="Time spent on review")

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Review Decision: {self.decision.upper()}",
            f"  Reviewer notes: {self.reviewer_notes}",
            f"  Review time: {self.review_time_seconds:.1f}s",
        ]
        if self.suggested_revision:
            lines.append(f"  Suggested revision: {self.suggested_revision}")
        return "\n".join(lines)


class ReviewTriggerResult(BaseModel):
    """Result of checking if rule needs review."""

    trigger_type: str = Field(description="Primary trigger type (e.g., 'confidence_borderline')")
    priority: Literal["critical", "high", "medium", "low"] = Field(description="Priority level")
    reason: str = Field(description="Human-readable explanation")
    all_triggers: List[str] = Field(description="All triggers that fired (not just primary)")

    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"{self.priority.upper()}: {self.reason} (triggers: {', '.join(self.all_triggers)})"


class RuleImpact(BaseModel):
    """Assessment of rule's impact on existing knowledge base."""

    affects_rules: int = Field(description="Number of existing rules potentially affected")
    affects_test_cases: int = Field(description="Number of test cases potentially affected")
    predicate_usage_frequency: Dict[str, int] = Field(
        default_factory=dict,
        description="How often predicates are used in existing rules",
    )
    novelty_score: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="How novel/unusual this rule is (0=common, 1=very novel)",
    )

    def is_high_impact(self, threshold: float = 0.1) -> bool:
        """Check if impact exceeds threshold."""
        # Placeholder - in real implementation, compare to total rule count
        return self.affects_rules > 0 or self.novelty_score > threshold


class ReviewQueueStats(BaseModel):
    """Statistics about the review queue."""

    total_items: int = Field(description="Total items in queue")
    pending: int = Field(description="Items awaiting review")
    in_review: int = Field(description="Items currently being reviewed")
    reviewed: int = Field(description="Items that have been reviewed")
    by_priority: Dict[str, int] = Field(description="Count of items by priority level")
    average_review_time_seconds: float = Field(description="Average time to complete review")
    oldest_pending: Optional[datetime] = Field(
        default=None, description="Timestamp of oldest pending item"
    )

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Review Queue Statistics",
            f"  Total: {self.total_items}",
            f"  Pending: {self.pending}",
            f"  In Review: {self.in_review}",
            f"  Reviewed: {self.reviewed}",
            f"  Avg Review Time: {self.average_review_time_seconds:.1f}s",
        ]
        if self.by_priority:
            lines.append("  By Priority:")
            for priority, count in self.by_priority.items():
                lines.append(f"    {priority}: {count}")
        if self.oldest_pending:
            age = (datetime.now() - self.oldest_pending).total_seconds() / 3600
            lines.append(f"  Oldest Pending: {age:.1f}h ago")
        return "\n".join(lines)


class ReviewConfig(BaseModel):
    """Configuration for review triggers and thresholds."""

    confidence_variance_threshold: float = Field(
        default=0.15,
        description="Variance threshold for triggering review",
    )
    consensus_strength_threshold: float = Field(
        default=0.6,
        description="Minimum consensus strength to avoid review",
    )
    impact_threshold: float = Field(
        default=0.1,
        description="Fraction of rules affected to trigger high-impact review",
    )
    constitutional_predicates: List[str] = Field(
        default_factory=lambda: [
            "fundamental_rights",
            "due_process",
            "core_principles",
            "constitutional",
        ],
        description="Predicates that require human approval",
    )
    enable_novelty_detection: bool = Field(
        default=True,
        description="Whether to flag novel predicates for review",
    )
