"""
Integration helpers for review system with validation pipeline.

Provides utilities to connect review queue and triggers with
the validation pipeline without modifying pipeline code.
"""

from typing import Optional, Set

from loguru import logger

from loft.neural.rule_schemas import GeneratedRule
from loft.validation.review_queue import ReviewQueue
from loft.validation.review_schemas import ReviewDecision
from loft.validation.review_trigger import ReviewTrigger
from loft.validation.validation_schemas import ValidationReport


class ReviewIntegration:
    """
    Integration layer between validation pipeline and review system.

    Handles automatic flagging of rules for review and application
    of review decisions.
    """

    def __init__(
        self,
        review_queue: ReviewQueue,
        review_trigger: ReviewTrigger,
    ):
        """
        Initialize review integration.

        Args:
            review_queue: Queue for managing reviews
            review_trigger: Trigger for detecting review needs
        """
        self.review_queue = review_queue
        self.review_trigger = review_trigger
        logger.debug("Initialized ReviewIntegration")

    def check_and_queue(
        self,
        rule: GeneratedRule,
        validation_report: ValidationReport,
        existing_predicates: Optional[Set[str]] = None,
        total_rules_count: int = 0,
        allow_human_review: bool = True,
    ) -> bool:
        """
        Check if rule needs review and add to queue if needed.

        Args:
            rule: Generated rule
            validation_report: Validation results
            existing_predicates: Set of known predicates
            total_rules_count: Total rules in knowledge base
            allow_human_review: Whether to allow queueing (can disable for auto-accept)

        Returns:
            True if rule was queued for review, False otherwise

        Example:
            >>> integration = ReviewIntegration(queue, trigger)
            >>> was_queued = integration.check_and_queue(rule, report)
            >>> if was_queued:
            ...     print("Rule flagged for human review")
        """
        if not allow_human_review:
            return False

        # Check if review is needed
        trigger_result = self.review_trigger.should_review(
            rule=rule,
            validation_report=validation_report,
            existing_predicates=existing_predicates,
            total_rules_count=total_rules_count,
        )

        if not trigger_result:
            logger.debug(f"No review needed for rule: {rule.asp_rule[:50]}...")
            return False

        # Add to review queue
        self.review_queue.add(
            rule=rule,
            validation_report=validation_report,
            priority=trigger_result.priority,
            reason=trigger_result.reason,
            metadata={"triggers": trigger_result.all_triggers},
        )

        logger.info(
            f"Rule flagged for review: {trigger_result.reason} (priority: {trigger_result.priority})"
        )
        return True

    def apply_decision(
        self,
        review_decision: ReviewDecision,
        callback_accept: Optional[callable] = None,
        callback_reject: Optional[callable] = None,
        callback_revise: Optional[callable] = None,
    ) -> bool:
        """
        Apply human review decision.

        Args:
            review_decision: The review decision to apply
            callback_accept: Function to call if accepted (e.g., add to ASP core)
            callback_reject: Function to call if rejected (e.g., log rejection)
            callback_revise: Function to call if revised (e.g., re-validate)

        Returns:
            True if decision was successfully applied

        Example:
            >>> def on_accept(rule_asp):
            ...     asp_core.add_rule(rule_asp)
            >>>
            >>> integration = ReviewIntegration(queue, trigger)
            >>> integration.apply_decision(
            ...     decision,
            ...     callback_accept=on_accept
            ... )
        """
        # Get the review item
        item = self.review_queue.get_item(review_decision.item_id)
        if not item:
            logger.error(f"Review item {review_decision.item_id} not found")
            return False

        if review_decision.decision == "accept":
            logger.info(f"Applying ACCEPT decision for {review_decision.item_id}")
            if callback_accept:
                callback_accept(item.rule.asp_rule)
            return True

        elif review_decision.decision == "reject":
            logger.info(
                f"Applying REJECT decision for {review_decision.item_id}: {review_decision.reviewer_notes}"
            )
            if callback_reject:
                callback_reject(item.rule.asp_rule, review_decision.reviewer_notes)
            return True

        elif review_decision.decision == "revise":
            logger.info(f"Applying REVISE decision for {review_decision.item_id}")
            if callback_revise:
                callback_revise(
                    item.rule.asp_rule,
                    review_decision.suggested_revision,
                    review_decision.reviewer_notes,
                )
            return True

        return False

    def get_pending_count(self) -> int:
        """Get number of pending review items."""
        stats = self.review_queue.get_statistics()
        return stats.pending

    def has_pending_critical(self) -> bool:
        """Check if there are any critical priority items pending."""
        critical_items = self.review_queue.get_by_priority("critical")
        return any(item.status == "pending" for item in critical_items)


def create_review_workflow(
    review_queue: Optional[ReviewQueue] = None,
    review_trigger: Optional[ReviewTrigger] = None,
) -> ReviewIntegration:
    """
    Create a complete review workflow with default components.

    Args:
        review_queue: Optional ReviewQueue (creates default if None)
        review_trigger: Optional ReviewTrigger (creates default if None)

    Returns:
        ReviewIntegration ready to use

    Example:
        >>> workflow = create_review_workflow()
        >>> if workflow.check_and_queue(rule, report):
        ...     print("Rule needs review!")
    """
    queue = review_queue or ReviewQueue()
    trigger = review_trigger or ReviewTrigger()

    return ReviewIntegration(
        review_queue=queue,
        review_trigger=trigger,
    )
