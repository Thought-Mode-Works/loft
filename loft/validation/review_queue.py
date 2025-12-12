"""
Review queue system for human-in-the-loop rule review.

Manages queue of rules requiring human oversight, prioritizes items,
and tracks review decisions.
"""

import json
import statistics
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from loft.neural.rule_schemas import GeneratedRule
from loft.validation.review_schemas import (
    ReviewConfig,
    ReviewDecision,
    ReviewItem,
    ReviewQueueStats,
)
from loft.validation.validation_schemas import ValidationReport


class ReviewStorage:
    """Simple file-based storage for review queue."""

    def __init__(self, base_dir: Path = Path(".loft/review_queue")):
        """
        Initialize review storage.

        Args:
            base_dir: Directory for storing review items
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def store(self, item_id: str, data: Dict) -> None:
        """Store review item data."""
        file_path = self.base_dir / f"{item_id}.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, item_id: str) -> Optional[Dict]:
        """Load review item data."""
        file_path = self.base_dir / f"{item_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r") as f:
            return json.load(f)

    def list_all(self) -> List[str]:
        """List all review item IDs."""
        return [f.stem for f in self.base_dir.glob("*.json")]

    def delete(self, item_id: str) -> bool:
        """Delete review item."""
        file_path = self.base_dir / f"{item_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False


class ReviewQueue:
    """
    Queue for rules requiring human review.

    Manages prioritization, assignment to reviewers, and decision tracking.
    """

    def __init__(
        self,
        storage: Optional[ReviewStorage] = None,
        config: Optional[ReviewConfig] = None,
    ):
        """
        Initialize review queue.

        Args:
            storage: Storage backend (creates default if None)
            config: Review configuration
        """
        self.storage = storage or ReviewStorage()
        self.config = config or ReviewConfig()
        self.priority_levels = {
            "critical": 0,  # Constitutional layer, system errors
            "high": 1,  # High impact, low confidence
            "medium": 2,  # Borderline confidence
            "low": 3,  # FYI, informational
        }
        logger.debug("Initialized ReviewQueue")

    def add(
        self,
        rule: GeneratedRule,
        validation_report: ValidationReport,
        priority: str,
        reason: str,
        metadata: Optional[Dict] = None,
    ) -> ReviewItem:
        """
        Add rule to review queue.

        Args:
            rule: Generated rule needing review
            validation_report: Full validation results
            priority: Urgency level (critical/high/medium/low)
            reason: Why human review is needed
            metadata: Additional context

        Returns:
            ReviewItem with unique ID

        Example:
            >>> queue = ReviewQueue()
            >>> item = queue.add(
            ...     rule=generated_rule,
            ...     validation_report=report,
            ...     priority="high",
            ...     reason="High impact rule affecting constitutional layer"
            ... )
            >>> assert item.status == "pending"
        """
        item_id = self._generate_id()

        item = ReviewItem(
            id=item_id,
            rule=rule,
            validation_report=validation_report,
            priority=priority,
            reason=reason,
            status="pending",
            metadata=metadata or {},
            created_at=datetime.now(),
        )

        self.storage.store(item_id, item.model_dump(mode="json"))
        logger.info(f"Added {priority} priority review item: {item_id} - {reason}")

        return item

    def get_next(self, reviewer_id: str) -> Optional[ReviewItem]:
        """
        Get next item for review, ordered by priority.

        Args:
            reviewer_id: ID of the reviewer

        Returns:
            Next ReviewItem to review, or None if queue is empty

        Example:
            >>> queue = ReviewQueue()
            >>> item = queue.get_next(reviewer_id="alice")
            >>> if item:
            ...     print(f"Reviewing: {item.id}")
        """
        items = []
        for item_id in self.storage.list_all():
            data = self.storage.load(item_id)
            if data and data["status"] == "pending":
                items.append(ReviewItem(**data))

        if not items:
            logger.debug("No pending items in review queue")
            return None

        # Sort by priority then creation time
        items.sort(key=lambda x: (self.priority_levels[x.priority], x.created_at))

        # Assign to reviewer
        item = items[0]
        item.status = "in_review"
        item.reviewer_id = reviewer_id
        item.review_started_at = datetime.now()

        self.storage.store(item.id, item.model_dump(mode="json"))
        logger.info(f"Assigned review item {item.id} to {reviewer_id}")

        return item

    def submit_review(
        self,
        item_id: str,
        decision: str,
        reviewer_notes: str,
        suggested_revision: Optional[str] = None,
    ) -> ReviewDecision:
        """
        Submit human review decision.

        Args:
            item_id: Review item ID
            decision: Accept/reject/revise
            reviewer_notes: Human explanation
            suggested_revision: If revise, suggested improvement

        Returns:
            ReviewDecision with outcome

        Example:
            >>> queue = ReviewQueue()
            >>> decision = queue.submit_review(
            ...     item_id="abc123",
            ...     decision="accept",
            ...     reviewer_notes="Rule looks good, validated manually"
            ... )
        """
        data = self.storage.load(item_id)
        if not data:
            raise ValueError(f"Review item {item_id} not found")

        item = ReviewItem(**data)

        if not item.review_started_at:
            review_time = 0.0
        else:
            review_time = (datetime.now() - item.review_started_at).total_seconds()

        review_decision = ReviewDecision(
            item_id=item_id,
            decision=decision,
            reviewer_notes=reviewer_notes,
            suggested_revision=suggested_revision,
            reviewed_at=datetime.now(),
            review_time_seconds=review_time,
        )

        item.status = "reviewed"
        item.review_decision = review_decision

        self.storage.store(item_id, item.model_dump(mode="json"))
        logger.info(
            f"Review completed for {item_id}: {decision} (time: {review_time:.1f}s)"
        )

        return review_decision

    def get_item(self, item_id: str) -> Optional[ReviewItem]:
        """
        Get specific review item by ID.

        Args:
            item_id: Review item ID

        Returns:
            ReviewItem if found, None otherwise
        """
        data = self.storage.load(item_id)
        if not data:
            return None
        return ReviewItem(**data)

    def get_pending(self) -> List[ReviewItem]:
        """
        Get all pending review items.

        Returns:
            List of pending ReviewItems
        """
        items = []
        for item_id in self.storage.list_all():
            data = self.storage.load(item_id)
            if data and data["status"] == "pending":
                items.append(ReviewItem(**data))

        # Sort by priority
        items.sort(key=lambda x: self.priority_levels[x.priority])
        return items

    def get_by_priority(self, priority: str) -> List[ReviewItem]:
        """
        Get all items with specific priority.

        Args:
            priority: Priority level (critical/high/medium/low)

        Returns:
            List of ReviewItems with that priority
        """
        items = []
        for item_id in self.storage.list_all():
            data = self.storage.load(item_id)
            if data and data["priority"] == priority:
                items.append(ReviewItem(**data))

        items.sort(key=lambda x: x.created_at)
        return items

    def get_statistics(self) -> ReviewQueueStats:
        """
        Get queue statistics.

        Returns:
            ReviewQueueStats with current state

        Example:
            >>> queue = ReviewQueue()
            >>> stats = queue.get_statistics()
            >>> print(stats.summary())
        """
        items = []
        for item_id in self.storage.list_all():
            data = self.storage.load(item_id)
            if data:
                items.append(ReviewItem(**data))

        if not items:
            return ReviewQueueStats(
                total_items=0,
                pending=0,
                in_review=0,
                reviewed=0,
                by_priority={},
                average_review_time_seconds=0.0,
            )

        pending = [i for i in items if i.status == "pending"]
        in_review = [i for i in items if i.status == "in_review"]
        reviewed = [i for i in items if i.status == "reviewed"]

        # Calculate average review time
        review_times = [
            i.review_decision.review_time_seconds for i in reviewed if i.review_decision
        ]
        avg_time = statistics.mean(review_times) if review_times else 0.0

        # Oldest pending item
        oldest_pending = None
        if pending:
            oldest_pending = min(p.created_at for p in pending)

        return ReviewQueueStats(
            total_items=len(items),
            pending=len(pending),
            in_review=len(in_review),
            reviewed=len(reviewed),
            by_priority={
                priority: sum(1 for i in items if i.priority == priority)
                for priority in self.priority_levels.keys()
            },
            average_review_time_seconds=avg_time,
            oldest_pending=oldest_pending,
        )

    def clear_reviewed(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear reviewed items from queue.

        Args:
            older_than_days: Only clear items older than this many days

        Returns:
            Number of items cleared
        """
        cleared = 0
        cutoff_date = None
        if older_than_days:
            from datetime import timedelta

            cutoff_date = datetime.now() - timedelta(days=older_than_days)

        for item_id in self.storage.list_all():
            data = self.storage.load(item_id)
            if data and data["status"] == "reviewed":
                item = ReviewItem(**data)
                if cutoff_date is None or item.created_at < cutoff_date:
                    self.storage.delete(item_id)
                    cleared += 1

        logger.info(f"Cleared {cleared} reviewed items from queue")
        return cleared

    def _generate_id(self) -> str:
        """Generate unique review item ID."""
        return f"review_{uuid.uuid4().hex[:12]}"

    def export_reviews(self, output_file: Path) -> None:
        """
        Export all reviews to JSON file.

        Args:
            output_file: Path to output file
        """
        items = []
        for item_id in self.storage.list_all():
            data = self.storage.load(item_id)
            if data:
                items.append(data)

        with open(output_file, "w") as f:
            json.dump(items, f, indent=2, default=str)

        logger.info(f"Exported {len(items)} review items to {output_file}")
