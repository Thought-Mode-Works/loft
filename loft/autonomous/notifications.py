"""
Notification Manager for Autonomous Test Harness.

This module handles sending notifications to Slack and other
webhook endpoints during autonomous runs.

Features:
- Slack webhook integration
- Configurable notification events
- Rate limiting to prevent spam
- Message formatting
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from loft.autonomous.config import NotificationConfig
from loft.autonomous.schemas import CycleResult, RunResult

logger = logging.getLogger(__name__)


@dataclass
class NotificationMessage:
    """A notification message.

    Attributes:
        title: Message title
        text: Message body
        color: Color for Slack attachment
        fields: Key-value fields
        timestamp: When the event occurred
    """

    title: str
    text: str
    color: str = "#36a64f"  # Green
    fields: Dict[str, str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.fields is None:
            self.fields = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


class NotificationManager:
    """Sends notifications for key autonomous run events.

    Supports Slack webhooks and can be extended for other
    notification providers.

    Attributes:
        config: Notification configuration
    """

    def __init__(self, config: NotificationConfig):
        """Initialize the notification manager.

        Args:
            config: Notification configuration
        """
        self._config = config
        self._last_notification_time: Optional[datetime] = None
        self._min_notification_interval = timedelta(seconds=30)
        self._notification_count = 0

    @property
    def config(self) -> NotificationConfig:
        """Get configuration."""
        return self._config

    @property
    def is_enabled(self) -> bool:
        """Check if notifications are enabled."""
        return bool(self._config.slack_webhook_url)

    def notify_started(self, run_id: str, config: Dict[str, Any]) -> bool:
        """Send notification when run starts.

        Args:
            run_id: Run identifier
            config: Run configuration

        Returns:
            True if notification sent successfully
        """
        if not self._config.notify_on_start:
            return False

        message = NotificationMessage(
            title=f"Autonomous Run Started: {run_id}",
            text="A new autonomous test run has begun.",
            color="#36a64f",  # Green
            fields={
                "Max Duration": f"{config.get('max_duration_hours', 4)} hours",
                "Checkpoint Interval": f"{config.get('checkpoint_interval_minutes', 15)} min",
                "LLM Model": config.get("llm_model", "claude-3-5-haiku"),
                "Max Cases": str(config.get("max_cases", 0)) or "Unlimited",
            },
        )

        return self._send_slack_notification(message)

    def notify_milestone(
        self,
        milestone: str,
        run_id: str,
        metrics: Dict[str, Any],
    ) -> bool:
        """Send notification for processing milestone.

        Args:
            milestone: Milestone description
            run_id: Run identifier
            metrics: Current metrics

        Returns:
            True if notification sent successfully
        """
        if not self._config.notify_on_milestone:
            return False

        message = NotificationMessage(
            title=f"Milestone: {milestone}",
            text=f"Run {run_id} has reached a milestone.",
            color="#439FE0",  # Blue
            fields={
                "Cases Processed": str(metrics.get("cases_processed", 0)),
                "Current Accuracy": f"{metrics.get('accuracy', 0):.2%}",
                "Elapsed Time": self._format_duration(
                    metrics.get("elapsed_seconds", 0)
                ),
                "Improvement Cycles": str(metrics.get("total_cycles", 0)),
            },
        )

        return self._send_slack_notification(message)

    def notify_improvement_cycle(
        self,
        run_id: str,
        cycle_result: CycleResult,
    ) -> bool:
        """Send notification after improvement cycle.

        Args:
            run_id: Run identifier
            cycle_result: Cycle result

        Returns:
            True if notification sent successfully
        """
        if not self._config.notify_on_cycle_complete:
            return False

        accuracy_delta = cycle_result.accuracy_delta
        delta_str = f"{accuracy_delta:+.2%}"
        color = "#36a64f" if accuracy_delta >= 0 else "#E01E5A"  # Green or Red

        message = NotificationMessage(
            title=f"Improvement Cycle {cycle_result.cycle_number} Complete",
            text=f"Run {run_id} completed improvement cycle.",
            color=color,
            fields={
                "Status": cycle_result.status.value,
                "Improvements Applied": str(cycle_result.improvements_applied),
                "Accuracy Change": delta_str,
                "Patterns Found": str(len(cycle_result.failure_patterns)),
            },
        )

        return self._send_slack_notification(message)

    def notify_completion(self, result: RunResult) -> bool:
        """Send notification when run completes.

        Args:
            result: Final run result

        Returns:
            True if notification sent successfully
        """
        if not self._config.notify_on_completion:
            return False

        color = "#36a64f" if result.was_successful else "#E01E5A"  # Green or Red

        message = NotificationMessage(
            title=f"Autonomous Run Complete: {result.run_id}",
            text=f"Run finished with status: {result.status.value}",
            color=color,
            fields={
                "Duration": f"{result.duration_hours:.2f} hours",
                "Final Accuracy": f"{result.final_metrics.overall_accuracy:.2%}",
                "Improvement Cycles": str(
                    result.final_metrics.improvement_cycles_completed
                ),
                "Rules Generated": str(result.final_metrics.rules_generated_total),
            },
        )

        return self._send_slack_notification(message)

    def notify_error(
        self,
        run_id: str,
        error: Exception,
        context: Dict[str, Any],
    ) -> bool:
        """Send notification on error.

        Args:
            run_id: Run identifier
            error: Exception that occurred
            context: Error context

        Returns:
            True if notification sent successfully
        """
        if not self._config.notify_on_error:
            return False

        message = NotificationMessage(
            title=f"Error in Run: {run_id}",
            text=f"An error occurred: {str(error)[:200]}",
            color="#E01E5A",  # Red
            fields={
                "Error Type": type(error).__name__,
                "Cases Processed": str(context.get("cases_processed", 0)),
                "Current Accuracy": f"{context.get('accuracy', 0):.2%}",
                "Last Checkpoint": context.get("last_checkpoint", "None"),
            },
        )

        return self._send_slack_notification(message)

    def notify_checkpoint(
        self,
        run_id: str,
        checkpoint_number: int,
        metrics: Dict[str, Any],
    ) -> bool:
        """Send notification when checkpoint is created.

        Args:
            run_id: Run identifier
            checkpoint_number: Checkpoint number
            metrics: Current metrics

        Returns:
            True if notification sent successfully
        """
        message = NotificationMessage(
            title=f"Checkpoint {checkpoint_number} Created",
            text=f"Run {run_id} created checkpoint.",
            color="#36a64f",  # Green
            fields={
                "Cases Processed": str(metrics.get("cases_processed", 0)),
                "Current Accuracy": f"{metrics.get('accuracy', 0):.2%}",
                "Elapsed Time": self._format_duration(
                    metrics.get("elapsed_seconds", 0)
                ),
            },
        )

        return self._send_slack_notification(message)

    def _send_slack_notification(self, message: NotificationMessage) -> bool:
        """Send a notification to Slack.

        Args:
            message: Notification message

        Returns:
            True if sent successfully
        """
        if not self._config.slack_webhook_url:
            logger.debug("No Slack webhook URL configured")
            return False

        if not self._should_send_notification():
            logger.debug("Rate limiting notification")
            return False

        payload = self._format_slack_payload(message)

        try:
            request = Request(
                self._config.slack_webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urlopen(request, timeout=10) as response:
                if response.status == 200:
                    self._last_notification_time = datetime.now()
                    self._notification_count += 1
                    logger.debug(f"Sent notification: {message.title}")
                    return True
                else:
                    logger.warning(f"Slack notification failed: {response.status}")
                    return False

        except URLError as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending notification: {e}")
            return False

    def _format_slack_payload(self, message: NotificationMessage) -> Dict[str, Any]:
        """Format message as Slack webhook payload.

        Args:
            message: Notification message

        Returns:
            Slack payload dictionary
        """
        fields = [
            {"title": key, "value": value, "short": True}
            for key, value in message.fields.items()
        ]

        attachment = {
            "color": message.color,
            "title": message.title,
            "text": message.text,
            "fields": fields,
            "ts": int(message.timestamp.timestamp()),
            "footer": "LOFT Autonomous Runner",
        }

        return {"attachments": [attachment]}

    def _should_send_notification(self) -> bool:
        """Check if notification should be sent (rate limiting).

        Returns:
            True if notification should be sent
        """
        if self._last_notification_time is None:
            return True

        return (
            datetime.now() - self._last_notification_time
            >= self._min_notification_interval
        )

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string
        """
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"


def create_notification_manager(config: NotificationConfig) -> NotificationManager:
    """Factory function to create a notification manager.

    Args:
        config: Notification configuration

    Returns:
        Configured NotificationManager
    """
    return NotificationManager(config)
