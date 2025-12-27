"""
Coverage tracking for knowledge metrics over time.

Stores and retrieves metrics snapshots to track trends.

Issue #274: Knowledge Coverage Metrics
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from loft.knowledge.coverage_calculator import CoverageCalculator
from loft.knowledge.coverage_schemas import CoverageMetrics, MetricsTrend
from loft.knowledge.database import KnowledgeDatabase

logger = logging.getLogger(__name__)


class CoverageTracker:
    """
    Track coverage metrics over time.

    Stores periodic snapshots and analyzes trends.
    """

    def __init__(
        self,
        knowledge_db: KnowledgeDatabase,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize coverage tracker.

        Args:
            knowledge_db: Knowledge database instance
            storage_path: Path to store metric snapshots (JSON)
        """
        self.db = knowledge_db
        self.calculator = CoverageCalculator(knowledge_db)

        # Default storage path
        if storage_path is None:
            storage_path = Path("metrics_history.json")

        self.storage_path = storage_path

        # Load existing history
        self.history: List[dict] = self._load_history()

    def _load_history(self) -> List[dict]:
        """
        Load metrics history from storage.

        Returns:
            List of metric snapshots
        """
        if not self.storage_path.exists():
            return []

        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                return data.get("snapshots", [])
        except Exception as e:
            logger.error(f"Failed to load history from {self.storage_path}: {e}")
            return []

    def _save_history(self):
        """Save metrics history to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.storage_path, "w") as f:
                json.dump(
                    {
                        "snapshots": self.history,
                        "last_updated": datetime.utcnow().isoformat(),
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Saved {len(self.history)} snapshots to {self.storage_path}")

        except Exception as e:
            logger.error(f"Failed to save history to {self.storage_path}: {e}")

    def take_snapshot(self) -> CoverageMetrics:
        """
        Take a snapshot of current coverage metrics.

        Returns:
            Current CoverageMetrics
        """
        logger.info("Taking coverage metrics snapshot")

        # Calculate current metrics
        metrics = self.calculator.calculate_metrics()

        # Store snapshot
        snapshot = metrics.to_dict()
        self.history.append(snapshot)

        # Save to disk
        self._save_history()

        logger.info(
            f"Snapshot taken: {metrics.total_rules} rules, "
            f"{metrics.domain_count} domains"
        )

        return metrics

    def get_latest_snapshot(self) -> Optional[CoverageMetrics]:
        """
        Get most recent snapshot.

        Returns:
            Latest CoverageMetrics or None
        """
        if not self.history:
            return None

        latest = self.history[-1]
        # Reconstruct CoverageMetrics from dict
        # (simplified - would need full reconstruction)
        metrics = CoverageMetrics()
        metrics.total_rules = latest.get("total_rules", 0)
        metrics.active_rules = latest.get("active_rules", 0)
        metrics.archived_rules = latest.get("archived_rules", 0)
        metrics.total_questions = latest.get("total_questions", 0)
        metrics.answered_questions = latest.get("answered_questions", 0)

        return metrics

    def get_snapshots_since(self, since: datetime) -> List[dict]:
        """
        Get all snapshots since a given time.

        Args:
            since: Get snapshots after this time

        Returns:
            List of snapshots
        """
        snapshots = []

        for snapshot in self.history:
            timestamp_str = snapshot.get("timestamp")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp >= since:
                    snapshots.append(snapshot)

        return snapshots

    def get_trend(
        self,
        metric_name: str,
        domain: Optional[str] = None,
        days: int = 30,
    ) -> MetricsTrend:
        """
        Get trend for a specific metric.

        Args:
            metric_name: Name of metric to track
            domain: Optional domain filter
            days: Number of days to analyze

        Returns:
            MetricsTrend with historical data
        """
        since = datetime.utcnow() - timedelta(days=days)
        snapshots = self.get_snapshots_since(since)

        trend = MetricsTrend(metric_name=metric_name, domain=domain)

        for snapshot in snapshots:
            timestamp_str = snapshot.get("timestamp")
            if not timestamp_str:
                continue

            timestamp = datetime.fromisoformat(timestamp_str)

            # Extract metric value
            value = None

            if domain:
                # Domain-specific metric
                domain_data = snapshot.get("domains", {}).get(domain)
                if domain_data:
                    value = domain_data.get(metric_name)
            else:
                # Overall metric
                value = snapshot.get(metric_name)

                # Check in quality metrics
                if value is None:
                    quality = snapshot.get("quality", {})
                    value = quality.get(metric_name)

            if value is not None:
                trend.add_sample(timestamp, float(value))

        return trend

    def get_domain_trends(self, domain: str, days: int = 30) -> List[MetricsTrend]:
        """
        Get all trends for a domain.

        Args:
            domain: Domain to analyze
            days: Number of days

        Returns:
            List of MetricsTrends
        """
        metrics_to_track = [
            "rule_count",
            "accuracy",
            "avg_confidence",
            "coverage_score",
        ]

        trends = []
        for metric_name in metrics_to_track:
            trend = self.get_trend(metric_name, domain=domain, days=days)
            if trend.values:  # Only include if we have data
                trends.append(trend)

        return trends

    def get_quality_trends(self, days: int = 30) -> List[MetricsTrend]:
        """
        Get quality metric trends.

        Args:
            days: Number of days

        Returns:
            List of MetricsTrends
        """
        metrics_to_track = [
            "avg_confidence",
            "quality_score",
            "high_confidence_rules",
        ]

        trends = []
        for metric_name in metrics_to_track:
            trend = self.get_trend(metric_name, domain=None, days=days)
            if trend.values:
                trends.append(trend)

        return trends

    def compare_snapshots(self, before: datetime, after: datetime) -> dict:
        """
        Compare metrics between two time points.

        Args:
            before: Earlier timestamp
            after: Later timestamp

        Returns:
            Dict with comparison metrics
        """
        # Get snapshots closest to these times
        before_snapshot = None
        after_snapshot = None

        for snapshot in self.history:
            timestamp_str = snapshot.get("timestamp")
            if not timestamp_str:
                continue

            timestamp = datetime.fromisoformat(timestamp_str)

            # Find closest before
            if timestamp <= before:
                if before_snapshot is None:
                    before_snapshot = snapshot
                else:
                    before_ts = datetime.fromisoformat(before_snapshot["timestamp"])
                    if timestamp > before_ts:
                        before_snapshot = snapshot

            # Find closest after
            if timestamp >= after:
                if after_snapshot is None:
                    after_snapshot = snapshot
                else:
                    after_ts = datetime.fromisoformat(after_snapshot["timestamp"])
                    if timestamp < after_ts:
                        after_snapshot = snapshot

        if not before_snapshot or not after_snapshot:
            return {}

        # Calculate changes
        comparison = {
            "before_timestamp": before_snapshot.get("timestamp"),
            "after_timestamp": after_snapshot.get("timestamp"),
            "changes": {},
        }

        # Compare rule counts
        before_rules = before_snapshot.get("total_rules", 0)
        after_rules = after_snapshot.get("total_rules", 0)
        comparison["changes"]["total_rules"] = {
            "before": before_rules,
            "after": after_rules,
            "delta": after_rules - before_rules,
            "percent_change": (
                ((after_rules - before_rules) / before_rules * 100)
                if before_rules > 0
                else 0
            ),
        }

        # Compare accuracy
        before_accuracy = before_snapshot.get("overall_accuracy")
        after_accuracy = after_snapshot.get("overall_accuracy")

        if before_accuracy is not None and after_accuracy is not None:
            comparison["changes"]["overall_accuracy"] = {
                "before": before_accuracy,
                "after": after_accuracy,
                "delta": after_accuracy - before_accuracy,
            }

        # Compare domain counts
        before_domains = len(before_snapshot.get("domains", {}))
        after_domains = len(after_snapshot.get("domains", {}))
        comparison["changes"]["domain_count"] = {
            "before": before_domains,
            "after": after_domains,
            "delta": after_domains - before_domains,
        }

        return comparison

    def get_snapshot_count(self) -> int:
        """Get number of stored snapshots."""
        return len(self.history)

    def clear_old_snapshots(self, days: int = 90):
        """
        Remove snapshots older than specified days.

        Args:
            days: Keep snapshots from last N days
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        filtered = []
        for snapshot in self.history:
            timestamp_str = snapshot.get("timestamp")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp >= cutoff:
                    filtered.append(snapshot)

        removed = len(self.history) - len(filtered)
        self.history = filtered

        if removed > 0:
            self._save_history()
            logger.info(f"Removed {removed} old snapshots (older than {days} days)")
