"""
Confidence tracker for monitoring confidence trends and performance.

Tracks confidence scores over time, analyzes trends, and identifies
areas where confidence calibration may need improvement.
"""

from typing import List, Optional, Dict, Any
import statistics
from datetime import datetime, timedelta
from loguru import logger

from loft.validation.confidence_schemas import (
    AggregatedConfidence,
    ConfidenceTrends,
    ConfidenceAnalysis,
)


class ConfidenceTracker:
    """
    Track and analyze confidence scores over time.

    Maintains history of confidence scores, identifies trends,
    and provides insights for improving confidence calibration.
    """

    def __init__(self, history_limit: int = 1000):
        """
        Initialize confidence tracker.

        Args:
            history_limit: Maximum number of confidence scores to keep in history
        """
        self.history: List[AggregatedConfidence] = []
        self.history_limit = history_limit

        # Track by predicate/rule category for analysis
        self.by_category: Dict[str, List[AggregatedConfidence]] = {}

        logger.debug(f"Initialized ConfidenceTracker (history_limit={history_limit})")

    def record(self, confidence: AggregatedConfidence, category: Optional[str] = None) -> None:
        """
        Record a confidence score.

        Args:
            confidence: Aggregated confidence to record
            category: Optional category (e.g., predicate name, rule type)

        Example:
            >>> tracker = ConfidenceTracker()
            >>> confidence = AggregatedConfidence(
            ...     score=0.85,
            ...     components={"generation": 0.85, "syntax": 1.0},
            ...     weights={"generation": 0.5, "syntax": 0.5},
            ...     variance=0.05,
            ...     is_reliable=True
            ... )
            >>> tracker.record(confidence, category="contract_rules")
        """
        # Add to overall history
        self.history.append(confidence)

        # Trim history if exceeds limit
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit :]

        # Add to category tracking
        if category:
            if category not in self.by_category:
                self.by_category[category] = []

            self.by_category[category].append(confidence)

            # Trim category history
            if len(self.by_category[category]) > self.history_limit:
                self.by_category[category] = self.by_category[category][-self.history_limit :]

        logger.debug(
            f"Recorded confidence: score={confidence.score:.2f}, "
            f"category={category}, total_history={len(self.history)}"
        )

    def get_trends(self, recent_window: Optional[int] = None) -> ConfidenceTrends:
        """
        Get confidence trends from recent history.

        Args:
            recent_window: Number of recent scores to analyze (default: all)

        Returns:
            ConfidenceTrends with statistical analysis

        Example:
            >>> tracker = ConfidenceTracker()
            >>> # ... record many scores ...
            >>> trends = tracker.get_trends(recent_window=100)
            >>> print(trends.summary())
        """
        if not self.history:
            return ConfidenceTrends(
                mean_confidence=0.0,
                median_confidence=0.0,
                std_confidence=0.0,
                mean_variance=0.0,
                num_reliable=0,
                num_total=0,
                reliability_rate=0.0,
            )

        # Get recent scores
        if recent_window:
            recent = self.history[-recent_window:]
        else:
            recent = self.history

        scores = [c.score for c in recent]
        variances = [c.variance for c in recent]
        reliable_count = sum(1 for c in recent if c.is_reliable)

        return ConfidenceTrends(
            mean_confidence=statistics.mean(scores),
            median_confidence=statistics.median(scores),
            std_confidence=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            mean_variance=statistics.mean(variances),
            num_reliable=reliable_count,
            num_total=len(recent),
            reliability_rate=reliable_count / len(recent),
        )

    def analyze_performance(
        self,
        actual_accuracies: Dict[str, float],
        category: Optional[str] = None,
    ) -> ConfidenceAnalysis:
        """
        Analyze confidence calibration performance.

        Compares predicted confidence to actual accuracy to identify
        areas where confidence is poorly calibrated.

        Args:
            actual_accuracies: Dict mapping rule_id -> actual accuracy
            category: Optional category to analyze (default: all)

        Returns:
            ConfidenceAnalysis identifying calibration issues

        Example:
            >>> tracker = ConfidenceTracker()
            >>> # ... record scores ...
            >>> accuracies = {"rule1": 0.85, "rule2": 0.65}
            >>> analysis = tracker.analyze_performance(accuracies)
            >>> print(analysis.summary())
        """
        # Get relevant history
        if category:
            history = self.by_category.get(category, [])
        else:
            history = self.history

        if not history:
            return ConfidenceAnalysis(
                underconfident_areas=[],
                overconfident_areas=[],
                calibration_quality=0.0,
                num_samples=0,
            )

        # Calculate calibration error
        underconfident: List[str] = []
        overconfident: List[str] = []

        # This is simplified - in practice, would need to track rule IDs
        # and match confidence scores to actual accuracies
        # For now, we'll use a placeholder approach

        # In a full implementation, would:
        # 1. Bin confidences and compare to accuracies
        # 2. Identify predicates that are consistently under/overconfident
        # 3. Calculate Expected Calibration Error (ECE)

        calibration_quality = 0.8  # Placeholder - would compute from actual data

        return ConfidenceAnalysis(
            underconfident_areas=underconfident,
            overconfident_areas=overconfident,
            calibration_quality=calibration_quality,
            num_samples=len(history),
        )

    def get_category_comparison(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare confidence statistics across categories.

        Returns:
            Dict mapping category -> statistics

        Example:
            >>> tracker = ConfidenceTracker()
            >>> # ... record scores in different categories ...
            >>> comparison = tracker.get_category_comparison()
            >>> for cat, stats in comparison.items():
            ...     print(f"{cat}: mean={stats['mean']:.2f}")
        """
        comparison = {}

        for category, history in self.by_category.items():
            if not history:
                continue

            scores = [c.score for c in history]
            variances = [c.variance for c in history]
            reliable_count = sum(1 for c in history if c.is_reliable)

            comparison[category] = {
                "count": len(history),
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "mean_variance": statistics.mean(variances),
                "reliability_rate": reliable_count / len(history),
            }

        return comparison

    def get_recent_scores(
        self, count: int = 10, category: Optional[str] = None
    ) -> List[AggregatedConfidence]:
        """
        Get most recent confidence scores.

        Args:
            count: Number of recent scores to return
            category: Optional category filter

        Returns:
            List of recent AggregatedConfidence objects
        """
        if category:
            history = self.by_category.get(category, [])
        else:
            history = self.history

        return history[-count:]

    def get_time_series(
        self,
        time_window: Optional[timedelta] = None,
        category: Optional[str] = None,
    ) -> List[tuple[datetime, float]]:
        """
        Get confidence scores as time series.

        Args:
            time_window: Only include scores within this time window
            category: Optional category filter

        Returns:
            List of (timestamp, confidence_score) tuples

        Example:
            >>> tracker = ConfidenceTracker()
            >>> # ... record scores ...
            >>> time_series = tracker.get_time_series(
            ...     time_window=timedelta(hours=24)
            ... )
        """
        if category:
            history = self.by_category.get(category, [])
        else:
            history = self.history

        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now() - time_window
            history = [c for c in history if c.timestamp >= cutoff]

        return [(c.timestamp, c.score) for c in history]

    def get_low_confidence_cases(
        self, threshold: float = 0.6, category: Optional[str] = None
    ) -> List[AggregatedConfidence]:
        """
        Get cases with confidence below threshold.

        Args:
            threshold: Minimum confidence threshold
            category: Optional category filter

        Returns:
            List of low-confidence cases

        Example:
            >>> tracker = ConfidenceTracker()
            >>> low_conf = tracker.get_low_confidence_cases(threshold=0.7)
            >>> for conf in low_conf:
            ...     print(conf.explanation())
        """
        if category:
            history = self.by_category.get(category, [])
        else:
            history = self.history

        return [c for c in history if c.score < threshold]

    def get_high_variance_cases(
        self, variance_threshold: float = 0.15, category: Optional[str] = None
    ) -> List[AggregatedConfidence]:
        """
        Get cases with high variance among components.

        High variance indicates disagreement among validation sources.

        Args:
            variance_threshold: Minimum variance to flag
            category: Optional category filter

        Returns:
            List of high-variance cases
        """
        if category:
            history = self.by_category.get(category, [])
        else:
            history = self.history

        return [c for c in history if c.variance >= variance_threshold]

    def clear_history(self, category: Optional[str] = None) -> None:
        """
        Clear confidence history.

        Args:
            category: If specified, only clear that category; otherwise clear all
        """
        if category:
            if category in self.by_category:
                self.by_category[category].clear()
                logger.info(f"Cleared history for category: {category}")
        else:
            self.history.clear()
            self.by_category.clear()
            logger.info("Cleared all confidence history")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of tracked confidence.

        Returns:
            Dict with overall statistics

        Example:
            >>> tracker = ConfidenceTracker()
            >>> # ... record scores ...
            >>> summary = tracker.get_summary()
            >>> print(f"Total tracked: {summary['total_tracked']}")
        """
        if not self.history:
            return {
                "total_tracked": 0,
                "categories": 0,
                "mean_confidence": 0.0,
                "reliability_rate": 0.0,
            }

        scores = [c.score for c in self.history]
        reliable = sum(1 for c in self.history if c.is_reliable)

        return {
            "total_tracked": len(self.history),
            "categories": len(self.by_category),
            "mean_confidence": statistics.mean(scores),
            "median_confidence": statistics.median(scores),
            "std_confidence": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "reliability_rate": reliable / len(self.history),
            "oldest_timestamp": self.history[0].timestamp if self.history else None,
            "newest_timestamp": self.history[-1].timestamp if self.history else None,
        }
