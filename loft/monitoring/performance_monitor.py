"""
Performance monitoring and regression detection.

Tracks system performance over time, detects regressions, analyzes trends,
and generates comprehensive performance reports.
"""

import statistics
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger

from loft.monitoring.performance_schemas import (
    PerformanceAlert,
    PerformanceReport,
    PerformanceSnapshot,
    RegressionAlert,
    TrendAnalysis,
)


class PerformanceMonitor:
    """
    Monitor system performance and detect regressions.

    Workflow:
    1. Capture periodic performance snapshots
    2. Store snapshots in history
    3. Detect regressions by comparing recent metrics
    4. Analyze trends over configurable lookback window
    5. Generate alerts for concerning patterns
    6. Create comprehensive reports
    """

    def __init__(
        self,
        regression_threshold: float = 0.02,
        trend_lookback_days: int = 7,
        max_snapshots: int = 1000,
    ):
        """
        Initialize performance monitor.

        Args:
            regression_threshold: Minimum degradation to trigger regression alert (default 2%)
            trend_lookback_days: Days of history to analyze for trends
            max_snapshots: Maximum snapshots to retain in history
        """
        self.regression_threshold = regression_threshold
        self.trend_lookback_days = trend_lookback_days
        self.max_snapshots = max_snapshots
        self.snapshot_history: List[PerformanceSnapshot] = []
        self.active_alerts: List[PerformanceAlert] = []

        logger.info(
            f"Initialized PerformanceMonitor (threshold: {regression_threshold:.1%}, "
            f"lookback: {trend_lookback_days} days)"
        )

    def capture_snapshot(
        self,
        core_version_id: str,
        overall_accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        total_rules: int,
        rules_by_layer: Dict[str, int],
        avg_confidence: float,
        query_latency_ms: float = 0.0,
        memory_usage_mb: float = 0.0,
        logical_consistency_score: float = 1.0,
        stratification_violations: int = 0,
        rules_incorporated_today: int = 0,
        rollbacks_today: int = 0,
        test_cases_passing: int = 0,
        test_cases_total: int = 0,
    ) -> PerformanceSnapshot:
        """
        Capture a performance snapshot.

        Args:
            core_version_id: Version identifier for the symbolic core
            overall_accuracy: Overall accuracy metric
            precision: Precision metric
            recall: Recall metric
            f1_score: F1 score metric
            total_rules: Total number of rules in system
            rules_by_layer: Rule count by stratification layer
            avg_confidence: Average rule confidence
            query_latency_ms: Query latency in milliseconds
            memory_usage_mb: Memory usage in MB
            logical_consistency_score: Logical consistency score
            stratification_violations: Count of stratification violations
            rules_incorporated_today: Rules incorporated today
            rollbacks_today: Rollbacks today
            test_cases_passing: Number of passing test cases
            test_cases_total: Total number of test cases

        Returns:
            PerformanceSnapshot object
        """
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            core_version_id=core_version_id,
            overall_accuracy=overall_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_rules=total_rules,
            rules_by_layer=rules_by_layer,
            avg_confidence=avg_confidence,
            query_latency_ms=query_latency_ms,
            memory_usage_mb=memory_usage_mb,
            logical_consistency_score=logical_consistency_score,
            stratification_violations=stratification_violations,
            rules_incorporated_today=rules_incorporated_today,
            rollbacks_today=rollbacks_today,
            test_cases_passing=test_cases_passing,
            test_cases_total=test_cases_total,
        )

        self.snapshot_history.append(snapshot)

        # Trim history if needed
        if len(self.snapshot_history) > self.max_snapshots:
            self.snapshot_history = self.snapshot_history[-self.max_snapshots :]

        logger.debug(
            f"Captured snapshot: accuracy={overall_accuracy:.2%}, "
            f"rules={total_rules}, violations={stratification_violations}"
        )

        return snapshot

    def detect_regressions(
        self,
        current_snapshot: PerformanceSnapshot,
        baseline_snapshot: Optional[PerformanceSnapshot] = None,
    ) -> List[RegressionAlert]:
        """
        Detect regressions by comparing current snapshot to baseline.

        Args:
            current_snapshot: Current performance snapshot
            baseline_snapshot: Baseline to compare against (defaults to most recent previous)

        Returns:
            List of detected regression alerts
        """
        if baseline_snapshot is None:
            if len(self.snapshot_history) < 2:
                return []
            baseline_snapshot = self.snapshot_history[-2]

        regressions = []

        # Check each critical metric
        metrics_to_check = [
            ("overall_accuracy", "Overall Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1 Score"),
            ("logical_consistency_score", "Logical Consistency"),
        ]

        for attr, name in metrics_to_check:
            baseline_value = getattr(baseline_snapshot, attr)
            current_value = getattr(current_snapshot, attr)
            degradation = baseline_value - current_value

            if degradation > self.regression_threshold:
                severity = "critical" if degradation > self.regression_threshold * 2 else "warning"

                alert = RegressionAlert(
                    metric=name,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    degradation=degradation,
                    threshold=self.regression_threshold,
                    severity=severity,
                    detected_at=datetime.now(),
                )

                regressions.append(alert)

                logger.warning(
                    f"Regression detected in {name}: {baseline_value:.2%} -> "
                    f"{current_value:.2%} (degradation: {degradation:.2%})"
                )

        return regressions

    def analyze_trends(
        self, metric_name: str, lookback_days: Optional[int] = None
    ) -> Optional[TrendAnalysis]:
        """
        Analyze trends for a specific metric.

        Args:
            metric_name: Name of metric to analyze
            lookback_days: Days to look back (defaults to configured lookback)

        Returns:
            TrendAnalysis object or None if insufficient data
        """
        if len(self.snapshot_history) < 2:
            return None

        lookback = lookback_days if lookback_days is not None else self.trend_lookback_days
        cutoff_date = datetime.now() - timedelta(days=lookback)

        # Filter snapshots within lookback window
        recent_snapshots = [s for s in self.snapshot_history if s.timestamp >= cutoff_date]

        if len(recent_snapshots) < 2:
            return None

        # Map metric name to attribute
        metric_map = {
            "Overall Accuracy": "overall_accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1 Score": "f1_score",
            "Logical Consistency": "logical_consistency_score",
            "Query Latency": "query_latency_ms",
            "Memory Usage": "memory_usage_mb",
        }

        attr = metric_map.get(metric_name)
        if attr is None:
            return None

        # Extract values
        values = [getattr(s, attr) for s in recent_snapshots]
        current_value = values[-1]

        # Calculate linear regression slope (change rate)
        n = len(values)
        x_values = list(range(n))
        mean_x = statistics.mean(x_values)
        mean_y = statistics.mean(values)

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, values))
        denominator = sum((x - mean_x) ** 2 for x in x_values)

        change_rate = numerator / denominator if denominator != 0 else 0.0

        # Scale change rate to per-day
        if n > 1:
            days_span = (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).days
            if days_span > 0:
                change_rate = change_rate * n / days_span

        # Determine trend direction
        if abs(change_rate) < 0.001:
            trend_direction = "stable"
        elif change_rate > 0:
            trend_direction = "improving"
        else:
            trend_direction = "degrading"

        # Calculate confidence based on data consistency
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        confidence = max(0.5, 1.0 - std_dev)

        # Determine alert level
        if trend_direction == "degrading":
            if abs(change_rate) > 0.01:
                alert_level = "critical"
            elif abs(change_rate) > 0.005:
                alert_level = "warning"
            else:
                alert_level = "watch"
        else:
            alert_level = "none"

        return TrendAnalysis(
            metric_name=metric_name,
            current_value=current_value,
            trend_direction=trend_direction,
            change_rate=change_rate,
            confidence=confidence,
            alert_level=alert_level,
        )

    def generate_report(self) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Returns:
            PerformanceReport with current status, trends, and recommendations
        """
        current_snapshot = self.snapshot_history[-1] if self.snapshot_history else None

        # Detect regressions
        regressions = []
        if current_snapshot and len(self.snapshot_history) >= 2:
            regressions = self.detect_regressions(current_snapshot)

        # Analyze trends for key metrics
        trends = {}
        metric_names = [
            "Overall Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Logical Consistency",
        ]

        for metric in metric_names:
            trend = self.analyze_trends(metric)
            if trend:
                trends[metric] = trend

        # Generate recommendations
        recommendations = self._generate_recommendations(current_snapshot, trends, regressions)

        report = PerformanceReport(
            generated_at=datetime.now(),
            current_snapshot=current_snapshot,
            trends=trends,
            regressions=regressions,
            active_alerts=self.active_alerts.copy(),
            recommendations=recommendations,
        )

        return report

    def add_alert(
        self,
        severity: str,
        category: str,
        message: str,
        metric: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> PerformanceAlert:
        """
        Add a performance alert.

        Args:
            severity: Alert severity ("low", "medium", "high", "critical")
            category: Alert category ("regression", "integrity", "performance", etc.)
            message: Alert message
            metric: Optional metric name
            details: Optional additional details

        Returns:
            Created PerformanceAlert
        """
        alert = PerformanceAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=message,
            metric=metric,
            details=details or {},
            resolved=False,
        )

        self.active_alerts.append(alert)
        logger.warning(f"Performance alert [{severity}] {category}: {message}")

        return alert

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve a performance alert.

        Args:
            alert_id: ID of alert to resolve

        Returns:
            True if alert found and resolved
        """
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"Resolved alert: {alert.message}")
                return True
        return False

    def get_snapshot_history(
        self, limit: Optional[int] = None, since: Optional[datetime] = None
    ) -> List[PerformanceSnapshot]:
        """
        Get snapshot history.

        Args:
            limit: Maximum number of snapshots to return
            since: Only return snapshots after this datetime

        Returns:
            List of performance snapshots
        """
        snapshots = self.snapshot_history

        if since:
            snapshots = [s for s in snapshots if s.timestamp >= since]

        if limit:
            snapshots = snapshots[-limit:]

        return snapshots

    def _generate_recommendations(
        self,
        current_snapshot: Optional[PerformanceSnapshot],
        trends: Dict[str, TrendAnalysis],
        regressions: List[RegressionAlert],
    ) -> List[str]:
        """
        Generate recommendations based on current state.

        Args:
            current_snapshot: Current performance snapshot
            trends: Trend analysis results
            regressions: Detected regressions

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Regression recommendations
        if regressions:
            critical_regressions = [r for r in regressions if r.severity == "critical"]
            if critical_regressions:
                recommendations.append(
                    f"URGENT: {len(critical_regressions)} critical regression(s) detected. "
                    "Consider rolling back recent changes."
                )
            else:
                recommendations.append(
                    f"{len(regressions)} regression(s) detected. Monitor closely and "
                    "investigate root causes."
                )

        # Trend recommendations
        degrading_trends = [t for t in trends.values() if t.trend_direction == "degrading"]
        if degrading_trends:
            high_concern = [t for t in degrading_trends if t.alert_level in ["warning", "critical"]]
            if high_concern:
                metrics = ", ".join([t.metric_name for t in high_concern])
                recommendations.append(
                    f"Degrading trends in: {metrics}. Review recent rule incorporations."
                )

        # Stratification violation recommendations
        if current_snapshot and current_snapshot.stratification_violations > 0:
            recommendations.append(
                f"{current_snapshot.stratification_violations} stratification violation(s) detected. "
                "Audit predicate dependencies and layer assignments."
            )

        # Rollback rate recommendations
        if current_snapshot and current_snapshot.rollbacks_today > 0:
            if current_snapshot.rules_incorporated_today > 0:
                rollback_rate = (
                    current_snapshot.rollbacks_today / current_snapshot.rules_incorporated_today
                )
                if rollback_rate > 0.2:
                    recommendations.append(
                        f"High rollback rate ({rollback_rate:.1%}). "
                        "Consider stricter rule validation or confidence thresholds."
                    )

        # Test coverage recommendations
        if current_snapshot and current_snapshot.test_cases_total > 0:
            pass_rate = current_snapshot.test_cases_passing / current_snapshot.test_cases_total
            if pass_rate < 0.95:
                recommendations.append(
                    f"Test pass rate is {pass_rate:.1%}. "
                    "Address failing tests before incorporating more rules."
                )

        # Default recommendation
        if not recommendations:
            recommendations.append("System performance is stable. Continue monitoring.")

        return recommendations
