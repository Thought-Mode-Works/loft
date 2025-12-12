"""
Schemas for performance monitoring and metrics tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""

    timestamp: datetime
    core_version_id: str

    # Accuracy metrics
    overall_accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Rule metrics
    total_rules: int
    rules_by_layer: Dict[str, int]  # Use string keys for JSON serialization
    avg_confidence: float

    # Performance metrics
    query_latency_ms: float
    memory_usage_mb: float

    # Consistency metrics
    logical_consistency_score: float
    stratification_violations: int

    # Experiential learning
    rules_incorporated_today: int
    rollbacks_today: int

    # Test coverage
    test_cases_passing: int
    test_cases_total: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "core_version_id": self.core_version_id,
            "overall_accuracy": self.overall_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "total_rules": self.total_rules,
            "rules_by_layer": self.rules_by_layer,
            "avg_confidence": self.avg_confidence,
            "query_latency_ms": self.query_latency_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "logical_consistency_score": self.logical_consistency_score,
            "stratification_violations": self.stratification_violations,
            "rules_incorporated_today": self.rules_incorporated_today,
            "rollbacks_today": self.rollbacks_today,
            "test_cases_passing": self.test_cases_passing,
            "test_cases_total": self.test_cases_total,
        }


@dataclass
class TrendAnalysis:
    """Analysis of performance trends over time."""

    metric_name: str
    current_value: float
    trend_direction: Literal["improving", "stable", "degrading"]
    change_rate: float  # per day
    confidence: float
    alert_level: Literal["none", "watch", "warning", "critical"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "trend_direction": self.trend_direction,
            "change_rate": self.change_rate,
            "confidence": self.confidence,
            "alert_level": self.alert_level,
        }


@dataclass
class RegressionAlert:
    """Alert for detected regression in metrics."""

    metric: str
    baseline_value: float
    current_value: float
    degradation: float
    threshold: float
    severity: Literal["warning", "critical"]
    detected_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric": self.metric,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "degradation": self.degradation,
            "threshold": self.threshold,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class PerformanceAlert:
    """Alert for performance issues."""

    alert_id: str
    timestamp: datetime
    severity: Literal["low", "medium", "high", "critical"]
    category: str  # "regression", "integrity", "performance", etc.
    message: str
    metric: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "metric": self.metric,
            "details": self.details,
            "resolved": self.resolved,
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""

    generated_at: datetime
    current_snapshot: Optional[PerformanceSnapshot]
    trends: Dict[str, TrendAnalysis]
    regressions: List[RegressionAlert]
    active_alerts: List[PerformanceAlert]
    recommendations: List[str]

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Performance Report",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Current Status",
        ]

        if self.current_snapshot:
            status = "✅ Stable"
            if self.regressions:
                status = "⚠️ Regressions Detected"
            elif any(a.severity in ["high", "critical"] for a in self.active_alerts):
                status = "🔴 Critical Issues"

            lines.append(f"**Status**: {status}")
            lines.append("")
            lines.append("### Key Metrics")
            lines.append(f"- Accuracy: {self.current_snapshot.overall_accuracy:.1%}")
            lines.append(f"- Precision: {self.current_snapshot.precision:.1%}")
            lines.append(f"- Recall: {self.current_snapshot.recall:.1%}")
            lines.append(f"- F1 Score: {self.current_snapshot.f1_score:.1%}")
            lines.append(f"- Total Rules: {self.current_snapshot.total_rules}")
            lines.append(
                f"- Test Coverage: {self.current_snapshot.test_cases_passing}/{self.current_snapshot.test_cases_total}"
            )
            lines.append("")

        if self.trends:
            lines.append("### Trends")
            for metric, trend in self.trends.items():
                emoji = {
                    "improving": "↑",
                    "stable": "→",
                    "degrading": "↓",
                }[trend.trend_direction]
                alert_emoji = {
                    "none": "✅",
                    "watch": "👀",
                    "warning": "🟡",
                    "critical": "🔴",
                }[trend.alert_level]
                lines.append(
                    f"- {metric}: {trend.trend_direction} {emoji} "
                    f"({trend.change_rate:+.4f}/day) {alert_emoji}"
                )
            lines.append("")

        if self.regressions:
            lines.append("### Regressions Detected")
            for reg in self.regressions:
                lines.append(
                    f"- **{reg.metric}**: {reg.baseline_value:.2%} → "
                    f"{reg.current_value:.2%} ({reg.severity.upper()})"
                )
            lines.append("")

        if self.active_alerts:
            lines.append("### Active Alerts")
            for alert in self.active_alerts:
                severity_emoji = {
                    "low": "ℹ️",
                    "medium": "🟡",
                    "high": "🟠",
                    "critical": "🔴",
                }[alert.severity]
                lines.append(
                    f"- {severity_emoji} **{alert.category}**: {alert.message}"
                )
            lines.append("")

        lines.append("### Recommendations")
        for rec in self.recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "current_snapshot": (
                self.current_snapshot.to_dict() if self.current_snapshot else None
            ),
            "trends": {k: v.to_dict() for k, v in self.trends.items()},
            "regressions": [r.to_dict() for r in self.regressions],
            "active_alerts": [a.to_dict() for a in self.active_alerts],
            "recommendations": self.recommendations,
        }
