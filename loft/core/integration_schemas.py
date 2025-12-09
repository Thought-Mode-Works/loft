"""
Schemas for self-modifying system integration.

Defines data structures for tracking improvement cycles, self-analysis,
and end-to-end system operation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loft.core.incorporation import IncorporationResult
from loft.monitoring.performance_schemas import TrendAnalysis


@dataclass
class KnowledgeGap:
    """A gap in the system's knowledge."""

    gap_id: str
    description: str
    missing_predicate: Optional[str] = None
    severity: str = "medium"  # "low", "medium", "high", "critical"
    context: Dict[str, Any] = field(default_factory=dict)
    identified_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gap_id": self.gap_id,
            "description": self.description,
            "missing_predicate": self.missing_predicate,
            "severity": self.severity,
            "context": self.context,
            "identified_at": self.identified_at.isoformat(),
        }


@dataclass
class ImprovementCycleResult:
    """Result of a complete self-improvement cycle."""

    cycle_number: int
    timestamp: datetime
    gaps_identified: int
    variants_generated: int
    rules_incorporated: int
    rules_pending_review: int
    baseline_accuracy: float
    final_accuracy: float
    overall_improvement: float
    successful_incorporations: List[IncorporationResult] = field(default_factory=list)
    status: str = "success"  # "success", "no_improvements", "failure"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp.isoformat(),
            "gaps_identified": self.gaps_identified,
            "variants_generated": self.variants_generated,
            "rules_incorporated": self.rules_incorporated,
            "rules_pending_review": self.rules_pending_review,
            "baseline_accuracy": self.baseline_accuracy,
            "final_accuracy": self.final_accuracy,
            "overall_improvement": self.overall_improvement,
            "status": self.status,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Improvement Cycle #{self.cycle_number}",
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Status: {self.status}",
            "",
            "Metrics:",
            f"  Gaps identified: {self.gaps_identified}",
            f"  Variants generated: {self.variants_generated}",
            f"  Rules incorporated: {self.rules_incorporated}",
            f"  Rules pending review: {self.rules_pending_review}",
            "",
            "Performance:",
            f"  Baseline accuracy: {self.baseline_accuracy:.2%}",
            f"  Final accuracy: {self.final_accuracy:.2%}",
            f"  Improvement: {self.overall_improvement:+.2%}",
        ]
        return "\n".join(lines)


@dataclass
class SelfAnalysisReport:
    """System's self-analysis and reflection."""

    generated_at: datetime
    narrative: str
    incorporation_success_rate: float
    best_strategy: Optional[str]
    performance_trends: Dict[str, TrendAnalysis]
    identified_weaknesses: List[str]
    confidence_in_self: float

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Self-Analysis Report",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Narrative",
            self.narrative,
            "",
            "## Key Metrics",
            f"- Incorporation success rate: {self.incorporation_success_rate:.1%}",
            f"- Best strategy: {self.best_strategy or 'N/A'}",
            f"- Self-confidence: {self.confidence_in_self:.1%}",
            "",
        ]

        if self.performance_trends:
            lines.append("## Performance Trends")
            for metric, trend in self.performance_trends.items():
                emoji = {"improving": "↑", "stable": "→", "degrading": "↓"}[
                    trend.trend_direction
                ]
                lines.append(f"- {metric}: {trend.trend_direction} {emoji}")
            lines.append("")

        if self.identified_weaknesses:
            lines.append("## Identified Weaknesses")
            for weakness in self.identified_weaknesses:
                lines.append(f"- {weakness}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "narrative": self.narrative,
            "incorporation_success_rate": self.incorporation_success_rate,
            "best_strategy": self.best_strategy,
            "performance_trends": {
                k: v.to_dict() for k, v in self.performance_trends.items()
            },
            "identified_weaknesses": self.identified_weaknesses,
            "confidence_in_self": self.confidence_in_self,
        }


@dataclass
class SystemHealthReport:
    """Comprehensive system health check."""

    generated_at: datetime
    overall_health: str  # "healthy", "degraded", "critical"
    components_status: Dict[str, str]
    total_rules: int
    rules_by_layer: Dict[str, int]
    recent_incorporations: int
    recent_rollbacks: int
    active_alerts: int
    pending_reviews: int
    recommendations: List[str]

    def to_markdown(self) -> str:
        """Generate markdown report."""
        health_emoji = {"healthy": "✅", "degraded": "⚠️", "critical": "🔴"}[
            self.overall_health
        ]

        lines = [
            "# System Health Report",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Overall Health: {self.overall_health} {health_emoji}",
            "",
            "## Component Status",
        ]

        for component, status in self.components_status.items():
            status_emoji = {"healthy": "✅", "degraded": "⚠️", "critical": "🔴"}.get(
                status, "❓"
            )
            lines.append(f"- {component}: {status} {status_emoji}")

        lines.extend(
            [
                "",
                "## System Metrics",
                f"- Total rules: {self.total_rules}",
                "- Rules by layer:",
            ]
        )

        for layer, count in self.rules_by_layer.items():
            lines.append(f"  - {layer}: {count}")

        lines.extend(
            [
                f"- Recent incorporations: {self.recent_incorporations}",
                f"- Recent rollbacks: {self.recent_rollbacks}",
                f"- Active alerts: {self.active_alerts}",
                f"- Pending reviews: {self.pending_reviews}",
                "",
                "## Recommendations",
            ]
        )

        for rec in self.recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "overall_health": self.overall_health,
            "components_status": self.components_status,
            "total_rules": self.total_rules,
            "rules_by_layer": self.rules_by_layer,
            "recent_incorporations": self.recent_incorporations,
            "recent_rollbacks": self.recent_rollbacks,
            "active_alerts": self.active_alerts,
            "pending_reviews": self.pending_reviews,
            "recommendations": self.recommendations,
        }
