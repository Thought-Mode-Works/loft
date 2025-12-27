"""
Data schemas for knowledge coverage metrics.

Defines data structures for tracking coverage, quality, and trends.

Issue #274: Knowledge Coverage Metrics
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class DomainMetrics:
    """
    Metrics for a specific legal domain.

    Tracks rules, questions, accuracy, and quality for one domain.
    """

    domain: str
    rule_count: int = 0
    active_rule_count: int = 0
    archived_rule_count: int = 0
    question_count: int = 0
    answered_question_count: int = 0
    avg_confidence: float = 0.0
    accuracy: Optional[float] = None  # % correct answers
    avg_rule_quality: float = 0.0
    doctrines: List[str] = field(default_factory=list)
    jurisdictions: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def coverage_score(self) -> float:
        """
        Calculate overall coverage score (0.0-1.0).

        Combines rule count, accuracy, and confidence.
        """
        if self.rule_count == 0:
            return 0.0

        # Weight factors
        rule_factor = min(self.rule_count / 100.0, 1.0)  # Cap at 100 rules
        confidence_factor = self.avg_confidence
        accuracy_factor = self.accuracy if self.accuracy is not None else 0.5

        # Weighted average
        score = (
            (rule_factor * 0.4) + (confidence_factor * 0.3) + (accuracy_factor * 0.3)
        )
        return min(max(score, 0.0), 1.0)

    def __str__(self) -> str:
        """String representation."""
        return (
            f"DomainMetrics({self.domain}): "
            f"{self.rule_count} rules, "
            f"{self.accuracy:.1%} accuracy"
            if self.accuracy
            else "no accuracy data"
        )


@dataclass
class DoctrineMetrics:
    """
    Metrics for a specific legal doctrine.

    Tracks coverage for doctrines within domains.
    """

    doctrine: str
    domain: str
    rule_count: int = 0
    avg_confidence: float = 0.0
    question_count: int = 0
    accuracy: Optional[float] = None

    def __str__(self) -> str:
        """String representation."""
        return f"DoctrineMetrics({self.doctrine}): {self.rule_count} rules"


@dataclass
class JurisdictionMetrics:
    """
    Metrics for a specific jurisdiction.

    Tracks rules across jurisdictions (federal, state-level, etc.).
    """

    jurisdiction: str
    rule_count: int = 0
    domains: List[str] = field(default_factory=list)
    avg_confidence: float = 0.0

    def __str__(self) -> str:
        """String representation."""
        return (
            f"JurisdictionMetrics({self.jurisdiction}): "
            f"{self.rule_count} rules across {len(self.domains)} domains"
        )


@dataclass
class QualityMetrics:
    """
    Quality metrics for rules.

    Tracks rule quality indicators.
    """

    total_rules: int = 0
    high_confidence_rules: int = 0  # confidence >= 0.9
    medium_confidence_rules: int = 0  # 0.7 <= confidence < 0.9
    low_confidence_rules: int = 0  # confidence < 0.7
    avg_confidence: float = 0.0
    rules_with_reasoning: int = 0
    rules_with_sources: int = 0
    validated_rules: int = 0  # validation_count > 0

    @property
    def quality_score(self) -> float:
        """
        Calculate overall quality score (0.0-1.0).

        Based on confidence distribution and metadata completeness.
        """
        if self.total_rules == 0:
            return 0.0

        # Confidence score
        confidence_score = (
            (self.high_confidence_rules * 1.0)
            + (self.medium_confidence_rules * 0.7)
            + (self.low_confidence_rules * 0.3)
        ) / self.total_rules

        # Metadata completeness
        reasoning_ratio = self.rules_with_reasoning / self.total_rules
        sources_ratio = self.rules_with_sources / self.total_rules

        # Weighted average
        quality = (
            (confidence_score * 0.5) + (reasoning_ratio * 0.3) + (sources_ratio * 0.2)
        )

        return min(max(quality, 0.0), 1.0)

    def __str__(self) -> str:
        """String representation."""
        return (
            f"QualityMetrics: {self.total_rules} rules, "
            f"quality score {self.quality_score:.2f}"
        )


@dataclass
class CoverageMetrics:
    """
    Overall knowledge base coverage metrics.

    Aggregates metrics across all domains, doctrines, and jurisdictions.
    """

    total_rules: int = 0
    active_rules: int = 0
    archived_rules: int = 0
    total_questions: int = 0
    answered_questions: int = 0
    domains: Dict[str, DomainMetrics] = field(default_factory=dict)
    doctrines: Dict[str, DoctrineMetrics] = field(default_factory=dict)
    jurisdictions: Dict[str, JurisdictionMetrics] = field(default_factory=dict)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def domain_count(self) -> int:
        """Number of domains covered."""
        return len(self.domains)

    @property
    def avg_confidence(self) -> float:
        """Average confidence across all rules."""
        return self.quality.avg_confidence

    @property
    def overall_accuracy(self) -> Optional[float]:
        """Overall accuracy across all answered questions."""
        if self.answered_questions == 0:
            return None

        # Aggregate accuracy from domains
        total_correct = 0
        total_answered = 0

        for domain_metrics in self.domains.values():
            if (
                domain_metrics.accuracy is not None
                and domain_metrics.answered_question_count > 0
            ):
                correct = int(
                    domain_metrics.accuracy * domain_metrics.answered_question_count
                )
                total_correct += correct
                total_answered += domain_metrics.answered_question_count

        if total_answered == 0:
            return None

        return total_correct / total_answered

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_rules": self.total_rules,
            "active_rules": self.active_rules,
            "archived_rules": self.archived_rules,
            "total_questions": self.total_questions,
            "answered_questions": self.answered_questions,
            "domain_count": self.domain_count,
            "domains": {
                name: {
                    "rule_count": m.rule_count,
                    "accuracy": m.accuracy,
                    "avg_confidence": m.avg_confidence,
                    "coverage_score": m.coverage_score,
                }
                for name, m in self.domains.items()
            },
            "quality": {
                "avg_confidence": self.quality.avg_confidence,
                "quality_score": self.quality.quality_score,
                "high_confidence_rules": self.quality.high_confidence_rules,
            },
            "overall_accuracy": self.overall_accuracy,
            "timestamp": self.timestamp.isoformat(),
        }

    def __str__(self) -> str:
        """String representation."""
        accuracy_str = (
            f"{self.overall_accuracy:.1%}"
            if self.overall_accuracy is not None
            else "N/A"
        )
        return (
            f"CoverageMetrics: {self.total_rules} rules, "
            f"{self.domain_count} domains, "
            f"{accuracy_str} accuracy"
        )


@dataclass
class MetricsTrend:
    """
    Trend data for metrics over time.

    Tracks how metrics change across time periods.
    """

    metric_name: str
    timestamps: List[datetime] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    domain: Optional[str] = None

    def add_sample(self, timestamp: datetime, value: float):
        """Add a sample to the trend."""
        self.timestamps.append(timestamp)
        self.values.append(value)

    @property
    def latest_value(self) -> Optional[float]:
        """Get most recent value."""
        return self.values[-1] if self.values else None

    @property
    def trend_direction(self) -> str:
        """
        Determine trend direction.

        Returns: "increasing", "decreasing", or "stable"
        """
        if len(self.values) < 2:
            return "stable"

        # Compare first half to second half
        midpoint = len(self.values) // 2
        first_half_avg = sum(self.values[:midpoint]) / midpoint if midpoint > 0 else 0
        second_half_avg = (
            sum(self.values[midpoint:]) / (len(self.values) - midpoint)
            if len(self.values) > midpoint
            else 0
        )

        # 5% threshold for change
        threshold = 0.05
        if second_half_avg > first_half_avg * (1 + threshold):
            return "increasing"
        elif second_half_avg < first_half_avg * (1 - threshold):
            return "decreasing"
        else:
            return "stable"

    def __str__(self) -> str:
        """String representation."""
        return (
            f"MetricsTrend({self.metric_name}): "
            f"{len(self.values)} samples, {self.trend_direction}"
        )


@dataclass
class CoverageGap:
    """
    Identified gap in knowledge coverage.

    Represents areas where coverage is insufficient.
    """

    area: str  # Domain, doctrine, or jurisdiction
    gap_type: str  # "missing_rules", "low_accuracy", "low_confidence"
    severity: float  # 0.0-1.0, higher is more severe
    description: str
    suggested_action: str

    def __str__(self) -> str:
        """String representation."""
        return f"CoverageGap({self.gap_type} in {self.area}): {self.description}"


@dataclass
class CoverageReport:
    """
    Comprehensive coverage report.

    Aggregates metrics, trends, and gaps for reporting.
    """

    metrics: CoverageMetrics
    trends: List[MetricsTrend] = field(default_factory=list)
    gaps: List[CoverageGap] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_markdown(self) -> str:
        """Format report as markdown."""
        lines = []
        lines.append("# Knowledge Coverage Report")
        lines.append("")
        lines.append(f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Overall metrics
        lines.append("## Overall Metrics")
        lines.append("")
        lines.append(f"- **Total Rules**: {self.metrics.total_rules}")
        lines.append(f"- **Active Rules**: {self.metrics.active_rules}")
        lines.append(f"- **Domains Covered**: {self.metrics.domain_count}")
        lines.append(f"- **Average Confidence**: {self.metrics.avg_confidence:.2%}")
        if self.metrics.overall_accuracy:
            lines.append(f"- **Overall Accuracy**: {self.metrics.overall_accuracy:.2%}")
        lines.append("")

        # Domain breakdown
        lines.append("## Coverage by Domain")
        lines.append("")
        for domain_name in sorted(self.metrics.domains.keys()):
            domain = self.metrics.domains[domain_name]
            lines.append(f"### {domain_name}")
            lines.append(f"- Rules: {domain.rule_count}")
            lines.append(f"- Confidence: {domain.avg_confidence:.2%}")
            if domain.accuracy:
                lines.append(f"- Accuracy: {domain.accuracy:.2%}")
            lines.append(f"- Coverage Score: {domain.coverage_score:.2%}")
            lines.append("")

        # Quality metrics
        lines.append("## Quality Metrics")
        lines.append("")
        lines.append(f"- Quality Score: {self.metrics.quality.quality_score:.2%}")
        lines.append(
            f"- High Confidence Rules: {self.metrics.quality.high_confidence_rules}"
        )
        lines.append(
            f"- Medium Confidence Rules: {self.metrics.quality.medium_confidence_rules}"
        )
        lines.append(
            f"- Low Confidence Rules: {self.metrics.quality.low_confidence_rules}"
        )
        lines.append("")

        # Gaps
        if self.gaps:
            lines.append("## Coverage Gaps")
            lines.append("")
            for gap in sorted(self.gaps, key=lambda g: g.severity, reverse=True):
                lines.append(f"- **{gap.area}** ({gap.gap_type}): {gap.description}")
                lines.append(f"  - Severity: {gap.severity:.2%}")
                lines.append(f"  - Action: {gap.suggested_action}")
                lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation."""
        return (
            f"CoverageReport: {self.metrics.domain_count} domains, "
            f"{len(self.gaps)} gaps identified"
        )
