"""
Pydantic schemas for confidence scoring and calibration system.

Defines structured outputs for confidence aggregation, calibration,
gating decisions, and tracking.
"""

from datetime import datetime
from typing import Dict, List, Literal
from pydantic import BaseModel, Field


class AggregatedConfidence(BaseModel):
    """
    Aggregated confidence from multiple sources.

    Combines confidence scores from generation, validation stages,
    empirical testing, and consensus voting into a single score.
    """

    score: float = Field(ge=0.0, le=1.0, description="Aggregate confidence score")
    components: Dict[str, float] = Field(
        description="Individual confidence scores by source"
    )
    weights: Dict[str, float] = Field(description="Weights used in aggregation")
    variance: float = Field(ge=0.0, description="Variance among component scores")
    is_reliable: bool = Field(
        description="Whether variance is low enough to trust score"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When confidence was computed"
    )

    def explanation(self) -> str:
        """Generate human-readable confidence explanation."""
        lines = [f"Overall Confidence: {self.score:.2f}"]
        lines.append("\nBreakdown:")
        for source, score in self.components.items():
            weight = self.weights.get(source, 0.0)
            contribution = weight * score
            lines.append(
                f"  {source.title()}: {score:.2f} "
                f"(weight: {weight:.2f}, contribution: {contribution:.2f})"
            )
        lines.append(
            f"\nVariance: {self.variance:.3f} ({'reliable' if self.is_reliable else 'unreliable'})"
        )
        return "\n".join(lines)


class CalibrationPoint(BaseModel):
    """
    Single data point for confidence calibration.

    Records predicted confidence vs actual accuracy for a rule.
    """

    predicted: float = Field(ge=0.0, le=1.0, description="Predicted confidence")
    actual: float = Field(ge=0.0, le=1.0, description="Actual accuracy")
    rule_id: str = Field(description="Identifier for the rule")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When point was recorded"
    )


class CalibrationReport(BaseModel):
    """
    Report on confidence calibration quality.

    Compares calibration performance before and after adjustment.
    """

    method: str = Field(description="Calibration method used")
    num_points: int = Field(description="Number of calibration points")
    before_mse: float = Field(description="MSE before calibration")
    after_mse: float = Field(description="MSE after calibration")
    before_ece: float = Field(description="Expected Calibration Error before")
    after_ece: float = Field(description="Expected Calibration Error after")
    improvement: float = Field(default=0.0, description="Relative improvement in ECE")

    def __init__(self, **data) -> None:
        super().__init__(**data)
        # Calculate improvement
        if self.before_ece > 0:
            self.improvement = (self.before_ece - self.after_ece) / self.before_ece
        else:
            self.improvement = 0.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Calibration Report ({self.method})",
            f"  Data Points: {self.num_points}",
            f"  Before: MSE={self.before_mse:.4f}, ECE={self.before_ece:.4f}",
            f"  After:  MSE={self.after_mse:.4f}, ECE={self.after_ece:.4f}",
            f"  Improvement: {self.improvement:.1%}",
        ]
        return "\n".join(lines)


class GateDecision(BaseModel):
    """
    Decision from confidence gating.

    Determines whether to accept, reject, or flag a rule based on confidence.
    """

    action: Literal["accept", "reject", "flag_for_review"] = Field(
        description="Action to take"
    )
    reasoning: str = Field(description="Explanation for decision")
    raw_confidence: float = Field(ge=0.0, le=1.0, description="Uncalibrated score")
    calibrated_confidence: float = Field(ge=0.0, le=1.0, description="Calibrated score")
    threshold: float = Field(ge=0.0, le=1.0, description="Threshold used")
    target_layer: str = Field(
        description="Stratification layer (operational/tactical/strategic)"
    )
    rule_impact: str = Field(description="Estimated impact (low/medium/high)")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When decision was made"
    )


class ConfidenceTrends(BaseModel):
    """
    Trends in confidence scores over time.

    Analyzes recent confidence scores to identify patterns.
    """

    mean_confidence: float = Field(description="Average confidence")
    median_confidence: float = Field(description="Median confidence")
    std_confidence: float = Field(description="Standard deviation")
    mean_variance: float = Field(description="Average variance among components")
    num_reliable: int = Field(description="Number of reliable scores")
    num_total: int = Field(description="Total number of scores")
    reliability_rate: float = Field(description="Fraction of scores that are reliable")

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Confidence Trends",
            f"  Mean: {self.mean_confidence:.2f} ± {self.std_confidence:.2f}",
            f"  Median: {self.median_confidence:.2f}",
            f"  Reliability: {self.reliability_rate:.1%} ({self.num_reliable}/{self.num_total})",
            f"  Avg Variance: {self.mean_variance:.3f}",
        ]
        return "\n".join(lines)


class ConfidenceAnalysis(BaseModel):
    """
    Analysis of confidence scoring performance.

    Identifies areas where confidence is poorly calibrated.
    """

    underconfident_areas: List[str] = Field(
        default_factory=list, description="Predicates with consistently low confidence"
    )
    overconfident_areas: List[str] = Field(
        default_factory=list,
        description="Predicates where confidence exceeds actual accuracy",
    )
    calibration_quality: float = Field(
        description="Overall calibration quality (1 - ECE)"
    )
    num_samples: int = Field(description="Number of samples analyzed")
    timestamp: datetime = Field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Confidence Analysis",
            f"  Calibration Quality: {self.calibration_quality:.1%}",
            f"  Samples: {self.num_samples}",
        ]

        if self.underconfident_areas:
            lines.append(
                f"  Underconfident ({len(self.underconfident_areas)}): "
                f"{', '.join(self.underconfident_areas[:3])}"
            )

        if self.overconfident_areas:
            lines.append(
                f"  Overconfident ({len(self.overconfident_areas)}): "
                f"{', '.join(self.overconfident_areas[:3])}"
            )

        return "\n".join(lines)
