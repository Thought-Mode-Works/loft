"""
Schemas for A/B testing framework.

Defines data structures for testing competing rule variants and analyzing
strategy performance.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loft.neural.rule_schemas import GeneratedRule


class SelectionCriteria(str, Enum):
    """Criteria for selecting best rule variant."""

    ACCURACY = "accuracy"  # Highest test accuracy
    PRECISION = "precision"  # Fewest false positives
    RECALL = "recall"  # Fewest false negatives
    F1_SCORE = "f1"  # Balanced precision/recall
    CONFIDENCE = "confidence"  # Highest calibrated confidence
    SIMPLICITY = "simplicity"  # Fewest predicates (Occam's razor)


@dataclass
class RuleVariant:
    """A variant of a rule for A/B testing."""

    variant_id: str
    rule: GeneratedRule
    strategy: str  # "conservative", "permissive", "balanced", etc.
    description: str


@dataclass
class VariantPerformance:
    """Performance metrics for a single rule variant."""

    variant: RuleVariant
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_confidence: float
    num_predicates: int
    test_failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "variant_id": self.variant.variant_id,
            "strategy": self.variant.strategy,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "avg_confidence": self.avg_confidence,
            "num_predicates": self.num_predicates,
            "test_failures": self.test_failures,
        }


@dataclass
class ABTestResult:
    """Results of A/B testing multiple rule variants."""

    winner: RuleVariant
    all_results: List[VariantPerformance]
    selection_criterion: SelectionCriteria
    confidence_in_winner: float
    performance_gap: float  # How much better was winner?
    test_id: Optional[str] = None

    def summary(self) -> str:
        """Generate human-readable summary of results."""
        lines = [
            "A/B Test Results:",
            f"  Winner: {self.winner.variant_id} (strategy: {self.winner.strategy})",
            f"  Selection criterion: {self.selection_criterion.value}",
            f"  Confidence in winner: {self.confidence_in_winner:.1%}",
            f"  Performance gap: {self.performance_gap:.3f}",
            "",
            "All variants:",
        ]

        # Sort by performance
        sorted_results = sorted(
            self.all_results,
            key=lambda r: self._get_variant_score(r),
            reverse=True,
        )

        for i, result in enumerate(sorted_results, 1):
            score = self._get_variant_score(result)
            lines.append(
                f"  {i}. {result.variant.variant_id} ({result.variant.strategy}): {score:.3f}"
            )

        return "\n".join(lines)

    def _get_variant_score(self, result: VariantPerformance) -> float:
        """Get score for variant based on selection criterion."""
        if self.selection_criterion == SelectionCriteria.ACCURACY:
            return result.accuracy
        elif self.selection_criterion == SelectionCriteria.PRECISION:
            return result.precision
        elif self.selection_criterion == SelectionCriteria.RECALL:
            return result.recall
        elif self.selection_criterion == SelectionCriteria.F1_SCORE:
            return result.f1_score
        elif self.selection_criterion == SelectionCriteria.CONFIDENCE:
            return result.avg_confidence
        elif self.selection_criterion == SelectionCriteria.SIMPLICITY:
            return 1.0 / (1.0 + result.num_predicates)
        else:
            return result.f1_score  # Default


@dataclass
class StrategyStats:
    """Statistics for a rule generation strategy."""

    strategy: str
    total_tests: int
    wins: int
    win_rate: float
    avg_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy,
            "total_tests": self.total_tests,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "avg_score": self.avg_score,
        }
