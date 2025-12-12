"""
Confidence aggregator for multi-source confidence scoring.

Combines confidence from generation, validation, empirical testing,
and consensus into a single weighted score with variance analysis.
"""

from typing import TYPE_CHECKING, Dict, Optional
from loguru import logger

from loft.validation.confidence_schemas import AggregatedConfidence

if TYPE_CHECKING:
    from loft.validation.validation_schemas import ValidationReport


class ConfidenceAggregator:
    """
    Aggregate confidence from multiple sources.

    Weights different confidence sources according to their reliability
    and importance for decision making.
    """

    # Default weights based on empirical importance
    DEFAULT_WEIGHTS = {
        "generation": 0.15,  # LLM self-assessment
        "syntax": 0.10,  # Syntax validation
        "semantic": 0.15,  # Semantic validation
        "empirical": 0.40,  # Highest weight - actual performance
        "consensus": 0.20,  # Multi-LLM agreement
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        variance_threshold: float = 0.1,
    ):
        """
        Initialize confidence aggregator.

        Args:
            weights: Custom weights for each source (must sum to 1.0)
            variance_threshold: Max variance for "reliable" classification
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.variance_threshold = variance_threshold

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            logger.info(f"Normalizing weights (sum={total:.3f})")
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.debug(
            f"Initialized ConfidenceAggregator with weights: {self.weights}, "
            f"variance_threshold={variance_threshold}"
        )

    def aggregate(
        self,
        generation_confidence: float,
        syntax_valid: bool,
        semantic_valid: bool,
        empirical_accuracy: Optional[float] = None,
        consensus_strength: Optional[float] = None,
    ) -> AggregatedConfidence:
        """
        Compute weighted confidence score.

        Args:
            generation_confidence: LLM's confidence in generated rule
            syntax_valid: Did rule pass syntax validation
            semantic_valid: Did rule pass semantic validation
            empirical_accuracy: Accuracy on test cases (optional)
            consensus_strength: Agreement among voter LLMs (optional)

        Returns:
            AggregatedConfidence with score and breakdown

        Example:
            >>> aggregator = ConfidenceAggregator()
            >>> confidence = aggregator.aggregate(
            ...     generation_confidence=0.87,
            ...     syntax_valid=True,
            ...     semantic_valid=True,
            ...     empirical_accuracy=0.82,
            ...     consensus_strength=0.90
            ... )
            >>> assert 0.8 <= confidence.score <= 0.9
        """
        # Convert validation booleans to scores
        syntax_score = 1.0 if syntax_valid else 0.0
        semantic_score = 1.0 if semantic_valid else 0.0

        # Build components dict
        components = {
            "generation": generation_confidence,
            "syntax": syntax_score,
            "semantic": semantic_score,
        }

        # Add optional components
        if empirical_accuracy is not None:
            components["empirical"] = empirical_accuracy
        if consensus_strength is not None:
            components["consensus"] = consensus_strength

        # Adjust weights if some components missing
        active_weights = {k: v for k, v in self.weights.items() if k in components}
        total_weight = sum(active_weights.values())

        if total_weight > 0:
            active_weights = {k: v / total_weight for k, v in active_weights.items()}
        else:
            logger.warning("No active weights, using uniform distribution")
            active_weights = {k: 1.0 / len(components) for k in components}

        # Weighted sum
        aggregate_score = sum(
            active_weights.get(source, 0.0) * score
            for source, score in components.items()
        )

        # Compute variance to measure agreement
        variance = self._compute_variance(components)

        # Check reliability based on variance
        is_reliable = variance < self.variance_threshold

        logger.debug(
            f"Aggregated confidence: score={aggregate_score:.3f}, "
            f"variance={variance:.3f}, reliable={is_reliable}"
        )

        return AggregatedConfidence(
            score=aggregate_score,
            components=components,
            weights=active_weights,
            variance=variance,
            is_reliable=is_reliable,
        )

    def aggregate_from_report(
        self, validation_report: "ValidationReport"
    ) -> AggregatedConfidence:
        """
        Aggregate confidence from a ValidationReport.

        Convenience method that extracts confidence from validation pipeline results.

        Args:
            validation_report: ValidationReport from validation pipeline

        Returns:
            AggregatedConfidence
        """
        # Extract components from report
        syntax_valid = False
        semantic_valid = False
        empirical_accuracy = None
        consensus_strength = None

        # Syntactic validation
        if "syntactic" in validation_report.stage_results:
            syntactic = validation_report.stage_results["syntactic"]
            syntax_valid = syntactic.is_valid

        # Semantic validation
        if "semantic" in validation_report.stage_results:
            semantic = validation_report.stage_results["semantic"]
            semantic_valid = semantic.is_valid

        # Empirical validation
        if "empirical" in validation_report.stage_results:
            empirical = validation_report.stage_results["empirical"]
            empirical_accuracy = empirical.accuracy

        # Consensus validation
        if "consensus" in validation_report.stage_results:
            consensus = validation_report.stage_results["consensus"]
            consensus_strength = consensus.consensus_strength

        # Generation confidence (stored in metadata if available)
        generation_confidence = validation_report.metadata.get(
            "generation_confidence", 0.5
        )

        return self.aggregate(
            generation_confidence=generation_confidence,
            syntax_valid=syntax_valid,
            semantic_valid=semantic_valid,
            empirical_accuracy=empirical_accuracy,
            consensus_strength=consensus_strength,
        )

    def _compute_variance(self, components: Dict[str, float]) -> float:
        """
        Measure disagreement among confidence sources.

        High variance indicates components disagree about confidence,
        which suggests the aggregate may be unreliable.

        Args:
            components: Dict of source -> score

        Returns:
            Variance of component scores
        """
        if not components:
            return 0.0

        scores = list(components.values())
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)

        return variance

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Update aggregation weights.

        Args:
            weights: New weights (will be normalized to sum to 1.0)
        """
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        logger.info(f"Updated weights: {self.weights}")

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()
