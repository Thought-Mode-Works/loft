"""
Metrics for ontological bridge fidelity.

Calculates translation fidelity, information loss, and hallucination rates.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from tests.integration.ontological_bridge.utils.semantic_similarity import (
    SemanticSimilarityCalculator,
)


@dataclass
class FidelityMetrics:
    """Metrics for translation fidelity."""

    semantic_similarity: float  # 0.0-1.0
    information_preservation: float  # 0.0-1.0
    hallucination_rate: float  # 0.0-1.0
    structural_accuracy: float  # 0.0-1.0
    overall_fidelity: float  # 0.0-1.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "semantic_similarity": self.semantic_similarity,
            "information_preservation": self.information_preservation,
            "hallucination_rate": self.hallucination_rate,
            "structural_accuracy": self.structural_accuracy,
            "overall_fidelity": self.overall_fidelity,
        }


class FidelityCalculator:
    """Calculate translation fidelity metrics."""

    def __init__(
        self,
        *,
        use_embeddings: Optional[bool] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize fidelity calculator with semantic similarity backend.

        Args:
            use_embeddings: Force enable/disable embeddings (overrides env flag).
            model_name: Embedding model name.
        """
        self._similarity_calculator = SemanticSimilarityCalculator(model_name=model_name)

        if use_embeddings is not None:
            self._similarity_calculator.embedding_calculator.enable_embeddings = (
                use_embeddings
                and self._similarity_calculator.embedding_calculator.model is not None
            )

    def calculate_fidelity(
        self,
        original_text: str,
        translated_text: str,
        semantic_similarity_score: Optional[float] = None,
        *,
        use_embeddings: Optional[bool] = None,
        similarity_calculator: Optional[SemanticSimilarityCalculator] = None,
    ) -> FidelityMetrics:
        """
        Calculate comprehensive fidelity metrics.

        Args:
            original_text: Original text
            translated_text: Translated text
            semantic_similarity_score: Pre-calculated semantic similarity. If None,
                calculate using the configured similarity backend.
            use_embeddings: Optional override for enabling embeddings at call time.
            similarity_calculator: Custom similarity calculator to use (primarily for
                testing/benchmarking).

        Returns:
            Fidelity metrics
        """
        semantic_similarity_score = self._get_semantic_similarity(
            original_text=original_text,
            translated_text=translated_text,
            semantic_similarity_score=semantic_similarity_score,
            use_embeddings=use_embeddings,
            similarity_calculator=similarity_calculator,
        )

        # Calculate information preservation (based on length and content)
        info_preservation = self._calculate_information_preservation(
            original_text, translated_text
        )

        # Estimate hallucination rate (content in translation not in original)
        hallucination = self._estimate_hallucination(original_text, translated_text)

        # Structural accuracy (simplified - based on sentence count)
        structural = self._calculate_structural_accuracy(original_text, translated_text)

        # Overall fidelity (weighted average)
        overall = (
            0.4 * semantic_similarity_score
            + 0.3 * info_preservation
            + 0.2 * (1.0 - hallucination)  # Lower hallucination is better
            + 0.1 * structural
        )

        return FidelityMetrics(
            semantic_similarity=semantic_similarity_score,
            information_preservation=info_preservation,
            hallucination_rate=hallucination,
            structural_accuracy=structural,
            overall_fidelity=overall,
        )

    def _get_semantic_similarity(
        self,
        original_text: str,
        translated_text: str,
        semantic_similarity_score: Optional[float],
        use_embeddings: Optional[bool],
        similarity_calculator: Optional[SemanticSimilarityCalculator],
    ) -> float:
        """Calculate or return provided semantic similarity."""
        if semantic_similarity_score is not None:
            return semantic_similarity_score

        calculator = similarity_calculator or self._similarity_calculator

        if use_embeddings is not None:
            calculator.embedding_calculator.enable_embeddings = (
                use_embeddings
                and calculator.embedding_calculator.model is not None
            )

        return calculator.calculate_similarity(original_text, translated_text)

    def _calculate_information_preservation(
        self, original: str, translated: str
    ) -> float:
        """
        Estimate how much information is preserved.

        Uses token overlap and length ratio as proxy.
        """
        original_tokens = set(original.lower().split())
        translated_tokens = set(translated.lower().split())

        if not original_tokens:
            return 0.0

        # How many original tokens appear in translation
        overlap = len(original_tokens.intersection(translated_tokens))
        preservation = overlap / len(original_tokens)

        # Adjust for length difference (penalize if translation is too short)
        length_ratio = (
            len(translated.split()) / len(original.split()) if original.split() else 1.0
        )
        length_factor = min(1.0, length_ratio)

        return (preservation + length_factor) / 2.0

    def _estimate_hallucination(self, original: str, translated: str) -> float:
        """
        Estimate hallucination rate.

        Measures how much content in translation is not in original.
        """
        original_tokens = set(original.lower().split())
        translated_tokens = set(translated.lower().split())

        if not translated_tokens:
            return 0.0

        # Tokens in translation but not in original
        hallucinated = translated_tokens - original_tokens

        # Ignore common stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
        }
        hallucinated = hallucinated - stopwords

        return len(hallucinated) / len(translated_tokens)

    def _calculate_structural_accuracy(self, original: str, translated: str) -> float:
        """
        Calculate structural accuracy.

        Simplified version based on sentence count similarity.
        """
        original_sentences = (
            original.count(".") + original.count("!") + original.count("?")
        )
        translated_sentences = (
            translated.count(".") + translated.count("!") + translated.count("?")
        )

        if original_sentences == 0:
            original_sentences = 1

        # How similar are the sentence counts
        ratio = min(original_sentences, translated_sentences) / max(
            original_sentences, translated_sentences
        )

        return ratio


def calculate_fidelity(
    original_text: str,
    translated_text: str,
    semantic_similarity_score: Optional[float] = None,
    *,
    use_embeddings: Optional[bool] = None,
    similarity_calculator: Optional[SemanticSimilarityCalculator] = None,
) -> FidelityMetrics:
    """
    Quick function to calculate fidelity metrics.

    Args:
        original_text: Original text
        translated_text: Translated text
        semantic_similarity_score: Pre-calculated semantic similarity. If None,
            the function will calculate using embeddings if available.
        use_embeddings: Optional override to enable/disable embeddings.
        similarity_calculator: Custom similarity calculator to use.

    Returns:
        Fidelity metrics
    """
    calculator = FidelityCalculator(use_embeddings=use_embeddings)
    return calculator.calculate_fidelity(
        original_text,
        translated_text,
        semantic_similarity_score,
        use_embeddings=use_embeddings,
        similarity_calculator=similarity_calculator,
    )


def aggregate_metrics(metrics_list: List[FidelityMetrics]) -> Dict[str, Any]:
    """
    Aggregate metrics from multiple test cases.

    Args:
        metrics_list: List of fidelity metrics

    Returns:
        Aggregated statistics
    """
    if not metrics_list:
        return {}

    n = len(metrics_list)

    return {
        "count": n,
        "avg_semantic_similarity": sum(m.semantic_similarity for m in metrics_list) / n,
        "avg_information_preservation": sum(
            m.information_preservation for m in metrics_list
        )
        / n,
        "avg_hallucination_rate": sum(m.hallucination_rate for m in metrics_list) / n,
        "avg_structural_accuracy": sum(m.structural_accuracy for m in metrics_list) / n,
        "avg_overall_fidelity": sum(m.overall_fidelity for m in metrics_list) / n,
        "min_fidelity": min(m.overall_fidelity for m in metrics_list),
        "max_fidelity": max(m.overall_fidelity for m in metrics_list),
    }
