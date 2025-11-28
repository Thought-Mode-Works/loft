"""
Semantic similarity utilities for translation validation.

Uses sentence embeddings to measure semantic similarity between
natural language texts.
"""

from typing import List, Tuple
import numpy as np

# Try to import sentence-transformers, fall back to simple similarity if unavailable
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False


class SemanticSimilarityCalculator:
    """Calculate semantic similarity between texts."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize similarity calculator.

        Args:
            model_name: Name of sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None

        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self.model = None

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0-1.0)
        """
        if self.model is not None:
            return self._calculate_with_embeddings(text1, text2)
        else:
            # Fallback to simple token-based similarity
            return self._calculate_simple_similarity(text1, text2)

    def _calculate_with_embeddings(self, text1: str, text2: str) -> float:
        """Calculate similarity using sentence embeddings."""
        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def _calculate_simple_similarity(self, text1: str, text2: str) -> float:
        """
        Simple token-based similarity (fallback).

        Uses Jaccard similarity on word tokens.
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union)

    def calculate_batch_similarity(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Calculate similarity for multiple text pairs.

        Args:
            text_pairs: List of (text1, text2) tuples

        Returns:
            List of similarity scores
        """
        return [self.calculate_similarity(t1, t2) for t1, t2 in text_pairs]


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Quick function to calculate semantic similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score (0.0-1.0)
    """
    calculator = SemanticSimilarityCalculator()
    return calculator.calculate_similarity(text1, text2)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (-1.0 to 1.0)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))
