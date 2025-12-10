"""
Semantic similarity utilities for translation validation.

Uses sentence embeddings to measure semantic similarity between
natural language texts.
"""

import os
from typing import List, Tuple, TYPE_CHECKING, Any, Dict

# Try to import optional dependencies
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    if not TYPE_CHECKING:
        # For type checking, pretend numpy exists
        np = None  # type: ignore
        SentenceTransformer = None  # type: ignore

if TYPE_CHECKING:
    import numpy as np


def _use_embeddings_default() -> bool:
    """Return whether embedding similarity is enabled by configuration."""
    value = os.getenv("LOFT_USE_EMBEDDING_SIMILARITY", "true").lower()
    return value not in {"0", "false", "no"}


class TokenSimilarity:
    """Token-based Jaccard similarity (fallback)."""

    @staticmethod
    def calculate(text1: str, text2: str) -> float:
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union)


class EmbeddingSemanticSimilarity:
    """Neural embedding-based semantic similarity calculator with caching."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        enable_embeddings: bool | None = None,
    ):
        """
        Initialize embedding-based similarity.

        Args:
            model_name: HuggingFace sentence-transformer model name.
            enable_embeddings: Override config flag for enabling embeddings.
        """
        self.model_name = model_name
        self.enable_embeddings = (
            _use_embeddings_default()
            if enable_embeddings is None
            else enable_embeddings
        )
        self.model = None
        self._cache: Dict[str, "np.ndarray"] = {}

        if self.enable_embeddings and SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self.model = None
                self.enable_embeddings = False
        else:
            # Disable embeddings if dependencies are missing or explicitly turned off.
            self.enable_embeddings = False

        self._token_similarity = TokenSimilarity()

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between text embeddings.

        Falls back to token overlap if embeddings are unavailable or disabled.
        """
        if self.enable_embeddings and self.model is not None and np is not None:
            emb1 = self._get_embedding(text1)
            emb2 = self._get_embedding(text2)
            return cosine_similarity(emb1, emb2)

        return self._token_similarity.calculate(text1, text2)

    def calculate_batch_similarity(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Efficiently calculate similarity for multiple pairs with caching.

        Preloads unseen embeddings and reuses cached vectors to minimize latency.
        """
        if self.enable_embeddings and self.model is not None and np is not None:
            unique_texts = {text for pair in pairs for text in pair}
            self._preload_embeddings(list(unique_texts))
            return [
                cosine_similarity(self._cache[t1], self._cache[t2]) for t1, t2 in pairs
            ]

        return [self._token_similarity.calculate(t1, t2) for t1, t2 in pairs]

    def _get_embedding(self, text: str) -> "np.ndarray":
        """Return cached embedding or compute and store it."""
        if text not in self._cache:
            self._cache[text] = self.model.encode(text)
        return self._cache[text]

    def _preload_embeddings(self, texts: List[str]) -> None:
        """Precompute embeddings for a list of texts that are not yet cached."""
        missing = [text for text in texts if text not in self._cache]
        if not missing:
            return

        embeddings = self.model.encode(missing)
        for text, embedding in zip(missing, embeddings):
            self._cache[text] = embedding


class SemanticSimilarityCalculator:
    """Calculate semantic similarity between texts."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize similarity calculator.

        Args:
            model_name: Name of sentence transformer model to use
        """
        self.model_name = model_name
        self.embedding_calculator = EmbeddingSemanticSimilarity(model_name=model_name)
        self.token_similarity = TokenSimilarity()

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0-1.0)
        """
        return self.embedding_calculator.calculate_similarity(text1, text2)

    def calculate_batch_similarity(
        self, text_pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """
        Calculate similarity for multiple text pairs.

        Args:
            text_pairs: List of (text1, text2) tuples

        Returns:
            List of similarity scores
        """
        return self.embedding_calculator.calculate_batch_similarity(text_pairs)


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


def cosine_similarity(vec1: Any, vec2: Any) -> float:
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
