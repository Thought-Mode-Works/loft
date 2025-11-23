"""
Translation quality metrics and fidelity tracking.

Measures quality and semantic preservation of ASP to NL translations.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Any
from dataclasses import dataclass
from loguru import logger

if TYPE_CHECKING:
    from loft.neural.llm_interface import LLMInterface

from .asp_to_nl import extract_predicates


@dataclass
class QualityMetrics:
    """Quality metrics for a translation."""

    completeness: float  # All predicates represented (0-1)
    readability: float  # Grammar and readability score (0-1)
    fidelity: float  # Semantic preservation score (0-1)
    overall: float  # Combined quality score (0-1)


def validate_translation_quality(asp_text: str, nl_text: str) -> float:
    """
    Score translation quality (0-1).

    Args:
        asp_text: Original ASP text
        nl_text: Natural language translation

    Returns:
        Quality score from 0.0 to 1.0

    Example:
        >>> asp = "contract(c1)."
        >>> nl = "c1 is a contract."
        >>> validate_translation_quality(asp, nl)
        1.0
    """
    scores = []

    # Check all predicates mentioned
    asp_predicates = extract_predicates(asp_text)
    nl_lower = nl_text.lower()

    for pred in asp_predicates:
        # Check if predicate name or humanized version appears in NL
        pred_lower = pred.lower()
        pred_humanized = pred.replace("_", " ")

        if pred_lower in nl_lower or pred_humanized in nl_lower:
            scores.append(1.0)
        else:
            scores.append(0.0)

    # Basic sentence structure check
    has_complete_sentence = nl_text.strip().endswith((".", "?", "!"))
    scores.append(1.0 if has_complete_sentence else 0.5)

    # Check for reasonable length (not too short)
    word_count = len(nl_text.split())
    length_score = min(1.0, word_count / 5.0)  # At least 5 words ideal
    scores.append(length_score)

    return sum(scores) / len(scores) if scores else 0.0


def check_grammar_with_llm(nl_text: str, llm_interface: LLMInterface) -> float:
    """
    Use LLM to check grammar quality.

    Args:
        nl_text: Natural language text to check
        llm_interface: LLM interface for grammar checking

    Returns:
        Grammar quality score (0-1)
    """
    prompt = f"""
    Rate the grammatical correctness of this sentence on a scale of 0.0 to 1.0:

    "{nl_text}"

    Provide only a number between 0.0 (completely incorrect) and 1.0 (perfect grammar).
    """

    try:
        response = llm_interface.query(prompt, temperature=0.0, max_tokens=10)
        score_text = response.raw_text.strip()

        # Extract numeric score
        import re

        match = re.search(r"(0\.\d+|1\.0|0|1)", score_text)
        if match:
            return float(match.group(1))
        else:
            logger.warning(f"Could not parse grammar score from: {score_text}")
            return 0.7  # Default moderate score

    except Exception as e:
        logger.error(f"Error checking grammar with LLM: {e}")
        return 0.7  # Default moderate score


def compute_fidelity(
    asp_original: str, asp_reconstructed: str, asp_core: Optional[Any] = None
) -> float:
    """
    Compute semantic fidelity between original and reconstructed ASP.

    This is used for roundtrip testing: ASP -> NL -> ASP.

    Args:
        asp_original: Original ASP text
        asp_reconstructed: Reconstructed ASP after roundtrip
        asp_core: Optional ASP core for semantic equivalence checking

    Returns:
        Fidelity score (0-1)

    Note:
        Full semantic equivalence checking requires ASP reasoning,
        which will be implemented when NL -> ASP translation is added (#9).
    """
    # For now, use syntactic similarity as a proxy
    # This will be enhanced when NL -> ASP translation is implemented

    # Normalize whitespace
    norm_original = " ".join(asp_original.split())
    norm_reconstructed = " ".join(asp_reconstructed.split())

    # Check exact match
    if norm_original == norm_reconstructed:
        return 1.0

    # Check predicate preservation
    preds_original = set(extract_predicates(asp_original))
    preds_reconstructed = set(extract_predicates(asp_reconstructed))

    if not preds_original:
        return 0.0

    # Compute Jaccard similarity of predicates
    intersection = preds_original & preds_reconstructed
    union = preds_original | preds_reconstructed

    jaccard = len(intersection) / len(union) if union else 0.0

    return jaccard


def compute_quality_metrics(
    asp_text: str,
    nl_text: str,
    llm_interface: Optional[LLMInterface] = None,
) -> QualityMetrics:
    """
    Compute comprehensive quality metrics for a translation.

    Args:
        asp_text: Original ASP text
        nl_text: Natural language translation
        llm_interface: Optional LLM interface for advanced checks

    Returns:
        QualityMetrics with all scores
    """
    # Completeness: are all predicates represented?
    asp_predicates = extract_predicates(asp_text)
    nl_lower = nl_text.lower()

    completeness_scores = []
    for pred in asp_predicates:
        pred_humanized = pred.replace("_", " ")
        if pred.lower() in nl_lower or pred_humanized in nl_lower:
            completeness_scores.append(1.0)
        else:
            completeness_scores.append(0.0)

    completeness = (
        sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
    )

    # Readability: basic checks
    word_count = len(nl_text.split())
    has_proper_ending = nl_text.strip().endswith((".", "?", "!"))
    readability_scores = [
        min(1.0, word_count / 5.0),  # Length check
        1.0 if has_proper_ending else 0.5,  # Punctuation
    ]

    # Use LLM for grammar if available
    if llm_interface:
        grammar_score = check_grammar_with_llm(nl_text, llm_interface)
        readability_scores.append(grammar_score)

    readability = sum(readability_scores) / len(readability_scores)

    # Fidelity: for now, same as completeness (will be enhanced with roundtrip testing)
    fidelity = completeness

    # Overall score
    overall = (completeness + readability + fidelity) / 3.0

    return QualityMetrics(
        completeness=completeness,
        readability=readability,
        fidelity=fidelity,
        overall=overall,
    )
