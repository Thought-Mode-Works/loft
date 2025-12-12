"""
Translation fidelity measurement for ASP ↔ Natural Language conversion.

This module measures how well meaning is preserved when translating between
symbolic (ASP) and neural (natural language) representations.
"""

from typing import Tuple, List, Dict, Any, Callable
from dataclasses import dataclass
import clingo
from loguru import logger


@dataclass
class FidelityTestResult:
    """Results from a fidelity test."""

    original_asp: str
    natural_language: str
    reconstructed_asp: str
    fidelity_score: float  # 0-1, where 1 is perfect fidelity
    is_semantically_equivalent: bool
    explanation: str
    errors: List[str]


class FidelityValidator:
    """Measure semantic fidelity of ASP ↔ NL translation."""

    def test_roundtrip_fidelity(
        self,
        asp_original: str,
        asp_to_nl_fn: Callable[[str], str],
        nl_to_asp_fn: Callable[[str], str],
    ) -> FidelityTestResult:
        """
        Test: ASP → NL → ASP preserves meaning.

        Args:
            asp_original: Original ASP text
            asp_to_nl_fn: Function to convert ASP to natural language
            nl_to_asp_fn: Function to convert natural language to ASP

        Returns:
            FidelityTestResult with score and analysis

        Example:
            >>> validator = FidelityValidator()
            >>> result = validator.test_roundtrip_fidelity(
            ...     "fact(a).",
            ...     lambda asp: "Fact a is true",
            ...     lambda nl: "fact(a)."
            ... )
            >>> assert result.fidelity_score > 0.9
        """
        errors: List[str] = []

        try:
            # ASP to NL
            nl_text = asp_to_nl_fn(asp_original)
            logger.debug(f"ASP→NL: '{asp_original}' → '{nl_text}'")

            # NL back to ASP
            asp_reconstructed = nl_to_asp_fn(nl_text)
            logger.debug(f"NL→ASP: '{nl_text}' → '{asp_reconstructed}'")

            # Measure semantic equivalence
            is_equivalent, fidelity = self._semantic_equivalence(
                asp_original, asp_reconstructed
            )

            explanation = f"""
Original ASP: {asp_original}
Natural Language: {nl_text}
Reconstructed ASP: {asp_reconstructed}
Fidelity: {fidelity:.2%}
Semantically Equivalent: {is_equivalent}
            """.strip()

            return FidelityTestResult(
                original_asp=asp_original,
                natural_language=nl_text,
                reconstructed_asp=asp_reconstructed,
                fidelity_score=fidelity,
                is_semantically_equivalent=is_equivalent,
                explanation=explanation,
                errors=errors,
            )

        except Exception as e:
            error_msg = f"Fidelity test failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

            return FidelityTestResult(
                original_asp=asp_original,
                natural_language="",
                reconstructed_asp="",
                fidelity_score=0.0,
                is_semantically_equivalent=False,
                explanation=error_msg,
                errors=errors,
            )

    def _semantic_equivalence(self, asp1: str, asp2: str) -> Tuple[bool, float]:
        """
        Check if two ASP expressions are semantically equivalent.

        Uses ASP reasoning to test logical equivalence by comparing answer sets.

        Args:
            asp1: First ASP program
            asp2: Second ASP program

        Returns:
            Tuple of (are_equivalent, fidelity_score)
            - are_equivalent: True if programs have same answer sets
            - fidelity_score: Similarity score 0-1

        Example:
            >>> validator = FidelityValidator()
            >>> # These are equivalent (same meaning, different order)
            >>> equiv, score = validator._semantic_equivalence(
            ...     "a. b.",
            ...     "b. a."
            ... )
            >>> assert equiv and score == 1.0
        """
        try:
            # Get answer sets for both programs
            answer_sets_1 = self._get_answer_sets(asp1)
            answer_sets_2 = self._get_answer_sets(asp2)

            # If both have same number of answer sets, compare them
            if len(answer_sets_1) != len(answer_sets_2):
                # Different number of answer sets = not equivalent
                # But compute partial score based on overlap
                overlap = self._compute_answer_set_overlap(answer_sets_1, answer_sets_2)
                return (False, overlap)

            # Compare answer sets (order doesn't matter)
            if self._answer_sets_match(answer_sets_1, answer_sets_2):
                logger.debug("ASP programs are semantically equivalent")
                return (True, 1.0)
            else:
                # Same number but different content
                overlap = self._compute_answer_set_overlap(answer_sets_1, answer_sets_2)
                logger.debug(f"ASP programs differ, overlap score: {overlap:.2%}")
                return (False, overlap)

        except Exception as e:
            logger.error(f"Semantic equivalence check failed: {str(e)}")
            return (False, 0.0)

    def _get_answer_sets(self, asp_program: str, max_sets: int = 10) -> List[frozenset]:
        """
        Get answer sets for an ASP program.

        Args:
            asp_program: ASP program text
            max_sets: Maximum number of answer sets to retrieve

        Returns:
            List of answer sets (each as frozenset of symbol strings)
        """
        answer_sets = []

        try:
            # Configure clingo to enumerate all models (0 = all)
            ctl = clingo.Control(["0"])
            ctl.add("base", [], asp_program)
            ctl.ground([("base", [])])

            def on_model(model: clingo.Model) -> bool:
                # Convert symbols to string representations
                symbols = frozenset(str(atom) for atom in model.symbols(shown=True))
                answer_sets.append(symbols)
                # Return False to stop solving after max_sets models
                return len(answer_sets) < max_sets

            ctl.solve(on_model=on_model)

            return answer_sets

        except Exception as e:
            logger.error(f"Error getting answer sets: {str(e)}")
            return []

    def _answer_sets_match(
        self, sets1: List[frozenset], sets2: List[frozenset]
    ) -> bool:
        """
        Check if two lists of answer sets are equivalent.

        Answer sets can be in different order but must have same content.

        Args:
            sets1: First list of answer sets
            sets2: Second list of answer sets

        Returns:
            True if answer sets match (ignoring order)
        """
        if len(sets1) != len(sets2):
            return False

        # Convert to sets for comparison (order doesn't matter)
        set_of_sets_1 = set(sets1)
        set_of_sets_2 = set(sets2)

        return set_of_sets_1 == set_of_sets_2

    def _compute_answer_set_overlap(
        self, sets1: List[frozenset], sets2: List[frozenset]
    ) -> float:
        """
        Compute similarity score between two lists of answer sets.

        Uses Jaccard similarity on the sets of answer sets.

        Args:
            sets1: First list of answer sets
            sets2: Second list of answer sets

        Returns:
            Overlap score between 0 and 1
        """
        if not sets1 and not sets2:
            return 1.0  # Both empty = perfect match

        if not sets1 or not sets2:
            return 0.0  # One empty, one not = no match

        set_of_sets_1 = set(sets1)
        set_of_sets_2 = set(sets2)

        intersection = set_of_sets_1 & set_of_sets_2
        union = set_of_sets_1 | set_of_sets_2

        if not union:
            return 0.0

        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity

    def batch_test_fidelity(
        self,
        asp_examples: List[str],
        asp_to_nl_fn: Callable[[str], str],
        nl_to_asp_fn: Callable[[str], str],
    ) -> Dict[str, Any]:
        """
        Test fidelity on a batch of ASP examples.

        Args:
            asp_examples: List of ASP program strings to test
            asp_to_nl_fn: ASP to natural language translator
            nl_to_asp_fn: Natural language to ASP translator

        Returns:
            Dictionary with batch statistics:
            {
                "total": int,
                "passed": int,  # fidelity > 0.95
                "average_fidelity": float,
                "min_fidelity": float,
                "max_fidelity": float,
                "results": List[FidelityTestResult]
            }

        Example:
            >>> validator = FidelityValidator()
            >>> examples = ["fact(a).", "rule(x) :- fact(x)."]
            >>> stats = validator.batch_test_fidelity(
            ...     examples,
            ...     lambda asp: asp,  # Identity for testing
            ...     lambda nl: nl
            ... )
            >>> assert stats["average_fidelity"] > 0.9
        """
        results = []

        for asp_example in asp_examples:
            result = self.test_roundtrip_fidelity(
                asp_example, asp_to_nl_fn, nl_to_asp_fn
            )
            results.append(result)

        fidelity_scores = [r.fidelity_score for r in results]
        passed = sum(1 for score in fidelity_scores if score > 0.95)

        stats = {
            "total": len(results),
            "passed": passed,
            "average_fidelity": (
                sum(fidelity_scores) / len(fidelity_scores) if fidelity_scores else 0.0
            ),
            "min_fidelity": min(fidelity_scores) if fidelity_scores else 0.0,
            "max_fidelity": max(fidelity_scores) if fidelity_scores else 0.0,
            "results": results,
        }

        logger.info(
            f"Batch fidelity test: {passed}/{len(results)} passed "
            f"(avg={stats['average_fidelity']:.2%})"
        )

        return stats


def compute_translation_fidelity(
    asp_program: str,
    asp_to_nl_fn: Callable[[str], str],
    nl_to_asp_fn: Callable[[str], str],
) -> float:
    """
    Convenience function to compute fidelity score for a single ASP program.

    Args:
        asp_program: ASP program text
        asp_to_nl_fn: Function to convert ASP to natural language
        nl_to_asp_fn: Function to convert natural language to ASP

    Returns:
        Fidelity score between 0 and 1

    Example:
        >>> fidelity = compute_translation_fidelity(
        ...     "fact(a).",
        ...     lambda asp: "A is a fact",
        ...     lambda nl: "fact(a)."
        ... )
        >>> assert 0 <= fidelity <= 1
    """
    validator = FidelityValidator()
    result = validator.test_roundtrip_fidelity(asp_program, asp_to_nl_fn, nl_to_asp_fn)
    return result.fidelity_score
