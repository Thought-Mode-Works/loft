"""
Translation Fidelity Tests

Tests that NL -> ASP -> NL round-trip translation preserves semantic meaning.
Validates the ontological bridge maintains fidelity across representations.
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loft.translation.nl_to_asp import NLToASPTranslator
from loft.neural.llm_interface import LLMInterface
from loft.neural.providers import AnthropicProvider

from tests.integration.ontological_bridge.utils.semantic_similarity import (
    SemanticSimilarityCalculator,
)
from tests.integration.ontological_bridge.utils.metrics import (
    calculate_fidelity,
    aggregate_metrics,
)


# Test cases for translation fidelity
TRANSLATION_TEST_CASES = [
    "A contract requires offer, acceptance, and consideration",
    "Land sale contracts must be in writing",
    "Contracts for goods over $500 require a written memorandum",
    "Part performance can satisfy the statute of frauds",
    "Suretyship agreements must be in writing",
    "Electronic signatures are valid under the ESIGN Act",
    "Specially manufactured goods are exempt from the writing requirement",
    "Merchant confirmation creates enforceable contract",
    "Contracts that cannot be performed within one year require writing",
    "Admission in court satisfies the statute of frauds",
    # Additional test cases
    "A contract is valid if it has mutual assent",
    "The statute of frauds applies to land transfers",
    "Oral contracts are generally enforceable",
    "Written contracts provide better evidence",
    "Consideration must have legal value",
    "A promise to pay another's debt must be written",
    "Contracts in consideration of marriage require writing",
    "The main purpose exception applies to suretyship",
    "Promissory estoppel can overcome statute of frauds",
    "Growing crops are considered goods under UCC",
]


@pytest.fixture(scope="module")
def translation_components():
    """Initialize translation components once for all tests."""
    # Use Haiku for cost-effectiveness in testing
    provider = AnthropicProvider(model_name="claude-haiku-3-5-20241022")
    llm = LLMInterface(provider=provider)

    nl_to_asp_translator = NLToASPTranslator(llm=llm)
    # asp_to_nl is a function, not a class - just return it as-is
    from loft.translation.asp_to_nl import asp_to_nl as asp_to_nl_func

    return nl_to_asp_translator, asp_to_nl_func


@pytest.fixture(scope="module")
def similarity_calculator():
    """Initialize semantic similarity calculator."""
    return SemanticSimilarityCalculator()


class TestTranslationFidelity:
    """Test translation fidelity with round-trip translation."""

    @pytest.mark.parametrize("original_text", TRANSLATION_TEST_CASES[:5])
    def test_round_trip_translation(
        self, original_text, translation_components, similarity_calculator
    ):
        """
        Test that NL -> ASP -> NL round-trip preserves meaning.

        Acceptance criteria: Semantic similarity > 0.7
        """
        nl_to_asp, asp_to_nl = translation_components

        # Translate NL -> ASP
        asp_result = nl_to_asp.translate(original_text)
        assert asp_result.asp_code, "ASP translation should not be empty"

        # Translate ASP -> NL
        nl_result = asp_to_nl.translate(asp_result.asp_code)
        assert nl_result.natural_language, "NL translation should not be empty"

        # Calculate semantic similarity
        similarity = similarity_calculator.calculate_similarity(
            original_text, nl_result.natural_language
        )

        # Calculate fidelity metrics
        metrics = calculate_fidelity(original_text, nl_result.natural_language, similarity)

        # Assert fidelity thresholds
        assert metrics.semantic_similarity >= 0.7, (
            f"Semantic similarity too low: {metrics.semantic_similarity:.2f}\n"
            f"Original: {original_text}\n"
            f"Round-trip: {nl_result.natural_language}"
        )

        assert metrics.overall_fidelity >= 0.65, (
            f"Overall fidelity too low: {metrics.overall_fidelity:.2f}\n"
            f"Metrics: {metrics.to_dict()}"
        )

    def test_batch_translation_fidelity(self, translation_components, similarity_calculator):
        """
        Test fidelity across multiple translations.

        Validates that average fidelity meets thresholds.
        """
        nl_to_asp, asp_to_nl = translation_components
        metrics_list = []

        # Test first 5 cases for speed
        for original_text in TRANSLATION_TEST_CASES[:5]:
            # Round-trip translation
            asp_result = nl_to_asp.translate(original_text)
            nl_result = asp_to_nl.translate(asp_result.asp_code)

            # Calculate metrics
            similarity = similarity_calculator.calculate_similarity(
                original_text, nl_result.natural_language
            )
            metrics = calculate_fidelity(original_text, nl_result.natural_language, similarity)
            metrics_list.append(metrics)

        # Aggregate metrics
        aggregated = aggregate_metrics(metrics_list)

        # Assert aggregate thresholds
        assert aggregated["avg_semantic_similarity"] >= 0.7, (
            f"Average semantic similarity below threshold: "
            f"{aggregated['avg_semantic_similarity']:.2f}"
        )

        assert aggregated["avg_hallucination_rate"] <= 0.3, (
            f"Average hallucination rate too high: {aggregated['avg_hallucination_rate']:.2f}"
        )

        assert aggregated["avg_overall_fidelity"] >= 0.65, (
            f"Average overall fidelity below threshold: {aggregated['avg_overall_fidelity']:.2f}"
        )

    def test_information_loss(self, translation_components):
        """Test that critical information is not lost in translation."""
        nl_to_asp, asp_to_nl = translation_components

        # Test case with specific requirements
        original = "A contract for the sale of land must be in writing and signed by the party to be charged"

        # Round-trip
        asp_result = nl_to_asp.translate(original)
        nl_result = asp_to_nl.translate(asp_result.asp_code)

        # Check that key concepts are preserved
        result_lower = nl_result.natural_language.lower()

        # These concepts should be present (relaxed checks)
        key_concepts = ["land", "writing", "contract"]

        found_concepts = sum(1 for concept in key_concepts if concept in result_lower)

        # At least 2 out of 3 key concepts should be preserved
        assert found_concepts >= 2, (
            f"Too many key concepts lost. Found {found_concepts}/3\n"
            f"Original: {original}\n"
            f"Round-trip: {nl_result.natural_language}"
        )

    def test_translation_confidence(self, translation_components):
        """Test that translations have reasonable confidence scores."""
        nl_to_asp, asp_to_nl = translation_components

        test_text = "Contracts for goods over $500 require a written memorandum"

        # Translate
        asp_result = nl_to_asp.translate(test_text)

        # Confidence should be reasonable
        assert 0.0 <= asp_result.confidence <= 1.0, (
            f"Confidence out of range: {asp_result.confidence}"
        )

        # For this simple case, confidence should be relatively high
        assert asp_result.confidence >= 0.5, (
            f"Confidence too low for straightforward case: {asp_result.confidence}"
        )


class TestTranslationConsistency:
    """Test that translation is consistent across multiple runs."""

    def test_translation_determinism(self, translation_components):
        """
        Test that same input produces similar outputs.

        Note: With LLMs, exact determinism is not expected, but results
        should be semantically similar.
        """
        nl_to_asp, _ = translation_components
        similarity_calc = SemanticSimilarityCalculator()

        test_text = "A contract requires offer and acceptance"

        # Translate twice with same input
        result1 = nl_to_asp.translate(test_text)
        result2 = nl_to_asp.translate(test_text)

        # Results should be similar (but may not be identical due to LLM variance)
        similarity = similarity_calc.calculate_similarity(result1.asp_code, result2.asp_code)

        # Relaxed threshold due to LLM variance
        assert similarity >= 0.6, (
            f"Translations not consistent enough: {similarity:.2f}\n"
            f"Result 1: {result1.asp_code}\n"
            f"Result 2: {result2.asp_code}"
        )


@pytest.mark.slow
class TestExtensiveTranslation:
    """Extensive translation tests (marked slow for optional execution)."""

    def test_all_translation_cases(self, translation_components, similarity_calculator):
        """
        Test all translation cases.

        This test is marked slow as it processes all test cases.
        """
        nl_to_asp, asp_to_nl = translation_components
        metrics_list = []
        failures = []

        for i, original_text in enumerate(TRANSLATION_TEST_CASES):
            try:
                # Round-trip
                asp_result = nl_to_asp.translate(original_text)
                nl_result = asp_to_nl.translate(asp_result.asp_code)

                # Calculate metrics
                similarity = similarity_calculator.calculate_similarity(
                    original_text, nl_result.natural_language
                )
                metrics = calculate_fidelity(original_text, nl_result.natural_language, similarity)
                metrics_list.append(metrics)

                # Track failures
                if metrics.overall_fidelity < 0.6:
                    failures.append(
                        {
                            "index": i,
                            "original": original_text,
                            "round_trip": nl_result.natural_language,
                            "fidelity": metrics.overall_fidelity,
                        }
                    )

            except Exception as e:
                failures.append(
                    {
                        "index": i,
                        "original": original_text,
                        "error": str(e),
                    }
                )

        # Aggregate results
        aggregated = aggregate_metrics(metrics_list)

        # Report results
        print("\n\nTranslation Fidelity Report:")
        print(f"Total cases: {len(TRANSLATION_TEST_CASES)}")
        print(f"Successful: {len(metrics_list)}")
        print(f"Failures: {len(failures)}")
        print("\nAggregate Metrics:")
        for key, value in aggregated.items():
            print(f"  {key}: {value:.3f}")

        if failures:
            print("\nFailures:")
            for failure in failures[:5]:  # Show first 5
                print(f"  {failure}")

        # Assert overall quality
        assert aggregated["avg_overall_fidelity"] >= 0.6, (
            "Overall fidelity below threshold across all cases"
        )
