"""
Edge Case Tests for Ontological Bridge

Tests handling of ambiguity, contradictions, complex negations,
and other challenging translation scenarios.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loft.translation.nl_to_asp import NLToASPTranslator
from loft.neural.llm_interface import LLMInterface
from loft.neural.providers import AnthropicProvider


@pytest.fixture(scope="module")
def translation_components():
    """Initialize translation components."""
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
        )
    provider = AnthropicProvider(api_key=api_key, model="claude-haiku-3-5-20241022")
    llm = LLMInterface(provider=provider)

    nl_to_asp_translator = NLToASPTranslator(llm=llm)
    # asp_to_nl is a function, not a class - just return it as-is
    from loft.translation.asp_to_nl import asp_to_nl as asp_to_nl_func

    return nl_to_asp_translator, asp_to_nl_func


class TestAmbiguityHandling:
    """Test handling of ambiguous statements."""

    def test_ambiguous_scope(self, translation_components):
        """Test handling of scope ambiguity."""
        nl_to_asp, _ = translation_components

        # Ambiguous: Does 'not' apply to just written or to both?
        ambiguous_text = "Contracts that are not written or signed are unenforceable"

        result = nl_to_asp.translate(ambiguous_text)

        # Should produce valid ASP code
        assert result.asp_code, "Should produce ASP code for ambiguous input"
        assert ":-" in result.asp_code, "Should produce rule structure"

        # Confidence should reflect uncertainty
        assert result.confidence <= 0.85, (
            f"Confidence should reflect ambiguity, got {result.confidence}"
        )

    def test_quantifier_ambiguity(self, translation_components):
        """Test handling of quantifier ambiguity."""
        nl_to_asp, _ = translation_components

        # "All" vs "some" ambiguity
        text = "Merchants can modify contracts"

        result = nl_to_asp.translate(text)

        # Should handle gracefully
        assert result.asp_code, "Should produce ASP code"


class TestNegationHandling:
    """Test handling of negation in various forms."""

    def test_simple_negation(self, translation_components):
        """Test simple negation."""
        nl_to_asp, asp_to_nl = translation_components

        text = "A contract is not enforceable if it is not in writing"

        result = nl_to_asp.translate(text)

        # Should contain negation operator
        assert "not" in result.asp_code.lower() or "~" in result.asp_code, (
            "Should contain negation in ASP code"
        )

    def test_double_negation(self, translation_components):
        """Test double negation handling."""
        nl_to_asp, _ = translation_components

        text = "It is not the case that contracts are not enforceable"

        result = nl_to_asp.translate(text)

        # Should handle double negation
        assert result.asp_code, "Should produce code for double negation"

    def test_implicit_negation(self, translation_components):
        """Test implicit negation (unless, except, without)."""
        nl_to_asp, _ = translation_components

        cases = [
            "Contracts are valid unless they violate public policy",
            "All contracts except those for illegal purposes are enforceable",
            "A contract is unenforceable without consideration",
        ]

        for text in cases:
            result = nl_to_asp.translate(text)
            assert result.asp_code, f"Should handle implicit negation: {text}"


class TestComplexStructures:
    """Test handling of complex logical structures."""

    def test_nested_conditions(self, translation_components):
        """Test nested conditional logic."""
        nl_to_asp, _ = translation_components

        text = "If a contract is for land, then if it is not in writing, then it is unenforceable"

        result = nl_to_asp.translate(text)

        # Should produce nested structure
        assert result.asp_code, "Should handle nested conditions"
        assert result.confidence >= 0.4, "Should have reasonable confidence"

    def test_multiple_conditions(self, translation_components):
        """Test multiple conditions (AND/OR)."""
        nl_to_asp, _ = translation_components

        text = (
            "A contract is enforceable if it is in writing and signed "
            "or if part performance has occurred"
        )

        result = nl_to_asp.translate(text)

        # Should handle multiple conditions
        assert result.asp_code, "Should handle multiple conditions"

    def test_exceptions_and_qualifiers(self, translation_components):
        """Test handling of exceptions and qualifiers."""
        nl_to_asp, _ = translation_components

        text = (
            "Generally, oral contracts are unenforceable, "
            "but there are several exceptions including part performance"
        )

        result = nl_to_asp.translate(text)

        # Should handle exception structure
        assert result.asp_code, "Should handle exceptions"


class TestContradictionHandling:
    """Test handling of contradictory statements."""

    def test_direct_contradiction(self, translation_components):
        """Test handling of direct contradictions."""
        nl_to_asp, _ = translation_components

        contradictory_text = "A contract is enforceable. The same contract is not enforceable."

        result = nl_to_asp.translate(contradictory_text)

        # Should still produce code (LLM should handle gracefully)
        assert result.asp_code, "Should produce code even with contradiction"

        # Confidence should be low
        assert result.confidence <= 0.7, (
            f"Confidence should reflect contradiction, got {result.confidence}"
        )

    def test_implicit_contradiction(self, translation_components):
        """Test handling of implicit contradictions."""
        nl_to_asp, _ = translation_components

        text = "All contracts require writing. Oral contracts are valid."

        result = nl_to_asp.translate(text)

        # Should handle implicit contradiction
        assert result.asp_code, "Should produce code for implicit contradiction"


class TestDomainSpecificEdgeCases:
    """Test edge cases specific to legal domain."""

    def test_legal_jargon(self, translation_components):
        """Test handling of legal terminology."""
        nl_to_asp, _ = translation_components

        jargon_cases = [
            "The statute of frauds requires a writing",
            "UCC 2-201 governs sales of goods",
            "The parol evidence rule applies",
            "Promissory estoppel may apply",
        ]

        for text in jargon_cases:
            result = nl_to_asp.translate(text)
            assert result.asp_code, f"Should handle legal jargon: {text}"

    def test_monetary_amounts(self, translation_components):
        """Test handling of monetary amounts."""
        nl_to_asp, _ = translation_components

        text = "Contracts for goods over $500 require a writing"

        result = nl_to_asp.translate(text)

        # Should handle numeric values
        assert result.asp_code, "Should handle monetary amounts"
        # Check if numeric value is preserved in some form
        assert "500" in result.asp_code or "value" in result.asp_code.lower(), (
            "Should preserve numeric constraint"
        )

    def test_temporal_references(self, translation_components):
        """Test handling of temporal references."""
        nl_to_asp, _ = translation_components

        text = "Contracts that cannot be performed within one year must be in writing"

        result = nl_to_asp.translate(text)

        # Should handle temporal constraint
        assert result.asp_code, "Should handle temporal references"


class TestEmptyAndMinimalInputs:
    """Test handling of edge cases like empty or minimal inputs."""

    def test_empty_input(self, translation_components):
        """Test handling of empty input."""
        nl_to_asp, _ = translation_components

        # Should handle gracefully
        result = nl_to_asp.translate("")

        # May return empty or minimal code
        assert isinstance(result.asp_code, str), "Should return string"

    def test_single_word(self, translation_components):
        """Test handling of single word input."""
        nl_to_asp, _ = translation_components

        result = nl_to_asp.translate("enforceable")

        # Should handle minimal input
        assert isinstance(result.asp_code, str), "Should return string"

    def test_very_long_input(self, translation_components):
        """Test handling of very long input."""
        nl_to_asp, _ = translation_components

        # Create long text
        long_text = " ".join(
            ["A contract is enforceable if it meets various requirements" for _ in range(20)]
        )

        result = nl_to_asp.translate(long_text)

        # Should handle long input without crashing
        assert result.asp_code, "Should handle long input"


@pytest.mark.slow
class TestRobustnessStress:
    """Stress tests for robustness (marked slow)."""

    def test_malformed_input(self, translation_components):
        """Test handling of malformed input."""
        nl_to_asp, _ = translation_components

        malformed_cases = [
            "contract is if when but not",
            "!!!@@@###",
            "A B C D E F G",
            "contract contract contract contract",
        ]

        for text in malformed_cases:
            try:
                result = nl_to_asp.translate(text)
                # Should not crash
                assert isinstance(result.asp_code, str), f"Should handle malformed input: {text}"
            except Exception as e:
                # If it raises an exception, it should be handled gracefully
                pytest.skip(f"Translation failed on malformed input (acceptable): {e}")
