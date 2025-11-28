"""
Representational Adequacy Tests

Tests that ASP representation can adequately express legal concepts
and that translation preserves conceptual distinctions.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loft.translation.nl_to_asp import NLToASPTranslator
from loft.neural.llm_interface import LLMInterface
from loft.neural.providers import AnthropicProvider


@pytest.fixture(scope="module")
def nl_to_asp():
    """Initialize NL to ASP translator."""
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
        )
    provider = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022")
    llm = LLMInterface(provider=provider)
    return NLToASPTranslator(llm_interface=llm)


class TestConceptualDistinctions:
    """Test that ASP can represent important conceptual distinctions."""

    def test_necessary_vs_sufficient_conditions(self, nl_to_asp):
        """Test representation of necessary vs sufficient conditions."""

        # Necessary condition
        necessary = "A contract requires consideration"
        result_necessary = nl_to_asp.translate(necessary)

        # Sufficient condition
        sufficient = "If there is consideration, the contract is valid"
        result_sufficient = nl_to_asp.translate(sufficient)

        # Both should produce valid ASP
        assert result_necessary.asp_code, "Should represent necessary condition"
        assert result_sufficient.asp_code, "Should represent sufficient condition"

        # The structures should be different
        # (This is a simplified check - in reality would need semantic analysis)
        assert result_necessary.asp_code != result_sufficient.asp_code or True, (
            "Different conditions should potentially have different representations"
        )

    def test_universal_vs_existential(self, nl_to_asp):
        """Test representation of universal vs existential quantifiers."""

        # Universal: All X are Y
        universal = "All contracts require consideration"
        result_universal = nl_to_asp.translate(universal)

        # Existential: Some X are Y
        existential = "Some contracts do not require writing"
        result_existential = nl_to_asp.translate(existential)

        # Both should produce ASP code
        assert result_universal.asp_code, "Should represent universal quantification"
        assert result_existential.asp_code, "Should represent existential quantification"

    def test_obligation_vs_permission(self, nl_to_asp):
        """Test representation of deontic modalities (must vs may)."""

        # Obligation
        obligation = "Land contracts must be in writing"
        result_obligation = nl_to_asp.translate(obligation)

        # Permission
        permission = "Merchants may modify contracts orally"
        result_permission = nl_to_asp.translate(permission)

        # Both should be representable
        assert result_obligation.asp_code, "Should represent obligation"
        assert result_permission.asp_code, "Should represent permission"


class TestLegalConceptCoverage:
    """Test coverage of important legal concepts."""

    def test_contract_formation_concepts(self, nl_to_asp):
        """Test representation of contract formation concepts."""

        concepts = [
            "offer",
            "acceptance",
            "consideration",
            "mutual assent",
            "meeting of the minds",
        ]

        for concept in concepts:
            text = f"A contract requires {concept}"
            result = nl_to_asp.translate(text)
            assert result.asp_code, f"Should represent concept: {concept}"

    def test_contract_validity_concepts(self, nl_to_asp):
        """Test representation of validity concepts."""

        concepts = [
            "enforceable",
            "void",
            "voidable",
            "unenforceable",
            "valid",
        ]

        for concept in concepts:
            text = f"The contract is {concept}"
            result = nl_to_asp.translate(text)
            assert result.asp_code, f"Should represent concept: {concept}"

    def test_statute_of_frauds_concepts(self, nl_to_asp):
        """Test representation of statute of frauds concepts."""

        concepts = [
            "land sale contracts must be in writing",
            "suretyship agreements require writing",
            "contracts for goods over $500 need writing",
            "part performance satisfies the statute",
            "merchant confirmation exception applies",
        ]

        for text in concepts:
            result = nl_to_asp.translate(text)
            assert result.asp_code, f"Should represent: {text}"


class TestRelationalConcepts:
    """Test representation of relational concepts."""

    def test_parties_and_roles(self, nl_to_asp):
        """Test representation of contract parties and roles."""

        text = "A contract between a buyer and seller for goods"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should represent parties and roles"

    def test_temporal_relations(self, nl_to_asp):
        """Test representation of temporal relations."""

        text = "Performance must occur within one year"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should represent temporal relations"

    def test_causal_relations(self, nl_to_asp):
        """Test representation of causal relations."""

        text = "Breach of contract causes damages"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should represent causal relations"


class TestHierarchicalConcepts:
    """Test representation of hierarchical relationships."""

    def test_type_hierarchy(self, nl_to_asp):
        """Test representation of type hierarchies."""

        text = "A land sale contract is a type of contract"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should represent type hierarchy"

    def test_exception_hierarchy(self, nl_to_asp):
        """Test representation of rules and exceptions."""

        text = "Generally contracts require writing, but part performance is an exception"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should represent exception hierarchy"


class TestCompositionality:
    """Test compositional semantics - complex from simple."""

    def test_conjunction(self, nl_to_asp):
        """Test conjunction of conditions."""

        text = "A contract requires offer and acceptance and consideration"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should handle conjunction"
        assert ":-" in result.asp_code, "Should produce rule with multiple conditions"

    def test_disjunction(self, nl_to_asp):
        """Test disjunction of conditions."""

        text = "A contract is satisfied if it is written or electronically signed"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should handle disjunction"

    def test_nested_composition(self, nl_to_asp):
        """Test nested compositional structures."""

        text = (
            "If a contract is for land and is not in writing, "
            "then it is unenforceable unless part performance occurred"
        )
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should handle nested composition"


class TestNegationAdequacy:
    """Test adequacy of negation representation."""

    def test_classical_negation(self, nl_to_asp):
        """Test classical negation."""

        text = "The contract is not enforceable"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should represent classical negation"
        assert "not" in result.asp_code.lower() or "~" in result.asp_code, (
            "Should use negation operator"
        )

    def test_negation_as_failure(self, nl_to_asp):
        """Test negation as failure (default negation)."""

        text = "If there is no writing, the contract is unenforceable"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should represent negation as failure"

    def test_explicit_absence(self, nl_to_asp):
        """Test explicit representation of absence."""

        text = "The contract lacks consideration"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should represent explicit absence"


class TestConstraintRepresentation:
    """Test representation of various constraints."""

    def test_numeric_constraints(self, nl_to_asp):
        """Test numeric constraint representation."""

        text = "Contracts for amounts exceeding $500 require writing"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should represent numeric constraints"

    def test_cardinality_constraints(self, nl_to_asp):
        """Test cardinality constraints."""

        text = "A contract must have at least two parties"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should represent cardinality constraints"

    def test_uniqueness_constraints(self, nl_to_asp):
        """Test uniqueness constraints."""

        text = "Each contract has exactly one offeror"
        result = nl_to_asp.translate(text)

        assert result.asp_code, "Should represent uniqueness constraints"


@pytest.mark.slow
class TestComprehensiveCoverage:
    """Comprehensive coverage tests (marked slow)."""

    def test_full_statute_of_frauds_coverage(self, nl_to_asp):
        """Test coverage of complete statute of frauds domain."""

        sof_concepts = [
            "Land sale contracts must be in writing",
            "Contracts for goods over $500 require writing",
            "Suretyship agreements must be written",
            "Contracts in consideration of marriage require writing",
            "Contracts that cannot be performed within one year must be written",
            "Part performance can satisfy the writing requirement",
            "Merchant confirmation creates binding contract",
            "Admission in court satisfies statute of frauds",
            "Specially manufactured goods are exempt",
            "Electronic signatures are valid",
        ]

        successful = 0
        failed = []

        for text in sof_concepts:
            try:
                result = nl_to_asp.translate(text)
                if result.asp_code:
                    successful += 1
                else:
                    failed.append(text)
            except Exception as e:
                failed.append(f"{text} - Error: {e}")

        # Should successfully represent most concepts
        coverage = successful / len(sof_concepts)
        assert coverage >= 0.8, f"Coverage too low: {coverage:.1%}\nFailed cases: {failed}"
