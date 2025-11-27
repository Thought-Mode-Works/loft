"""
Unit tests for translation quality metrics and fidelity tracking.

Tests quality validation, fidelity computation, and roundtrip testing.
"""

import pytest
from unittest.mock import Mock, MagicMock
from loft.translation.quality import (
    QualityMetrics,
    validate_translation_quality,
    check_grammar_with_llm,
    compute_fidelity,
    compute_quality_metrics,
    _question_to_statement,
    roundtrip_fidelity_test,
    compute_asp_equivalence,
    _compute_semantic_equivalence_with_solver,
)
from loft.translation.asp_to_nl import extract_predicates


class TestQualityMetrics:
    """Test QualityMetrics dataclass."""

    def test_quality_metrics_creation(self):
        """Test creating quality metrics."""
        metrics = QualityMetrics(completeness=0.9, readability=0.8, fidelity=0.85, overall=0.85)
        assert metrics.completeness == 0.9
        assert metrics.readability == 0.8
        assert metrics.fidelity == 0.85
        assert metrics.overall == 0.85


class TestValidateTranslationQuality:
    """Test validate_translation_quality function."""

    def test_perfect_translation(self):
        """Test quality score for perfect translation."""
        asp = "contract(c1)."
        nl = "c1 is a contract."
        score = validate_translation_quality(asp, nl)
        assert score > 0.8

    def test_all_predicates_present(self):
        """Test when all predicates are mentioned."""
        asp = "contract(c1), enforceable(c1)."
        nl = "c1 is a contract and c1 is enforceable."
        score = validate_translation_quality(asp, nl)
        assert score > 0.8

    def test_missing_predicates(self):
        """Test when some predicates are missing."""
        asp = "contract(c1), enforceable(c1)."
        nl = "c1 is a contract."
        score = validate_translation_quality(asp, nl)
        assert score < 1.0

    def test_humanized_predicate_names(self):
        """Test that humanized predicate names are recognized."""
        asp = "statute_of_frauds(c1)."
        nl = "c1 has statute of frauds."
        score = validate_translation_quality(asp, nl)
        assert score > 0.5

    def test_incomplete_sentence(self):
        """Test penalty for incomplete sentences."""
        asp = "contract(c1)."
        nl = "c1 is a contract"  # No period
        score = validate_translation_quality(asp, nl)
        # Should be penalized but not zero
        assert 0.3 < score < 1.0

    def test_short_translation(self):
        """Test penalty for too-short translations."""
        asp = "contract(c1)."
        nl = "c1."  # Very short
        score = validate_translation_quality(asp, nl)
        assert score < 0.8

    def test_empty_translation(self):
        """Test score for empty translation."""
        asp = "contract(c1)."
        nl = ""
        score = validate_translation_quality(asp, nl)
        assert score < 0.5

    def test_no_predicates(self):
        """Test with no predicates in ASP."""
        asp = ""
        nl = "Some text."
        score = validate_translation_quality(asp, nl)
        # Should handle gracefully
        assert 0.0 <= score <= 1.0


class TestCheckGrammarWithLLM:
    """Test check_grammar_with_llm function."""

    def test_grammar_check_success(self):
        """Test successful grammar checking."""
        llm = Mock()
        mock_response = Mock()
        mock_response.raw_text = "0.95"
        llm.query.return_value = mock_response

        score = check_grammar_with_llm("This is a well-formed sentence.", llm)
        assert score == 0.95
        llm.query.assert_called_once()

    def test_grammar_check_whole_number(self):
        """Test parsing whole number scores."""
        llm = Mock()
        mock_response = Mock()
        mock_response.raw_text = "1"
        llm.query.return_value = mock_response

        score = check_grammar_with_llm("Perfect grammar.", llm)
        assert score == 1.0

    def test_grammar_check_with_text(self):
        """Test parsing score from text response."""
        llm = Mock()
        mock_response = Mock()
        mock_response.raw_text = "The score is 0.87 out of 1.0"
        llm.query.return_value = mock_response

        score = check_grammar_with_llm("Some sentence.", llm)
        assert score == 0.87

    def test_grammar_check_unparseable(self):
        """Test fallback when score cannot be parsed."""
        llm = Mock()
        mock_response = Mock()
        mock_response.raw_text = "Good grammar"
        llm.query.return_value = mock_response

        score = check_grammar_with_llm("Some sentence.", llm)
        assert score == 0.7  # Default fallback

    def test_grammar_check_error(self):
        """Test error handling."""
        llm = Mock()
        llm.query.side_effect = Exception("API error")

        score = check_grammar_with_llm("Some sentence.", llm)
        assert score == 0.7  # Default fallback


class TestComputeFidelity:
    """Test compute_fidelity function."""

    def test_identical_asp(self):
        """Test fidelity of identical ASP."""
        asp = "contract(c1)."
        fidelity = compute_fidelity(asp, asp)
        assert fidelity == 1.0

    def test_whitespace_normalization(self):
        """Test that whitespace is normalized."""
        asp1 = "contract(c1)."
        asp2 = "contract( c1 ) ."
        fidelity = compute_fidelity(asp1, asp2)
        assert fidelity == 1.0

    def test_same_predicates_different_order(self):
        """Test fidelity with same predicates in different order."""
        asp1 = "contract(c1), enforceable(c1)."
        asp2 = "enforceable(c1), contract(c1)."
        fidelity = compute_fidelity(asp1, asp2)
        assert fidelity == 1.0  # Jaccard similarity of identical sets

    def test_partial_overlap(self):
        """Test fidelity with partial predicate overlap."""
        asp1 = "contract(c1), enforceable(c1)."
        asp2 = "contract(c1), void(c1)."
        fidelity = compute_fidelity(asp1, asp2)
        # Jaccard: {contract, enforceable} vs {contract, void}
        # Intersection: {contract}, Union: {contract, enforceable, void}
        # Score: 1/3 ≈ 0.33
        assert 0.3 <= fidelity <= 0.4

    def test_no_overlap(self):
        """Test fidelity with no predicate overlap."""
        asp1 = "contract(c1)."
        asp2 = "void(c1)."
        fidelity = compute_fidelity(asp1, asp2)
        assert fidelity == 0.0

    def test_empty_asp(self):
        """Test fidelity with empty ASP."""
        asp1 = ""
        asp2 = "contract(c1)."
        fidelity = compute_fidelity(asp1, asp2)
        assert fidelity == 0.0


class TestComputeQualityMetrics:
    """Test compute_quality_metrics function."""

    def test_quality_metrics_structure(self):
        """Test that all metric fields are present."""
        asp = "contract(c1)."
        nl = "c1 is a contract."
        metrics = compute_quality_metrics(asp, nl)

        assert isinstance(metrics, QualityMetrics)
        assert 0.0 <= metrics.completeness <= 1.0
        assert 0.0 <= metrics.readability <= 1.0
        assert 0.0 <= metrics.fidelity <= 1.0
        assert 0.0 <= metrics.overall <= 1.0

    def test_high_quality_metrics(self):
        """Test metrics for high quality translation."""
        asp = "contract(c1)."
        nl = "c1 is a contract."
        metrics = compute_quality_metrics(asp, nl)

        assert metrics.completeness > 0.8
        assert metrics.readability > 0.5
        assert metrics.overall > 0.6

    def test_low_quality_metrics(self):
        """Test metrics for low quality translation."""
        asp = "contract(c1), enforceable(c1), valid(c1)."
        nl = "Something."
        metrics = compute_quality_metrics(asp, nl)

        assert metrics.completeness < 0.5
        assert metrics.overall < 0.7

    def test_metrics_with_llm(self):
        """Test metrics computation with LLM interface."""
        llm = Mock()
        mock_response = Mock()
        mock_response.raw_text = "0.9"
        llm.query.return_value = mock_response

        asp = "contract(c1)."
        nl = "c1 is a contract."
        metrics = compute_quality_metrics(asp, nl, llm_interface=llm)

        # Should include LLM grammar check in readability
        assert metrics.readability > 0.5
        llm.query.assert_called_once()

    def test_metrics_without_llm(self):
        """Test metrics computation without LLM."""
        asp = "contract(c1)."
        nl = "c1 is a contract."
        metrics = compute_quality_metrics(asp, nl, llm_interface=None)

        # Should still compute all metrics
        assert 0.0 <= metrics.readability <= 1.0


class TestQuestionToStatement:
    """Test _question_to_statement function."""

    def test_is_a_pattern(self):
        """Test 'Is X a Y?' conversion."""
        question = "Is contract c1 a contract?"
        statement = _question_to_statement(question)
        assert statement == "contract c1 is a contract"

    def test_is_an_pattern(self):
        """Test 'Is X an Y?' conversion."""
        question = "Is c1 an enforceable contract?"
        statement = _question_to_statement(question)
        assert statement == "c1 is a enforceable contract"

    def test_signed_by_pattern(self):
        """Test 'Is X signed by Y?' conversion."""
        question = "Is writing w1 signed by john?"
        statement = _question_to_statement(question)
        assert statement == "writing w1 was signed by john"

    def test_was_signed_by_pattern(self):
        """Test 'Was X signed by Y?' conversion."""
        question = "Was document d1 signed by mary?"
        statement = _question_to_statement(question)
        assert statement == "document d1 was signed by mary"

    def test_does_hold_pattern(self):
        """Test 'Does X hold for Y?' conversion."""
        question = "Does writing hold for writing w1?"
        statement = _question_to_statement(question)
        assert "w1 is a writing" in statement

    def test_does_have_pattern(self):
        """Test 'Does X have Y?' conversion."""
        question = "Does contract c1 have consideration?"
        statement = _question_to_statement(question)
        assert statement == "contract c1 has consideration"

    def test_is_adjective_pattern(self):
        """Test 'Is X Y?' (adjective) conversion."""
        question = "Is contract c1 valid?"
        statement = _question_to_statement(question)
        assert statement == "contract c1 is valid"

    def test_unrecognized_pattern(self):
        """Test unrecognized patterns are returned as-is."""
        question = "What is the status of contract c1?"
        statement = _question_to_statement(question)
        assert statement == question


class TestRoundtripFidelityTest:
    """Test roundtrip_fidelity_test function."""

    def test_perfect_roundtrip(self):
        """Test roundtrip with perfect reconstruction."""
        asp_to_nl = Mock()
        nl_to_asp = Mock()

        # Mock ASP → NL
        nl_result = Mock()
        nl_result.natural_language = "Is contract c1 a contract?"
        asp_to_nl.translate_query.return_value = nl_result

        # Mock NL → ASP
        asp_result = Mock()
        asp_result.asp_facts = ["contract(c1)."]
        nl_to_asp.translate_to_facts.return_value = asp_result

        fidelity, explanation = roundtrip_fidelity_test("contract(c1).", asp_to_nl, nl_to_asp)

        assert fidelity == 1.0
        assert "contract(c1)" in explanation
        assert "Fidelity" in explanation

    def test_partial_roundtrip(self):
        """Test roundtrip with partial reconstruction."""
        asp_to_nl = Mock()
        nl_to_asp = Mock()

        nl_result = Mock()
        nl_result.natural_language = "Is contract c1 a contract?"
        asp_to_nl.translate_query.return_value = nl_result

        asp_result = Mock()
        asp_result.asp_facts = ["void(c1)."]  # Different predicate
        nl_to_asp.translate_to_facts.return_value = asp_result

        fidelity, explanation = roundtrip_fidelity_test("contract(c1).", asp_to_nl, nl_to_asp)

        assert fidelity < 1.0
        assert "Fidelity" in explanation

    def test_roundtrip_with_question_conversion(self):
        """Test that questions are converted to statements."""
        asp_to_nl = Mock()
        nl_to_asp = Mock()

        nl_result = Mock()
        nl_result.natural_language = "Does contract c1 have consideration?"
        asp_to_nl.translate_query.return_value = nl_result

        asp_result = Mock()
        asp_result.asp_facts = ["has_consideration(c1)."]
        nl_to_asp.translate_to_facts.return_value = asp_result

        fidelity, explanation = roundtrip_fidelity_test(
            "has_consideration(c1).", asp_to_nl, nl_to_asp
        )

        # Verify question was converted to statement
        assert "statement" in explanation.lower()


class TestComputeAspEquivalence:
    """Test compute_asp_equivalence function."""

    def test_identical_asp_no_solver(self):
        """Test equivalence of identical ASP without solver."""
        asp1 = "contract(c1)."
        asp2 = "contract(c1)."
        score = compute_asp_equivalence(asp1, asp2)
        assert score == 1.0

    def test_different_asp_no_solver(self):
        """Test equivalence of different ASP without solver."""
        asp1 = "contract(c1)."
        asp2 = "void(c1)."
        score = compute_asp_equivalence(asp1, asp2)
        assert score == 0.0

    def test_with_asp_core(self):
        """Test equivalence with ASP core."""
        # Mock ASP core
        asp_core = Mock()
        asp1 = "contract(c1)."
        asp2 = "contract(c1)."

        # Should attempt semantic equivalence then fall back
        score = compute_asp_equivalence(asp1, asp2, asp_core)
        assert 0.0 <= score <= 1.0


class TestComputeSemanticEquivalenceWithSolver:
    """Test _compute_semantic_equivalence_with_solver function."""

    def test_identical_facts(self):
        """Test semantic equivalence of identical facts."""
        try:
            from clingo import Control

            asp1 = "contract(c1)."
            asp2 = "contract(c1)."
            asp_core = Mock()

            score = _compute_semantic_equivalence_with_solver(asp1, asp2, asp_core)
            assert score == 1.0
        except ImportError:
            pytest.skip("Clingo not available")

    def test_both_unsatisfiable(self):
        """Test equivalence when both are unsatisfiable."""
        try:
            from clingo import Control

            asp1 = "contract(c1). :- contract(c1)."
            asp2 = "void(c1). :- void(c1)."
            asp_core = Mock()

            score = _compute_semantic_equivalence_with_solver(asp1, asp2, asp_core)
            assert score == 1.0  # Both have no models
        except ImportError:
            pytest.skip("Clingo not available")

    def test_one_satisfiable_one_not(self):
        """Test when one is satisfiable and other is not."""
        try:
            from clingo import Control

            asp1 = "contract(c1)."
            asp2 = ":- contract(c1)."  # Unsatisfiable if contract(c1) is added
            asp_core = Mock()

            # This test depends on clingo behavior
            score = _compute_semantic_equivalence_with_solver(asp1, asp2, asp_core)
            assert 0.0 <= score <= 1.0
        except ImportError:
            pytest.skip("Clingo not available")

    def test_different_models(self):
        """Test equivalence with different answer sets."""
        try:
            from clingo import Control

            asp1 = "contract(c1)."
            asp2 = "void(c1)."
            asp_core = Mock()

            score = _compute_semantic_equivalence_with_solver(asp1, asp2, asp_core)
            # Different answer sets
            assert score < 1.0
        except ImportError:
            pytest.skip("Clingo not available")

    def test_clingo_not_available(self):
        """Test fallback when clingo is not available."""
        # Save original import
        import sys

        original_modules = sys.modules.copy()

        try:
            # Mock clingo as unavailable
            sys.modules["clingo"] = None

            asp1 = "contract(c1)."
            asp2 = "contract(c1)."
            asp_core = Mock()

            # Should fall back to syntactic comparison
            score = _compute_semantic_equivalence_with_solver(asp1, asp2, asp_core)
            assert score == 1.0
        finally:
            # Restore modules
            sys.modules = original_modules
