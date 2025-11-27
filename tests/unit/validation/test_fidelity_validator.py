"""
Unit tests for fidelity validator.

Tests translation fidelity measurement for ASP ↔ Natural Language conversion.
"""

import pytest
from loft.validation.fidelity import FidelityValidator, FidelityTestResult, compute_translation_fidelity


class TestFidelityValidator:
    """Tests for FidelityValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a fidelity validator for testing."""
        return FidelityValidator()

    def test_perfect_fidelity_identity_translation(self, validator):
        """Test perfect fidelity with identity translation."""
        # Identity translation: ASP → ASP → ASP
        asp_original = "fact(a)."

        result = validator.test_roundtrip_fidelity(
            asp_original=asp_original,
            asp_to_nl_fn=lambda asp: asp,  # Identity: return ASP as-is
            nl_to_asp_fn=lambda nl: nl,  # Identity: return ASP as-is
        )

        assert isinstance(result, FidelityTestResult)
        assert result.fidelity_score == 1.0
        assert result.is_semantically_equivalent
        assert result.original_asp == asp_original
        assert result.reconstructed_asp == asp_original
        assert len(result.errors) == 0

    def test_perfect_fidelity_simple_fact(self, validator):
        """Test perfect fidelity with simple fact translation."""
        asp_original = "contract(c1)."

        # Simple translation that preserves semantics
        def asp_to_nl(asp: str) -> str:
            return "Contract c1 exists"

        def nl_to_asp(nl: str) -> str:
            return "contract(c1)."

        result = validator.test_roundtrip_fidelity(
            asp_original=asp_original,
            asp_to_nl_fn=asp_to_nl,
            nl_to_asp_fn=nl_to_asp,
        )

        assert result.fidelity_score == 1.0
        assert result.is_semantically_equivalent

    def test_imperfect_fidelity_information_loss(self, validator):
        """Test imperfect fidelity when information is lost."""
        asp_original = "enforceable(c1). void(c1)."

        # Translation loses void(c1)
        def asp_to_nl(asp: str) -> str:
            return "Contract c1 is enforceable"

        def nl_to_asp(nl: str) -> str:
            return "enforceable(c1)."  # Missing void(c1)

        result = validator.test_roundtrip_fidelity(
            asp_original=asp_original,
            asp_to_nl_fn=asp_to_nl,
            nl_to_asp_fn=nl_to_asp,
        )

        assert result.fidelity_score < 1.0
        assert not result.is_semantically_equivalent
        assert len(result.errors) == 0  # No translation errors, just semantic loss

    def test_fidelity_with_reordering(self, validator):
        """Test that reordering facts doesn't affect fidelity."""
        asp_original = "a. b. c."

        # Translation that reorders
        def asp_to_nl(asp: str) -> str:
            return "c, b, and a"

        def nl_to_asp(nl: str) -> str:
            return "c. b. a."  # Different order

        result = validator.test_roundtrip_fidelity(
            asp_original=asp_original,
            asp_to_nl_fn=asp_to_nl,
            nl_to_asp_fn=nl_to_asp,
        )

        # Should still be semantically equivalent (order doesn't matter in ASP)
        assert result.is_semantically_equivalent
        assert result.fidelity_score == 1.0

    def test_fidelity_with_rule(self, validator):
        """Test fidelity with ASP rules."""
        asp_original = "enforceable(C) :- contract(C), not void(C)."

        # Identity translation
        result = validator.test_roundtrip_fidelity(
            asp_original=asp_original,
            asp_to_nl_fn=lambda asp: asp,
            nl_to_asp_fn=lambda nl: nl,
        )

        assert result.fidelity_score == 1.0
        assert result.is_semantically_equivalent

    def test_fidelity_translation_error(self, validator):
        """Test handling of translation errors."""
        asp_original = "fact(a)."

        # Translation function that raises exception
        def faulty_translation(asp: str) -> str:
            raise ValueError("Translation failed")

        result = validator.test_roundtrip_fidelity(
            asp_original=asp_original,
            asp_to_nl_fn=faulty_translation,
            nl_to_asp_fn=lambda nl: nl,
        )

        assert result.fidelity_score == 0.0
        assert not result.is_semantically_equivalent
        assert len(result.errors) > 0
        assert "Translation failed" in result.errors[0]

    def test_semantic_equivalence_same_program(self, validator):
        """Test semantic equivalence of identical programs."""
        asp1 = "fact(a). fact(b)."
        asp2 = "fact(a). fact(b)."

        is_equiv, score = validator._semantic_equivalence(asp1, asp2)

        assert is_equiv
        assert score == 1.0

    def test_semantic_equivalence_different_order(self, validator):
        """Test semantic equivalence with different fact order."""
        asp1 = "fact(a). fact(b)."
        asp2 = "fact(b). fact(a)."

        is_equiv, score = validator._semantic_equivalence(asp1, asp2)

        assert is_equiv
        assert score == 1.0

    def test_semantic_equivalence_different_programs(self, validator):
        """Test semantic equivalence of different programs."""
        asp1 = "fact(a)."
        asp2 = "fact(b)."

        is_equiv, score = validator._semantic_equivalence(asp1, asp2)

        assert not is_equiv
        assert score < 1.0

    def test_semantic_equivalence_subset(self, validator):
        """Test semantic equivalence when one is subset of other."""
        asp1 = "fact(a). fact(b)."
        asp2 = "fact(a)."

        is_equiv, score = validator._semantic_equivalence(asp1, asp2)

        assert not is_equiv
        # Different number of answer sets leads to 0 overlap
        # (both have 1 answer set, but with different content)
        assert score == 0.0  # No overlap because answer sets differ

    def test_get_answer_sets_simple(self, validator):
        """Test getting answer sets from simple program."""
        asp_program = "a. b."

        answer_sets = validator._get_answer_sets(asp_program)

        assert len(answer_sets) == 1
        # Should contain 'a' and 'b'
        assert 'a' in answer_sets[0]
        assert 'b' in answer_sets[0]

    def test_get_answer_sets_with_choice(self, validator):
        """Test getting multiple answer sets with choice rule."""
        asp_program = "{a}. b."

        answer_sets = validator._get_answer_sets(asp_program, max_sets=5)

        # Should have 2 answer sets: {b} and {a, b}
        assert len(answer_sets) == 2

    def test_get_answer_sets_unsatisfiable(self, validator):
        """Test getting answer sets from unsatisfiable program."""
        asp_program = "a. :- a."  # Constraint that rejects 'a'

        answer_sets = validator._get_answer_sets(asp_program)

        # No answer sets (unsatisfiable)
        assert len(answer_sets) == 0

    def test_get_answer_sets_with_rule(self, validator):
        """Test getting answer sets with rules."""
        asp_program = "b :- a. a."

        answer_sets = validator._get_answer_sets(asp_program)

        assert len(answer_sets) == 1
        # Both 'a' and 'b' should be derived
        assert 'a' in answer_sets[0]
        assert 'b' in answer_sets[0]

    def test_answer_sets_match_identical(self, validator):
        """Test answer set matching with identical sets."""
        sets1 = [frozenset(['a', 'b']), frozenset(['c'])]
        sets2 = [frozenset(['a', 'b']), frozenset(['c'])]

        assert validator._answer_sets_match(sets1, sets2)

    def test_answer_sets_match_different_order(self, validator):
        """Test answer set matching with different order."""
        sets1 = [frozenset(['a', 'b']), frozenset(['c'])]
        sets2 = [frozenset(['c']), frozenset(['a', 'b'])]  # Different order

        assert validator._answer_sets_match(sets1, sets2)

    def test_answer_sets_match_different_content(self, validator):
        """Test answer set matching with different content."""
        sets1 = [frozenset(['a', 'b'])]
        sets2 = [frozenset(['c', 'd'])]

        assert not validator._answer_sets_match(sets1, sets2)

    def test_answer_sets_match_different_length(self, validator):
        """Test answer set matching with different lengths."""
        sets1 = [frozenset(['a']), frozenset(['b'])]
        sets2 = [frozenset(['a'])]

        assert not validator._answer_sets_match(sets1, sets2)

    def test_compute_answer_set_overlap_perfect(self, validator):
        """Test overlap computation with perfect match."""
        sets1 = [frozenset(['a', 'b'])]
        sets2 = [frozenset(['a', 'b'])]

        overlap = validator._compute_answer_set_overlap(sets1, sets2)

        assert overlap == 1.0

    def test_compute_answer_set_overlap_none(self, validator):
        """Test overlap computation with no overlap."""
        sets1 = [frozenset(['a'])]
        sets2 = [frozenset(['b'])]

        overlap = validator._compute_answer_set_overlap(sets1, sets2)

        assert overlap == 0.0

    def test_compute_answer_set_overlap_partial(self, validator):
        """Test overlap computation with partial overlap."""
        sets1 = [frozenset(['a', 'b']), frozenset(['c'])]
        sets2 = [frozenset(['a', 'b']), frozenset(['d'])]

        overlap = validator._compute_answer_set_overlap(sets1, sets2)

        # Intersection: {frozenset(['a', 'b'])} size 1
        # Union: {frozenset(['a', 'b']), frozenset(['c']), frozenset(['d'])} size 3
        # Jaccard: 1/3 ≈ 0.333
        assert 0.3 < overlap < 0.4

    def test_compute_answer_set_overlap_both_empty(self, validator):
        """Test overlap computation with both empty."""
        sets1 = []
        sets2 = []

        overlap = validator._compute_answer_set_overlap(sets1, sets2)

        assert overlap == 1.0  # Both empty = perfect match

    def test_compute_answer_set_overlap_one_empty(self, validator):
        """Test overlap computation with one empty."""
        sets1 = [frozenset(['a'])]
        sets2 = []

        overlap = validator._compute_answer_set_overlap(sets1, sets2)

        assert overlap == 0.0

    def test_batch_test_fidelity(self, validator):
        """Test batch fidelity testing."""
        asp_examples = [
            "fact(a).",
            "fact(b).",
            "rule(x) :- condition(x).",
        ]

        # Identity translation (perfect fidelity)
        stats = validator.batch_test_fidelity(
            asp_examples=asp_examples,
            asp_to_nl_fn=lambda asp: asp,
            nl_to_asp_fn=lambda nl: nl,
        )

        assert stats["total"] == 3
        assert stats["passed"] == 3  # All should pass (fidelity > 0.95)
        assert stats["average_fidelity"] == 1.0
        assert stats["min_fidelity"] == 1.0
        assert stats["max_fidelity"] == 1.0
        assert len(stats["results"]) == 3

    def test_batch_test_fidelity_mixed_quality(self, validator):
        """Test batch fidelity with mixed quality translations."""
        asp_examples = [
            "a.",  # Will translate perfectly
            "b.",  # Will translate perfectly
        ]

        # First translation perfect, second loses information
        call_count = [0]

        def nl_to_asp(nl: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "a."
            else:
                return ""  # Lost information

        stats = validator.batch_test_fidelity(
            asp_examples=asp_examples,
            asp_to_nl_fn=lambda asp: asp,
            nl_to_asp_fn=nl_to_asp,
        )

        assert stats["total"] == 2
        assert stats["passed"] < stats["total"]  # Not all passed
        assert stats["average_fidelity"] < 1.0

    def test_batch_test_fidelity_empty_list(self, validator):
        """Test batch fidelity with empty list."""
        stats = validator.batch_test_fidelity(
            asp_examples=[],
            asp_to_nl_fn=lambda asp: asp,
            nl_to_asp_fn=lambda nl: nl,
        )

        assert stats["total"] == 0
        assert stats["passed"] == 0
        assert stats["average_fidelity"] == 0.0
        assert stats["min_fidelity"] == 0.0
        assert stats["max_fidelity"] == 0.0

    def test_compute_translation_fidelity_function(self):
        """Test standalone compute_translation_fidelity function."""
        fidelity = compute_translation_fidelity(
            asp_program="fact(a).",
            asp_to_nl_fn=lambda asp: asp,
            nl_to_asp_fn=lambda nl: nl,
        )

        assert 0 <= fidelity <= 1.0
        assert fidelity == 1.0  # Identity should be perfect

    def test_fidelity_result_structure(self, validator):
        """Test that FidelityTestResult has all expected fields."""
        result = validator.test_roundtrip_fidelity(
            asp_original="a.",
            asp_to_nl_fn=lambda asp: "a",
            nl_to_asp_fn=lambda nl: "a.",
        )

        assert hasattr(result, "original_asp")
        assert hasattr(result, "natural_language")
        assert hasattr(result, "reconstructed_asp")
        assert hasattr(result, "fidelity_score")
        assert hasattr(result, "is_semantically_equivalent")
        assert hasattr(result, "explanation")
        assert hasattr(result, "errors")

    def test_fidelity_with_complex_rule(self, validator):
        """Test fidelity with complex ASP rule."""
        asp_original = """
        enforceable(C) :-
            contract(C),
            has_writing(C),
            not void(C),
            not fraud(C).
        """

        # Identity translation
        result = validator.test_roundtrip_fidelity(
            asp_original=asp_original,
            asp_to_nl_fn=lambda asp: asp,
            nl_to_asp_fn=lambda nl: nl,
        )

        assert result.fidelity_score == 1.0
        assert result.is_semantically_equivalent

    def test_fidelity_with_negation(self, validator):
        """Test fidelity preservation with negation."""
        asp_original = "valid(C) :- contract(C), not invalid(C)."

        result = validator.test_roundtrip_fidelity(
            asp_original=asp_original,
            asp_to_nl_fn=lambda asp: asp,
            nl_to_asp_fn=lambda nl: nl,
        )

        assert result.fidelity_score == 1.0

    def test_fidelity_with_constraint(self, validator):
        """Test fidelity with ASP constraints."""
        asp_original = ":- enforceable(C), void(C)."

        result = validator.test_roundtrip_fidelity(
            asp_original=asp_original,
            asp_to_nl_fn=lambda asp: asp,
            nl_to_asp_fn=lambda nl: nl,
        )

        assert result.fidelity_score == 1.0

    def test_explanation_content(self, validator):
        """Test that explanation contains useful information."""
        asp_original = "fact(a)."

        result = validator.test_roundtrip_fidelity(
            asp_original=asp_original,
            asp_to_nl_fn=lambda asp: "Natural language: " + asp,
            nl_to_asp_fn=lambda nl: "fact(a).",
        )

        assert "Original ASP" in result.explanation
        assert "Natural Language" in result.explanation
        assert "Reconstructed ASP" in result.explanation
        assert "Fidelity" in result.explanation
        assert "Semantically Equivalent" in result.explanation

    def test_max_answer_sets_limit(self, validator):
        """Test that max_sets parameter limits answer sets."""
        # Program with many answer sets
        asp_program = "{a}. {b}. {c}."  # 2^3 = 8 answer sets

        answer_sets = validator._get_answer_sets(asp_program, max_sets=3)

        # Should stop at max_sets
        assert len(answer_sets) <= 3

    def test_semantic_equivalence_with_derived_facts(self, validator):
        """Test semantic equivalence when facts are derived vs explicit."""
        asp1 = "a. b :- a."  # 'b' is derived
        asp2 = "a. b."  # 'b' is explicit

        is_equiv, score = validator._semantic_equivalence(asp1, asp2)

        # Both should derive same answer set {a, b}
        assert is_equiv
        assert score == 1.0
