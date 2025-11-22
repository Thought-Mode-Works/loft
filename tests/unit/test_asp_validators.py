"""
Unit tests for ASP validators.

Tests syntax and semantic validation of ASP programs.
"""

import pytest
from loft.validation import (
    ASPSyntaxValidator,
    ASPSemanticValidator,
    validate_asp_program,
)


class TestASPSyntaxValidator:
    """Tests for ASPSyntaxValidator."""

    def test_valid_program(self) -> None:
        """Test that valid ASP program passes syntax validation."""
        validator = ASPSyntaxValidator()
        is_valid, error = validator.validate_program("fact(a).")

        assert is_valid
        assert error is None

    def test_valid_rule(self) -> None:
        """Test that valid ASP rule passes syntax validation."""
        validator = ASPSyntaxValidator()
        program = "head(X) :- body(X)."
        is_valid, error = validator.validate_program(program)

        assert is_valid
        assert error is None

    def test_invalid_syntax(self) -> None:
        """Test that invalid syntax fails validation."""
        validator = ASPSyntaxValidator()
        program = "this is not valid ASP syntax"
        is_valid, error = validator.validate_program(program)

        assert not is_valid
        assert error is not None
        assert "syntax" in error.lower() or "error" in error.lower()

    def test_empty_program(self) -> None:
        """Test that empty program is valid."""
        validator = ASPSyntaxValidator()
        is_valid, error = validator.validate_program("")

        assert is_valid
        assert error is None

    def test_complex_program(self) -> None:
        """Test validation of complex ASP program."""
        validator = ASPSyntaxValidator()
        program = """
        % Facts
        contract(c1).
        party(john).

        % Rules
        enforceable(C) :- contract(C), not unenforceable(C).
        unenforceable(C) :- contract(C), missing_requirement(C).

        % Constraints
        :- enforceable(C), unenforceable(C).
        """
        is_valid, error = validator.validate_program(program)

        assert is_valid
        assert error is None


class TestASPSemanticValidator:
    """Tests for ASPSemanticValidator."""

    def test_consistent_program(self) -> None:
        """Test that consistent program passes validation."""
        validator = ASPSemanticValidator()
        program = "fact(a). fact(b)."
        is_consistent, msg = validator.check_consistency(program)

        assert is_consistent
        assert "answer set" in msg.lower()

    def test_inconsistent_program(self) -> None:
        """Test that inconsistent program fails validation."""
        validator = ASPSemanticValidator()
        # Program with contradiction
        program = """
        a.
        -a.
        :- a, -a.
        """
        is_consistent, msg = validator.check_consistency(program)

        assert not is_consistent

    def test_count_answer_sets_deterministic(self) -> None:
        """Test counting answer sets for deterministic program."""
        validator = ASPSemanticValidator()
        program = "fact(a)."
        count = validator.count_answer_sets(program)

        assert count == 1

    def test_count_answer_sets_nondeterministic(self) -> None:
        """Test counting answer sets for non-deterministic program."""
        validator = ASPSemanticValidator()
        # Choice rule creates multiple answer sets
        program = "{a; b}."
        count = validator.count_answer_sets(program)

        # Should have 4 answer sets: {}, {a}, {b}, {a,b}
        assert count == 4

    def test_rule_composition_compatible(self) -> None:
        """Test that compatible rules compose correctly."""
        validator = ASPSemanticValidator()
        rule1 = "a :- b."
        rule2 = "b :- c."

        assert validator.check_rule_composition(rule1, rule2)

    def test_rule_composition_incompatible(self) -> None:
        """Test that incompatible rules are detected."""
        validator = ASPSemanticValidator()
        rule1 = "a."
        rule2 = "-a."

        # These create a contradiction
        assert not validator.check_rule_composition(rule1, rule2)

    def test_get_answer_sets(self) -> None:
        """Test retrieving answer sets."""
        validator = ASPSemanticValidator()
        program = "fact(a). fact(b)."
        answer_sets = validator.get_answer_sets(program)

        assert len(answer_sets) == 1
        # Answer set should contain both facts
        symbols = answer_sets[0]
        assert len(symbols) == 2

    def test_detect_contradictions(self) -> None:
        """Test contradiction detection."""
        validator = ASPSemanticValidator()

        # Consistent program
        program1 = "a. b."
        contradictions1 = validator.detect_contradictions(program1)
        assert len(contradictions1) == 0

        # Inconsistent program
        program2 = "a. -a. :- a, -a."
        contradictions2 = validator.detect_contradictions(program2)
        assert len(contradictions2) > 0


class TestValidateASPProgram:
    """Tests for the comprehensive validate_asp_program function."""

    def test_validate_valid_program(self) -> None:
        """Test comprehensive validation of valid program."""
        program = "fact(a). rule(X) :- fact(X)."
        results = validate_asp_program(program)

        assert results["syntax_valid"]
        assert results["is_consistent"]
        assert results["overall_valid"]
        assert results["answer_set_count"] >= 1
        assert len(results["contradictions"]) == 0

    def test_validate_invalid_syntax(self) -> None:
        """Test validation catches syntax errors."""
        program = "invalid syntax here"
        results = validate_asp_program(program)

        assert not results["syntax_valid"]
        assert not results["overall_valid"]
        assert results["syntax_error"] is not None

    def test_validate_inconsistent_program(self) -> None:
        """Test validation catches inconsistencies."""
        program = "a. -a. :- a, -a."
        results = validate_asp_program(program)

        # Syntax is valid but semantics are not
        assert results["syntax_valid"]
        assert not results["is_consistent"]
        assert not results["overall_valid"]

    def test_validate_empty_program(self) -> None:
        """Test validation of empty program."""
        results = validate_asp_program("")

        assert results["syntax_valid"]
        assert results["is_consistent"]
        assert results["overall_valid"]


class TestASPValidatorIntegration:
    """Integration tests for ASP validators."""

    def test_negation_as_failure(self) -> None:
        """Test validators handle negation-as-failure correctly."""
        validator = ASPSemanticValidator()

        program = """
        a :- not b.
        b :- not a.
        """

        # This should have 2 answer sets: {a} and {b}
        count = validator.count_answer_sets(program)
        assert count == 2

        is_consistent, _ = validator.check_consistency(program)
        assert is_consistent

    def test_constraints(self) -> None:
        """Test validators handle constraints correctly."""
        validator = ASPSemanticValidator()

        # Program with constraint
        program = """
        {a; b}.
        :- a, b.
        """

        # Should have 3 answer sets: {}, {a}, {b} (not {a,b} due to constraint)
        count = validator.count_answer_sets(program)
        assert count == 3

    def test_complex_legal_rules(self) -> None:
        """Test validators on legal reasoning example."""
        program = """
        % Contract types
        contract(c1).
        land_sale(c1).

        % Within statute if land sale
        within_statute(C) :- land_sale(C).

        % Has writing
        has_writing(c1).

        % Satisfies statute of frauds
        satisfies_statute_of_frauds(C) :-
            within_statute(C),
            has_writing(C).

        % Enforceable if satisfies statute
        enforceable(C) :- satisfies_statute_of_frauds(C).
        """

        results = validate_asp_program(program)

        assert results["overall_valid"]
        assert results["syntax_valid"]
        assert results["is_consistent"]

        # Get answer sets and verify enforceable(c1) is derived
        validator = ASPSemanticValidator()
        answer_sets = validator.get_answer_sets(program)
        assert len(answer_sets) == 1

        # Check that enforceable(c1) is in the answer set
        symbols_str = [str(s) for s in answer_sets[0]]
        assert any("enforceable(c1)" in s for s in symbols_str)
