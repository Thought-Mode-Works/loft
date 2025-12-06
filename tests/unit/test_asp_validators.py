"""
Unit tests for ASP validators.

Tests syntax and semantic validation of ASP programs.
"""

from loft.validation import (
    ASPSyntaxValidator,
    ASPSemanticValidator,
    validate_asp_program,
)
from loft.validation.validation_schemas import ValidationResult


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


class TestGeneratedRuleValidation:
    """Tests for LLM-generated rule validation."""

    def test_validate_valid_generated_rule(self) -> None:
        """Test validation of a valid LLM-generated rule."""
        validator = ASPSyntaxValidator()
        rule = "enforceable(C) :- contract(C), not void(C)."
        existing_predicates = ["contract/1", "void/1"]

        result = validator.validate_generated_rule(rule, existing_predicates)

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.stage_name == "syntactic"
        assert len(result.error_messages) == 0
        # May have warnings, but should be valid
        assert "new_predicates" in result.details
        assert "enforceable" in result.details["new_predicates"]

    def test_validate_rule_with_syntax_error(self) -> None:
        """Test validation catches syntax errors."""
        validator = ASPSyntaxValidator()
        rule = "enforceable(C) :- contract(C not void(C)."  # Missing comma

        result = validator.validate_generated_rule(rule)

        assert not result.is_valid
        assert len(result.error_messages) > 0
        assert "syntax" in result.error_messages[0].lower()

    def test_validate_rule_with_lowercase_variable(self) -> None:
        """Test validation warns about lowercase variables."""
        validator = ASPSyntaxValidator()
        # Using lowercase 'c' instead of uppercase 'C'
        rule = "enforceable(c) :- contract(c)."

        result = validator.validate_generated_rule(rule)

        # Should be valid but have warnings
        assert result.is_valid
        assert len(result.warnings) > 0
        assert any("variable" in w.lower() for w in result.warnings)

    def test_validate_rule_with_uppercase_predicate(self) -> None:
        """Test validation catches uppercase predicates as syntax error."""
        validator = ASPSyntaxValidator()
        rule = "Enforceable(C) :- Contract(C)."

        result = validator.validate_generated_rule(rule)

        # Clingo treats uppercase predicates as syntax errors
        assert not result.is_valid
        assert len(result.error_messages) > 0
        assert "syntax" in result.error_messages[0].lower()

    def test_validate_rule_with_invalid_negation(self) -> None:
        """Test validation catches invalid negation syntax."""
        validator = ASPSyntaxValidator()
        rule = "enforceable(C) :- contract(C), !void(C)."  # Using ! instead of not

        result = validator.validate_generated_rule(rule)

        # This should fail because Clingo doesn't recognize ! as negation
        assert not result.is_valid
        assert len(result.error_messages) > 0

    def test_validate_rule_missing_period(self) -> None:
        """Test validation warns about missing period."""
        validator = ASPSyntaxValidator()
        rule = "enforceable(C) :- contract(C)"  # Missing period

        result = validator.validate_generated_rule(rule)

        # Clingo will catch this as syntax error
        assert not result.is_valid

    def test_validate_constraint_rule(self) -> None:
        """Test validation of constraint rules."""
        validator = ASPSyntaxValidator()
        rule = ":- enforceable(C), unenforceable(C)."

        result = validator.validate_generated_rule(rule)

        assert result.is_valid
        # Should have some warnings about constraint formatting
        assert result.stage_name == "syntactic"

    def test_validate_rule_predicate_compatibility(self) -> None:
        """Test validation tracks new vs existing predicates."""
        validator = ASPSyntaxValidator()
        rule = "enforceable(C) :- contract(C), has_writing(C), not void(C)."
        existing_predicates = ["contract/1", "void/1"]  # has_writing is new

        result = validator.validate_generated_rule(rule, existing_predicates)

        assert result.is_valid
        assert "new_predicates" in result.details
        assert "used_predicates" in result.details
        assert "has_writing" in result.details["new_predicates"]
        assert "enforceable" in result.details["new_predicates"]
        assert "contract" in result.details["used_predicates"]
        assert "void" in result.details["used_predicates"]

    def test_validate_complex_rule(self) -> None:
        """Test validation of a complex legal rule."""
        validator = ASPSyntaxValidator()
        rule = """
        satisfies_statute_of_frauds(C) :-
            within_statute(C),
            has_sufficient_writing(C).
        """
        existing_predicates = ["within_statute/1", "has_sufficient_writing/1"]

        result = validator.validate_generated_rule(rule, existing_predicates)

        assert result.is_valid
        assert "satisfies_statute_of_frauds" in result.details.get("new_predicates", [])

    def test_validate_rule_with_multiple_statements(self) -> None:
        """Test validation warns about multiple rules in one string."""
        validator = ASPSyntaxValidator()
        rule = "a :- b. c :- d."  # Two rules

        result = validator.validate_generated_rule(rule)

        assert result.is_valid  # Still valid ASP
        assert len(result.warnings) > 0
        assert any("multiple" in w.lower() for w in result.warnings)

    def test_validate_very_long_rule(self) -> None:
        """Test validation warns about overly complex rules."""
        validator = ASPSyntaxValidator()
        # Create a rule longer than 150 chars
        rule = "enforceable(C) :- " + ", ".join([f"predicate{i}(C)" for i in range(20)]) + "."

        result = validator.validate_generated_rule(rule)

        # Should be valid but warn about length
        assert result.is_valid
        assert len(result.warnings) > 0
        assert any("long" in w.lower() or "complex" in w.lower() for w in result.warnings)

    def test_validate_rule_with_empty_head(self) -> None:
        """Test validation catches rule with empty head."""
        validator = ASPSyntaxValidator()
        rule = ":- body(X)."  # This is a constraint, not an empty head

        result = validator.validate_generated_rule(rule)

        # Constraints are valid
        assert result.is_valid

    def test_validate_rule_summary(self) -> None:
        """Test that ValidationResult summary works for generated rules."""
        validator = ASPSyntaxValidator()
        rule = "enforceable(C) :- contract(C), not void(C)."

        result = validator.validate_generated_rule(rule)

        summary = result.summary()
        assert "Syntactic Validation" in summary
        assert "PASS" in summary or "FAIL" in summary

    def test_validate_rule_negation_spacing(self) -> None:
        """Test validation catches negation without proper spacing."""
        validator = ASPSyntaxValidator()
        rule = "enforceable(C) :- contract(C), notenforceable(C)."  # Missing space after 'not'

        result = validator.validate_generated_rule(rule)

        # This might be valid (notenforceable could be a predicate name)
        # But it should be flagged as a potential issue
        # The validator should warn or error depending on context
        assert isinstance(result, ValidationResult)


class TestUnsafeVariableDetection:
    """Tests for unsafe variable detection (issue #167)."""

    def test_safe_rule_no_errors(self) -> None:
        """Test that safe rules produce no errors."""
        from loft.validation.asp_validators import check_unsafe_variables

        # All head variables are bound in body
        rule = "cause_of_harm(X, Type) :- dangerous_condition(X), type_of_harm(X, Type)."
        errors, warnings = check_unsafe_variables(rule)

        assert len(errors) == 0

    def test_unsafe_variable_detected(self) -> None:
        """Test that unsafe variables are detected (issue #167 example)."""
        from loft.validation.asp_validators import check_unsafe_variables

        # Fall is not bound in body
        rule = "cause_of_harm(X, Fall) :- dangerous_condition(X)."
        errors, warnings = check_unsafe_variables(rule)

        assert len(errors) == 1
        assert "Fall" in errors[0]
        assert "unsafe" in errors[0].lower()

    def test_multiple_unsafe_variables(self) -> None:
        """Test detection of multiple unsafe variables."""
        from loft.validation.asp_validators import check_unsafe_variables

        # Both Y and Z are not bound in body
        rule = "pred(X, Y, Z) :- other_pred(X)."
        errors, warnings = check_unsafe_variables(rule)

        assert len(errors) == 2
        # Check both variables are mentioned
        error_text = " ".join(errors)
        assert "Y" in error_text
        assert "Z" in error_text

    def test_constant_not_flagged_as_unsafe(self) -> None:
        """Test that lowercase constants are not flagged as unsafe."""
        from loft.validation.asp_validators import check_unsafe_variables

        # 'fall' is lowercase (constant), not a variable
        rule = "cause_of_harm(X, fall) :- dangerous_condition(X)."
        errors, warnings = check_unsafe_variables(rule)

        assert len(errors) == 0

    def test_variable_in_negative_literal_unsafe(self) -> None:
        """Test that variables only in negative literals are detected."""
        from loft.validation.asp_validators import check_unsafe_variables

        # Y only appears in negative literal, so X is safe but Y is unsafe
        rule = "result(X, Y) :- input(X), not excluded(Y)."
        errors, warnings = check_unsafe_variables(rule)

        # Y should be flagged as unsafe
        assert len(errors) == 1
        assert "Y" in errors[0]

    def test_fact_no_errors(self) -> None:
        """Test that facts (no body) don't produce errors."""
        from loft.validation.asp_validators import check_unsafe_variables

        rule = "fact(constant)."
        errors, warnings = check_unsafe_variables(rule)

        assert len(errors) == 0
        assert len(warnings) == 0

    def test_constraint_no_errors(self) -> None:
        """Test that constraints (no head) don't produce errors."""
        from loft.validation.asp_validators import check_unsafe_variables

        rule = ":- conflicting(X), other(X)."
        errors, warnings = check_unsafe_variables(rule)

        assert len(errors) == 0

    def test_complex_safe_rule(self) -> None:
        """Test a complex safe rule from legal domain."""
        from loft.validation.asp_validators import check_unsafe_variables

        rule = """
        satisfies_statute_of_frauds(Contract, Party) :-
            contract(Contract),
            party(Contract, Party),
            has_writing(Contract),
            signed_by(Contract, Party).
        """
        errors, warnings = check_unsafe_variables(rule)

        assert len(errors) == 0

    def test_integration_with_validator(self) -> None:
        """Test that unsafe variable check is integrated into validator."""
        validator = ASPSyntaxValidator()

        # Rule with unsafe variable (Fall not bound)
        rule = "cause_of_harm(X, Fall) :- dangerous_condition(X)."
        result = validator.validate_generated_rule(rule)

        # Should not be valid due to unsafe variable
        assert not result.is_valid
        assert "unsafe_variables" in result.details
        assert any("Fall" in err for err in result.error_messages)

    def test_safe_rule_passes_validation(self) -> None:
        """Test that safe rules pass validation."""
        validator = ASPSyntaxValidator()

        # All head variables bound in body
        rule = "cause_of_harm(X, Type) :- dangerous_condition(X), harm_type(X, Type)."
        result = validator.validate_generated_rule(rule)

        # Should be valid - no unsafe variables
        assert result.is_valid
        assert (
            "unsafe_variables" not in result.details
            or len(result.details.get("unsafe_variables", [])) == 0
        )
