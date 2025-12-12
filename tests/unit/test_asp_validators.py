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
        rule = (
            "enforceable(C) :- "
            + ", ".join([f"predicate{i}(C)" for i in range(20)])
            + "."
        )

        result = validator.validate_generated_rule(rule)

        # Should be valid but warn about length
        assert result.is_valid
        assert len(result.warnings) > 0
        assert any(
            "long" in w.lower() or "complex" in w.lower() for w in result.warnings
        )

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
    """Tests for unsafe variable detection (issue #167).

    Uses pytest.mark.parametrize for cleaner, more maintainable tests.
    """

    # Test data for SAFE rules (no errors expected)
    SAFE_RULE_CASES = [
        pytest.param(
            "cause_of_harm(X, Type) :- dangerous_condition(X), type_of_harm(X, Type).",
            id="all_head_vars_bound",
        ),
        pytest.param(
            "cause_of_harm(X, fall) :- dangerous_condition(X).",
            id="lowercase_constant_not_variable",
        ),
        pytest.param(
            "fact(constant).",
            id="fact_no_body",
        ),
        pytest.param(
            ":- conflicting(X), other(X).",
            id="constraint_no_head",
        ),
        pytest.param(
            """satisfies_statute_of_frauds(Contract, Party) :-
            contract(Contract),
            party(Contract, Party),
            has_writing(Contract),
            signed_by(Contract, Party).""",
            id="complex_legal_rule",
        ),
        pytest.param(
            "successor(X, Y) :- number(X), Y = X + 1.",
            id="arithmetic_expression_binds_variable",
        ),
        pytest.param(
            "result(X) :- input(X), not excluded(X).",
            id="negated_atom_same_variable",
        ),
        pytest.param(
            "valid(X) :- candidate(X), not rejected(X), not excluded(X).",
            id="multiple_negated_literals",
        ),
    ]

    # Test data for UNSAFE rules (errors expected)
    UNSAFE_RULE_CASES = [
        pytest.param(
            "cause_of_harm(X, Fall) :- dangerous_condition(X).",
            ["Fall"],
            id="single_unsafe_variable",
        ),
        pytest.param(
            "pred(X, Y, Z) :- other_pred(X).",
            ["Y", "Z"],
            id="multiple_unsafe_variables",
        ),
        pytest.param(
            "result(X, Y) :- input(X), not excluded(Y).",
            ["Y"],
            id="variable_only_in_negative_literal",
        ),
    ]

    @pytest.mark.parametrize("rule", SAFE_RULE_CASES)
    def test_safe_rule_no_errors(self, rule: str) -> None:
        """Test that safe rules produce no unsafe variable errors."""
        from loft.validation.asp_validators import check_unsafe_variables

        errors, warnings = check_unsafe_variables(rule)

        assert len(errors) == 0, f"Unexpected errors for safe rule: {errors}"

    @pytest.mark.parametrize("rule,expected_vars", UNSAFE_RULE_CASES)
    def test_unsafe_variable_detected(
        self, rule: str, expected_vars: list[str]
    ) -> None:
        """Test that unsafe variables are detected."""
        from loft.validation.asp_validators import check_unsafe_variables

        errors, warnings = check_unsafe_variables(rule)

        assert len(errors) == len(
            expected_vars
        ), f"Expected {len(expected_vars)} errors, got {len(errors)}: {errors}"
        error_text = " ".join(errors)
        for var in expected_vars:
            assert var in error_text, f"Expected '{var}' in errors, got: {errors}"

    def test_unsafe_variable_error_message_format(self) -> None:
        """Test that unsafe variable errors include appropriate messaging."""
        from loft.validation.asp_validators import check_unsafe_variables

        rule = "cause_of_harm(X, Fall) :- dangerous_condition(X)."
        errors, warnings = check_unsafe_variables(rule)

        assert "unsafe" in errors[0].lower()

    def test_integration_with_validator(self) -> None:
        """Test that unsafe variable check is integrated into validator."""
        validator = ASPSyntaxValidator()

        rule = "cause_of_harm(X, Fall) :- dangerous_condition(X)."
        result = validator.validate_generated_rule(rule)

        assert not result.is_valid
        assert "unsafe_variables" in result.details
        assert any("Fall" in err for err in result.error_messages)

    def test_safe_rule_passes_validation(self) -> None:
        """Test that safe rules pass validation."""
        validator = ASPSyntaxValidator()

        rule = "cause_of_harm(X, Type) :- dangerous_condition(X), harm_type(X, Type)."
        result = validator.validate_generated_rule(rule)

        assert result.is_valid
        assert (
            "unsafe_variables" not in result.details
            or len(result.details.get("unsafe_variables", [])) == 0
        )

    # Test data for _extract_variables helper function
    EXTRACT_VARIABLES_CASES = [
        pytest.param("pred(X, Y, Z)", {"X", "Y", "Z"}, id="basic_extraction"),
        pytest.param("pred(a, b, c)", set(), id="no_variables"),
        pytest.param("pred(X, constant, Y)", {"X", "Y"}, id="mixed"),
        pytest.param(
            "pred(My_Var, Another_One)",
            {"My_Var", "Another_One"},
            id="underscore_names",
        ),
    ]

    @pytest.mark.parametrize("text,expected", EXTRACT_VARIABLES_CASES)
    def test_helper_extract_variables(self, text: str, expected: set[str]) -> None:
        """Test the _extract_variables helper function."""
        from loft.validation.asp_validators import _extract_variables

        variables = _extract_variables(text)
        assert variables == expected

    def test_documented_limitations_choice_rules(self) -> None:
        """Document behavior with choice rules (known limitation)."""
        from loft.validation.asp_validators import check_unsafe_variables

        rule = "{selected(X)} :- candidate(X)."
        errors, warnings = check_unsafe_variables(rule)

        # Document current behavior - may or may not detect correctly
        assert isinstance(errors, list)
        assert isinstance(warnings, list)

    def test_documented_limitations_aggregates(self) -> None:
        """Document behavior with aggregates (known limitation)."""
        from loft.validation.asp_validators import check_unsafe_variables

        rule = "has_items(Group) :- group(Group), #count{X : item(Group, X)} > 0."
        errors, warnings = check_unsafe_variables(rule)

        # Document current behavior
        assert isinstance(errors, list)
        assert isinstance(warnings, list)


class TestEmbeddedPeriodDetection:
    """Tests for embedded period detection (issue #168).

    Uses pytest.mark.parametrize for cleaner, more maintainable tests.
    """

    # Test data for embedded period detection that SHOULD produce errors
    EMBEDDED_PERIOD_ERROR_CASES = [
        pytest.param(
            "physical_harm(Spectator.FoulBall) :- at_game(Spectator).",
            "Spectator.FoulBall",
            id="oop_style_dot_notation",
        ),
        pytest.param(
            "result(X) :- input.parse(X).",
            "input.parse",
            id="method_style_notation",
        ),
        pytest.param(
            "output(X.value) :- input(X).",
            "X.value",
            id="variable_dot_lowercase",
        ),
        pytest.param(
            "liable(Defendant.Negligence) :- duty_owed(Defendant), breach(Defendant).",
            "Defendant.Negligence",
            id="legal_context_embedded_period",
        ),
        pytest.param(
            "result(X) :- math.sqrt(X, Y).",
            "math.sqrt",
            id="namespace_style_notation",
        ),
    ]

    # Test data for valid rules that should NOT produce errors
    VALID_RULE_CASES = [
        pytest.param(
            "injured_by(Spectator, FoulBall) :- at_baseball_game(Spectator), foul_ball_strike(FoulBall).",
            id="comma_separated_args",
        ),
        pytest.param("fact(a).", id="simple_fact"),
        pytest.param("threshold(3.14).", id="floating_point_number"),
        pytest.param("person(john).", id="fact_with_constant"),
        pytest.param(":- conflict(X), valid(X).", id="constraint_rule"),
        pytest.param("", id="empty_string"),
    ]

    @pytest.mark.parametrize("rule,expected_pattern", EMBEDDED_PERIOD_ERROR_CASES)
    def test_embedded_period_detected(self, rule: str, expected_pattern: str) -> None:
        """Test that embedded periods are detected in various patterns."""
        from loft.validation.asp_validators import check_embedded_periods

        errors, warnings = check_embedded_periods(rule)

        assert len(errors) > 0, f"Expected errors for pattern: {expected_pattern}"
        assert any(
            expected_pattern in err for err in errors
        ), f"Expected '{expected_pattern}' in errors, got: {errors}"

    @pytest.mark.parametrize("rule", VALID_RULE_CASES)
    def test_valid_rule_no_errors(self, rule: str) -> None:
        """Test that valid ASP rules produce no embedded period errors."""
        from loft.validation.asp_validators import check_embedded_periods

        errors, warnings = check_embedded_periods(rule)

        assert len(errors) == 0, f"Unexpected errors for rule '{rule}': {errors}"

    def test_oop_style_error_message_format(self) -> None:
        """Test that OOP-style errors include appropriate messaging."""
        from loft.validation.asp_validators import check_embedded_periods

        rule = "physical_harm(Spectator.FoulBall) :- at_game(Spectator)."
        errors, warnings = check_embedded_periods(rule)

        assert any("OOP" in err or "dot notation" in err.lower() for err in errors)

    def test_multiple_embedded_periods(self) -> None:
        """Test detection of multiple embedded periods."""
        from loft.validation.asp_validators import check_embedded_periods

        rule = "result(A.B) :- input(C.D), process(E.F)."
        errors, warnings = check_embedded_periods(rule)

        # Should detect all three
        assert len(errors) >= 3

    def test_integration_with_validator(self) -> None:
        """Test that embedded period check is integrated into validator.

        Note: Clingo may catch some embedded period issues as syntax errors
        before our check runs. The embedded period check provides additional
        value for pre-validation and better error messages.
        """
        validator = ASPSyntaxValidator()

        rule = "result(X.Y) :- input(X)."
        result = validator.validate_generated_rule(rule)

        # Should not be valid - either due to syntax error or embedded period
        assert not result.is_valid
        assert "syntax_error" in result.details or "embedded_periods" in result.details

    def test_valid_rule_passes_validation(self) -> None:
        """Test that valid rules pass embedded period validation."""
        validator = ASPSyntaxValidator()

        rule = "injured_by(Person, Object) :- accident(Person), involved(Object)."
        result = validator.validate_generated_rule(rule)

        assert result.is_valid
        assert (
            "embedded_periods" not in result.details
            or len(result.details.get("embedded_periods", [])) == 0
        )


class TestInputValidationEdgeCases:
    """Tests for input validation edge cases (feedback from multi-agent review).

    Uses pytest.mark.parametrize for cleaner, more maintainable tests.
    These tests verify that functions handle edge cases gracefully:
    - Empty strings
    - Whitespace-only strings
    - Type errors for non-string inputs
    """

    # Test data for empty/whitespace inputs that should return empty results
    EMPTY_OR_WHITESPACE_INPUTS = [
        pytest.param("", id="empty_string"),
        pytest.param("   \n\t  ", id="whitespace_only"),
    ]

    # Test data for invalid type inputs that should raise TypeError
    INVALID_TYPE_INPUTS = [
        pytest.param(123, id="integer"),
        pytest.param(None, id="none"),
        pytest.param(["rule"], id="list"),
    ]

    @pytest.mark.parametrize("input_str", EMPTY_OR_WHITESPACE_INPUTS)
    def test_check_embedded_periods_empty_or_whitespace(self, input_str: str) -> None:
        """Test check_embedded_periods with empty/whitespace input."""
        from loft.validation.asp_validators import check_embedded_periods

        errors, warnings = check_embedded_periods(input_str)
        assert errors == []
        assert warnings == []

    @pytest.mark.parametrize("input_str", EMPTY_OR_WHITESPACE_INPUTS)
    def test_check_unsafe_variables_empty_or_whitespace(self, input_str: str) -> None:
        """Test check_unsafe_variables with empty/whitespace input."""
        from loft.validation.asp_validators import check_unsafe_variables

        errors, warnings = check_unsafe_variables(input_str)
        assert errors == []
        assert warnings == []

    @pytest.mark.parametrize("input_str", EMPTY_OR_WHITESPACE_INPUTS)
    def test_extract_variables_empty_or_whitespace(self, input_str: str) -> None:
        """Test _extract_variables with empty/whitespace input."""
        from loft.validation.asp_validators import _extract_variables

        variables = _extract_variables(input_str)
        assert variables == set()

    @pytest.mark.parametrize("invalid_input", INVALID_TYPE_INPUTS)
    def test_check_embedded_periods_type_error(self, invalid_input) -> None:  # type: ignore[no-untyped-def]
        """Test check_embedded_periods raises TypeError for non-string."""
        from loft.validation.asp_validators import check_embedded_periods

        with pytest.raises(TypeError, match="Expected string"):
            check_embedded_periods(invalid_input)  # type: ignore

    @pytest.mark.parametrize("invalid_input", INVALID_TYPE_INPUTS)
    def test_check_unsafe_variables_type_error(self, invalid_input) -> None:  # type: ignore[no-untyped-def]
        """Test check_unsafe_variables raises TypeError for non-string."""
        from loft.validation.asp_validators import check_unsafe_variables

        with pytest.raises(TypeError, match="Expected string"):
            check_unsafe_variables(invalid_input)  # type: ignore


class TestContextAwareDetection:
    """Tests for context-aware detection (issue #177).

    These tests verify that ASP validators skip detection inside:
    - Quoted strings (e.g., "Hello.World")
    - ASP comments (e.g., % This is a comment with.period)
    """

    def test_strip_asp_context_removes_quoted_strings(self) -> None:
        """Test that quoted strings are removed from rule text."""
        from loft.validation.asp_validators import strip_asp_context

        # Quoted string with embedded period
        rule = 'predicate("string.with.dot").'
        stripped = strip_asp_context(rule)

        assert "string.with.dot" not in stripped
        assert "predicate()." == stripped

    def test_strip_asp_context_removes_comments(self) -> None:
        """Test that ASP comments are removed from rule text."""
        from loft.validation.asp_validators import strip_asp_context

        # Comment with embedded period
        rule = "fact(X). % This is a.comment with.period"
        stripped = strip_asp_context(rule)

        assert "a.comment" not in stripped
        assert "with.period" not in stripped
        assert "fact(X)." in stripped

    def test_strip_asp_context_multiline_comments(self) -> None:
        """Test that multiline comments are all removed."""
        from loft.validation.asp_validators import strip_asp_context

        rule = """% First.line.comment
predicate(X) :- other(X). % Inline.comment
% Another.line.comment
fact(a)."""
        stripped = strip_asp_context(rule)

        assert "First.line.comment" not in stripped
        assert "Inline.comment" not in stripped
        assert "Another.line.comment" not in stripped
        assert "predicate(X) :- other(X)." in stripped
        assert "fact(a)." in stripped

    def test_strip_asp_context_preserves_valid_content(self) -> None:
        """Test that valid ASP content is preserved."""
        from loft.validation.asp_validators import strip_asp_context

        rule = "enforceable(C) :- contract(C), not void(C)."
        stripped = strip_asp_context(rule)

        assert stripped == rule

    def test_embedded_periods_in_quoted_string_not_flagged(self) -> None:
        """Test that periods inside quoted strings are not flagged as errors."""
        from loft.validation.asp_validators import check_embedded_periods

        # Period inside quoted string should be ignored
        rule = 'label(X, "Hello.World") :- entity(X).'
        errors, warnings = check_embedded_periods(rule)

        # Should not flag "Hello.World" as embedded period
        assert len(errors) == 0
        assert all("Hello.World" not in w for w in warnings)

    def test_embedded_periods_in_comment_not_flagged(self) -> None:
        """Test that periods inside comments are not flagged as errors."""
        from loft.validation.asp_validators import check_embedded_periods

        # Period inside comment should be ignored
        rule = "fact(X) :- input(X). % This is a.comment with Var.Name"
        errors, warnings = check_embedded_periods(rule)

        # Should not flag "a.comment" or "Var.Name" as embedded period
        assert len(errors) == 0
        assert all("a.comment" not in w for w in warnings)
        assert all("Var.Name" not in w for w in warnings)

    def test_unsafe_variables_in_quoted_string_not_flagged(self) -> None:
        """Test that variables inside quoted strings are not flagged as unsafe."""
        from loft.validation.asp_validators import check_unsafe_variables

        # Variable pattern inside quoted string should be ignored
        rule = 'message(X, "Hello Variable Y") :- entity(X).'
        errors, warnings = check_unsafe_variables(rule)

        # Y inside the quoted string should not be flagged as unsafe
        # Only X is a real variable, and it appears in body
        assert len(errors) == 0

    def test_unsafe_variables_in_comment_not_flagged(self) -> None:
        """Test that variables inside comments are not flagged as unsafe."""
        from loft.validation.asp_validators import check_unsafe_variables

        # Variable pattern inside comment should be ignored
        rule = "result(X) :- input(X). % Note: Y is unused"
        errors, warnings = check_unsafe_variables(rule)

        # Y inside the comment should not be flagged as unsafe
        assert len(errors) == 0

    def test_real_embedded_period_still_detected(self) -> None:
        """Test that real embedded periods are still detected with context stripping."""
        from loft.validation.asp_validators import check_embedded_periods

        # Real embedded period in code should still be detected
        rule = "result(X.Y) :- input(X). % This is valid: A.B in comment"
        errors, warnings = check_embedded_periods(rule)

        # Should detect X.Y in code but not A.B in comment
        assert len(errors) > 0
        assert any("X.Y" in err for err in errors)
        assert all("A.B" not in err for err in errors)

    def test_real_unsafe_variable_still_detected(self) -> None:
        """Test that real unsafe variables are still detected with context stripping."""
        from loft.validation.asp_validators import check_unsafe_variables

        # Real unsafe variable should still be detected
        rule = "result(X, Y) :- input(X). % Y is intentionally unsafe"
        errors, warnings = check_unsafe_variables(rule)

        # Y in the head is truly unsafe and should be detected
        assert len(errors) == 1
        assert "Y" in errors[0]

    def test_mixed_quoted_and_real_periods(self) -> None:
        """Test rules with both quoted and real embedded periods."""
        from loft.validation.asp_validators import check_embedded_periods

        # Mix of quoted (should ignore) and real (should detect) periods
        rule = 'result(Obj.Method, "string.safe") :- input(X).'
        errors, warnings = check_embedded_periods(rule)

        # Should detect Obj.Method but not string.safe
        assert len(errors) > 0
        assert any("Obj.Method" in err for err in errors)
        assert all("string.safe" not in err for err in errors)

    def test_comment_at_start_of_line(self) -> None:
        """Test comment at start of line with embedded periods."""
        from loft.validation.asp_validators import check_embedded_periods

        rule = """% This rule has A.Period in comment
valid(X) :- input(X)."""
        errors, warnings = check_embedded_periods(rule)

        # Should not flag A.Period in comment
        assert len(errors) == 0
        assert all("A.Period" not in w for w in warnings)

    def test_empty_quoted_string(self) -> None:
        """Test that empty quoted strings don't break processing."""
        from loft.validation.asp_validators import strip_asp_context

        rule = 'label(X, "") :- entity(X).'
        stripped = strip_asp_context(rule)

        assert stripped == "label(X, ) :- entity(X)."

    def test_integration_with_validator_quoted_string(self) -> None:
        """Test that validator correctly handles quoted strings with periods."""
        validator = ASPSyntaxValidator()

        # Rule with quoted string containing period - should be valid
        rule = 'config(X, "default.value") :- setting(X).'
        result = validator.validate_generated_rule(rule)

        # Should not flag embedded_periods for the quoted string
        if "embedded_periods" in result.details:
            assert all(
                "default.value" not in err for err in result.details["embedded_periods"]
            )

    def test_integration_with_validator_comment_period(self) -> None:
        """Test that validator correctly handles comments with periods."""
        validator = ASPSyntaxValidator()

        # Rule with comment containing period pattern - should be valid
        rule = "valid(X) :- input(X). % Uses input.method pattern"
        result = validator.validate_generated_rule(rule)

        # Should be valid - comment period shouldn't cause issues
        # Note: result.is_valid depends on other validation rules too
        if "embedded_periods" in result.details:
            assert all(
                "input.method" not in err for err in result.details["embedded_periods"]
            )


class TestEscapedQuotesAndEdgeCases:
    """Tests for escaped quotes and edge cases (multi-agent review feedback).

    These tests verify that ASP validators correctly handle:
    - Escaped quotes within strings (e.g., "He said \\"hello\\"")
    - Multiple consecutive strings
    - Strings with backslash escapes (\\n, \\t, etc.)
    - Edge cases like empty strings, adjacent strings
    """

    def test_strip_asp_context_handles_escaped_quotes(self) -> None:
        """Test that escaped quotes within strings are handled correctly."""
        from loft.validation.asp_validators import strip_asp_context

        # String with escaped quotes: "He said \"hello\""
        rule = r'message(X, "He said \"hello\"") :- entity(X).'
        stripped = strip_asp_context(rule)

        # The entire quoted string should be removed, including escaped quotes
        assert r'"He said \"hello\""' not in stripped
        assert "message(X, ) :- entity(X)." == stripped

    def test_strip_asp_context_handles_backslash_escapes(self) -> None:
        """Test that backslash escapes within strings are handled correctly."""
        from loft.validation.asp_validators import strip_asp_context

        # String with newline and tab escapes: "line1\nline2\ttab"
        rule = r'log(X, "line1\nline2\ttab") :- event(X).'
        stripped = strip_asp_context(rule)

        # The entire quoted string should be removed
        assert "log(X, ) :- event(X)." == stripped

    def test_strip_asp_context_handles_backslash_at_end(self) -> None:
        """Test that backslash before closing quote is handled."""
        from loft.validation.asp_validators import strip_asp_context

        # String ending with escaped backslash: "path\\to\\file\\"
        rule = r'path(X, "C:\\Users\\file") :- windows(X).'
        stripped = strip_asp_context(rule)

        # The entire quoted string should be removed
        assert "path(X, ) :- windows(X)." == stripped

    def test_strip_asp_context_handles_multiple_strings(self) -> None:
        """Test multiple quoted strings in one rule."""
        from loft.validation.asp_validators import strip_asp_context

        rule = 'pair("first.string", "second.string") :- condition(X).'
        stripped = strip_asp_context(rule)

        # Both strings should be removed
        assert "first.string" not in stripped
        assert "second.string" not in stripped
        assert "pair(, ) :- condition(X)." == stripped

    def test_strip_asp_context_handles_adjacent_strings(self) -> None:
        """Test adjacent quoted strings without separator."""
        from loft.validation.asp_validators import strip_asp_context

        # Adjacent strings (unusual but valid ASP)
        rule = 'concat("hello""world") :- true.'
        stripped = strip_asp_context(rule)

        # Both strings should be removed
        assert "hello" not in stripped
        assert "world" not in stripped
        assert "concat() :- true." == stripped

    def test_embedded_periods_with_escaped_quotes(self) -> None:
        """Test embedded period detection ignores periods in escaped quote strings."""
        from loft.validation.asp_validators import check_embedded_periods

        # Embedded period inside escaped quote context
        rule = r'message(X, "Path is C:\\Users\\file.txt") :- entity(X).'
        errors, warnings = check_embedded_periods(rule)

        # Should not flag "file.txt" as embedded period
        assert len(errors) == 0
        assert all("file.txt" not in w for w in warnings)

    def test_unsafe_variables_with_escaped_quotes(self) -> None:
        """Test unsafe variable detection ignores variables in escaped quote strings."""
        from loft.validation.asp_validators import check_unsafe_variables

        # Variable pattern inside string with escaped quotes
        rule = r'message(X, "Variable \"Y\" is undefined") :- entity(X).'
        errors, warnings = check_unsafe_variables(rule)

        # Y inside the quoted string should not be flagged as unsafe
        assert len(errors) == 0

    def test_asp_patterns_class_exists(self) -> None:
        """Test that ASPPatterns class is properly defined and accessible."""
        from loft.validation.asp_validators import ASPPatterns

        # Verify all expected patterns exist
        assert hasattr(ASPPatterns, "VARIABLE")
        assert hasattr(ASPPatterns, "NEGATIVE_LITERAL")
        assert hasattr(ASPPatterns, "NEGATIVE_ARGS")
        assert hasattr(ASPPatterns, "OOP_STYLE")
        assert hasattr(ASPPatterns, "METHOD_STYLE")
        assert hasattr(ASPPatterns, "GENERAL_PERIOD")
        assert hasattr(ASPPatterns, "DIGIT_BEFORE")
        assert hasattr(ASPPatterns, "DIGIT_AFTER")
        assert hasattr(ASPPatterns, "QUOTED_STRING")
        assert hasattr(ASPPatterns, "ASP_COMMENT")

    def test_asp_patterns_quoted_string_matches_escaped(self) -> None:
        """Test ASPPatterns.QUOTED_STRING correctly matches escaped quotes."""
        from loft.validation.asp_validators import ASPPatterns

        # Simple string
        assert ASPPatterns.QUOTED_STRING.search('"hello"') is not None

        # String with escaped quote
        match = ASPPatterns.QUOTED_STRING.search(r'"He said \"hello\""')
        assert match is not None
        assert match.group() == r'"He said \"hello\""'

        # String with escaped backslash
        match = ASPPatterns.QUOTED_STRING.search(r'"path\\to\\file"')
        assert match is not None
        assert match.group() == r'"path\\to\\file"'

    def test_string_with_period_and_escaped_quote(self) -> None:
        """Test string containing both period and escaped quotes."""
        from loft.validation.asp_validators import check_embedded_periods

        # Complex string with period and escaped quotes
        rule = r'info(X, "Error: \"file.not.found\"") :- error(X).'
        errors, warnings = check_embedded_periods(rule)

        # None of the periods inside the string should be flagged
        assert len(errors) == 0
        assert all("file.not.found" not in w for w in warnings)

    def test_real_period_with_escaped_quote_context(self) -> None:
        """Test that real embedded periods are still detected alongside escaped strings."""
        from loft.validation.asp_validators import check_embedded_periods

        # Real embedded period plus escaped quote string
        rule = r'result(Obj.Method, "String with \"quotes\"") :- input(X).'
        errors, warnings = check_embedded_periods(rule)

        # Should detect Obj.Method but not anything in the quoted string
        assert len(errors) > 0
        assert any("Obj.Method" in err for err in errors)
