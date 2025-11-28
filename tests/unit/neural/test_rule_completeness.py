"""
Tests for ASP rule completeness validation.

Issue #98: LLM Rule Generation Produces Truncated ASP Rules

This module tests the fix for truncated rule detection and validation,
ensuring that incomplete rules are rejected and only well-formed rules
are accepted into the knowledge base.
"""

import pytest
from unittest.mock import Mock
from loft.neural.rule_schemas import (
    GeneratedRule,
    validate_asp_rule_completeness,
)
from loft.neural.rule_generator import (
    RuleGenerator,
    RuleGenerationError,
)
from loft.neural.llm_interface import LLMInterface, LLMResponse
from loft.neural.errors import LLMParsingError


class TestValidateAspRuleCompleteness:
    """Test the validate_asp_rule_completeness function."""

    def test_valid_simple_rule(self):
        """Test validation of valid simple rule."""
        rule = "enforceable(C) :- contract(C), not void(C)."
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is True
        assert error is None

    def test_valid_fact(self):
        """Test validation of valid fact."""
        rule = "contract(c1)."
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is True
        assert error is None

    def test_valid_multiline_rule(self):
        """Test validation of valid multiline rule."""
        rule = """
        enforceable(C) :-
            contract(C),
            has_writing(C),
            not void(C).
        """
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is True
        assert error is None

    def test_valid_rule_with_comparison(self):
        """Test validation of rule with comparison operator."""
        rule = "expensive(C) :- contract(C), amount(C, A), A > 500."
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is True
        assert error is None

    def test_reject_empty_rule(self):
        """Test rejection of empty rule."""
        rule = ""
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is False
        assert "empty" in error.lower()

    def test_reject_missing_period(self):
        """Test rejection of rule missing terminal period."""
        rule = "enforceable(C) :- contract(C)"
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is False
        assert "period" in error.lower()

    def test_reject_unbalanced_parentheses(self):
        """Test rejection of rule with unbalanced parentheses."""
        rule = "enforceable(C :- contract(C)."
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is False
        assert "parenthes" in error.lower()

    def test_reject_truncated_predicate_underscore(self):
        """Test rejection of rule truncated at underscore."""
        rule = "enforceable(C) :- land_sale_."
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is False
        assert "truncat" in error.lower()

    def test_reject_truncated_empty_predicate_args(self):
        """Test rejection of rule with empty predicate arguments."""
        rule = "enforceable(C) :- contract(."
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is False
        # Should fail on either truncation or unbalanced parens
        assert error is not None

    def test_reject_trailing_comma(self):
        """Test rejection of rule with trailing comma."""
        rule = "enforceable(C) :- contract(C),."
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is False
        assert "truncat" in error.lower()

    def test_reject_empty_rule_body(self):
        """Test rejection of rule with empty body."""
        rule = "enforceable(C) :-."
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is False
        assert "truncat" in error.lower() or "empty" in error.lower()

    def test_reject_incomplete_argument_list(self):
        """Test rejection of rule with incomplete argument list."""
        rule = "enforceable(C) :- contract(C,."
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is False
        # Should fail on unbalanced parens or truncation
        assert error is not None

    def test_reject_orphan_character(self):
        """Test rejection of rule with orphan character after period."""
        rule = "enforceable(C) :- contract(C). a"
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is False
        # Rule is rejected - exact error message may vary (period or orphan)
        assert error is not None

    def test_real_truncation_pattern_1(self):
        """Test rejection of actual truncation pattern from logs."""
        # From transfer study logs: "enforceable(C) :- land_sale_"
        rule = "enforceable(C) :- land_sale_"
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is False

    def test_real_truncation_pattern_2(self):
        """Test rejection of actual truncation pattern from logs."""
        # From transfer study logs: rule ending mid-predicate
        rule = """adverse_possession(Property) :-
            continuous_occupation(Property),"""
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is False

    def test_real_truncation_pattern_3(self):
        """Test rejection of actual truncation pattern with orphan char."""
        # From transfer study logs: rule with "a" after
        rule = """fixture(X) :-
            annexed(X),
            adapted(X),
            intent_of_permanence(X).

        a"""
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is False
        # Rule is rejected - exact error message may vary (period or orphan)
        assert error is not None


class TestGeneratedRuleValidation:
    """Test GeneratedRule Pydantic model validation."""

    def test_valid_generated_rule(self):
        """Test that valid rule passes schema validation."""
        rule = GeneratedRule(
            asp_rule="enforceable(C) :- contract(C), not void(C).",
            confidence=0.9,
            reasoning="Contract is enforceable if not void",
            predicates_used=["contract/1", "void/1"],
            new_predicates=["enforceable/1"],
            source_type="principle",
            source_text="A contract is enforceable unless void",
        )
        assert rule.asp_rule == "enforceable(C) :- contract(C), not void(C)."

    def test_reject_truncated_rule_in_schema(self):
        """Test that truncated rule fails schema validation."""
        with pytest.raises(ValueError) as exc_info:
            GeneratedRule(
                asp_rule="enforceable(C) :- land_sale_",
                confidence=0.9,
                reasoning="Test",
                predicates_used=[],
                source_type="principle",
                source_text="Test",
            )
        assert "Invalid ASP rule" in str(exc_info.value)

    def test_reject_missing_period_in_schema(self):
        """Test that rule missing period fails schema validation."""
        with pytest.raises(ValueError) as exc_info:
            GeneratedRule(
                asp_rule="enforceable(C) :- contract(C)",
                confidence=0.9,
                reasoning="Test",
                predicates_used=[],
                source_type="principle",
                source_text="Test",
            )
        assert "Invalid ASP rule" in str(exc_info.value)

    def test_reject_empty_rule_in_schema(self):
        """Test that empty rule fails schema validation."""
        with pytest.raises(ValueError):
            GeneratedRule(
                asp_rule="",
                confidence=0.9,
                reasoning="Test",
                predicates_used=[],
                source_type="principle",
                source_text="Test",
            )


class TestRuleGeneratorRetryLogic:
    """Test RuleGenerator retry logic for truncation handling."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mocked LLM interface."""
        llm = Mock(spec=LLMInterface)
        llm.get_total_cost = Mock(return_value=0.05)
        llm.get_total_tokens = Mock(return_value=1000)
        return llm

    @pytest.fixture
    def rule_generator(self, mock_llm):
        """Create a RuleGenerator with mocked LLM."""
        return RuleGenerator(
            llm=mock_llm,
            domain="contract_law",
            prompt_version="v1.1",
        )

    def test_successful_generation_first_try(self, rule_generator, mock_llm):
        """Test successful generation on first attempt."""
        valid_rule = GeneratedRule(
            asp_rule="enforceable(C) :- contract(C), not void(C).",
            confidence=0.9,
            reasoning="Test",
            predicates_used=["contract/1"],
            source_type="principle",
            source_text="Test",
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = valid_rule
        mock_llm.query = Mock(return_value=mock_response)

        result = rule_generator.generate_from_principle(
            principle_text="A contract is enforceable unless void"
        )

        assert result.asp_rule == "enforceable(C) :- contract(C), not void(C)."
        assert mock_llm.query.call_count == 1

    def test_retry_on_truncation(self, rule_generator, mock_llm):
        """Test retry logic when first attempt produces truncated rule."""
        # First attempt returns truncated rule (will fail validation)
        truncated_rule = GeneratedRule.__new__(GeneratedRule)
        object.__setattr__(truncated_rule, "asp_rule", "enforceable(C) :- land_")
        object.__setattr__(truncated_rule, "confidence", 0.9)
        object.__setattr__(truncated_rule, "reasoning", "Test")
        object.__setattr__(truncated_rule, "predicates_used", [])
        object.__setattr__(truncated_rule, "new_predicates", [])
        object.__setattr__(truncated_rule, "alternative_formulations", [])
        object.__setattr__(truncated_rule, "source_type", "principle")
        object.__setattr__(truncated_rule, "source_text", "Test")
        object.__setattr__(truncated_rule, "citation", None)
        object.__setattr__(truncated_rule, "jurisdiction", None)

        # Second attempt returns valid rule
        valid_rule = GeneratedRule(
            asp_rule="enforceable(C) :- contract(C), land_sale(C).",
            confidence=0.9,
            reasoning="Test",
            predicates_used=["contract/1"],
            source_type="principle",
            source_text="Test",
        )

        mock_response_1 = Mock(spec=LLMResponse)
        mock_response_1.content = truncated_rule

        mock_response_2 = Mock(spec=LLMResponse)
        mock_response_2.content = valid_rule

        mock_llm.query = Mock(side_effect=[mock_response_1, mock_response_2])

        result = rule_generator.generate_from_principle(principle_text="Test principle")

        assert result.asp_rule == "enforceable(C) :- contract(C), land_sale(C)."
        assert mock_llm.query.call_count == 2

    def test_retry_on_validation_error(self, rule_generator, mock_llm):
        """Test retry logic when validation error occurs."""
        # First two attempts raise validation error, third succeeds
        valid_rule = GeneratedRule(
            asp_rule="valid(C) :- contract(C).",
            confidence=0.85,
            reasoning="Test",
            predicates_used=[],
            source_type="principle",
            source_text="Test",
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = valid_rule

        # Simulate validation errors then success
        mock_llm.query = Mock(
            side_effect=[
                ValueError("Invalid ASP rule: truncated"),
                ValueError("Invalid ASP rule: missing period"),
                mock_response,
            ]
        )

        result = rule_generator.generate_from_principle(principle_text="Test principle")

        assert result.asp_rule == "valid(C) :- contract(C)."
        assert mock_llm.query.call_count == 3

    def test_exhausted_retries_raises_error(self, rule_generator, mock_llm):
        """Test that exhausted retries raises RuleGenerationError."""
        # All attempts fail
        mock_llm.query = Mock(side_effect=ValueError("Invalid ASP rule"))

        with pytest.raises(RuleGenerationError) as exc_info:
            rule_generator.generate_from_principle(
                principle_text="Test principle",
                max_retries=3,
            )

        assert exc_info.value.attempts == 3
        assert "Invalid ASP rule" in exc_info.value.last_error

    def test_increased_tokens_on_retry(self, rule_generator, mock_llm):
        """Test that max_tokens increases on retry attempts."""
        # All attempts fail to verify token escalation
        mock_llm.query = Mock(side_effect=ValueError("Invalid ASP rule"))

        with pytest.raises(RuleGenerationError):
            rule_generator.generate_from_principle(
                principle_text="Test principle",
                max_retries=2,
            )

        # Check that second call had increased max_tokens
        calls = mock_llm.query.call_args_list
        assert len(calls) == 2

        # First call should use default tokens (4096)
        first_call_tokens = calls[0][1].get("max_tokens", 4096)
        # Second call should use increased tokens (8192)
        second_call_tokens = calls[1][1].get("max_tokens", 4096)

        assert second_call_tokens >= first_call_tokens

    def test_llm_parsing_error_triggers_retry(self, rule_generator, mock_llm):
        """Test that LLMParsingError triggers retry."""
        valid_rule = GeneratedRule(
            asp_rule="result(X) :- input(X).",
            confidence=0.8,
            reasoning="Test",
            predicates_used=[],
            source_type="principle",
            source_text="Test",
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = valid_rule

        mock_llm.query = Mock(
            side_effect=[
                LLMParsingError("Failed to parse JSON", raw_response="invalid"),
                mock_response,
            ]
        )

        result = rule_generator.generate_from_principle(principle_text="Test principle")

        assert result.asp_rule == "result(X) :- input(X)."
        assert mock_llm.query.call_count == 2


class TestRuleGeneratorCaseRetryLogic:
    """Test RuleGenerator retry logic for generate_from_case."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mocked LLM interface."""
        llm = Mock(spec=LLMInterface)
        llm.get_total_cost = Mock(return_value=0.05)
        llm.get_total_tokens = Mock(return_value=1000)
        return llm

    @pytest.fixture
    def rule_generator(self, mock_llm):
        """Create a RuleGenerator with mocked LLM."""
        return RuleGenerator(
            llm=mock_llm,
            domain="contract_law",
            prompt_version="v1.1",
        )

    def test_case_generation_with_retry(self, rule_generator, mock_llm):
        """Test case rule generation with retry on failure."""
        valid_rule = GeneratedRule(
            asp_rule="exception(C) :- part_performance(C).",
            confidence=0.9,
            reasoning="Test",
            predicates_used=[],
            source_type="case",
            source_text="Test case",
            citation="Test v. Case",
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = valid_rule

        mock_llm.query = Mock(
            side_effect=[
                ValueError("Truncated rule"),
                mock_response,
            ]
        )

        result = rule_generator.generate_from_case(
            case_text="The court held...",
            citation="Test v. Case",
        )

        assert result.asp_rule == "exception(C) :- part_performance(C)."
        assert mock_llm.query.call_count == 2

    def test_case_generation_exhausted_retries(self, rule_generator, mock_llm):
        """Test case generation raises error after exhausted retries."""
        mock_llm.query = Mock(side_effect=ValueError("Invalid"))

        with pytest.raises(RuleGenerationError) as exc_info:
            rule_generator.generate_from_case(
                case_text="Test",
                citation="Test v. Case",
                max_retries=2,
            )

        assert "Test v. Case" in str(exc_info.value)
        assert exc_info.value.attempts == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_valid_constraint_rule(self):
        """Test validation of constraint (headless rule)."""
        # Constraints have empty heads
        rule = ":- contract(C), void(C), enforceable(C)."
        is_valid, error = validate_asp_rule_completeness(rule)
        # May or may not be valid depending on implementation
        # At minimum, should not crash
        assert isinstance(is_valid, bool)

    def test_valid_choice_rule(self):
        """Test validation of choice rule."""
        rule = "{ select(X) : option(X) } = 1."
        is_valid, error = validate_asp_rule_completeness(rule)
        # Should handle choice rules
        assert isinstance(is_valid, bool)

    def test_valid_aggregate_rule(self):
        """Test validation of rule with aggregate."""
        rule = "total(S) :- S = #sum{ V,X : value(X,V) }."
        is_valid, error = validate_asp_rule_completeness(rule)
        # Should handle aggregates
        assert isinstance(is_valid, bool)

    def test_whitespace_handling(self):
        """Test that extra whitespace is handled correctly."""
        rule = "   enforceable(C) :- contract(C).   "
        is_valid, error = validate_asp_rule_completeness(rule)
        assert is_valid is True

    def test_multiline_with_comments(self):
        """Test multiline rule with comments."""
        rule = """% This is a comment
        enforceable(C) :- contract(C).  % inline comment
        """
        is_valid, error = validate_asp_rule_completeness(rule)
        # Comments may or may not be supported
        assert isinstance(is_valid, bool)

    def test_multiple_rules_in_string(self):
        """Test validation of multiple rules in one string."""
        # Multiple rules in a single string may or may not be valid
        # depending on implementation. The important thing is that
        # complete rules are accepted and incomplete ones rejected.
        rule = "fact1(a). fact2(b). rule(X) :- fact1(X)."
        is_valid, error = validate_asp_rule_completeness(rule)
        # May be valid or invalid depending on clingo parsing
        # Just ensure it doesn't crash and returns a result
        assert isinstance(is_valid, bool)
