"""
Unit tests for semantic validator.

Tests semantic validation of LLM-generated ASP rules.
"""

import pytest
from loft.validation.semantic_validator import SemanticValidator
from loft.validation.validation_schemas import ValidationResult


class TestSemanticValidator:
    """Tests for SemanticValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a semantic validator for testing."""
        return SemanticValidator()

    def test_validate_consistent_rule(self, validator):
        """Test validation of a consistent rule."""
        rule = "enforceable(C) :- contract(C), not void(C)."
        existing_rules = "contract(c1). contract(c2). void(c2)."

        result = validator.validate_rule(rule, existing_rules)

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.stage_name == "semantic"
        assert len(result.error_messages) == 0

    def test_validate_inconsistent_rule(self, validator):
        """Test validation catches inconsistent rule."""
        rule = "enforceable(C). unenforceable(C)."
        existing_rules = ":- enforceable(C), unenforceable(C)."

        result = validator.validate_rule(rule, existing_rules)

        assert not result.is_valid
        assert len(result.error_messages) > 0
        assert "inconsistency" in result.error_messages[0].lower()

    def test_validate_self_contradictory_rule(self, validator):
        """Test validation catches self-contradictory rule."""
        rule = "a. -a. :- a, -a."

        result = validator.validate_rule(rule)

        assert not result.is_valid
        assert len(result.error_messages) > 0
        assert "self-contradictory" in result.error_messages[0].lower()

    def test_validate_rule_operational_layer(self, validator):
        """Test validation for operational layer compatibility."""
        # Ground fact - should be compatible with operational
        rule = "contract(c1)."

        result = validator.validate_rule(rule, target_layer="operational")

        assert result.is_valid
        assert result.details["stratification"]["is_compatible"]

    def test_validate_rule_operational_layer_incompatible(self, validator):
        """Test validation warns about operational layer incompatibility."""
        # Rule with variables - not suitable for operational
        rule = "enforceable(C) :- contract(C)."

        result = validator.validate_rule(rule, target_layer="operational")

        # Should be valid but have warning
        assert len(result.warnings) > 0
        assert not result.details["stratification"]["is_compatible"]

    def test_validate_rule_tactical_layer(self, validator):
        """Test validation for tactical layer."""
        rule = "enforceable(C) :- contract(C), not void(C)."

        result = validator.validate_rule(rule, target_layer="tactical")

        assert result.is_valid
        assert result.details["stratification"]["is_compatible"]

    def test_validate_rule_strategic_layer(self, validator):
        """Test validation for strategic layer."""
        rule = "policy_compliant(X) :- rule(X), not exception(X)."

        result = validator.validate_rule(rule, target_layer="strategic")

        assert result.is_valid
        assert result.details["stratification"]["is_compatible"]

    def test_validate_rule_with_aggregation(self, validator):
        """Test validation of rule with aggregation."""
        rule = "total(S) :- S = #sum{X : value(X)}."

        # Should warn if targeting tactical layer
        result = validator.validate_rule(rule, target_layer="tactical")

        assert len(result.warnings) > 0
        assert not result.details["stratification"]["is_compatible"]

    def test_validate_redundant_rule(self, validator):
        """Test detection of redundant rules."""
        existing_rules = "a :- b. b."
        # This rule is redundant - a is already derivable
        rule = "a."

        result = validator.validate_rule(rule, existing_rules)

        # Should be valid but warn about redundancy
        assert result.is_valid
        if result.details.get("redundancy", {}).get("is_redundant"):
            assert len(result.warnings) > 0

    def test_validate_non_redundant_rule(self, validator):
        """Test that non-redundant rules pass."""
        existing_rules = "a. b :- a."
        rule = "c :- a."  # Adds new derivation (c is new)

        result = validator.validate_rule(rule, existing_rules)

        assert result.is_valid
        redundancy = result.details.get("redundancy", {})
        assert not redundancy.get("is_redundant", False)

    def test_validate_constraint_rule(self, validator):
        """Test validation of constraint rules."""
        rule = ":- enforceable(C), unenforceable(C)."

        result = validator.validate_rule(rule)

        assert result.is_valid
        assert result.stage_name == "semantic"

    def test_validate_rule_predicate_coherence(self, validator):
        """Test predicate coherence checking."""
        # Mixed domain predicates
        rule = "result(X) :- contract(X), server(X)."

        result = validator.validate_rule(rule, context={"domain": "legal"})

        # Should have coherence details
        assert "coherence" in result.details
        # May or may not have issues depending on heuristics

    def test_validate_rule_no_circularity(self, validator):
        """Test that non-circular rules pass."""
        rule = "a :- b. b :- c."

        result = validator.validate_rule(rule)

        assert result.is_valid
        circularity = result.details.get("circularity", {})
        assert not circularity.get("has_cycle", False)

    def test_validate_batch(self, validator):
        """Test batch validation of multiple rules."""
        rules = [
            "enforceable(C) :- contract(C).",
            "valid(C) :- enforceable(C), not void(C).",
            "void(c1).",
        ]
        existing_rules = "contract(c1). contract(c2)."

        results = validator.validate_batch(
            rules, existing_rules=existing_rules, target_layer="tactical"
        )

        assert len(results) == 3
        assert all(isinstance(r, ValidationResult) for r in results)
        # Most should be valid
        assert sum(1 for r in results if r.is_valid) >= 2

    def test_validate_rule_with_negation(self, validator):
        """Test validation of rule with negation."""
        rule = "exception_applies(C) :- contract(C), not has_writing(C)."
        existing_rules = "contract(c1). has_writing(c2)."

        result = validator.validate_rule(rule, existing_rules)

        assert result.is_valid
        assert result.stage_name == "semantic"

    def test_validate_complex_legal_rule(self, validator):
        """Test validation of complex legal reasoning rule."""
        rule = """
        satisfies_statute_of_frauds(C) :-
            within_statute(C),
            has_sufficient_writing(C).
        """
        existing_rules = """
        contract(c1).
        within_statute(C) :- land_sale_contract(C).
        land_sale_contract(c1).
        """

        result = validator.validate_rule(rule, existing_rules, target_layer="tactical")

        assert result.is_valid
        assert result.details["consistency"]["is_consistent"]

    def test_validation_result_summary(self, validator):
        """Test that ValidationResult summary works for semantic validation."""
        rule = "enforceable(C) :- contract(C)."

        result = validator.validate_rule(rule)

        summary = result.summary()
        assert "Semantic Validation" in summary
        assert "PASS" in summary or "FAIL" in summary

    def test_validate_rule_empty_existing(self, validator):
        """Test validation with no existing rules."""
        rule = "fact(a)."

        result = validator.validate_rule(rule, existing_rules=None)

        assert result.is_valid
        assert "consistency" not in result.details  # No base to check against

    def test_validate_rule_with_context(self, validator):
        """Test validation with additional context."""
        rule = "enforceable(C) :- contract(C)."
        context = {"domain": "contract_law", "jurisdiction": "CA"}

        result = validator.validate_rule(rule, context=context)

        assert result.is_valid
        # Context should be used in coherence checking
        assert "coherence" in result.details
