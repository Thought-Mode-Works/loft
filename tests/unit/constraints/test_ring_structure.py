"""
Unit tests for ring structure implementation.

Tests the ring algebraic structure for compositional rule combination,
verifying ring axioms and homomorphism properties.
"""

import pytest
from typing import Dict, Any

from loft.constraints.ring_structure import (
    BooleanRule,
    ConfidenceRule,
    RuleComposition,
    RingHomomorphism,
    RingPropertyVerifier,
    ComposedRule,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def boolean_test_cases() -> list:
    """Boolean test cases with all combinations."""
    return [
        {"a": True, "b": True, "c": True},
        {"a": True, "b": True, "c": False},
        {"a": True, "b": False, "c": True},
        {"a": True, "b": False, "c": False},
        {"a": False, "b": True, "c": True},
        {"a": False, "b": True, "c": False},
        {"a": False, "b": False, "c": True},
        {"a": False, "b": False, "c": False},
    ]


@pytest.fixture
def confidence_test_cases() -> list:
    """Confidence test cases with varying values."""
    return [
        {"conf_a": 0.8, "conf_b": 0.6, "conf_c": 0.9},
        {"conf_a": 0.0, "conf_b": 0.5, "conf_c": 0.5},
        {"conf_a": 1.0, "conf_b": 1.0, "conf_c": 1.0},
        {"conf_a": 0.5, "conf_b": 0.5, "conf_c": 0.5},
        {"conf_a": 0.3, "conf_b": 0.7, "conf_c": 0.2},
    ]


@pytest.fixture
def rule_a() -> BooleanRule:
    """Boolean rule checking 'a' field."""
    return BooleanRule(lambda c: c["a"], "a")


@pytest.fixture
def rule_b() -> BooleanRule:
    """Boolean rule checking 'b' field."""
    return BooleanRule(lambda c: c["b"], "b")


@pytest.fixture
def rule_c() -> BooleanRule:
    """Boolean rule checking 'c' field."""
    return BooleanRule(lambda c: c["c"], "c")


@pytest.fixture
def conf_rule_a() -> ConfidenceRule:
    """Confidence rule reading 'conf_a' field."""
    return ConfidenceRule(lambda c: c["conf_a"], "conf_a")


@pytest.fixture
def conf_rule_b() -> ConfidenceRule:
    """Confidence rule reading 'conf_b' field."""
    return ConfidenceRule(lambda c: c["conf_b"], "conf_b")


@pytest.fixture
def conf_rule_c() -> ConfidenceRule:
    """Confidence rule reading 'conf_c' field."""
    return ConfidenceRule(lambda c: c["conf_c"], "conf_c")


# ============================================================================
# BooleanRule Tests
# ============================================================================


class TestBooleanRule:
    """Tests for BooleanRule class."""

    def test_evaluate_true(self, rule_a: BooleanRule) -> None:
        """Test evaluation returns True when predicate matches."""
        assert rule_a.evaluate({"a": True}) is True

    def test_evaluate_false(self, rule_a: BooleanRule) -> None:
        """Test evaluation returns False when predicate doesn't match."""
        assert rule_a.evaluate({"a": False}) is False

    def test_conjunction_truth_table(
        self, rule_a: BooleanRule, rule_b: BooleanRule
    ) -> None:
        """Test AND operation follows boolean conjunction truth table."""
        combined = rule_a * rule_b

        # T AND T = T
        assert combined.evaluate({"a": True, "b": True}) is True
        # T AND F = F
        assert combined.evaluate({"a": True, "b": False}) is False
        # F AND T = F
        assert combined.evaluate({"a": False, "b": True}) is False
        # F AND F = F
        assert combined.evaluate({"a": False, "b": False}) is False

    def test_disjunction_truth_table(
        self, rule_a: BooleanRule, rule_b: BooleanRule
    ) -> None:
        """Test OR operation follows boolean disjunction truth table."""
        combined = rule_a + rule_b

        # T OR T = T
        assert combined.evaluate({"a": True, "b": True}) is True
        # T OR F = T
        assert combined.evaluate({"a": True, "b": False}) is True
        # F OR T = T
        assert combined.evaluate({"a": False, "b": True}) is True
        # F OR F = F
        assert combined.evaluate({"a": False, "b": False}) is False

    def test_negation(self, rule_a: BooleanRule) -> None:
        """Test NOT operation."""
        negated = -rule_a

        assert negated.evaluate({"a": True}) is False
        assert negated.evaluate({"a": False}) is True

    def test_double_negation(self, rule_a: BooleanRule) -> None:
        """Test double negation returns original value."""
        double_neg = -(-rule_a)

        assert double_neg.evaluate({"a": True}) is True
        assert double_neg.evaluate({"a": False}) is False

    def test_zero_element(self) -> None:
        """Test zero element is always false."""
        zero = BooleanRule.zero()

        assert zero.is_zero() is True
        assert zero.is_one() is False
        assert zero.evaluate({"a": True}) is False
        assert zero.evaluate({"a": False}) is False

    def test_one_element(self) -> None:
        """Test one element is always true."""
        one = BooleanRule.one()

        assert one.is_one() is True
        assert one.is_zero() is False
        assert one.evaluate({"a": True}) is True
        assert one.evaluate({"a": False}) is True

    def test_identity_with_zero(self, rule_a: BooleanRule) -> None:
        """Test a + 0 = a (additive identity)."""
        zero = BooleanRule.zero()
        result = rule_a + zero

        assert result.evaluate({"a": True}) == rule_a.evaluate({"a": True})
        assert result.evaluate({"a": False}) == rule_a.evaluate({"a": False})

    def test_identity_with_one(self, rule_a: BooleanRule) -> None:
        """Test a * 1 = a (multiplicative identity)."""
        one = BooleanRule.one()
        result = rule_a * one

        assert result.evaluate({"a": True}) == rule_a.evaluate({"a": True})
        assert result.evaluate({"a": False}) == rule_a.evaluate({"a": False})

    def test_annihilation_with_zero(self, rule_a: BooleanRule) -> None:
        """Test a * 0 = 0."""
        zero = BooleanRule.zero()
        result = rule_a * zero

        assert result.is_zero() is True
        assert result.evaluate({"a": True}) is False

    def test_absorption_with_one(self, rule_a: BooleanRule) -> None:
        """Test a + 1 = 1."""
        one = BooleanRule.one()
        result = rule_a + one

        assert result.is_one() is True
        assert result.evaluate({"a": False}) is True

    def test_rule_name_composition(
        self, rule_a: BooleanRule, rule_b: BooleanRule
    ) -> None:
        """Test rule names are composed correctly."""
        conj = rule_a * rule_b
        disj = rule_a + rule_b
        neg = -rule_a

        assert "∧" in conj.name
        assert "∨" in disj.name
        assert "¬" in neg.name

    def test_negation_of_zero(self) -> None:
        """Test negation of zero is one."""
        zero = BooleanRule.zero()
        neg_zero = -zero

        assert neg_zero.is_one() is True

    def test_negation_of_one(self) -> None:
        """Test negation of one is zero."""
        one = BooleanRule.one()
        neg_one = -one

        assert neg_one.is_zero() is True


# ============================================================================
# ConfidenceRule Tests
# ============================================================================


class TestConfidenceRule:
    """Tests for ConfidenceRule class."""

    def test_evaluate(self, conf_rule_a: ConfidenceRule) -> None:
        """Test basic evaluation."""
        assert conf_rule_a.evaluate({"conf_a": 0.8}) == 0.8

    def test_addition_is_max(
        self, conf_rule_a: ConfidenceRule, conf_rule_b: ConfidenceRule
    ) -> None:
        """Test addition returns max of confidences."""
        combined = conf_rule_a + conf_rule_b
        case = {"conf_a": 0.8, "conf_b": 0.6}

        assert combined.evaluate(case) == 0.8  # max(0.8, 0.6)

    def test_multiplication_is_product(
        self, conf_rule_a: ConfidenceRule, conf_rule_b: ConfidenceRule
    ) -> None:
        """Test multiplication returns product of confidences."""
        combined = conf_rule_a * conf_rule_b
        case = {"conf_a": 0.8, "conf_b": 0.6}

        assert abs(combined.evaluate(case) - 0.48) < 0.01  # 0.8 * 0.6

    def test_zero_element(self) -> None:
        """Test zero element."""
        zero = ConfidenceRule.zero()

        assert zero.is_zero() is True
        assert zero.evaluate({"conf_a": 0.5}) == 0.0

    def test_one_element(self) -> None:
        """Test one element."""
        one = ConfidenceRule.one()

        assert one.is_one() is True
        assert one.evaluate({"conf_a": 0.5}) == 1.0

    def test_identity_with_zero(self, conf_rule_a: ConfidenceRule) -> None:
        """Test a + 0 = a."""
        zero = ConfidenceRule.zero()
        result = conf_rule_a + zero

        assert result.evaluate({"conf_a": 0.8}) == 0.8

    def test_identity_with_one(self, conf_rule_a: ConfidenceRule) -> None:
        """Test a * 1 = a."""
        one = ConfidenceRule.one()
        result = conf_rule_a * one

        assert result.evaluate({"conf_a": 0.8}) == 0.8

    def test_annihilation_with_zero(self, conf_rule_a: ConfidenceRule) -> None:
        """Test a * 0 = 0."""
        zero = ConfidenceRule.zero()
        result = conf_rule_a * zero

        assert result.is_zero() is True

    def test_chained_multiplication(
        self,
        conf_rule_a: ConfidenceRule,
        conf_rule_b: ConfidenceRule,
        conf_rule_c: ConfidenceRule,
    ) -> None:
        """Test chained multiplication."""
        combined = conf_rule_a * conf_rule_b * conf_rule_c
        case = {"conf_a": 0.5, "conf_b": 0.5, "conf_c": 0.5}

        # 0.5 * 0.5 * 0.5 = 0.125
        assert abs(combined.evaluate(case) - 0.125) < 0.01

    def test_chained_addition(
        self,
        conf_rule_a: ConfidenceRule,
        conf_rule_b: ConfidenceRule,
        conf_rule_c: ConfidenceRule,
    ) -> None:
        """Test chained addition (max)."""
        combined = conf_rule_a + conf_rule_b + conf_rule_c
        case = {"conf_a": 0.3, "conf_b": 0.7, "conf_c": 0.5}

        # max(max(0.3, 0.7), 0.5) = 0.7
        assert combined.evaluate(case) == 0.7


# ============================================================================
# RuleComposition Tests
# ============================================================================


class TestRuleComposition:
    """Tests for RuleComposition utility class."""

    def test_exception_blocks_when_true(self) -> None:
        """Test exception blocks base rule when exception applies."""
        base = BooleanRule(lambda c: c["contract_valid"], "valid")
        exception = BooleanRule(lambda c: c["is_minor"], "minor")

        with_exception = RuleComposition.exception(base, exception)

        # Valid contract, is minor: exception blocks
        assert (
            with_exception.evaluate({"contract_valid": True, "is_minor": True}) is False
        )

    def test_exception_allows_when_false(self) -> None:
        """Test base rule applies when exception doesn't apply."""
        base = BooleanRule(lambda c: c["contract_valid"], "valid")
        exception = BooleanRule(lambda c: c["is_minor"], "minor")

        with_exception = RuleComposition.exception(base, exception)

        # Valid contract, not minor: applies
        assert (
            with_exception.evaluate({"contract_valid": True, "is_minor": False}) is True
        )

    def test_exception_invalid_base(self) -> None:
        """Test invalid base rule doesn't apply even without exception."""
        base = BooleanRule(lambda c: c["contract_valid"], "valid")
        exception = BooleanRule(lambda c: c["is_minor"], "minor")

        with_exception = RuleComposition.exception(base, exception)

        # Invalid contract: base rule fails
        assert (
            with_exception.evaluate({"contract_valid": False, "is_minor": False})
            is False
        )

    def test_conditional_true_antecedent(self) -> None:
        """Test conditional when condition is true."""
        condition = BooleanRule(lambda c: c["offer_made"], "offer")
        consequent = BooleanRule(lambda c: c["acceptance_given"], "accept")

        conditional = RuleComposition.conditional(condition, consequent)

        # Offer made, acceptance given: true
        assert (
            conditional.evaluate({"offer_made": True, "acceptance_given": True}) is True
        )
        # Offer made, no acceptance: false (condition true, consequent false)
        assert (
            conditional.evaluate({"offer_made": True, "acceptance_given": False})
            is False
        )

    def test_conditional_false_antecedent(self) -> None:
        """Test conditional when condition is false (vacuously true)."""
        condition = BooleanRule(lambda c: c["offer_made"], "offer")
        consequent = BooleanRule(lambda c: c["acceptance_given"], "accept")

        conditional = RuleComposition.conditional(condition, consequent)

        # No offer: vacuously true (material implication)
        assert (
            conditional.evaluate({"offer_made": False, "acceptance_given": False})
            is True
        )
        assert (
            conditional.evaluate({"offer_made": False, "acceptance_given": True})
            is True
        )

    def test_sequence(self) -> None:
        """Test sequence composition (conjunction)."""
        first = BooleanRule(lambda c: c["step1"], "step1")
        second = BooleanRule(lambda c: c["step2"], "step2")

        seq = RuleComposition.sequence(first, second)

        assert seq.evaluate({"step1": True, "step2": True}) is True
        assert seq.evaluate({"step1": True, "step2": False}) is False
        assert seq.evaluate({"step1": False, "step2": True}) is False

    def test_alternative(self) -> None:
        """Test alternative composition (disjunction)."""
        rule1 = BooleanRule(lambda c: c["option1"], "opt1")
        rule2 = BooleanRule(lambda c: c["option2"], "opt2")

        alt = RuleComposition.alternative(rule1, rule2)

        assert alt.evaluate({"option1": True, "option2": False}) is True
        assert alt.evaluate({"option1": False, "option2": True}) is True
        assert alt.evaluate({"option1": False, "option2": False}) is False


# ============================================================================
# RingPropertyVerifier Tests
# ============================================================================


class TestRingPropertyVerifier:
    """Tests for RingPropertyVerifier class."""

    def test_verify_distributivity_boolean(
        self,
        rule_a: BooleanRule,
        rule_b: BooleanRule,
        rule_c: BooleanRule,
        boolean_test_cases: list,
    ) -> None:
        """Test distributivity for boolean rules."""
        verifier = RingPropertyVerifier(boolean_test_cases)

        # a * (b + c) = (a * b) + (a * c)
        assert verifier.verify_distributivity(rule_a, rule_b, rule_c) is True

    def test_verify_associativity_add_boolean(
        self,
        rule_a: BooleanRule,
        rule_b: BooleanRule,
        rule_c: BooleanRule,
        boolean_test_cases: list,
    ) -> None:
        """Test associativity of addition for boolean rules."""
        verifier = RingPropertyVerifier(boolean_test_cases)

        # (a + b) + c = a + (b + c)
        assert verifier.verify_associativity_add(rule_a, rule_b, rule_c) is True

    def test_verify_associativity_mul_boolean(
        self,
        rule_a: BooleanRule,
        rule_b: BooleanRule,
        rule_c: BooleanRule,
        boolean_test_cases: list,
    ) -> None:
        """Test associativity of multiplication for boolean rules."""
        verifier = RingPropertyVerifier(boolean_test_cases)

        # (a * b) * c = a * (b * c)
        assert verifier.verify_associativity_mul(rule_a, rule_b, rule_c) is True

    def test_verify_commutativity_add_boolean(
        self, rule_a: BooleanRule, rule_b: BooleanRule, boolean_test_cases: list
    ) -> None:
        """Test commutativity of addition for boolean rules."""
        verifier = RingPropertyVerifier(boolean_test_cases)

        # a + b = b + a
        assert verifier.verify_commutativity_add(rule_a, rule_b) is True

    def test_verify_identity_boolean(
        self, rule_a: BooleanRule, boolean_test_cases: list
    ) -> None:
        """Test identity elements for boolean rules."""
        verifier = RingPropertyVerifier(boolean_test_cases)

        assert verifier.verify_identity(rule_a, BooleanRule) is True

    def test_verify_annihilation_boolean(
        self, rule_a: BooleanRule, boolean_test_cases: list
    ) -> None:
        """Test annihilation property a * 0 = 0."""
        verifier = RingPropertyVerifier(boolean_test_cases)

        assert verifier.verify_annihilation(rule_a, BooleanRule) is True

    def test_full_verification_boolean(
        self,
        rule_a: BooleanRule,
        rule_b: BooleanRule,
        rule_c: BooleanRule,
        boolean_test_cases: list,
    ) -> None:
        """Test full ring verification for boolean rules."""
        verifier = RingPropertyVerifier(boolean_test_cases)
        elements = [rule_a, rule_b, rule_c]

        report = verifier.full_verification(elements, BooleanRule)

        assert report.is_ring is True
        assert report.distributivity_holds is True
        assert report.associativity_add_holds is True
        assert report.associativity_mul_holds is True
        assert report.identity_holds is True
        assert report.commutativity_add_holds is True
        assert report.tests_passed == report.tests_run

    def test_full_verification_confidence(
        self,
        conf_rule_a: ConfidenceRule,
        conf_rule_b: ConfidenceRule,
        conf_rule_c: ConfidenceRule,
        confidence_test_cases: list,
    ) -> None:
        """Test full ring verification for confidence rules."""
        verifier = RingPropertyVerifier(confidence_test_cases)
        elements = [conf_rule_a, conf_rule_b, conf_rule_c]

        report = verifier.full_verification(elements, ConfidenceRule)

        # Note: Confidence semiring may not satisfy all ring axioms
        # but should satisfy associativity and identity
        assert report.associativity_add_holds is True
        assert report.associativity_mul_holds is True
        assert report.identity_holds is True

    def test_verification_report_structure(
        self,
        rule_a: BooleanRule,
        rule_b: BooleanRule,
        boolean_test_cases: list,
    ) -> None:
        """Test verification report contains expected fields."""
        verifier = RingPropertyVerifier(boolean_test_cases)

        report = verifier.full_verification([rule_a, rule_b], BooleanRule)

        assert hasattr(report, "distributivity_holds")
        assert hasattr(report, "associativity_add_holds")
        assert hasattr(report, "associativity_mul_holds")
        assert hasattr(report, "identity_holds")
        assert hasattr(report, "commutativity_add_holds")
        assert hasattr(report, "tests_run")
        assert hasattr(report, "tests_passed")
        assert hasattr(report, "violations")
        assert report.tests_run > 0


# ============================================================================
# RingHomomorphism Tests
# ============================================================================


class TestRingHomomorphism:
    """Tests for RingHomomorphism class."""

    def test_boolean_to_confidence_homomorphism(self, boolean_test_cases: list) -> None:
        """Verify boolean to confidence homomorphism."""

        def bool_to_conf(rule: BooleanRule) -> ConfidenceRule:
            rule_pred = rule.predicate
            return ConfidenceRule(
                evaluator=lambda c, r=rule_pred: 1.0 if r(c) else 0.0,
                name=f"conf({rule.name})",
            )

        hom = RingHomomorphism(BooleanRule, ConfidenceRule, bool_to_conf)

        rules = [
            BooleanRule(lambda c: c["a"], "a"),
            BooleanRule(lambda c: c["b"], "b"),
        ]

        report = hom.verify_homomorphism(rules, boolean_test_cases)

        assert report.is_homomorphism is True
        assert len(report.violations) == 0

    def test_apply_homomorphism(self) -> None:
        """Test applying homomorphism to a rule."""

        def double_conf(rule: ConfidenceRule) -> ConfidenceRule:
            rule_eval = rule.evaluator
            return ConfidenceRule(
                evaluator=lambda c, r=rule_eval: min(1.0, r(c) * 2),
                name=f"double({rule.name})",
            )

        hom = RingHomomorphism(ConfidenceRule, ConfidenceRule, double_conf)

        rule = ConfidenceRule(lambda c: c["conf"], "conf")
        transformed = hom.apply(rule)

        assert transformed.evaluate({"conf": 0.3}) == 0.6
        assert transformed.evaluate({"conf": 0.7}) == 1.0  # Capped at 1.0

    def test_non_homomorphism_detection(self) -> None:
        """Test detection of non-homomorphism."""
        # Add a constant: φ(a) = a + 0.1
        # φ(a + b) = max(a, b) + 0.1
        # φ(a) + φ(b) = max(a + 0.1, b + 0.1) = max(a, b) + 0.1
        # This preserves addition!
        # φ(a * b) = a*b + 0.1
        # φ(a) * φ(b) = (a + 0.1) * (b + 0.1) = ab + 0.1a + 0.1b + 0.01
        # These differ! Not a homomorphism.

        def add_constant(rule: ConfidenceRule) -> ConfidenceRule:
            rule_eval = rule.evaluator
            return ConfidenceRule(
                evaluator=lambda c, r=rule_eval: min(1.0, r(c) + 0.1),
                name=f"plus({rule.name})",
            )

        hom = RingHomomorphism(ConfidenceRule, ConfidenceRule, add_constant)

        rules = [
            ConfidenceRule(lambda c: 0.5, "half"),
            ConfidenceRule(lambda c: 0.4, "forty"),
        ]
        test_cases = [{"x": 1}]

        report = hom.verify_homomorphism(rules, test_cases)

        # Should detect multiplication violation
        assert report.is_homomorphism is False
        assert len(report.violations) > 0

    def test_homomorphism_verification_report_structure(self) -> None:
        """Test verification report contains expected fields."""

        def identity(rule: BooleanRule) -> BooleanRule:
            return rule

        hom = RingHomomorphism(BooleanRule, BooleanRule, identity)
        rules = [BooleanRule(lambda c: True, "true")]

        report = hom.verify_homomorphism(rules, [{}])

        assert hasattr(report, "is_homomorphism")
        assert hasattr(report, "violations")
        assert hasattr(report, "tests_run")
        assert hasattr(report, "tests_passed")


# ============================================================================
# ComposedRule Tests
# ============================================================================


class TestComposedRule:
    """Tests for ComposedRule class."""

    def test_from_boolean_rule(self, rule_a: BooleanRule) -> None:
        """Test creating composed rule from boolean rule."""
        composed = ComposedRule.from_boolean_rule(rule_a)

        assert composed.evaluate({"a": True}) is True
        assert composed.evaluate({"a": False}) is False
        assert composed.name == "a"

    def test_conjoin(self, rule_a: BooleanRule, rule_b: BooleanRule) -> None:
        """Test conjunction of composed rules."""
        composed_a = ComposedRule.from_boolean_rule(rule_a)
        composed_b = ComposedRule.from_boolean_rule(rule_b)

        conjoined = composed_a.conjoin(composed_b)

        assert conjoined.evaluate({"a": True, "b": True}) is True
        assert conjoined.evaluate({"a": True, "b": False}) is False
        assert "∧" in conjoined.name

    def test_disjoin(self, rule_a: BooleanRule, rule_b: BooleanRule) -> None:
        """Test disjunction of composed rules."""
        composed_a = ComposedRule.from_boolean_rule(rule_a)
        composed_b = ComposedRule.from_boolean_rule(rule_b)

        disjoined = composed_a.disjoin(composed_b)

        assert disjoined.evaluate({"a": True, "b": False}) is True
        assert disjoined.evaluate({"a": False, "b": True}) is True
        assert disjoined.evaluate({"a": False, "b": False}) is False
        assert "∨" in disjoined.name

    def test_composition_tree(self, rule_a: BooleanRule, rule_b: BooleanRule) -> None:
        """Test composition tree is built correctly."""
        composed_a = ComposedRule.from_boolean_rule(rule_a)
        composed_b = ComposedRule.from_boolean_rule(rule_b)

        conjoined = composed_a.conjoin(composed_b)

        assert conjoined.composition_tree["type"] == "conjunction"
        assert "left" in conjoined.composition_tree
        assert "right" in conjoined.composition_tree

    def test_explain_composition_leaf(self, rule_a: BooleanRule) -> None:
        """Test explanation of leaf rule."""
        composed = ComposedRule.from_boolean_rule(rule_a)

        explanation = composed.explain_composition()

        assert "Rule: a" in explanation

    def test_explain_composition_conjunction(
        self, rule_a: BooleanRule, rule_b: BooleanRule
    ) -> None:
        """Test explanation of conjunction."""
        composed_a = ComposedRule.from_boolean_rule(rule_a)
        composed_b = ComposedRule.from_boolean_rule(rule_b)

        conjoined = composed_a.conjoin(composed_b)
        explanation = conjoined.explain_composition()

        assert "AND" in explanation
        assert "Rule: a" in explanation
        assert "Rule: b" in explanation

    def test_explain_composition_disjunction(
        self, rule_a: BooleanRule, rule_b: BooleanRule
    ) -> None:
        """Test explanation of disjunction."""
        composed_a = ComposedRule.from_boolean_rule(rule_a)
        composed_b = ComposedRule.from_boolean_rule(rule_b)

        disjoined = composed_a.disjoin(composed_b)
        explanation = disjoined.explain_composition()

        assert "OR" in explanation

    def test_nested_composition(
        self, rule_a: BooleanRule, rule_b: BooleanRule, rule_c: BooleanRule
    ) -> None:
        """Test nested composition."""
        composed_a = ComposedRule.from_boolean_rule(rule_a)
        composed_b = ComposedRule.from_boolean_rule(rule_b)
        composed_c = ComposedRule.from_boolean_rule(rule_c)

        # (a AND b) OR c
        nested = composed_a.conjoin(composed_b).disjoin(composed_c)

        # True if (a AND b) OR c
        assert nested.evaluate({"a": True, "b": True, "c": False}) is True
        assert nested.evaluate({"a": False, "b": False, "c": True}) is True
        assert nested.evaluate({"a": True, "b": False, "c": False}) is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestRingIntegration:
    """Integration tests for ring structure."""

    def test_legal_rule_composition_scenario(self) -> None:
        """Test realistic legal rule composition scenario."""
        # Define individual legal conditions
        has_offer = BooleanRule(lambda c: c.get("offer", False), "has_offer")
        has_acceptance = BooleanRule(
            lambda c: c.get("acceptance", False), "has_acceptance"
        )
        has_consideration = BooleanRule(
            lambda c: c.get("consideration", False), "has_consideration"
        )
        is_minor = BooleanRule(lambda c: c.get("minor", False), "is_minor")

        # Contract valid = (offer AND acceptance AND consideration) AND NOT minor
        base_contract = has_offer * has_acceptance * has_consideration
        valid_contract = RuleComposition.exception(base_contract, is_minor)

        # Test cases
        test_cases = [
            # All elements present, not minor: valid
            {
                "offer": True,
                "acceptance": True,
                "consideration": True,
                "minor": False,
            },
            # Missing acceptance: invalid
            {
                "offer": True,
                "acceptance": False,
                "consideration": True,
                "minor": False,
            },
            # All present but minor: invalid (exception)
            {"offer": True, "acceptance": True, "consideration": True, "minor": True},
        ]

        assert valid_contract.evaluate(test_cases[0]) is True
        assert valid_contract.evaluate(test_cases[1]) is False
        assert valid_contract.evaluate(test_cases[2]) is False

    def test_ring_verification_with_legal_rules(self) -> None:
        """Test ring axioms hold for legal rule composition."""
        has_offer = BooleanRule(lambda c: c.get("offer", False), "offer")
        has_acceptance = BooleanRule(lambda c: c.get("acceptance", False), "accept")
        has_consideration = BooleanRule(
            lambda c: c.get("consideration", False), "consider"
        )

        test_cases = [
            {"offer": True, "acceptance": True, "consideration": True},
            {"offer": True, "acceptance": False, "consideration": True},
            {"offer": False, "acceptance": True, "consideration": False},
            {"offer": False, "acceptance": False, "consideration": False},
        ]

        verifier = RingPropertyVerifier(test_cases)
        report = verifier.full_verification(
            [has_offer, has_acceptance, has_consideration], BooleanRule
        )

        assert report.is_ring is True

    def test_confidence_weighted_rules(self) -> None:
        """Test confidence-weighted rule combination."""
        # High confidence rule
        high_conf = ConfidenceRule(lambda c: 0.9, "high")
        # Low confidence rule
        low_conf = ConfidenceRule(lambda c: 0.3, "low")

        # Combined via max (addition)
        combined_or = high_conf + low_conf
        # Combined via product (multiplication)
        combined_and = high_conf * low_conf

        case: Dict[str, Any] = {}

        # Max should give 0.9
        assert combined_or.evaluate(case) == 0.9
        # Product should give 0.27
        assert abs(combined_and.evaluate(case) - 0.27) < 0.01

    def test_de_morgan_laws(self, boolean_test_cases: list) -> None:
        """Test De Morgan's laws hold."""
        rule_a = BooleanRule(lambda c: c["a"], "a")
        rule_b = BooleanRule(lambda c: c["b"], "b")

        # NOT (a AND b) = (NOT a) OR (NOT b)
        lhs1 = -(rule_a * rule_b)
        rhs1 = (-rule_a) + (-rule_b)

        # NOT (a OR b) = (NOT a) AND (NOT b)
        lhs2 = -(rule_a + rule_b)
        rhs2 = (-rule_a) * (-rule_b)

        for case in boolean_test_cases:
            assert lhs1.evaluate(case) == rhs1.evaluate(case)
            assert lhs2.evaluate(case) == rhs2.evaluate(case)

    def test_idempotence(self, boolean_test_cases: list) -> None:
        """Test idempotence: a + a = a and a * a = a."""
        rule_a = BooleanRule(lambda c: c["a"], "a")

        or_self = rule_a + rule_a
        and_self = rule_a * rule_a

        for case in boolean_test_cases:
            assert or_self.evaluate(case) == rule_a.evaluate(case)
            assert and_self.evaluate(case) == rule_a.evaluate(case)

    def test_absorption_laws(self, boolean_test_cases: list) -> None:
        """Test absorption laws: a + (a * b) = a and a * (a + b) = a."""
        rule_a = BooleanRule(lambda c: c["a"], "a")
        rule_b = BooleanRule(lambda c: c["b"], "b")

        # a + (a * b) = a
        lhs1 = rule_a + (rule_a * rule_b)
        # a * (a + b) = a
        lhs2 = rule_a * (rule_a + rule_b)

        for case in boolean_test_cases:
            assert lhs1.evaluate(case) == rule_a.evaluate(case)
            assert lhs2.evaluate(case) == rule_a.evaluate(case)
