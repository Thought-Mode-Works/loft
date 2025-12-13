"""
Phase 7 Integration Tests: Geometric Constraints & Invariance.

This module tests the integration of all Phase 7 components:
- Equivariance (content-neutrality)
- Symmetry (party symmetry)
- Temporal consistency
- Measure theory (confidence)
- Ring structure (composition)
- Constitutional verification (safety)

It validates MVP criteria from ROADMAP.md and ensures all components
work together correctly.
"""

import pytest
from typing import Dict, Any, List
from datetime import datetime

from loft.constraints.equivariance import (
    EquivarianceVerifier,
    PartyPermutationEquivariance,
    ContentSubstitutionEquivariance,
)
from loft.constraints.temporal import (
    TemporalConsistencyTester,
    TemporalField,
)
from loft.constraints.measure_theory import (
    CaseSpace,
    CaseDimension,
    CaseDistribution,
    MonomialPotential,
)
from loft.constraints.ring_structure import (
    BooleanRule,
    ConfidenceRule,
    RuleComposition,
    RingPropertyVerifier,
    ComposedRule,
)
from loft.constraints.constitutional import (
    SystemState,
    Rule,
    Fact,
    create_verifier,
    create_guard,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def contract_test_cases() -> List[Dict[str, Any]]:
    """Standard contract test cases for integration testing."""
    return [
        {
            "plaintiff": "alice",
            "defendant": "bob",
            "amount": 600,
            "has_writing": True,
            "has_offer": True,
            "has_acceptance": True,
            "has_consideration": True,
            "goods_sale": True,
            "formation_date": "2020-01-15",
            "performance_date": "2020-06-15",
        },
        {
            "plaintiff": "charlie",
            "defendant": "diana",
            "amount": 400,
            "has_writing": False,
            "has_offer": True,
            "has_acceptance": True,
            "has_consideration": True,
            "goods_sale": True,
            "formation_date": "2020-03-01",
            "performance_date": "2020-09-01",
        },
        {
            "plaintiff": "eve",
            "defendant": "frank",
            "amount": 1000,
            "has_writing": True,
            "has_offer": True,
            "has_acceptance": False,
            "has_consideration": True,
            "goods_sale": False,
            "formation_date": "2021-01-01",
            "performance_date": "2021-12-31",
        },
    ]


@pytest.fixture
def boolean_test_cases() -> List[Dict[str, Any]]:
    """Boolean test cases for ring property verification."""
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
def temporal_fields() -> List[TemporalField]:
    """Standard temporal fields for testing."""
    return [
        TemporalField(path="formation_date", field_type="date"),
        TemporalField(path="performance_date", field_type="date"),
    ]


@pytest.fixture
def case_space() -> CaseSpace:
    """Standard case space for measure theory tests."""
    return CaseSpace(
        dimensions=[
            CaseDimension("has_offer", "boolean"),
            CaseDimension("has_acceptance", "boolean"),
            CaseDimension("has_consideration", "boolean"),
            CaseDimension("has_writing", "boolean"),
            CaseDimension("amount", "numeric", bounds=(0.0, 10000.0)),
        ]
    )


@pytest.fixture
def valid_system_state() -> SystemState:
    """A valid system state for constitutional verification."""
    return SystemState(
        rules=[
            Rule(
                rule_id="r1",
                confidence=0.85,
                metadata={"neutral": True},
                body="has_offer(X), has_acceptance(X)",
                head="contract_formed(X)",
            ),
            Rule(
                rule_id="r2",
                confidence=0.90,
                metadata={},
                body="contract_formed(X), has_consideration(X)",
                head="valid_contract(X)",
            ),
        ],
        facts=[
            Fact("has_offer"),
            Fact("has_acceptance"),
        ],
        metadata={
            "max_query_depth": 100,
            "previous_state_id": "init",
            "recent_outcomes": [
                {"result": True, "explanation": "Contract formation rule applied"}
            ],
        },
        timestamp=datetime.now().timestamp(),
    )


# ============================================================================
# MVP Validation Tests
# ============================================================================


class TestPhase7MVPCriteria:
    """Validate Phase 7 MVP criteria from ROADMAP.md."""

    def test_mvp_content_neutrality(
        self, contract_test_cases: List[Dict[str, Any]]
    ) -> None:
        """
        MVP: Rules satisfy content-neutrality constraints.

        Content-neutrality means outcomes depend on legal structure,
        not on arbitrary content like specific names or amounts.
        """

        # Create a content-neutral rule
        def neutral_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            # Result depends on legal elements, not specific values
            has_elements = (
                case.get("has_offer", False)
                and case.get("has_acceptance", False)
                and case.get("has_consideration", False)
            )
            return {"valid": has_elements}

        # Verify with equivariance constraints
        verifier = EquivarianceVerifier(
            [
                PartyPermutationEquivariance(),
                ContentSubstitutionEquivariance(),
            ]
        )

        report = verifier.verify_rule(neutral_rule, contract_test_cases)

        # Content-neutral rule should be equivariant
        assert (
            report.is_equivariant
        ), f"Content-neutral rule violates equivariance: {report.violations}"

    def test_mvp_party_swap_equivalence(
        self, contract_test_cases: List[Dict[str, Any]]
    ) -> None:
        """
        MVP: Party-swapping produces equivalent outcomes.

        For symmetric rules, swapping party identities should not
        change the legal analysis.
        """

        # Create a party-symmetric rule
        def symmetric_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            # Rule doesn't depend on specific party names
            has_parties = "plaintiff" in case and "defendant" in case
            return {"has_parties": has_parties}

        verifier = EquivarianceVerifier([PartyPermutationEquivariance()])

        report = verifier.verify_rule(symmetric_rule, contract_test_cases)

        assert (
            report.is_equivariant
        ), f"Symmetric rule violates party swap equivalence: {report.violations}"

    def test_mvp_temporal_consistency(
        self,
        temporal_fields: List[TemporalField],
    ) -> None:
        """
        MVP: Similar cases (same temporal relationships) -> similar outcomes.

        Shifting all dates by a constant should not change outcomes.
        """
        # Create test cases with adequate date gaps (30+ days for 0.5 scale)
        temporal_test_cases = [
            {
                "formation_date": "2020-01-01",
                "performance_date": "2020-07-01",  # 6 months gap
                "amount": 500,
            },
            {
                "formation_date": "2020-03-15",
                "performance_date": "2020-09-15",  # 6 months gap
                "amount": 750,
            },
        ]

        tester = TemporalConsistencyTester(temporal_fields)

        # Create a temporally consistent rule
        def temporal_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            # Rule only cares about relationship between dates, not absolute values
            return {
                "has_dates": "formation_date" in case and "performance_date" in case
            }

        report = tester.test_shift_invariance(temporal_rule, temporal_test_cases)

        assert (
            report.is_consistent
        ), f"Temporal consistency violated: {report.violations}"

    def test_mvp_ring_homomorphism(
        self, boolean_test_cases: List[Dict[str, Any]]
    ) -> None:
        """
        MVP: Rule composition follows ring homomorphism properties.

        Verify that combining rules preserves algebraic structure.
        """
        # Create test rules
        has_offer = BooleanRule(lambda c: c.get("a", False), "has_offer")
        has_acceptance = BooleanRule(lambda c: c.get("b", False), "has_acceptance")
        has_consideration = BooleanRule(
            lambda c: c.get("c", False), "has_consideration"
        )

        verifier = RingPropertyVerifier(boolean_test_cases)

        # Verify ring axioms
        report = verifier.full_verification(
            [has_offer, has_acceptance, has_consideration],
            BooleanRule,
        )

        assert report.is_ring, (
            f"Rule composition violates ring axioms: "
            f"distributivity={report.distributivity_holds}, "
            f"associativity_add={report.associativity_add_holds}, "
            f"associativity_mul={report.associativity_mul_holds}"
        )

    def test_mvp_constitutional_preservation(
        self, valid_system_state: SystemState
    ) -> None:
        """
        MVP: Constitutional constraints are provably preserved.

        Verify that all operations maintain constitutional properties.
        """
        verifier = create_verifier()

        # Verify all properties hold
        report = verifier.verify_all(valid_system_state)

        assert report.all_verified, (
            f"Constitutional violations found: "
            f"{[r.property_name for r in report.get_violations()]}"
        )


# ============================================================================
# Cross-Component Integration Tests
# ============================================================================


class TestCrossComponentIntegration:
    """Test interactions between Phase 7 components."""

    def test_symmetric_rules_compose_symmetrically(
        self, boolean_test_cases: List[Dict[str, Any]]
    ) -> None:
        """
        Symmetric rules combined via ring operations remain symmetric.
        """
        # Create symmetric rules (don't depend on specific party names)
        rule_a = BooleanRule(lambda c: c.get("a", False), "rule_a")
        rule_b = BooleanRule(lambda c: c.get("b", False), "rule_b")

        # Compose rules using ring operations
        composed_and = rule_a * rule_b  # Conjunction
        composed_or = rule_a + rule_b  # Disjunction

        # Verify composition preserves symmetry
        verifier = EquivarianceVerifier([PartyPermutationEquivariance()])

        # Original rules should be equivariant
        report_a = verifier.verify_rule(
            lambda c: {"result": rule_a.evaluate(c)}, boolean_test_cases
        )
        report_b = verifier.verify_rule(
            lambda c: {"result": rule_b.evaluate(c)}, boolean_test_cases
        )

        assert report_a.is_equivariant
        assert report_b.is_equivariant

        # Composed rules should also be equivariant
        report_and = verifier.verify_rule(
            lambda c: {"result": composed_and.evaluate(c)}, boolean_test_cases
        )
        report_or = verifier.verify_rule(
            lambda c: {"result": composed_or.evaluate(c)}, boolean_test_cases
        )

        assert report_and.is_equivariant, "Conjunction breaks symmetry"
        assert report_or.is_equivariant, "Disjunction breaks symmetry"

    def test_measure_theory_with_ring_structure(self, case_space: CaseSpace) -> None:
        """
        Measure-theoretic confidence integrates with ring composition.
        """
        # Create confidence rules
        offer_conf = ConfidenceRule(
            lambda c: 1.0 if c.get("has_offer", False) else 0.0, "offer"
        )
        acceptance_conf = ConfidenceRule(
            lambda c: 1.0 if c.get("has_acceptance", False) else 0.0, "acceptance"
        )

        # Compose via ring multiplication (independent conjunction)
        combined = offer_conf * acceptance_conf

        # Create sample cases
        samples = [
            {
                "has_offer": True,
                "has_acceptance": True,
                "has_consideration": True,
                "has_writing": True,
                "amount": 500,
            },
            {
                "has_offer": True,
                "has_acceptance": False,
                "has_consideration": True,
                "has_writing": False,
                "amount": 300,
            },
            {
                "has_offer": False,
                "has_acceptance": True,
                "has_consideration": False,
                "has_writing": True,
                "amount": 1000,
            },
        ]

        # Verify measure-theoretic properties: distribution can be computed
        distribution = CaseDistribution.from_samples(case_space, samples)
        assert distribution is not None

        # Combined confidence should equal product
        for sample in samples:
            expected = offer_conf.evaluate(sample) * acceptance_conf.evaluate(sample)
            actual = combined.evaluate(sample)
            assert abs(expected - actual) < 0.01, (
                f"Ring composition doesn't preserve measure: "
                f"expected {expected}, got {actual}"
            )

    def test_constitutional_guard_blocks_invalid_confidence(
        self, valid_system_state: SystemState
    ) -> None:
        """
        Constitutional guard blocks operations that introduce invalid confidence.
        """
        guard = create_guard()

        # Operation that would introduce invalid confidence
        def add_invalid_confidence_rule(state: SystemState) -> SystemState:
            return SystemState(
                rules=state.rules + [Rule(rule_id="bad", confidence=1.5)],  # Invalid!
                facts=state.facts,
                metadata=state.metadata,
                timestamp=state.timestamp + 1,
            )

        result_state, report = guard.guard(
            add_invalid_confidence_rule,
            valid_system_state,
            "add_invalid_confidence",
        )

        # Guard should block the operation
        assert not report.transition_safe
        # Original state should be preserved
        assert result_state == valid_system_state

    def test_constitutional_guard_blocks_contradiction(
        self, valid_system_state: SystemState
    ) -> None:
        """
        Constitutional guard blocks operations that introduce contradictions.
        """
        guard = create_guard()

        # Operation that would introduce contradiction
        def add_contradiction(state: SystemState) -> SystemState:
            return SystemState(
                rules=state.rules,
                facts=[Fact("p", negated=False), Fact("p", negated=True)],
                metadata=state.metadata,
                timestamp=state.timestamp + 1,
            )

        result_state, report = guard.guard(
            add_contradiction,
            valid_system_state,
            "add_contradiction",
        )

        assert not report.transition_safe
        assert any(v.property_name == "NO_CONTRADICTION" for v in report.violations)

    def test_ring_composition_with_exception(
        self, boolean_test_cases: List[Dict[str, Any]]
    ) -> None:
        """
        Rule composition with exceptions works correctly.
        """
        # Base rule: contract is valid
        base_rule = BooleanRule(
            lambda c: c.get("a", False) and c.get("b", False),
            "valid_contract",
        )

        # Exception: minor party
        exception_rule = BooleanRule(lambda c: c.get("c", False), "is_minor")

        # Compose: valid unless minor
        with_exception = RuleComposition.exception(base_rule, exception_rule)

        # Test cases
        test_cases = [
            {"a": True, "b": True, "c": False},  # Valid, not minor -> True
            {"a": True, "b": True, "c": True},  # Valid, but minor -> False
            {"a": True, "b": False, "c": False},  # Invalid -> False
        ]

        expected = [True, False, False]

        for case, exp in zip(test_cases, expected):
            result = with_exception.evaluate(case)
            assert result == exp, f"Exception composition failed for {case}"

    def test_composed_rule_explainability(self) -> None:
        """
        ComposedRule maintains explainability through composition tree.
        """
        rule_a = BooleanRule(lambda c: c.get("offer", False), "has_offer")
        rule_b = BooleanRule(lambda c: c.get("accept", False), "has_acceptance")
        rule_c = BooleanRule(lambda c: c.get("consider", False), "has_consideration")

        # Build composed rule
        composed_a = ComposedRule.from_boolean_rule(rule_a)
        composed_b = ComposedRule.from_boolean_rule(rule_b)
        composed_c = ComposedRule.from_boolean_rule(rule_c)

        # (offer AND acceptance) AND consideration
        final = composed_a.conjoin(composed_b).conjoin(composed_c)

        # Get explanation
        explanation = final.explain_composition()

        # Should contain all component names
        assert "has_offer" in explanation
        assert "has_acceptance" in explanation
        assert "has_consideration" in explanation
        assert "AND" in explanation


# ============================================================================
# Component Compatibility Tests
# ============================================================================


class TestComponentCompatibility:
    """Test that all Phase 7 components use compatible interfaces."""

    def test_all_constraints_modules_importable(self) -> None:
        """Verify all constraint modules can be imported."""
        from loft.constraints import equivariance
        from loft.constraints import symmetry
        from loft.constraints import temporal
        from loft.constraints import measure_theory
        from loft.constraints import ring_structure
        from loft.constraints import constitutional

        # All modules should have key classes
        assert hasattr(equivariance, "EquivarianceVerifier")
        assert hasattr(symmetry, "PartySymmetryTester")
        assert hasattr(temporal, "TemporalConsistencyTester")
        assert hasattr(measure_theory, "CaseSpace")
        assert hasattr(ring_structure, "BooleanRule")
        assert hasattr(constitutional, "ConstitutionalVerifier")

    def test_constraints_init_exports(self) -> None:
        """Verify __init__.py exports all necessary classes."""
        import loft.constraints as constraints_module

        # Check that all expected exports are available
        expected_exports = [
            # Equivariance
            "TransformationType",
            "EquivarianceConstraint",
            "PartyPermutationEquivariance",
            "AmountScalingEquivariance",
            "ContentSubstitutionEquivariance",
            "EquivarianceVerifier",
            # Symmetry
            "SymmetryType",
            "PartySymmetryConstraint",
            "PartySymmetryTester",
            # Temporal
            "TemporalTransformType",
            "TemporalField",
            "TemporalConsistencyTester",
            # Measure Theory
            "MeasurableOutcome",
            "CaseSpace",
            "CaseDimension",
            # Ring Structure
            "RingElement",
            "BooleanRule",
            "ConfidenceRule",
            "RuleComposition",
            "RingHomomorphism",
            "RingPropertyVerifier",
            # Constitutional
            "PropertyType",
            "VerificationResult",
            "ConstitutionalProperty",
            "ConstitutionalVerifier",
            "ConstitutionalGuard",
        ]

        for export in expected_exports:
            assert hasattr(constraints_module, export), f"Missing export: {export}"
            assert getattr(constraints_module, export) is not None

    def test_monomial_potential_integration(self, case_space: CaseSpace) -> None:
        """Test MonomialPotential with case space."""
        # Create a monomial potential for contract validity
        potential = MonomialPotential(
            elements=["offer", "acceptance", "consideration"],
            weights=[1.0, 1.0, 1.0],  # All equally required
        )

        # Test satisfaction evaluation
        full_satisfaction = {"offer": 1.0, "acceptance": 1.0, "consideration": 1.0}
        partial_satisfaction = {"offer": 1.0, "acceptance": 0.5, "consideration": 1.0}
        zero_satisfaction = {"offer": 0.0, "acceptance": 1.0, "consideration": 1.0}

        assert potential.evaluate(full_satisfaction) == 1.0
        assert 0.0 < potential.evaluate(partial_satisfaction) < 1.0
        assert potential.evaluate(zero_satisfaction) == 0.0

    def test_temporal_with_measure_theory(self, case_space: CaseSpace) -> None:
        """Test temporal consistency integrates with measure theory."""
        temporal_fields = [
            TemporalField(path="formation_date", field_type="date"),
        ]

        tester = TemporalConsistencyTester(temporal_fields)

        # Create temporally varying cases
        cases = [
            {"formation_date": "2020-01-01", "has_offer": True, "amount": 500},
            {"formation_date": "2020-07-01", "has_offer": True, "amount": 500},
        ]

        # A rule that should be temporally consistent
        def amount_based_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            return {"valid": case.get("amount", 0) >= 500}

        report = tester.test_shift_invariance(amount_based_rule, cases)
        assert report.is_consistent


# ============================================================================
# End-to-End Scenario Tests
# ============================================================================


class TestEndToEndScenarios:
    """End-to-end tests for realistic legal scenarios."""

    def test_contract_validity_full_pipeline(self) -> None:
        """
        Test complete contract validity analysis through all components.
        """
        # 1. Create contract formation rules using ring structure
        has_offer = BooleanRule(lambda c: c.get("offer", False), "offer")
        has_acceptance = BooleanRule(lambda c: c.get("acceptance", False), "acceptance")
        has_consideration = BooleanRule(
            lambda c: c.get("consideration", False), "consideration"
        )

        # Contract = offer AND acceptance AND consideration
        contract_valid = has_offer * has_acceptance * has_consideration

        # 2. Verify ring properties
        test_cases = [
            {"offer": True, "acceptance": True, "consideration": True},
            {"offer": True, "acceptance": False, "consideration": True},
            {"offer": False, "acceptance": True, "consideration": True},
            {"offer": False, "acceptance": False, "consideration": False},
        ]

        ring_verifier = RingPropertyVerifier(test_cases)
        ring_report = ring_verifier.full_verification(
            [has_offer, has_acceptance, has_consideration], BooleanRule
        )
        assert ring_report.is_ring

        # 3. Verify equivariance
        equiv_verifier = EquivarianceVerifier([PartyPermutationEquivariance()])
        equiv_report = equiv_verifier.verify_rule(
            lambda c: {"valid": contract_valid.evaluate(c)}, test_cases
        )
        assert equiv_report.is_equivariant

        # 4. Create system state and verify constitutional properties
        state = SystemState(
            rules=[
                Rule(rule_id="contract", confidence=0.95, metadata={"neutral": True})
            ],
            facts=[],
            metadata={
                "max_query_depth": 100,
                "previous_state_id": "init",
                "recent_outcomes": [{"result": True, "explanation": "Test"}],
            },
            timestamp=0.0,
        )

        const_verifier = create_verifier()
        const_report = const_verifier.verify_all(state)
        assert const_report.all_verified

        # 5. Evaluate contracts
        valid_contract = {"offer": True, "acceptance": True, "consideration": True}
        invalid_contract = {"offer": True, "acceptance": False, "consideration": True}

        assert contract_valid.evaluate(valid_contract) is True
        assert contract_valid.evaluate(invalid_contract) is False

    def test_statute_of_frauds_with_exception(self) -> None:
        """
        Test Statute of Frauds rule with part performance exception.
        """
        # Base rule: goods over $500 require writing
        requires_writing = BooleanRule(
            lambda c: c.get("amount", 0) >= 500 and c.get("goods_sale", False),
            "requires_writing",
        )
        has_writing = BooleanRule(lambda c: c.get("has_writing", False), "has_writing")

        # Exception: part performance
        part_performance = BooleanRule(
            lambda c: c.get("part_performance", False), "part_performance"
        )

        # SoF satisfied = has writing OR part performance OR doesn't require writing
        sof_satisfied = has_writing + part_performance + (-requires_writing)

        test_cases = [
            # Requires writing, has writing: satisfied
            {
                "amount": 600,
                "goods_sale": True,
                "has_writing": True,
                "part_performance": False,
            },
            # Requires writing, no writing, but part performance: satisfied
            {
                "amount": 600,
                "goods_sale": True,
                "has_writing": False,
                "part_performance": True,
            },
            # Requires writing, no writing, no exception: not satisfied
            {
                "amount": 600,
                "goods_sale": True,
                "has_writing": False,
                "part_performance": False,
            },
            # Doesn't require writing (under $500): satisfied
            {
                "amount": 400,
                "goods_sale": True,
                "has_writing": False,
                "part_performance": False,
            },
        ]

        expected = [True, True, False, True]

        for case, exp in zip(test_cases, expected):
            result = sof_satisfied.evaluate(case)
            assert result == exp, f"SoF analysis failed for {case}: expected {exp}"

    def test_multi_rule_validation_pipeline(self) -> None:
        """
        Test validation of multiple rules through the full constraint pipeline.
        """
        # Create multiple rules
        rules = [
            BooleanRule(lambda c: c.get("valid", False), "validity_rule"),
            BooleanRule(
                lambda c: c.get("signed", False) or c.get("electronic", False),
                "signature_rule",
            ),
            BooleanRule(lambda c: c.get("consideration", False), "consideration_rule"),
        ]

        test_cases = [
            {"valid": True, "signed": True, "electronic": False, "consideration": True},
            {"valid": True, "signed": False, "electronic": True, "consideration": True},
            {
                "valid": False,
                "signed": True,
                "electronic": True,
                "consideration": False,
            },
        ]

        # 1. Verify all rules satisfy ring properties
        ring_verifier = RingPropertyVerifier(test_cases)
        ring_report = ring_verifier.full_verification(rules, BooleanRule)
        assert ring_report.is_ring

        # 2. Verify all rules are equivariant
        equiv_verifier = EquivarianceVerifier([PartyPermutationEquivariance()])
        for rule in rules:
            report = equiv_verifier.verify_rule(
                lambda c, r=rule: {"result": r.evaluate(c)}, test_cases
            )
            assert report.is_equivariant, f"Rule {rule.name} is not equivariant"

        # 3. Compose all rules and verify composition
        composed = rules[0]
        for rule in rules[1:]:
            composed = composed * rule

        # Composed rule should be equivariant if all components are
        composed_report = equiv_verifier.verify_rule(
            lambda c: {"result": composed.evaluate(c)}, test_cases
        )
        assert composed_report.is_equivariant
