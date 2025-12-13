"""
Unit tests for constitutional layer verification.

Tests the constitutional layer formal verification system, ensuring
that core safety and correctness properties are preserved.
"""

import pytest
from typing import Dict

from loft.constraints.constitutional import (
    PropertyType,
    VerificationResult,
    Fact,
    Rule,
    SystemState,
    ConstitutionalProperty,
    ConstitutionalVerifier,
    create_standard_properties,
    create_verifier,
    create_guard,
    check_no_contradiction,
    check_party_neutrality,
    check_confidence_bounds,
    check_stratification,
    check_rollback_available,
    check_rule_monotonicity,
    check_query_termination,
    check_explainability,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def empty_state() -> SystemState:
    """An empty system state."""
    return SystemState(
        rules=[],
        facts=[],
        metadata={"max_query_depth": 100, "previous_state_id": "init"},
        timestamp=0.0,
    )


@pytest.fixture
def valid_state() -> SystemState:
    """A valid system state with no violations."""
    return SystemState(
        rules=[
            Rule(rule_id="r1", confidence=0.8, metadata={}, body="a(X)", head="b(X)"),
            Rule(rule_id="r2", confidence=0.9, metadata={}, body="b(X)", head="c(X)"),
        ],
        facts=[
            Fact("valid", False),
            Fact("enforceable", False),
        ],
        metadata={
            "max_query_depth": 100,
            "previous_state_id": "prev",
            "recent_outcomes": [{"result": True, "explanation": "Rule r1 applied"}],
        },
        timestamp=1.0,
    )


@pytest.fixture
def contradictory_state() -> SystemState:
    """A state with logical contradictions."""
    return SystemState(
        rules=[],
        facts=[
            Fact("valid", negated=False),
            Fact("valid", negated=True),  # Contradiction!
        ],
        metadata={"max_query_depth": 100, "previous_state_id": "prev"},
        timestamp=1.0,
    )


@pytest.fixture
def invalid_confidence_state() -> SystemState:
    """A state with invalid confidence scores."""
    return SystemState(
        rules=[
            Rule(rule_id="r1", confidence=1.5),  # Invalid: > 1.0
        ],
        facts=[],
        metadata={"max_query_depth": 100, "previous_state_id": "prev"},
        timestamp=1.0,
    )


@pytest.fixture
def asymmetric_neutral_rule_state() -> SystemState:
    """A state with an asymmetric neutral rule."""
    return SystemState(
        rules=[
            Rule(
                rule_id="r1",
                confidence=0.9,
                metadata={"neutral": True},
                body="plaintiff(X), damages(X, Y)",  # Only mentions plaintiff
                head="liable(X)",
            ),
        ],
        facts=[],
        metadata={"max_query_depth": 100, "previous_state_id": "prev"},
        timestamp=1.0,
    )


@pytest.fixture
def no_rollback_state() -> SystemState:
    """A state without rollback capability."""
    return SystemState(
        rules=[],
        facts=[],
        metadata={"max_query_depth": 100},  # No previous_state_id
        timestamp=1.0,
    )


@pytest.fixture
def standard_verifier() -> ConstitutionalVerifier:
    """A verifier with standard properties."""
    return create_verifier()


# ============================================================================
# Fact and Rule Tests
# ============================================================================


class TestFact:
    """Tests for Fact class."""

    def test_fact_creation(self) -> None:
        """Test basic fact creation."""
        fact = Fact("valid")
        assert fact.predicate == "valid"
        assert fact.negated is False

    def test_negated_fact(self) -> None:
        """Test negated fact creation."""
        fact = Fact("valid", negated=True)
        assert fact.predicate == "valid"
        assert fact.negated is True

    def test_fact_with_args(self) -> None:
        """Test fact with arguments."""
        fact = Fact("owns", args=("alice", "car"))
        assert fact.predicate == "owns"
        assert fact.args == ("alice", "car")

    def test_fact_equality(self) -> None:
        """Test fact equality."""
        fact1 = Fact("valid")
        fact2 = Fact("valid")
        fact3 = Fact("invalid")

        assert fact1 == fact2
        assert fact1 != fact3

    def test_fact_hash(self) -> None:
        """Test fact hashing for set operations."""
        fact1 = Fact("valid")
        fact2 = Fact("valid")

        fact_set = {fact1, fact2}
        assert len(fact_set) == 1


class TestRule:
    """Tests for Rule class."""

    def test_rule_creation(self) -> None:
        """Test basic rule creation."""
        rule = Rule(rule_id="r1", confidence=0.8)
        assert rule.rule_id == "r1"
        assert rule.confidence == 0.8

    def test_symmetric_rule_no_parties(self) -> None:
        """Test rule with no party terms is symmetric."""
        rule = Rule(rule_id="r1", body="valid(X), written(X)", head="enforceable(X)")
        assert rule.is_symmetric() is True

    def test_symmetric_rule_balanced_parties(self) -> None:
        """Test rule with balanced party terms is symmetric."""
        rule = Rule(
            rule_id="r1",
            body="plaintiff(X), defendant(Y), agreement(X, Y)",
            head="contract(X, Y)",
        )
        assert rule.is_symmetric() is True

    def test_asymmetric_rule_single_party(self) -> None:
        """Test rule with only one party is asymmetric."""
        rule = Rule(
            rule_id="r1",
            body="plaintiff(X), damages(X)",
            head="liable(X)",
        )
        assert rule.is_symmetric() is False

    def test_role_specific_rule_allowed(self) -> None:
        """Test that role-specific rules can mention single parties."""
        rule = Rule(
            rule_id="r1",
            body="plaintiff(X), damages(X)",
            head="liable(X)",
            metadata={"role_specific": True},
        )
        assert rule.is_symmetric() is True


# ============================================================================
# SystemState Tests
# ============================================================================


class TestSystemState:
    """Tests for SystemState class."""

    def test_get_rule_set(self, valid_state: SystemState) -> None:
        """Test getting rule identifiers."""
        rule_set = valid_state.get_rule_set()
        assert rule_set == {"r1", "r2"}

    def test_has_contradiction_false(self, valid_state: SystemState) -> None:
        """Test no contradiction detection."""
        assert valid_state.has_contradiction() is False

    def test_has_contradiction_true(self, contradictory_state: SystemState) -> None:
        """Test contradiction detection."""
        assert contradictory_state.has_contradiction() is True

    def test_get_contradicting_facts(self, contradictory_state: SystemState) -> None:
        """Test getting contradicting fact pairs."""
        contradictions = contradictory_state.get_contradicting_facts()
        assert len(contradictions) == 1
        assert contradictions[0][0].predicate == "valid"
        assert contradictions[0][1].predicate == "valid"


# ============================================================================
# Property Checker Tests
# ============================================================================


class TestPropertyCheckers:
    """Tests for individual property checker functions."""

    def test_check_no_contradiction_valid(self, valid_state: SystemState) -> None:
        """Test no contradiction check on valid state."""
        assert check_no_contradiction(valid_state) is True

    def test_check_no_contradiction_invalid(
        self, contradictory_state: SystemState
    ) -> None:
        """Test no contradiction check on contradictory state."""
        assert check_no_contradiction(contradictory_state) is False

    def test_check_confidence_bounds_valid(self, valid_state: SystemState) -> None:
        """Test confidence bounds on valid state."""
        assert check_confidence_bounds(valid_state) is True

    def test_check_confidence_bounds_invalid(
        self, invalid_confidence_state: SystemState
    ) -> None:
        """Test confidence bounds on invalid state."""
        assert check_confidence_bounds(invalid_confidence_state) is False

    def test_check_confidence_bounds_negative(self) -> None:
        """Test confidence bounds with negative value."""
        state = SystemState(
            rules=[Rule(rule_id="r1", confidence=-0.5)],
            facts=[],
            metadata={},
            timestamp=0.0,
        )
        assert check_confidence_bounds(state) is False

    def test_check_confidence_bounds_none(self) -> None:
        """Test confidence bounds with None value."""
        state = SystemState(
            rules=[Rule(rule_id="r1", confidence=None)],
            facts=[],
            metadata={},
            timestamp=0.0,
        )
        assert check_confidence_bounds(state) is True

    def test_check_party_neutrality_valid(self, valid_state: SystemState) -> None:
        """Test party neutrality on valid state."""
        assert check_party_neutrality(valid_state) is True

    def test_check_party_neutrality_invalid(
        self, asymmetric_neutral_rule_state: SystemState
    ) -> None:
        """Test party neutrality on asymmetric neutral rule."""
        assert check_party_neutrality(asymmetric_neutral_rule_state) is False

    def test_check_rollback_available_valid(self, valid_state: SystemState) -> None:
        """Test rollback availability on valid state."""
        assert check_rollback_available(valid_state) is True

    def test_check_rollback_available_invalid(
        self, no_rollback_state: SystemState
    ) -> None:
        """Test rollback availability on state without rollback."""
        assert check_rollback_available(no_rollback_state) is False

    def test_check_rollback_initial_state(self) -> None:
        """Test rollback availability for initial state."""
        state = SystemState(
            rules=[],
            facts=[],
            metadata={"is_initial_state": True},
            timestamp=0.0,
        )
        assert check_rollback_available(state) is True

    def test_check_stratification_valid(self) -> None:
        """Test valid stratification."""
        state = SystemState(
            rules=[
                Rule(rule_id="r1", body="a(X)", head="b(X)", stratification_level=0),
                Rule(
                    rule_id="r2", body="not b(X)", head="c(X)", stratification_level=1
                ),
            ],
            facts=[],
            metadata={},
            timestamp=0.0,
        )
        assert check_stratification(state) is True

    def test_check_stratification_invalid(self) -> None:
        """Test invalid stratification (same level negative dependency)."""
        state = SystemState(
            rules=[
                Rule(rule_id="r1", body="a(X)", head="b(X)", stratification_level=0),
                Rule(
                    rule_id="r2",
                    body="not b(X)",
                    head="c(X)",
                    stratification_level=0,  # Same level - invalid!
                ),
            ],
            facts=[],
            metadata={},
            timestamp=0.0,
        )
        assert check_stratification(state) is False

    def test_check_rule_monotonicity_valid(self) -> None:
        """Test rule monotonicity on valid state."""
        state = SystemState(
            rules=[Rule(rule_id="r1")],
            facts=[],
            metadata={"deprecated_rules": ["r2"]},  # r2 deprecated, not active
            timestamp=0.0,
        )
        assert check_rule_monotonicity(state) is True

    def test_check_rule_monotonicity_invalid(self) -> None:
        """Test rule monotonicity when deprecated rule still active."""
        state = SystemState(
            rules=[Rule(rule_id="r1")],
            facts=[],
            metadata={"deprecated_rules": ["r1"]},  # r1 marked deprecated but active!
            timestamp=0.0,
        )
        assert check_rule_monotonicity(state) is False

    def test_check_query_termination_depth_limit(self) -> None:
        """Test query termination with depth limit."""
        state = SystemState(
            rules=[],
            facts=[],
            metadata={"max_query_depth": 100},
            timestamp=0.0,
        )
        assert check_query_termination(state) is True

    def test_check_query_termination_timeout(self) -> None:
        """Test query termination with timeout."""
        state = SystemState(
            rules=[],
            facts=[],
            metadata={"query_timeout_ms": 5000},
            timestamp=0.0,
        )
        assert check_query_termination(state) is True

    def test_check_query_termination_missing(self) -> None:
        """Test query termination without limits."""
        state = SystemState(
            rules=[],
            facts=[],
            metadata={},
            timestamp=0.0,
        )
        assert check_query_termination(state) is False

    def test_check_explainability_valid(self) -> None:
        """Test explainability with explained outcomes."""
        state = SystemState(
            rules=[],
            facts=[],
            metadata={
                "recent_outcomes": [
                    {"result": True, "explanation": "Rule applied"},
                    {"result": False, "explanation": "No matching rule"},
                ]
            },
            timestamp=0.0,
        )
        assert check_explainability(state) is True

    def test_check_explainability_missing(self) -> None:
        """Test explainability with unexplained outcome."""
        state = SystemState(
            rules=[],
            facts=[],
            metadata={
                "recent_outcomes": [
                    {"result": True, "explanation": "Rule applied"},
                    {"result": False},  # No explanation!
                ]
            },
            timestamp=0.0,
        )
        assert check_explainability(state) is False

    def test_check_explainability_no_outcomes(self) -> None:
        """Test explainability with no outcomes."""
        state = SystemState(
            rules=[],
            facts=[],
            metadata={},
            timestamp=0.0,
        )
        assert check_explainability(state) is True


# ============================================================================
# ConstitutionalProperty Tests
# ============================================================================


class TestConstitutionalProperty:
    """Tests for ConstitutionalProperty class."""

    def test_property_creation(self) -> None:
        """Test creating a constitutional property."""
        prop = ConstitutionalProperty(
            name="TEST_PROP",
            description="A test property",
            property_type=PropertyType.LOGICAL,
            formal_spec="∀x: test(x)",
            checker=lambda s: True,
        )
        assert prop.name == "TEST_PROP"
        assert prop.property_type == PropertyType.LOGICAL

    def test_verify_runtime_passes(self, valid_state: SystemState) -> None:
        """Test runtime verification that passes."""
        prop = ConstitutionalProperty(
            name="TEST",
            description="Test",
            property_type=PropertyType.LOGICAL,
            formal_spec="",
            checker=lambda s: True,
        )
        assert prop.verify_runtime(valid_state) is True

    def test_verify_runtime_fails(self, valid_state: SystemState) -> None:
        """Test runtime verification that fails."""
        prop = ConstitutionalProperty(
            name="TEST",
            description="Test",
            property_type=PropertyType.LOGICAL,
            formal_spec="",
            checker=lambda s: False,
        )
        assert prop.verify_runtime(valid_state) is False

    def test_verify_runtime_exception(self, valid_state: SystemState) -> None:
        """Test runtime verification with exception."""

        def raise_error(s: SystemState) -> bool:
            raise ValueError("Test error")

        prop = ConstitutionalProperty(
            name="TEST",
            description="Test",
            property_type=PropertyType.LOGICAL,
            formal_spec="",
            checker=raise_error,
        )
        # Should return False, not raise
        assert prop.verify_runtime(valid_state) is False


# ============================================================================
# ConstitutionalVerifier Tests
# ============================================================================


class TestConstitutionalVerifier:
    """Tests for ConstitutionalVerifier class."""

    def test_verify_property_valid(
        self, standard_verifier: ConstitutionalVerifier, valid_state: SystemState
    ) -> None:
        """Test verifying a property on valid state."""
        result = standard_verifier.verify_property("NO_CONTRADICTION", valid_state)
        assert result.result == VerificationResult.VERIFIED

    def test_verify_property_invalid(
        self,
        standard_verifier: ConstitutionalVerifier,
        contradictory_state: SystemState,
    ) -> None:
        """Test verifying a property on invalid state."""
        result = standard_verifier.verify_property(
            "NO_CONTRADICTION", contradictory_state
        )
        assert result.result == VerificationResult.FALSIFIED

    def test_verify_property_unknown(
        self, standard_verifier: ConstitutionalVerifier, valid_state: SystemState
    ) -> None:
        """Test verifying unknown property."""
        result = standard_verifier.verify_property("NONEXISTENT", valid_state)
        assert result.result == VerificationResult.UNKNOWN

    def test_verify_all_valid(
        self, standard_verifier: ConstitutionalVerifier, valid_state: SystemState
    ) -> None:
        """Test verifying all properties on valid state."""
        report = standard_verifier.verify_all(valid_state)
        assert report.all_verified is True
        assert len(report.get_violations()) == 0

    def test_verify_all_invalid(
        self,
        standard_verifier: ConstitutionalVerifier,
        contradictory_state: SystemState,
    ) -> None:
        """Test verifying all properties on invalid state."""
        report = standard_verifier.verify_all(contradictory_state)
        assert report.all_verified is False
        violations = report.get_violations()
        assert len(violations) > 0
        assert any(v.property_name == "NO_CONTRADICTION" for v in violations)

    def test_verify_transition_safe(
        self, standard_verifier: ConstitutionalVerifier, valid_state: SystemState
    ) -> None:
        """Test safe transition verification."""
        before = valid_state
        after = SystemState(
            rules=valid_state.rules + [Rule(rule_id="r3", confidence=0.7)],
            facts=valid_state.facts,
            metadata=valid_state.metadata,
            timestamp=2.0,
        )

        report = standard_verifier.verify_transition(before, after, "add_rule")
        assert report.transition_safe is True
        assert len(report.violations) == 0

    def test_verify_transition_violation(
        self, standard_verifier: ConstitutionalVerifier, valid_state: SystemState
    ) -> None:
        """Test transition that introduces violation."""
        before = valid_state
        after = SystemState(
            rules=valid_state.rules,
            facts=[Fact("p", negated=False), Fact("p", negated=True)],  # Contradiction
            metadata=valid_state.metadata,
            timestamp=2.0,
        )

        report = standard_verifier.verify_transition(
            before, after, "introduce_contradiction"
        )
        assert report.transition_safe is False
        assert len(report.violations) > 0
        assert any(v.property_name == "NO_CONTRADICTION" for v in report.violations)

    def test_verification_cache(
        self, standard_verifier: ConstitutionalVerifier, valid_state: SystemState
    ) -> None:
        """Test that verification results are cached."""
        # First call
        result1 = standard_verifier.verify_property("NO_CONTRADICTION", valid_state)

        # Second call should use cache
        result2 = standard_verifier.verify_property("NO_CONTRADICTION", valid_state)

        assert result1.result == result2.result
        assert result2.explanation == "Cached result"

    def test_clear_cache(
        self, standard_verifier: ConstitutionalVerifier, valid_state: SystemState
    ) -> None:
        """Test cache clearing."""
        # Populate cache
        standard_verifier.verify_property("NO_CONTRADICTION", valid_state)

        # Clear cache
        standard_verifier.clear_cache()

        # Should not use cache
        result = standard_verifier.verify_property("NO_CONTRADICTION", valid_state)
        assert result.explanation != "Cached result"

    def test_report_to_markdown(
        self, standard_verifier: ConstitutionalVerifier, valid_state: SystemState
    ) -> None:
        """Test markdown report generation."""
        report = standard_verifier.verify_all(valid_state)
        markdown = report.to_markdown()

        assert "Constitutional Verification Report" in markdown
        assert "All Verified" in markdown
        assert "✅" in markdown


# ============================================================================
# ConstitutionalGuard Tests
# ============================================================================


class TestConstitutionalGuard:
    """Tests for ConstitutionalGuard class."""

    def test_guard_allows_safe_operation(self, valid_state: SystemState) -> None:
        """Test guard allows safe operations."""
        guard = create_guard()

        def safe_operation(state: SystemState) -> SystemState:
            return SystemState(
                rules=state.rules + [Rule(rule_id="r3", confidence=0.7)],
                facts=state.facts,
                metadata=state.metadata,
                timestamp=state.timestamp + 1,
            )

        new_state, report = guard.guard(safe_operation, valid_state, "add_rule")

        assert report.transition_safe is True
        assert len(new_state.rules) == len(valid_state.rules) + 1

    def test_guard_blocks_unsafe_operation(self, valid_state: SystemState) -> None:
        """Test guard blocks unsafe operations."""
        guard = create_guard()

        def unsafe_operation(state: SystemState) -> SystemState:
            return SystemState(
                rules=state.rules,
                facts=[Fact("p", negated=False), Fact("p", negated=True)],
                metadata=state.metadata,
                timestamp=state.timestamp + 1,
            )

        new_state, report = guard.guard(
            unsafe_operation, valid_state, "introduce_contradiction"
        )

        assert report.transition_safe is False
        # Original state should be returned
        assert new_state == valid_state

    def test_guard_handles_exception(self, valid_state: SystemState) -> None:
        """Test guard handles operation exceptions."""
        guard = create_guard()

        def failing_operation(state: SystemState) -> SystemState:
            raise ValueError("Operation failed")

        new_state, report = guard.guard(failing_operation, valid_state, "failing_op")

        assert report.transition_safe is False
        assert new_state == valid_state
        assert len(report.violations) > 0

    def test_get_blocked_operations(self, valid_state: SystemState) -> None:
        """Test getting blocked operations history."""
        guard = create_guard()

        def unsafe_operation(state: SystemState) -> SystemState:
            return SystemState(
                rules=state.rules,
                facts=[Fact("p", negated=False), Fact("p", negated=True)],
                metadata=state.metadata,
                timestamp=state.timestamp + 1,
            )

        guard.guard(unsafe_operation, valid_state, "op1")
        guard.guard(unsafe_operation, valid_state, "op2")

        blocked = guard.get_blocked_operations()
        assert len(blocked) == 2

    def test_clear_history(self, valid_state: SystemState) -> None:
        """Test clearing blocked operations history."""
        guard = create_guard()

        def unsafe_operation(state: SystemState) -> SystemState:
            return SystemState(
                rules=state.rules,
                facts=[Fact("p", negated=False), Fact("p", negated=True)],
                metadata=state.metadata,
                timestamp=state.timestamp + 1,
            )

        guard.guard(unsafe_operation, valid_state, "op1")
        guard.clear_history()

        assert len(guard.get_blocked_operations()) == 0


# ============================================================================
# Standard Properties Tests
# ============================================================================


class TestStandardProperties:
    """Tests for standard constitutional properties."""

    def test_all_standard_properties_exist(self) -> None:
        """Test all 8 standard properties are defined."""
        props = create_standard_properties()
        assert len(props) == 8

        names = {p.name for p in props}
        assert "NO_CONTRADICTION" in names
        assert "RULE_MONOTONICITY" in names
        assert "PARTY_NEUTRALITY" in names
        assert "QUERY_TERMINATION" in names
        assert "EXPLAINABILITY" in names
        assert "CONFIDENCE_BOUNDS" in names
        assert "STRATIFICATION_VALID" in names
        assert "ROLLBACK_AVAILABLE" in names

    def test_standard_properties_have_descriptions(self) -> None:
        """Test all standard properties have descriptions."""
        props = create_standard_properties()
        for prop in props:
            assert prop.description
            assert len(prop.description) > 10

    def test_standard_properties_have_formal_specs(self) -> None:
        """Test all standard properties have formal specifications."""
        props = create_standard_properties()
        for prop in props:
            assert prop.formal_spec
            assert len(prop.formal_spec) > 5

    def test_standard_properties_have_types(self) -> None:
        """Test all standard properties have property types."""
        props = create_standard_properties()
        type_counts: Dict[PropertyType, int] = {}
        for prop in props:
            type_counts[prop.property_type] = type_counts.get(prop.property_type, 0) + 1

        # Should have multiple types represented
        assert len(type_counts) >= 3


# ============================================================================
# Integration Tests
# ============================================================================


class TestConstitutionalIntegration:
    """Integration tests for constitutional verification."""

    def test_full_workflow(self) -> None:
        """Test complete verification workflow."""
        # Create initial valid state
        state = SystemState(
            rules=[Rule(rule_id="r1", confidence=0.8)],
            facts=[Fact("valid")],
            metadata={
                "max_query_depth": 100,
                "previous_state_id": "init",
                "recent_outcomes": [{"result": True, "explanation": "Test"}],
            },
            timestamp=0.0,
        )

        # Create guard
        guard = create_guard()

        # Verify initial state
        verifier = guard.verifier
        initial_report = verifier.verify_all(state)
        assert initial_report.all_verified is True

        # Apply safe modification
        def add_valid_rule(s: SystemState) -> SystemState:
            return SystemState(
                rules=s.rules + [Rule(rule_id="r2", confidence=0.9)],
                facts=s.facts,
                metadata=s.metadata,
                timestamp=s.timestamp + 1,
            )

        new_state, report = guard.guard(add_valid_rule, state, "add_rule")
        assert report.transition_safe is True

        # Try unsafe modification
        def add_invalid_rule(s: SystemState) -> SystemState:
            return SystemState(
                rules=s.rules
                + [Rule(rule_id="r3", confidence=2.0)],  # Invalid confidence
                facts=s.facts,
                metadata=s.metadata,
                timestamp=s.timestamp + 1,
            )

        blocked_state, blocked_report = guard.guard(
            add_invalid_rule, new_state, "add_invalid_rule"
        )
        assert blocked_report.transition_safe is False
        assert blocked_state == new_state  # Original state preserved

    def test_report_generation(self, valid_state: SystemState) -> None:
        """Test verification report generation."""
        verifier = create_verifier()
        report = verifier.verify_all(valid_state)

        # Test markdown generation
        markdown = report.to_markdown()
        assert "## Constitutional Verification Report" in markdown
        assert "NO_CONTRADICTION" in markdown or "Verification" in markdown

    def test_transition_report_generation(self, valid_state: SystemState) -> None:
        """Test transition report generation."""
        verifier = create_verifier()

        after = SystemState(
            rules=valid_state.rules,
            facts=[Fact("p", negated=False), Fact("p", negated=True)],
            metadata=valid_state.metadata,
            timestamp=2.0,
        )

        report = verifier.verify_transition(valid_state, after, "test_op")

        markdown = report.to_markdown()
        assert "## Transition Verification Report" in markdown
        assert "test_op" in markdown

    def test_multiple_violations(self) -> None:
        """Test detection of multiple violations."""
        state = SystemState(
            rules=[
                Rule(rule_id="r1", confidence=1.5),  # Invalid confidence
                Rule(
                    rule_id="r2",
                    confidence=0.8,
                    metadata={"neutral": True},
                    body="plaintiff(X)",  # Asymmetric neutral rule
                    head="result(X)",
                ),
            ],
            facts=[
                Fact("p", negated=False),
                Fact("p", negated=True),  # Contradiction
            ],
            metadata={},  # No rollback, no query limits
            timestamp=0.0,
        )

        verifier = create_verifier()
        report = verifier.verify_all(state)

        violations = report.get_violations()
        assert len(violations) >= 3  # At least 3 violations
