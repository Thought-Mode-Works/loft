"""
Unit tests for O(d)-Equivariance implementation.

Tests cover:
- TransformationType enum
- PartyPermutationEquivariance
- AmountScalingEquivariance
- ContentSubstitutionEquivariance
- EquivarianceVerifier
- EquivarianceReport
"""

import pytest
from typing import Any, Dict

from loft.constraints.equivariance import (
    TransformationType,
    EquivarianceViolation,
    EquivarianceReport,
    PartyPermutationEquivariance,
    AmountScalingEquivariance,
    ContentSubstitutionEquivariance,
    EquivarianceVerifier,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_contract_case() -> Dict[str, Any]:
    """Simple contract case with two parties."""
    return {
        "plaintiff": "alice",
        "defendant": "bob",
        "amount": 600,
        "has_offer": True,
        "has_acceptance": True,
    }


@pytest.fixture
def complex_contract_case() -> Dict[str, Any]:
    """More complex case with nested structures."""
    return {
        "parties": {
            "buyer": "alice",
            "seller": "bob",
        },
        "contract": {
            "amount": 1000,
            "description": "Sale of widgets",
            "id": "CONTRACT-123",
        },
        "terms": {
            "price": 1000,
            "quantity": 10,
        },
    }


@pytest.fixture
def multi_party_case() -> Dict[str, Any]:
    """Case with multiple parties."""
    return {
        "plaintiff": "alice",
        "defendant": "bob",
        "guarantor": "charlie",
        "amount": 500,
    }


@pytest.fixture
def test_cases(
    simple_contract_case: Dict[str, Any],
    complex_contract_case: Dict[str, Any],
) -> list:
    """Collection of test cases."""
    return [simple_contract_case, complex_contract_case]


# =============================================================================
# Test TransformationType
# =============================================================================


class TestTransformationType:
    """Tests for TransformationType enum."""

    def test_transformation_types_exist(self):
        """Verify all expected transformation types exist."""
        assert TransformationType.PARTY_PERMUTATION is not None
        assert TransformationType.AMOUNT_SCALING is not None
        assert TransformationType.TEMPORAL_SHIFT is not None
        assert TransformationType.CONTENT_SUBSTITUTION is not None

    def test_transformation_types_are_distinct(self):
        """Verify transformation types are distinct."""
        types = [
            TransformationType.PARTY_PERMUTATION,
            TransformationType.AMOUNT_SCALING,
            TransformationType.TEMPORAL_SHIFT,
            TransformationType.CONTENT_SUBSTITUTION,
        ]
        assert len(set(types)) == len(types)


# =============================================================================
# Test EquivarianceViolation
# =============================================================================


class TestEquivarianceViolation:
    """Tests for EquivarianceViolation dataclass."""

    def test_violation_creation(self):
        """Test creating an equivariance violation."""
        violation = EquivarianceViolation(
            constraint_name="party_permutation",
            transformation_type=TransformationType.PARTY_PERMUTATION,
            original_case={"plaintiff": "alice"},
            transformed_case={"plaintiff": "bob"},
            transformation_params={"party_mapping": {"alice": "bob"}},
            original_output={"valid": True},
            transformed_output={"valid": False},
            expected_output={"valid": True},
            explanation="Output changed when it shouldn't have",
        )
        assert violation.constraint_name == "party_permutation"
        assert violation.transformation_type == TransformationType.PARTY_PERMUTATION

    def test_violation_to_dict(self):
        """Test converting violation to dictionary."""
        violation = EquivarianceViolation(
            constraint_name="party_permutation",
            transformation_type=TransformationType.PARTY_PERMUTATION,
            original_case={"plaintiff": "alice"},
            transformed_case={"plaintiff": "bob"},
            transformation_params={"party_mapping": {"alice": "bob"}},
            original_output={"valid": True},
            transformed_output={"valid": False},
            expected_output={"valid": True},
            explanation="Output changed",
        )
        result = violation.to_dict()
        assert isinstance(result, dict)
        assert result["constraint_name"] == "party_permutation"
        assert result["transformation_type"] == "PARTY_PERMUTATION"


# =============================================================================
# Test EquivarianceReport
# =============================================================================


class TestEquivarianceReport:
    """Tests for EquivarianceReport dataclass."""

    def test_report_is_equivariant_when_no_failures(self):
        """Test that report shows equivariant when no failures."""
        report = EquivarianceReport(
            rule_name="test_rule",
            total_tests=10,
            passed_tests=10,
            failed_tests=[],
            constraint_breakdown={},
        )
        assert report.is_equivariant is True
        assert report.equivariance_score == 1.0

    def test_report_not_equivariant_with_failures(self):
        """Test that report shows non-equivariant with failures."""
        violation = EquivarianceViolation(
            constraint_name="test",
            transformation_type=TransformationType.PARTY_PERMUTATION,
            original_case={},
            transformed_case={},
            transformation_params={},
            original_output={},
            transformed_output={},
            expected_output={},
            explanation="failed",
        )
        report = EquivarianceReport(
            rule_name="test_rule",
            total_tests=10,
            passed_tests=8,
            failed_tests=[violation],
            constraint_breakdown={},
        )
        assert report.is_equivariant is False
        assert report.equivariance_score == 0.8

    def test_report_empty_tests(self):
        """Test report with no tests."""
        report = EquivarianceReport(
            rule_name="test_rule",
            total_tests=0,
            passed_tests=0,
            failed_tests=[],
            constraint_breakdown={},
        )
        assert report.is_equivariant is True
        assert report.equivariance_score == 1.0

    def test_report_to_markdown(self):
        """Test markdown generation."""
        report = EquivarianceReport(
            rule_name="test_rule",
            total_tests=10,
            passed_tests=10,
            failed_tests=[],
            constraint_breakdown={"party_permutation": {"passed": 5, "total": 5}},
        )
        markdown = report.to_markdown()
        assert "test_rule" in markdown
        assert "✅ Equivariant" in markdown
        assert "100.0%" in markdown


# =============================================================================
# Test PartyPermutationEquivariance
# =============================================================================


class TestPartyPermutationEquivariance:
    """Tests for PartyPermutationEquivariance constraint."""

    def test_party_detection(self, simple_contract_case: Dict[str, Any]):
        """Test that parties are correctly detected."""
        constraint = PartyPermutationEquivariance()
        parties = constraint._detect_parties(simple_contract_case)
        assert "alice" in parties
        assert "bob" in parties

    def test_party_detection_nested(self, complex_contract_case: Dict[str, Any]):
        """Test party detection in nested structures."""
        constraint = PartyPermutationEquivariance()
        parties = constraint._detect_parties(complex_contract_case)
        assert "alice" in parties
        assert "bob" in parties

    def test_apply_transform_simple(self, simple_contract_case: Dict[str, Any]):
        """Test applying party permutation transformation."""
        constraint = PartyPermutationEquivariance()
        params = {"party_mapping": {"alice": "bob", "bob": "alice"}}
        transformed = constraint.apply_transform(simple_contract_case, params)

        assert transformed["plaintiff"] == "bob"
        assert transformed["defendant"] == "alice"
        # Non-party fields unchanged
        assert transformed["amount"] == 600

    def test_apply_transform_no_mapping(self, simple_contract_case: Dict[str, Any]):
        """Test that empty mapping returns original case."""
        constraint = PartyPermutationEquivariance()
        transformed = constraint.apply_transform(simple_contract_case, {})
        assert transformed == simple_contract_case

    def test_generate_transformations(self, simple_contract_case: Dict[str, Any]):
        """Test transformation generation for two parties."""
        constraint = PartyPermutationEquivariance()
        transformations = constraint.generate_transformations(simple_contract_case)

        # For 2 parties, should generate 1 transformation (swap)
        assert len(transformations) == 1
        assert "party_mapping" in transformations[0]

    def test_generate_transformations_multi_party(self, multi_party_case: Dict[str, Any]):
        """Test transformation generation for multiple parties."""
        constraint = PartyPermutationEquivariance()
        transformations = constraint.generate_transformations(multi_party_case)

        # For 3 parties, should generate 5 transformations (3! - 1)
        assert len(transformations) == 5

    def test_generate_transformations_single_party(self):
        """Test that single party case generates no transformations."""
        constraint = PartyPermutationEquivariance()
        case = {"plaintiff": "alice", "amount": 500}
        transformations = constraint.generate_transformations(case)
        assert len(transformations) == 0

    def test_verify_equivariant_rule(self, simple_contract_case: Dict[str, Any]):
        """Test verification of an equivariant rule."""
        constraint = PartyPermutationEquivariance()

        # Content-neutral rule: doesn't depend on party names
        def equivariant_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "valid": case.get("has_offer") and case.get("has_acceptance"),
                "amount": case.get("amount"),
            }

        params = {"party_mapping": {"alice": "bob", "bob": "alice"}}
        is_equivariant, violation = constraint.verify_equivariance(
            equivariant_rule, simple_contract_case, params
        )

        assert is_equivariant is True
        assert violation is None

    def test_verify_non_equivariant_rule(self, simple_contract_case: Dict[str, Any]):
        """Test verification catches non-equivariant rule."""
        constraint = PartyPermutationEquivariance()

        # Non-equivariant rule: depends on specific party name
        def non_equivariant_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "valid": case.get("plaintiff") == "alice",
            }

        params = {"party_mapping": {"alice": "bob", "bob": "alice"}}
        is_equivariant, violation = constraint.verify_equivariance(
            non_equivariant_rule, simple_contract_case, params
        )

        assert is_equivariant is False
        assert violation is not None
        assert violation.constraint_name == "party_permutation"

    def test_transform_output(self):
        """Test that output is transformed correctly."""
        constraint = PartyPermutationEquivariance()
        output = {"winner": "alice", "loser": "bob", "valid": True}
        params = {"party_mapping": {"alice": "bob", "bob": "alice"}}

        transformed = constraint.transform_output(output, params)
        assert transformed["winner"] == "bob"
        assert transformed["loser"] == "alice"
        assert transformed["valid"] is True

    def test_custom_party_fields(self):
        """Test using custom party field names."""
        constraint = PartyPermutationEquivariance(party_fields={"custom_party_a", "custom_party_b"})
        case = {
            "custom_party_a": "alice",
            "custom_party_b": "bob",
            "plaintiff": "charlie",  # Not in custom fields
        }
        parties = constraint._detect_parties(case)
        assert "alice" in parties
        assert "bob" in parties
        assert "charlie" not in parties


# =============================================================================
# Test AmountScalingEquivariance
# =============================================================================


class TestAmountScalingEquivariance:
    """Tests for AmountScalingEquivariance constraint."""

    def test_amount_detection(self, simple_contract_case: Dict[str, Any]):
        """Test amount field detection."""
        constraint = AmountScalingEquivariance()
        amounts = constraint._detect_amounts(simple_contract_case)
        assert "amount" in amounts
        assert amounts["amount"] == 600

    def test_amount_detection_nested(self, complex_contract_case: Dict[str, Any]):
        """Test amount detection in nested structures."""
        constraint = AmountScalingEquivariance()
        amounts = constraint._detect_amounts(complex_contract_case)
        assert "contract.amount" in amounts
        assert "terms.price" in amounts

    def test_apply_scaling(self, simple_contract_case: Dict[str, Any]):
        """Test applying amount scaling."""
        constraint = AmountScalingEquivariance()
        params = {"scale_factor": 2.0}
        transformed = constraint.apply_transform(simple_contract_case, params)

        assert transformed["amount"] == 1200
        # Non-amount fields unchanged
        assert transformed["plaintiff"] == "alice"

    def test_apply_scaling_factor_one(self, simple_contract_case: Dict[str, Any]):
        """Test that scale factor 1.0 returns original."""
        constraint = AmountScalingEquivariance()
        params = {"scale_factor": 1.0}
        transformed = constraint.apply_transform(simple_contract_case, params)
        assert transformed == simple_contract_case

    def test_generate_transformations_preserving_threshold(self):
        """Test transformation generation preserves threshold relationships."""
        constraint = AmountScalingEquivariance(thresholds=[500.0])

        # Case with amount above threshold
        case = {"amount": 600}
        transformations = constraint.generate_transformations(case)

        # Should include scale factors that keep amount above 500
        for params in transformations:
            scaled_amount = 600 * params["scale_factor"]
            # Original is above 500, scaled should also be above 500
            original_above = 600 >= 500
            scaled_above = scaled_amount >= 500
            assert original_above == scaled_above

    def test_generate_transformations_no_amounts(self):
        """Test that case without amounts generates no transformations."""
        constraint = AmountScalingEquivariance()
        case = {"plaintiff": "alice", "valid": True}
        transformations = constraint.generate_transformations(case)
        assert len(transformations) == 0

    def test_verify_equivariant_threshold_rule(self):
        """Test verification of threshold-based rule (invariant under scaling)."""
        constraint = AmountScalingEquivariance(thresholds=[500.0])

        # Rule that checks if amount >= 500 (threshold-based, scaling-invariant)
        def threshold_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            amount = case.get("amount", 0)
            return {"requires_writing": amount >= 500}

        case = {"amount": 600}  # Above threshold
        # Scale by 2 keeps it above threshold
        params = {"scale_factor": 2.0}

        is_equivariant, violation = constraint.verify_equivariance(threshold_rule, case, params)

        # Should be equivariant since both original and scaled are above threshold
        assert is_equivariant is True

    def test_custom_amount_fields(self):
        """Test using custom amount field names."""
        constraint = AmountScalingEquivariance(amount_fields={"custom_value", "special_price"})
        case = {"custom_value": 100, "amount": 500}

        amounts = constraint._detect_amounts(case)
        assert "custom_value" in amounts
        assert "amount" not in amounts


# =============================================================================
# Test ContentSubstitutionEquivariance
# =============================================================================


class TestContentSubstitutionEquivariance:
    """Tests for ContentSubstitutionEquivariance constraint."""

    def test_apply_substitution(self):
        """Test applying content substitution."""
        constraint = ContentSubstitutionEquivariance()
        case = {
            "id": "CONTRACT-123",
            "description": "Sale of widgets",
            "amount": 500,
        }
        params = {
            "content_mapping": {
                "CONTRACT-123": "CONTENT_0",
                "Sale of widgets": "CONTENT_1",
            }
        }

        transformed = constraint.apply_transform(case, params)
        assert transformed["id"] == "CONTENT_0"
        assert transformed["description"] == "CONTENT_1"
        assert transformed["amount"] == 500  # Non-content field unchanged

    def test_generate_transformations(self):
        """Test content substitution transformation generation."""
        constraint = ContentSubstitutionEquivariance()
        case = {"id": "ABC-123", "description": "Test item", "amount": 100}

        transformations = constraint.generate_transformations(case)
        assert len(transformations) == 1
        assert "content_mapping" in transformations[0]

    def test_exclude_fields(self):
        """Test that excluded fields are not substituted."""
        constraint = ContentSubstitutionEquivariance(exclude_fields={"legal_term"})
        case = {
            "id": "CONTRACT-123",
            "legal_term": "consideration",  # Should not be substituted
        }
        params = {
            "content_mapping": {
                "CONTRACT-123": "CONTENT_0",
                "consideration": "CONTENT_1",
            }
        }

        transformed = constraint.apply_transform(case, params)
        assert transformed["id"] == "CONTENT_0"
        assert transformed["legal_term"] == "consideration"  # Preserved

    def test_verify_equivariant_rule(self):
        """Test verification of content-neutral rule."""
        constraint = ContentSubstitutionEquivariance()

        # Content-neutral rule: doesn't depend on specific content values
        def content_neutral_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            return {"has_id": "id" in case and case["id"] is not None}

        case = {"id": "CONTRACT-123", "description": "Test"}
        params = {"content_mapping": {"CONTRACT-123": "X", "Test": "Y"}}

        is_equivariant, violation = constraint.verify_equivariance(
            content_neutral_rule, case, params
        )
        assert is_equivariant is True


# =============================================================================
# Test EquivarianceVerifier
# =============================================================================


class TestEquivarianceVerifier:
    """Tests for EquivarianceVerifier class."""

    def test_verify_rule_equivariant(self, test_cases: list):
        """Test verification of fully equivariant rule."""
        verifier = EquivarianceVerifier([PartyPermutationEquivariance()])

        # Fully equivariant rule
        def equivariant_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            return {"valid": True}

        report = verifier.verify_rule(equivariant_rule, test_cases)
        assert report.is_equivariant is True
        assert report.equivariance_score == 1.0

    def test_verify_rule_non_equivariant(self, simple_contract_case: Dict[str, Any]):
        """Test verification catches non-equivariant rule."""
        verifier = EquivarianceVerifier([PartyPermutationEquivariance()])

        # Non-equivariant rule
        def non_equivariant_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            return {"valid": case.get("plaintiff") == "alice"}

        report = verifier.verify_rule(non_equivariant_rule, [simple_contract_case])
        assert report.is_equivariant is False
        assert len(report.failed_tests) > 0

    def test_verify_rule_multiple_constraints(self, simple_contract_case: Dict[str, Any]):
        """Test verification with multiple constraints."""
        verifier = EquivarianceVerifier(
            [
                PartyPermutationEquivariance(),
                AmountScalingEquivariance(),
            ]
        )

        def equivariant_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            return {"valid": True}

        report = verifier.verify_rule(equivariant_rule, [simple_contract_case])
        assert report.is_equivariant is True
        assert "party_permutation" in report.constraint_breakdown
        assert "amount_scaling" in report.constraint_breakdown

    def test_verify_rules_multiple(self, simple_contract_case: Dict[str, Any]):
        """Test verifying multiple rules at once."""
        verifier = EquivarianceVerifier([PartyPermutationEquivariance()])

        rules = [
            ("equivariant", lambda c: {"valid": True}),
            ("non_equivariant", lambda c: {"valid": c.get("plaintiff") == "alice"}),
        ]

        reports = verifier.verify_rules(rules, [simple_contract_case])

        assert len(reports) == 2
        assert reports["equivariant"].is_equivariant is True
        assert reports["non_equivariant"].is_equivariant is False

    def test_generate_transformed_cases(self, simple_contract_case: Dict[str, Any]):
        """Test generating all transformed versions of a case."""
        verifier = EquivarianceVerifier(
            [
                PartyPermutationEquivariance(),
                AmountScalingEquivariance(thresholds=[500.0]),
            ]
        )

        transformed = verifier.generate_transformed_cases(simple_contract_case)
        assert len(transformed) > 0

        # Check structure
        for case, transform_type, params in transformed:
            assert isinstance(case, dict)
            assert isinstance(transform_type, TransformationType)
            assert isinstance(params, dict)

    def test_verify_rule_empty_test_cases(self):
        """Test verification with empty test cases."""
        verifier = EquivarianceVerifier([PartyPermutationEquivariance()])

        def rule(case: Dict[str, Any]) -> Dict[str, Any]:
            return {"valid": True}

        report = verifier.verify_rule(rule, [])
        assert report.is_equivariant is True
        assert report.total_tests == 0

    def test_verify_rule_custom_name(self, simple_contract_case: Dict[str, Any]):
        """Test verification with custom rule name."""
        verifier = EquivarianceVerifier([PartyPermutationEquivariance()])

        report = verifier.verify_rule(
            lambda c: {"valid": True},
            [simple_contract_case],
            rule_name="my_custom_rule",
        )
        assert report.rule_name == "my_custom_rule"


# =============================================================================
# Integration Tests
# =============================================================================


class TestEquivarianceIntegration:
    """Integration tests for equivariance module."""

    def test_full_verification_workflow(self):
        """Test complete verification workflow."""
        # Create test cases
        test_cases = [
            {
                "plaintiff": "alice",
                "defendant": "bob",
                "amount": 600,
                "has_offer": True,
                "has_acceptance": True,
                "description": "Sale of goods",
            },
            {
                "plaintiff": "charlie",
                "defendant": "david",
                "amount": 1000,
                "has_offer": True,
                "has_acceptance": False,
                "description": "Service agreement",
            },
        ]

        # Create verifier with multiple constraints
        verifier = EquivarianceVerifier(
            [
                PartyPermutationEquivariance(),
                AmountScalingEquivariance(thresholds=[500.0]),
                ContentSubstitutionEquivariance(),
            ]
        )

        # Define an equivariant rule
        def contract_validity_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            has_offer = case.get("has_offer", False)
            has_acceptance = case.get("has_acceptance", False)
            amount = case.get("amount", 0)

            return {
                "valid": has_offer and has_acceptance,
                "requires_writing": amount >= 500,
            }

        # Verify
        report = verifier.verify_rule(
            contract_validity_rule,
            test_cases,
            rule_name="contract_validity",
        )

        # Should be equivariant
        assert report.is_equivariant is True
        assert report.rule_name == "contract_validity"

        # Check breakdown
        assert "party_permutation" in report.constraint_breakdown
        assert "amount_scaling" in report.constraint_breakdown
        assert "content_substitution" in report.constraint_breakdown

    def test_threshold_crossing_not_tested(self):
        """Verify that transformations crossing thresholds are not generated."""
        constraint = AmountScalingEquivariance(thresholds=[500.0])

        # Case at threshold boundary
        case = {"amount": 500}
        transformations = constraint.generate_transformations(case)

        # Scale factors should only be those that don't cross threshold
        for params in transformations:
            scaled = 500 * params["scale_factor"]
            # Original is at threshold (>=500 is True)
            # Scaled should also satisfy same threshold relationship
            assert (scaled >= 500) == (500 >= 500)

    def test_report_markdown_with_violations(self):
        """Test markdown report generation with violations."""
        verifier = EquivarianceVerifier([PartyPermutationEquivariance()])

        # Non-equivariant rule: checks for specific party name "alice"
        # This is non-equivariant because swapping parties changes validity
        def bad_rule(case: Dict[str, Any]) -> Dict[str, Any]:
            # Rule that favors "alice" as plaintiff - NOT content-neutral
            is_alice_plaintiff = case.get("plaintiff") == "alice"
            return {"valid": is_alice_plaintiff, "reason": "alice_favored"}

        case = {"plaintiff": "alice", "defendant": "bob"}
        report = verifier.verify_rule(bad_rule, [case], rule_name="bad_rule")

        markdown = report.to_markdown()
        assert "bad_rule" in markdown
        assert "❌ Non-equivariant" in markdown
        assert "Violations" in markdown
