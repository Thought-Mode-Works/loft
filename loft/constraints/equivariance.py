"""
O(d)-Equivariance Implementation for Content-Neutrality.

This module implements equivariance constraints to ensure legal rules are
content-neutral - the outcome should depend on the legal structure of a case,
not on arbitrary content labels like party names, specific dollar amounts,
or irrelevant identifiers.

Mathematical Foundation:
-----------------------
A function f: X → Y is O(d)-equivariant if for all orthogonal transformations g ∈ O(d):
    f(g · x) = g · f(x)

For legal reasoning, this translates to:
- Party permutation equivariance: Swapping parties doesn't change legal relationships
- Amount scaling equivariance: Proportional scaling preserves legal thresholds
- Content substitution equivariance: Replacing arbitrary labels preserves analysis

Content-neutrality is formalized as equivariance under content-permutation groups:
1. Party permutation group S_n acts on n-party contracts
2. Amount scaling group R+ acts on monetary values
3. Content substitution acts on arbitrary string labels

A rule is content-neutral iff it commutes with all these group actions.

Example:
-------
    >>> from loft.constraints.equivariance import (
    ...     EquivarianceVerifier,
    ...     PartyPermutationEquivariance
    ... )
    >>> verifier = EquivarianceVerifier([PartyPermutationEquivariance()])
    >>> rule = lambda case: {"valid": case.get("has_offer") and case.get("has_acceptance")}
    >>> report = verifier.verify_rule(rule, [{"plaintiff": "alice", "defendant": "bob", "has_offer": True, "has_acceptance": True}])
    >>> report.is_equivariant
    True
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple
import copy
import itertools

from loguru import logger


class TransformationType(Enum):
    """Types of equivariant transformations for legal reasoning."""

    PARTY_PERMUTATION = auto()  # Swap party identities
    AMOUNT_SCALING = auto()  # Scale monetary values proportionally
    TEMPORAL_SHIFT = (
        auto()
    )  # Shift dates by constant offset (implemented in temporal.py)
    CONTENT_SUBSTITUTION = auto()  # Replace content labels with equivalent ones


@dataclass
class EquivarianceViolation:
    """Records a specific equivariance violation found during verification."""

    constraint_name: str
    transformation_type: TransformationType
    original_case: Dict[str, Any]
    transformed_case: Dict[str, Any]
    transformation_params: Dict[str, Any]
    original_output: Any
    transformed_output: Any
    expected_output: Any
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary for serialization."""
        return {
            "constraint_name": self.constraint_name,
            "transformation_type": self.transformation_type.name,
            "original_case": self.original_case,
            "transformed_case": self.transformed_case,
            "transformation_params": self.transformation_params,
            "original_output": str(self.original_output),
            "transformed_output": str(self.transformed_output),
            "expected_output": str(self.expected_output),
            "explanation": self.explanation,
        }


@dataclass
class EquivarianceReport:
    """Report on equivariance verification results for a rule."""

    rule_name: str
    total_tests: int
    passed_tests: int
    failed_tests: List[EquivarianceViolation] = field(default_factory=list)
    constraint_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def is_equivariant(self) -> bool:
        """Check if rule passed all equivariance tests."""
        return len(self.failed_tests) == 0

    @property
    def equivariance_score(self) -> float:
        """Calculate equivariance score as fraction of passed tests."""
        if self.total_tests == 0:
            return 1.0
        return self.passed_tests / self.total_tests

    def to_markdown(self) -> str:
        """Generate markdown summary of verification results."""
        lines = [
            f"## Equivariance Report: {self.rule_name}",
            "",
            f"**Status**: {'✅ Equivariant' if self.is_equivariant else '❌ Non-equivariant'}",
            f"**Score**: {self.equivariance_score:.1%} ({self.passed_tests}/{self.total_tests} tests passed)",
            "",
        ]

        if self.constraint_breakdown:
            lines.append("### Breakdown by Constraint")
            lines.append("")
            for constraint_name, breakdown in self.constraint_breakdown.items():
                status = (
                    "✅"
                    if breakdown.get("passed", 0) == breakdown.get("total", 0)
                    else "❌"
                )
                lines.append(
                    f"- {status} **{constraint_name}**: "
                    f"{breakdown.get('passed', 0)}/{breakdown.get('total', 0)} passed"
                )
            lines.append("")

        if self.failed_tests:
            lines.append("### Violations")
            lines.append("")
            for i, violation in enumerate(self.failed_tests[:5], 1):  # Show first 5
                lines.append(f"**Violation {i}**: {violation.explanation}")
                lines.append(f"- Constraint: {violation.constraint_name}")
                lines.append(f"- Transform: {violation.transformation_type.name}")
                lines.append("")

            if len(self.failed_tests) > 5:
                lines.append(f"*... and {len(self.failed_tests) - 5} more violations*")

        return "\n".join(lines)


class EquivarianceConstraint(ABC):
    """
    Abstract base class for equivariance constraints.

    An equivariance constraint defines:
    1. A transformation type (how to modify cases)
    2. How to apply the transformation to a case
    3. How to transform the expected output
    4. How to verify equivariance for a given rule
    """

    def __init__(
        self,
        name: str,
        transformation_type: TransformationType,
        description: str,
    ):
        self.name = name
        self.transformation_type = transformation_type
        self.description = description

    @abstractmethod
    def apply_transform(
        self,
        case: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply transformation to a case.

        Args:
            case: The original case dictionary
            params: Transformation parameters (e.g., party_mapping for permutation)

        Returns:
            Transformed case dictionary
        """
        pass

    @abstractmethod
    def transform_output(
        self,
        output: Any,
        params: Dict[str, Any],
    ) -> Any:
        """
        Transform expected output according to the same transformation.

        For equivariance, if we transform input by g, we expect output to
        transform by g as well: f(g·x) = g·f(x)

        Args:
            output: The original rule output
            params: Same transformation parameters used for input

        Returns:
            Expected transformed output
        """
        pass

    @abstractmethod
    def generate_transformations(
        self,
        case: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate transformation parameters for testing a case.

        Args:
            case: The case to generate transformations for

        Returns:
            List of transformation parameter dictionaries
        """
        pass

    def verify_equivariance(
        self,
        rule: Callable[[Dict[str, Any]], Any],
        case: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[bool, Optional[EquivarianceViolation]]:
        """
        Verify rule produces equivariant output for a specific transformation.

        Args:
            rule: Function that takes a case and produces output
            case: Original case
            params: Transformation parameters

        Returns:
            (is_equivariant, violation_if_any)
        """
        try:
            # Get original output
            original_output = rule(case)

            # Transform case
            transformed_case = self.apply_transform(case, params)

            # Get transformed output
            transformed_output = rule(transformed_case)

            # Get expected output (original output transformed)
            expected_output = self.transform_output(original_output, params)

            # Check if outputs match
            if self._outputs_equivalent(transformed_output, expected_output):
                return True, None
            else:
                violation = EquivarianceViolation(
                    constraint_name=self.name,
                    transformation_type=self.transformation_type,
                    original_case=case,
                    transformed_case=transformed_case,
                    transformation_params=params,
                    original_output=original_output,
                    transformed_output=transformed_output,
                    expected_output=expected_output,
                    explanation=(
                        f"Rule output changed under {self.transformation_type.name} "
                        f"when it should have been invariant or equivariant. "
                        f"Expected {expected_output}, got {transformed_output}"
                    ),
                )
                return False, violation

        except Exception as e:
            logger.warning(f"Error during equivariance verification: {e}")
            violation = EquivarianceViolation(
                constraint_name=self.name,
                transformation_type=self.transformation_type,
                original_case=case,
                transformed_case={},
                transformation_params=params,
                original_output=None,
                transformed_output=None,
                expected_output=None,
                explanation=f"Exception during verification: {str(e)}",
            )
            return False, violation

    def _outputs_equivalent(self, output1: Any, output2: Any) -> bool:
        """
        Check if two outputs are equivalent.

        Override for custom equivalence logic.
        """
        if output1 == output2:
            return True

        # Handle dict comparison with tolerance for nested structures
        if isinstance(output1, dict) and isinstance(output2, dict):
            if set(output1.keys()) != set(output2.keys()):
                return False
            for key in output1:
                if not self._outputs_equivalent(output1[key], output2[key]):
                    return False
            return True

        # Handle numeric comparison with tolerance
        if isinstance(output1, (int, float)) and isinstance(output2, (int, float)):
            return abs(output1 - output2) < 1e-9

        return False


class PartyPermutationEquivariance(EquivarianceConstraint):
    """
    Ensures party swapping doesn't change legal analysis.

    For symmetric legal relationships, swapping party identities should not
    change the legal outcome. This constraint tests that rules are invariant
    under party permutations.

    Example:
        If contract_valid(alice, bob) is true, then contract_valid(bob, alice)
        should also be true for symmetric contracts.
    """

    # Common party-related field names to detect
    PARTY_FIELD_PATTERNS: FrozenSet[str] = frozenset(
        {
            "plaintiff",
            "defendant",
            "buyer",
            "seller",
            "lessor",
            "lessee",
            "promisor",
            "promisee",
            "offeror",
            "offeree",
            "party",
            "party1",
            "party2",
            "party_a",
            "party_b",
            "grantor",
            "grantee",
            "assignor",
            "assignee",
            "licensor",
            "licensee",
            "principal",
            "agent",
            "employer",
            "employee",
            "guarantor",
            "beneficiary",
            "trustee",
            "settlor",
        }
    )

    def __init__(self, party_fields: Optional[Set[str]] = None):
        """
        Initialize party permutation equivariance constraint.

        Args:
            party_fields: Optional set of field names that contain party references.
                         If None, will auto-detect based on common patterns.
        """
        super().__init__(
            name="party_permutation",
            transformation_type=TransformationType.PARTY_PERMUTATION,
            description="Swapping party identities preserves legal relationships",
        )
        self.party_fields = party_fields

    def _detect_parties(self, case: Dict[str, Any]) -> Set[str]:
        """
        Detect unique party values in a case.

        Args:
            case: Case dictionary

        Returns:
            Set of unique party values found
        """
        parties: Set[str] = set()
        self._scan_for_parties(case, parties)
        return parties

    def _scan_for_parties(
        self, obj: Any, found: Set[str], current_key: str = ""
    ) -> None:
        """Recursively scan for party values."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                is_party_field = (
                    self.party_fields is not None and key in self.party_fields
                ) or (
                    self.party_fields is None
                    and key.lower() in self.PARTY_FIELD_PATTERNS
                )

                if is_party_field and isinstance(value, str):
                    found.add(value)
                else:
                    self._scan_for_parties(value, found, key)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._scan_for_parties(item, found, current_key)

    def apply_transform(
        self,
        case: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Swap party identities in case facts.

        Args:
            case: Original case
            params: Must contain 'party_mapping' dict mapping old->new party names

        Returns:
            Case with party names swapped
        """
        party_mapping = params.get("party_mapping", {})
        if not party_mapping:
            return case

        return self._apply_mapping_recursive(copy.deepcopy(case), party_mapping)

    def _apply_mapping_recursive(
        self,
        obj: Any,
        mapping: Dict[str, str],
    ) -> Any:
        """Recursively apply party mapping to object."""
        if isinstance(obj, str):
            return mapping.get(obj, obj)
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                # Check if this is a party field
                is_party_field = (
                    self.party_fields is not None and key in self.party_fields
                ) or (
                    self.party_fields is None
                    and key.lower() in self.PARTY_FIELD_PATTERNS
                )

                if is_party_field and isinstance(value, str):
                    result[key] = mapping.get(value, value)
                else:
                    result[key] = self._apply_mapping_recursive(value, mapping)
            return result
        elif isinstance(obj, list):
            return [self._apply_mapping_recursive(item, mapping) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._apply_mapping_recursive(item, mapping) for item in obj)
        else:
            return obj

    def transform_output(
        self,
        output: Any,
        params: Dict[str, Any],
    ) -> Any:
        """
        Transform output according to party permutation.

        For most legal rules, the output structure should be invariant
        (same validity, same reasons) but party references in output
        should be transformed.
        """
        party_mapping = params.get("party_mapping", {})
        if not party_mapping:
            return output

        return self._apply_mapping_recursive(copy.deepcopy(output), party_mapping)

    def generate_transformations(
        self,
        case: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate party permutation transformations for a case.

        For n parties, generates all n! permutations (excluding identity).
        For large n, limits to pairwise swaps for efficiency.
        """
        parties = list(self._detect_parties(case))

        if len(parties) < 2:
            return []  # Need at least 2 parties to permute

        transformations = []

        if len(parties) <= 4:
            # For small party count, test all permutations
            for perm in itertools.permutations(parties):
                if perm != tuple(parties):  # Exclude identity
                    mapping = dict(zip(parties, perm))
                    transformations.append({"party_mapping": mapping})
        else:
            # For larger party counts, just test pairwise swaps
            for i in range(len(parties)):
                for j in range(i + 1, len(parties)):
                    mapping = {parties[i]: parties[j], parties[j]: parties[i]}
                    transformations.append({"party_mapping": mapping})

        return transformations


class AmountScalingEquivariance(EquivarianceConstraint):
    """
    Ensures proportional scaling of amounts preserves threshold relationships.

    Legal rules often have amount thresholds (e.g., $500 for UCC statute of frauds).
    This constraint verifies that scaling all amounts proportionally doesn't
    change the legal analysis - what matters is whether amounts are above/below
    thresholds, not their absolute values.

    Note: This only tests invariance for scaling that preserves threshold
    relationships. Scaling that crosses thresholds is expected to change outcomes.
    """

    # Common amount-related field patterns
    AMOUNT_FIELD_PATTERNS: FrozenSet[str] = frozenset(
        {
            "amount",
            "price",
            "value",
            "cost",
            "payment",
            "consideration",
            "damages",
            "fee",
            "rent",
            "salary",
            "sum",
            "total",
        }
    )

    def __init__(
        self,
        amount_fields: Optional[Set[str]] = None,
        thresholds: Optional[List[float]] = None,
    ):
        """
        Initialize amount scaling equivariance constraint.

        Args:
            amount_fields: Field names containing monetary amounts
            thresholds: Known legal thresholds to preserve (e.g., [500] for UCC)
        """
        super().__init__(
            name="amount_scaling",
            transformation_type=TransformationType.AMOUNT_SCALING,
            description="Proportional scaling preserves legal threshold relationships",
        )
        self.amount_fields = amount_fields
        self.thresholds = thresholds or [500.0]  # Default UCC threshold

    def _detect_amounts(self, case: Dict[str, Any]) -> Dict[str, float]:
        """Detect amount fields and their values in a case."""
        amounts: Dict[str, float] = {}
        self._scan_for_amounts(case, "", amounts)
        return amounts

    def _scan_for_amounts(
        self,
        obj: Any,
        path: str,
        found: Dict[str, float],
    ) -> None:
        """Recursively scan for amount values."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                is_amount_field = (
                    self.amount_fields is not None and key in self.amount_fields
                ) or (
                    self.amount_fields is None
                    and key.lower() in self.AMOUNT_FIELD_PATTERNS
                )

                if is_amount_field and isinstance(value, (int, float)):
                    found[new_path] = float(value)
                else:
                    self._scan_for_amounts(value, new_path, found)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                self._scan_for_amounts(item, f"{path}[{i}]", found)

    def apply_transform(
        self,
        case: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Scale amounts in case by given factor.

        Args:
            case: Original case
            params: Must contain 'scale_factor' (float > 0)

        Returns:
            Case with amounts scaled
        """
        scale_factor = params.get("scale_factor", 1.0)
        if scale_factor == 1.0:
            return case

        return self._apply_scaling_recursive(copy.deepcopy(case), scale_factor)

    def _apply_scaling_recursive(
        self,
        obj: Any,
        scale_factor: float,
    ) -> Any:
        """Recursively apply scaling to amount fields."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                is_amount_field = (
                    self.amount_fields is not None and key in self.amount_fields
                ) or (
                    self.amount_fields is None
                    and key.lower() in self.AMOUNT_FIELD_PATTERNS
                )

                if is_amount_field and isinstance(value, (int, float)):
                    result[key] = value * scale_factor
                else:
                    result[key] = self._apply_scaling_recursive(value, scale_factor)
            return result
        elif isinstance(obj, list):
            return [self._apply_scaling_recursive(item, scale_factor) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(
                self._apply_scaling_recursive(item, scale_factor) for item in obj
            )
        else:
            return obj

    def transform_output(
        self,
        output: Any,
        params: Dict[str, Any],
    ) -> Any:
        """
        Transform output for amount scaling.

        For threshold-based rules, the output structure (valid/invalid, reasons)
        should be invariant when scaling preserves threshold relationships.
        Amount values in output should be scaled.
        """
        scale_factor = params.get("scale_factor", 1.0)
        if scale_factor == 1.0:
            return output

        # Scale any amounts in output, but preserve structure
        return self._apply_scaling_recursive(copy.deepcopy(output), scale_factor)

    def generate_transformations(
        self,
        case: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate scaling transformations that preserve threshold relationships.

        Only generates scalings where all amounts stay on the same side of
        all thresholds after scaling.
        """
        amounts = self._detect_amounts(case)
        if not amounts:
            return []

        transformations = []

        # Test various scale factors
        test_factors = [0.5, 0.9, 1.1, 2.0]

        for factor in test_factors:
            # Check if this scaling preserves threshold relationships
            preserves_thresholds = True
            for amount in amounts.values():
                for threshold in self.thresholds:
                    original_side = amount >= threshold
                    scaled_side = (amount * factor) >= threshold
                    if original_side != scaled_side:
                        preserves_thresholds = False
                        break
                if not preserves_thresholds:
                    break

            if preserves_thresholds:
                transformations.append({"scale_factor": factor})

        return transformations


class ContentSubstitutionEquivariance(EquivarianceConstraint):
    """
    Ensures arbitrary content labels can be substituted without changing analysis.

    Legal rules should not depend on specific string values that are legally
    irrelevant (e.g., contract ID, reference numbers, product names in generic
    sale of goods).

    This constraint verifies that substituting arbitrary content with
    equivalent placeholders doesn't change the legal outcome.
    """

    # Fields that are typically content-neutral
    CONTENT_FIELD_PATTERNS: FrozenSet[str] = frozenset(
        {
            "id",
            "reference",
            "description",
            "name",
            "title",
            "product",
            "item",
            "goods",
            "subject",
            "address",
            "location",
        }
    )

    def __init__(
        self,
        content_fields: Optional[Set[str]] = None,
        exclude_fields: Optional[Set[str]] = None,
    ):
        """
        Initialize content substitution equivariance constraint.

        Args:
            content_fields: Fields containing substitutable content
            exclude_fields: Fields to exclude from substitution (e.g., legal terms)
        """
        super().__init__(
            name="content_substitution",
            transformation_type=TransformationType.CONTENT_SUBSTITUTION,
            description="Substituting arbitrary content labels preserves legal analysis",
        )
        self.content_fields = content_fields
        self.exclude_fields = exclude_fields or set()

    def apply_transform(
        self,
        case: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Substitute content values in case.

        Args:
            case: Original case
            params: Must contain 'content_mapping' dict mapping old->new content

        Returns:
            Case with content substituted
        """
        content_mapping = params.get("content_mapping", {})
        if not content_mapping:
            return case

        return self._apply_substitution_recursive(copy.deepcopy(case), content_mapping)

    def _apply_substitution_recursive(
        self,
        obj: Any,
        mapping: Dict[str, str],
    ) -> Any:
        """Recursively apply content substitution."""
        if isinstance(obj, str):
            return mapping.get(obj, obj)
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key in self.exclude_fields:
                    result[key] = value
                    continue

                is_content_field = (
                    self.content_fields is not None and key in self.content_fields
                ) or (
                    self.content_fields is None
                    and key.lower() in self.CONTENT_FIELD_PATTERNS
                )

                if is_content_field and isinstance(value, str):
                    result[key] = mapping.get(value, value)
                else:
                    result[key] = self._apply_substitution_recursive(value, mapping)
            return result
        elif isinstance(obj, list):
            return [self._apply_substitution_recursive(item, mapping) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(
                self._apply_substitution_recursive(item, mapping) for item in obj
            )
        else:
            return obj

    def transform_output(
        self,
        output: Any,
        params: Dict[str, Any],
    ) -> Any:
        """
        Transform output for content substitution.

        Output structure should be invariant; content values should be substituted.
        """
        content_mapping = params.get("content_mapping", {})
        if not content_mapping:
            return output

        return self._apply_substitution_recursive(
            copy.deepcopy(output), content_mapping
        )

    def generate_transformations(
        self,
        case: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate content substitution transformations.

        Creates a mapping that replaces all content values with placeholder strings.
        """
        content_values: Set[str] = set()
        self._scan_for_content(case, content_values)

        if not content_values:
            return []

        # Create substitution mapping
        mapping = {}
        for i, value in enumerate(sorted(content_values)):
            mapping[value] = f"CONTENT_{i}"

        return [{"content_mapping": mapping}]

    def _scan_for_content(
        self, obj: Any, found: Set[str], current_key: str = ""
    ) -> None:
        """Recursively scan for content values."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in self.exclude_fields:
                    continue

                is_content_field = (
                    self.content_fields is not None and key in self.content_fields
                ) or (
                    self.content_fields is None
                    and key.lower() in self.CONTENT_FIELD_PATTERNS
                )

                if is_content_field and isinstance(value, str):
                    found.add(value)
                else:
                    self._scan_for_content(value, found, key)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._scan_for_content(item, found, current_key)


class EquivarianceVerifier:
    """
    Verifies O(d)-equivariance properties of legal rules.

    This class orchestrates equivariance verification by:
    1. Accepting a list of equivariance constraints
    2. For each test case, generating transformations
    3. Verifying rule output is equivariant under each transformation
    4. Producing a comprehensive report

    Example:
        >>> verifier = EquivarianceVerifier([
        ...     PartyPermutationEquivariance(),
        ...     AmountScalingEquivariance()
        ... ])
        >>> report = verifier.verify_rule(my_rule, test_cases)
        >>> print(report.is_equivariant)
    """

    def __init__(self, constraints: List[EquivarianceConstraint]):
        """
        Initialize verifier with equivariance constraints.

        Args:
            constraints: List of equivariance constraints to verify
        """
        self.constraints = constraints

    def verify_rule(
        self,
        rule: Callable[[Dict[str, Any]], Any],
        test_cases: List[Dict[str, Any]],
        rule_name: Optional[str] = None,
    ) -> EquivarianceReport:
        """
        Verify a rule satisfies all equivariance constraints.

        Args:
            rule: Function that takes a case dict and returns output
            test_cases: List of test cases to verify against
            rule_name: Optional name for the rule (for reporting)

        Returns:
            EquivarianceReport with verification results
        """
        rule_name = rule_name or getattr(rule, "__name__", "anonymous_rule")

        total_tests = 0
        passed_tests = 0
        failed_tests: List[EquivarianceViolation] = []
        constraint_breakdown: Dict[str, Dict[str, Any]] = {}

        for constraint in self.constraints:
            constraint_total = 0
            constraint_passed = 0
            constraint_failures: List[EquivarianceViolation] = []

            for case in test_cases:
                transformations = constraint.generate_transformations(case)

                for params in transformations:
                    constraint_total += 1
                    total_tests += 1

                    is_equivariant, violation = constraint.verify_equivariance(
                        rule, case, params
                    )

                    if is_equivariant:
                        constraint_passed += 1
                        passed_tests += 1
                    else:
                        if violation:
                            constraint_failures.append(violation)
                            failed_tests.append(violation)

            constraint_breakdown[constraint.name] = {
                "total": constraint_total,
                "passed": constraint_passed,
                "failed": len(constraint_failures),
                "transformation_type": constraint.transformation_type.name,
            }

        return EquivarianceReport(
            rule_name=rule_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            constraint_breakdown=constraint_breakdown,
        )

    def verify_rules(
        self,
        rules: List[Tuple[str, Callable[[Dict[str, Any]], Any]]],
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, EquivarianceReport]:
        """
        Verify multiple rules against equivariance constraints.

        Args:
            rules: List of (name, rule_function) tuples
            test_cases: Test cases to verify against

        Returns:
            Dictionary mapping rule names to their reports
        """
        reports = {}
        for name, rule in rules:
            reports[name] = self.verify_rule(rule, test_cases, rule_name=name)
        return reports

    def generate_transformed_cases(
        self,
        case: Dict[str, Any],
    ) -> List[Tuple[Dict[str, Any], TransformationType, Dict[str, Any]]]:
        """
        Generate all transformed versions of a case for testing.

        Args:
            case: Original case

        Returns:
            List of (transformed_case, transformation_type, params) tuples
        """
        transformed_cases = []

        for constraint in self.constraints:
            transformations = constraint.generate_transformations(case)
            for params in transformations:
                transformed = constraint.apply_transform(case, params)
                transformed_cases.append(
                    (transformed, constraint.transformation_type, params)
                )

        return transformed_cases
