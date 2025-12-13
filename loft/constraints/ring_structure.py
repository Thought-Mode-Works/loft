"""
Ring Algebraic Structure for Compositional Rule Combination (Phase 7).

This module implements a ring algebraic structure for composing legal rules,
enabling principled combination with well-defined mathematical properties.
The ring structure allows proving that certain properties are preserved
under rule composition.

Ring Structure Definition:
    A ring (R, +, ·, 0, 1) satisfies:
    1. (R, +, 0) is an abelian group (addition)
    2. (R, ·, 1) is a monoid (multiplication)
    3. Multiplication distributes over addition: a·(b+c) = a·b + a·c

For legal rules:
    - Addition (+): Disjunction (OR)
    - Multiplication (·): Conjunction/sequence (AND, THEN)
    - Zero (0): Always-false rule
    - One (1): Always-true rule (tautology)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Generic, TypeVar
from abc import ABC, abstractmethod


T = TypeVar("T")


class RingElement(ABC, Generic[T]):
    """
    Abstract base for elements in a rule ring.

    Must support addition (disjunction) and multiplication (conjunction).
    """

    @abstractmethod
    def __add__(self, other: "RingElement[T]") -> "RingElement[T]":
        """Disjunction: self OR other."""
        pass

    @abstractmethod
    def __mul__(self, other: "RingElement[T]") -> "RingElement[T]":
        """Conjunction: self AND other."""
        pass

    @abstractmethod
    def evaluate(self, case: Dict[str, Any]) -> T:
        """Evaluate the rule on a case."""
        pass

    @abstractmethod
    def is_zero(self) -> bool:
        """Check if this is the zero element (always false)."""
        pass

    @abstractmethod
    def is_one(self) -> bool:
        """Check if this is the one element (always true)."""
        pass


@dataclass
class BooleanRule(RingElement[bool]):
    """
    A legal rule in the boolean ring.

    The boolean ring ({true, false}, XOR, AND) is the simplest rule ring.
    We use (OR, AND) which forms a bounded distributive lattice.
    """

    predicate: Callable[[Dict[str, Any]], bool]
    name: str = ""
    _is_zero: bool = False
    _is_one: bool = False

    def __add__(self, other: "BooleanRule") -> "BooleanRule":
        """Disjunction (OR)."""
        if self.is_zero():
            return other
        if other.is_zero():
            return self
        if self.is_one() or other.is_one():
            return BooleanRule.one()

        # Capture self and other in closure
        self_pred = self.predicate
        other_pred = other.predicate
        return BooleanRule(
            predicate=lambda c, s=self_pred, o=other_pred: s(c) or o(c),
            name=f"({self.name} ∨ {other.name})",
        )

    def __mul__(self, other: "BooleanRule") -> "BooleanRule":
        """Conjunction (AND)."""
        if self.is_zero() or other.is_zero():
            return BooleanRule.zero()
        if self.is_one():
            return other
        if other.is_one():
            return self

        # Capture predicates in closure
        self_pred = self.predicate
        other_pred = other.predicate
        return BooleanRule(
            predicate=lambda c, s=self_pred, o=other_pred: s(c) and o(c),
            name=f"({self.name} ∧ {other.name})",
        )

    def __neg__(self) -> "BooleanRule":
        """Negation (NOT)."""
        if self.is_zero():
            return BooleanRule.one()
        if self.is_one():
            return BooleanRule.zero()

        self_pred = self.predicate
        return BooleanRule(
            predicate=lambda c, s=self_pred: not s(c),
            name=f"¬{self.name}",
        )

    def evaluate(self, case: Dict[str, Any]) -> bool:
        """Evaluate the rule on a case."""
        return self.predicate(case)

    def is_zero(self) -> bool:
        """Check if this is the zero element (always false)."""
        return self._is_zero

    def is_one(self) -> bool:
        """Check if this is the one element (always true)."""
        return self._is_one

    @classmethod
    def zero(cls) -> "BooleanRule":
        """The always-false rule (additive identity)."""
        return cls(predicate=lambda c: False, name="⊥", _is_zero=True)

    @classmethod
    def one(cls) -> "BooleanRule":
        """The always-true rule (multiplicative identity)."""
        return cls(predicate=lambda c: True, name="⊤", _is_one=True)


@dataclass
class ConfidenceRule(RingElement[float]):
    """
    A rule in the confidence ring [0, 1].

    Uses:
    - Addition: max(a, b) (optimistic combination)
    - Multiplication: a * b (independent conjunction)

    This forms a semiring structure commonly used in probabilistic reasoning.
    """

    evaluator: Callable[[Dict[str, Any]], float]
    name: str = ""
    _is_zero: bool = False
    _is_one: bool = False

    def __add__(self, other: "ConfidenceRule") -> "ConfidenceRule":
        """Optimistic combination: max confidence."""
        if self._is_zero:
            return other
        if other._is_zero:
            return self
        if self._is_one or other._is_one:
            return ConfidenceRule.one()

        self_eval = self.evaluator
        other_eval = other.evaluator
        return ConfidenceRule(
            evaluator=lambda c, s=self_eval, o=other_eval: max(s(c), o(c)),
            name=f"max({self.name}, {other.name})",
        )

    def __mul__(self, other: "ConfidenceRule") -> "ConfidenceRule":
        """Independent conjunction: product of confidences."""
        if self._is_zero or other._is_zero:
            return ConfidenceRule.zero()
        if self._is_one:
            return other
        if other._is_one:
            return self

        self_eval = self.evaluator
        other_eval = other.evaluator
        return ConfidenceRule(
            evaluator=lambda c, s=self_eval, o=other_eval: s(c) * o(c),
            name=f"({self.name} × {other.name})",
        )

    def evaluate(self, case: Dict[str, Any]) -> float:
        """Evaluate the rule on a case."""
        return self.evaluator(case)

    def is_zero(self) -> bool:
        """Check if this is the zero element."""
        return self._is_zero

    def is_one(self) -> bool:
        """Check if this is the one element."""
        return self._is_one

    @classmethod
    def zero(cls) -> "ConfidenceRule":
        """The always-zero rule (additive identity)."""
        return cls(evaluator=lambda c: 0.0, name="0", _is_zero=True)

    @classmethod
    def one(cls) -> "ConfidenceRule":
        """The always-one rule (multiplicative identity)."""
        return cls(evaluator=lambda c: 1.0, name="1", _is_one=True)


class RuleComposition:
    """
    Utilities for composing rules while preserving ring properties.
    """

    @staticmethod
    def exception(base_rule: BooleanRule, exception: BooleanRule) -> BooleanRule:
        """
        Apply base_rule unless exception applies.

        Semantics: base_rule AND NOT exception
        """
        return base_rule * (-exception)

    @staticmethod
    def conditional(condition: BooleanRule, consequent: BooleanRule) -> BooleanRule:
        """
        If condition then consequent.

        Semantics: NOT condition OR consequent (material implication)
        """
        return (-condition) + consequent

    @staticmethod
    def sequence(first: BooleanRule, second: BooleanRule) -> BooleanRule:
        """
        Apply first, then second (conjunction in boolean ring).
        """
        return first * second

    @staticmethod
    def alternative(rule1: BooleanRule, rule2: BooleanRule) -> BooleanRule:
        """
        Either rule1 or rule2 (disjunction).
        """
        return rule1 + rule2


@dataclass
class HomomorphismViolation:
    """A violation of homomorphism properties."""

    property_name: str
    element_a: str
    element_b: str
    case: Dict[str, Any]
    expected: Any
    actual: Any


@dataclass
class HomomorphismVerificationReport:
    """Report on homomorphism verification."""

    is_homomorphism: bool
    violations: List[HomomorphismViolation] = field(default_factory=list)
    tests_run: int = 0
    tests_passed: int = 0


class RingHomomorphism:
    """
    A structure-preserving map between rule rings.

    Used to transform rules while preserving composition properties.
    """

    def __init__(
        self,
        source_ring: type,
        target_ring: type,
        map_fn: Callable[[RingElement], RingElement],
    ):
        self.source_ring = source_ring
        self.target_ring = target_ring
        self.map_fn = map_fn

    def apply(self, element: RingElement) -> RingElement:
        """Apply the homomorphism to a ring element."""
        return self.map_fn(element)

    def verify_homomorphism(
        self,
        test_elements: List[RingElement],
        test_cases: List[Dict[str, Any]],
    ) -> HomomorphismVerificationReport:
        """
        Verify that this map is actually a homomorphism.

        Checks:
        - φ(a + b) = φ(a) + φ(b)
        - φ(a · b) = φ(a) · φ(b)
        """
        violations: List[HomomorphismViolation] = []
        tests_run = 0
        tests_passed = 0

        for a in test_elements:
            for b in test_elements:
                # Check addition preservation: φ(a + b) = φ(a) + φ(b)
                phi_sum = self.apply(a + b)
                sum_phi = self.apply(a) + self.apply(b)

                for case in test_cases:
                    tests_run += 1
                    phi_sum_val = phi_sum.evaluate(case)
                    sum_phi_val = sum_phi.evaluate(case)

                    # Handle float comparison for ConfidenceRule
                    if isinstance(phi_sum_val, float):
                        is_equal = abs(phi_sum_val - sum_phi_val) < 1e-9
                    else:
                        is_equal = phi_sum_val == sum_phi_val

                    if is_equal:
                        tests_passed += 1
                    else:
                        violations.append(
                            HomomorphismViolation(
                                property_name="addition",
                                element_a=getattr(a, "name", str(a)),
                                element_b=getattr(b, "name", str(b)),
                                case=case,
                                expected=phi_sum_val,
                                actual=sum_phi_val,
                            )
                        )

                # Check multiplication preservation: φ(a · b) = φ(a) · φ(b)
                phi_prod = self.apply(a * b)
                prod_phi = self.apply(a) * self.apply(b)

                for case in test_cases:
                    tests_run += 1
                    phi_prod_val = phi_prod.evaluate(case)
                    prod_phi_val = prod_phi.evaluate(case)

                    # Handle float comparison
                    if isinstance(phi_prod_val, float):
                        is_equal = abs(phi_prod_val - prod_phi_val) < 1e-9
                    else:
                        is_equal = phi_prod_val == prod_phi_val

                    if is_equal:
                        tests_passed += 1
                    else:
                        violations.append(
                            HomomorphismViolation(
                                property_name="multiplication",
                                element_a=getattr(a, "name", str(a)),
                                element_b=getattr(b, "name", str(b)),
                                case=case,
                                expected=phi_prod_val,
                                actual=prod_phi_val,
                            )
                        )

        return HomomorphismVerificationReport(
            is_homomorphism=len(violations) == 0,
            violations=violations,
            tests_run=tests_run,
            tests_passed=tests_passed,
        )


@dataclass
class RingVerificationReport:
    """Report on ring axiom verification."""

    distributivity_holds: bool
    associativity_add_holds: bool
    associativity_mul_holds: bool
    identity_holds: bool
    commutativity_add_holds: bool
    tests_run: int = 0
    tests_passed: int = 0
    violations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_ring(self) -> bool:
        """Check if all ring axioms hold."""
        return all(
            [
                self.distributivity_holds,
                self.associativity_add_holds,
                self.associativity_mul_holds,
                self.identity_holds,
            ]
        )


class RingPropertyVerifier:
    """
    Verifies that rule compositions satisfy ring axioms.
    """

    def __init__(self, test_cases: List[Dict[str, Any]]):
        self.test_cases = test_cases

    def _values_equal(self, v1: Any, v2: Any) -> bool:
        """Compare values, handling float comparisons."""
        if isinstance(v1, float) and isinstance(v2, float):
            return abs(v1 - v2) < 1e-9
        return v1 == v2

    def verify_distributivity(
        self,
        a: RingElement,
        b: RingElement,
        c: RingElement,
    ) -> bool:
        """
        Verify a · (b + c) = a·b + a·c
        """
        lhs = a * (b + c)
        rhs = (a * b) + (a * c)

        for case in self.test_cases:
            if not self._values_equal(lhs.evaluate(case), rhs.evaluate(case)):
                return False
        return True

    def verify_associativity_add(
        self,
        a: RingElement,
        b: RingElement,
        c: RingElement,
    ) -> bool:
        """Verify (a + b) + c = a + (b + c)."""
        lhs = (a + b) + c
        rhs = a + (b + c)

        return all(
            self._values_equal(lhs.evaluate(case), rhs.evaluate(case))
            for case in self.test_cases
        )

    def verify_associativity_mul(
        self,
        a: RingElement,
        b: RingElement,
        c: RingElement,
    ) -> bool:
        """Verify (a · b) · c = a · (b · c)."""
        lhs = (a * b) * c
        rhs = a * (b * c)

        return all(
            self._values_equal(lhs.evaluate(case), rhs.evaluate(case))
            for case in self.test_cases
        )

    def verify_commutativity_add(
        self,
        a: RingElement,
        b: RingElement,
    ) -> bool:
        """Verify a + b = b + a."""
        lhs = a + b
        rhs = b + a

        return all(
            self._values_equal(lhs.evaluate(case), rhs.evaluate(case))
            for case in self.test_cases
        )

    def verify_identity(self, a: RingElement, ring_class: type) -> bool:
        """Verify a + 0 = a and a · 1 = a."""
        zero = ring_class.zero()
        one = ring_class.one()

        return all(
            self._values_equal((a + zero).evaluate(case), a.evaluate(case))
            and self._values_equal((a * one).evaluate(case), a.evaluate(case))
            for case in self.test_cases
        )

    def verify_annihilation(self, a: RingElement, ring_class: type) -> bool:
        """Verify a · 0 = 0."""
        zero = ring_class.zero()
        product = a * zero

        return all(
            self._values_equal(product.evaluate(case), zero.evaluate(case))
            for case in self.test_cases
        )

    def full_verification(
        self,
        elements: List[RingElement],
        ring_class: type,
    ) -> RingVerificationReport:
        """Run all ring axiom verifications."""
        distributivity_results: List[bool] = []
        associativity_add_results: List[bool] = []
        associativity_mul_results: List[bool] = []
        commutativity_add_results: List[bool] = []
        identity_results: List[bool] = []
        violations: List[Dict[str, Any]] = []
        tests_run = 0
        tests_passed = 0

        for a in elements:
            # Test identity
            tests_run += 1
            identity_ok = self.verify_identity(a, ring_class)
            identity_results.append(identity_ok)
            if identity_ok:
                tests_passed += 1
            else:
                violations.append(
                    {"property": "identity", "element": getattr(a, "name", str(a))}
                )

            for b in elements:
                # Test commutativity of addition
                tests_run += 1
                comm_ok = self.verify_commutativity_add(a, b)
                commutativity_add_results.append(comm_ok)
                if comm_ok:
                    tests_passed += 1
                else:
                    violations.append(
                        {
                            "property": "commutativity_add",
                            "a": getattr(a, "name", str(a)),
                            "b": getattr(b, "name", str(b)),
                        }
                    )

                for c in elements:
                    # Test distributivity
                    tests_run += 1
                    dist_ok = self.verify_distributivity(a, b, c)
                    distributivity_results.append(dist_ok)
                    if dist_ok:
                        tests_passed += 1
                    else:
                        violations.append(
                            {
                                "property": "distributivity",
                                "a": getattr(a, "name", str(a)),
                                "b": getattr(b, "name", str(b)),
                                "c": getattr(c, "name", str(c)),
                            }
                        )

                    # Test associativity of addition
                    tests_run += 1
                    assoc_add_ok = self.verify_associativity_add(a, b, c)
                    associativity_add_results.append(assoc_add_ok)
                    if assoc_add_ok:
                        tests_passed += 1
                    else:
                        violations.append(
                            {
                                "property": "associativity_add",
                                "a": getattr(a, "name", str(a)),
                                "b": getattr(b, "name", str(b)),
                                "c": getattr(c, "name", str(c)),
                            }
                        )

                    # Test associativity of multiplication
                    tests_run += 1
                    assoc_mul_ok = self.verify_associativity_mul(a, b, c)
                    associativity_mul_results.append(assoc_mul_ok)
                    if assoc_mul_ok:
                        tests_passed += 1
                    else:
                        violations.append(
                            {
                                "property": "associativity_mul",
                                "a": getattr(a, "name", str(a)),
                                "b": getattr(b, "name", str(b)),
                                "c": getattr(c, "name", str(c)),
                            }
                        )

        return RingVerificationReport(
            distributivity_holds=all(distributivity_results),
            associativity_add_holds=all(associativity_add_results),
            associativity_mul_holds=all(associativity_mul_results),
            identity_holds=all(identity_results),
            commutativity_add_holds=all(commutativity_add_results),
            tests_run=tests_run,
            tests_passed=tests_passed,
            violations=violations,
        )


@dataclass
class ComposedRule:
    """
    A rule composed from multiple base rules with ring operations.

    Tracks composition history for explainability.
    """

    root: RingElement
    composition_tree: Dict[str, Any] = field(default_factory=dict)
    name: str = ""

    @classmethod
    def from_boolean_rule(cls, rule: BooleanRule) -> "ComposedRule":
        """Create a composed rule from a single boolean rule."""
        return cls(
            root=rule,
            composition_tree={"type": "leaf", "rule": rule.name},
            name=rule.name,
        )

    def conjoin(self, other: "ComposedRule") -> "ComposedRule":
        """Create conjunction of two composed rules."""
        new_root = self.root * other.root
        return ComposedRule(
            root=new_root,
            composition_tree={
                "type": "conjunction",
                "left": self.composition_tree,
                "right": other.composition_tree,
            },
            name=f"({self.name} ∧ {other.name})",
        )

    def disjoin(self, other: "ComposedRule") -> "ComposedRule":
        """Create disjunction of two composed rules."""
        new_root = self.root + other.root
        return ComposedRule(
            root=new_root,
            composition_tree={
                "type": "disjunction",
                "left": self.composition_tree,
                "right": other.composition_tree,
            },
            name=f"({self.name} ∨ {other.name})",
        )

    def evaluate(self, case: Dict[str, Any]) -> Any:
        """Evaluate the composed rule on a case."""
        return self.root.evaluate(case)

    def explain_composition(self) -> str:
        """Generate human-readable explanation of rule composition."""

        def _explain_tree(tree: Dict[str, Any], depth: int = 0) -> str:
            indent = "  " * depth
            if tree.get("type") == "leaf":
                return f"{indent}Rule: {tree.get('rule', 'unknown')}"
            elif tree.get("type") == "conjunction":
                left = _explain_tree(tree["left"], depth + 1)
                right = _explain_tree(tree["right"], depth + 1)
                return f"{indent}AND:\n{left}\n{right}"
            elif tree.get("type") == "disjunction":
                left = _explain_tree(tree["left"], depth + 1)
                right = _explain_tree(tree["right"], depth + 1)
                return f"{indent}OR:\n{left}\n{right}"
            return f"{indent}Unknown composition"

        return _explain_tree(self.composition_tree)
