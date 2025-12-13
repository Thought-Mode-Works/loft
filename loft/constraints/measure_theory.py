"""
Measure-Theoretic Representation of Legal Rules (Phase 7).

This module implements a measure-theoretic framework for legal rules, enabling
formal probability and confidence reasoning over legal outcomes. It defines
measurable spaces, functions, and probability distributions for legal cases.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum, auto


class MeasurableOutcome(Enum):
    """Possible outcomes of a legal rule."""

    TRUE = auto()  # Rule applies, conclusion holds
    FALSE = auto()  # Rule applies, conclusion fails
    UNDEFINED = auto()  # Rule does not apply (outside domain)


@dataclass
class CaseDimension:
    """A single dimension of the case space."""

    name: str
    dim_type: str  # "boolean" | "categorical" | "numeric" | "ordinal"
    categories: Optional[List[str]] = None  # For categorical
    bounds: Optional[Tuple[float, float]] = None  # For numeric

    def encode(self, value: Any) -> float:
        """Encode value to [0, 1] range."""
        if self.dim_type == "boolean":
            return 1.0 if value else 0.0
        elif self.dim_type == "categorical":
            if not self.categories:
                return 0.0
            try:
                return self.categories.index(value) / max(1, len(self.categories) - 1)
            except ValueError:
                return 0.0
        elif self.dim_type == "numeric":
            if not self.bounds:
                return float(value)
            lo, hi = self.bounds
            if hi == lo:
                return 0.0
            return (float(value) - lo) / (hi - lo)
        return 0.0

    def decode(self, encoded: float) -> Any:
        """Decode from [0, 1] back to original type."""
        if self.dim_type == "boolean":
            return encoded > 0.5
        elif self.dim_type == "categorical":
            if not self.categories:
                return None
            idx = int(round(encoded * (len(self.categories) - 1)))
            idx = max(0, min(idx, len(self.categories) - 1))
            return self.categories[idx]
        elif self.dim_type == "numeric":
            if not self.bounds:
                return encoded
            lo, hi = self.bounds
            return encoded * (hi - lo) + lo
        return encoded


@dataclass
class CaseSpace:
    """
    Defines the measurable space of legal cases.

    A case is a point in this space, specified by values for each dimension.
    """

    dimensions: List[CaseDimension]
    _dim_index: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self._dim_index = {d.name: i for i, d in enumerate(self.dimensions)}

    @property
    def ndim(self) -> int:
        return len(self.dimensions)

    def case_to_vector(self, case: Dict[str, Any]) -> np.ndarray:
        """Convert case dict to vector in case space."""
        vec = np.zeros(self.ndim)
        for dim in self.dimensions:
            if dim.name in case:
                vec[self._dim_index[dim.name]] = dim.encode(case[dim.name])
        return vec

    def vector_to_case(self, vec: np.ndarray) -> Dict[str, Any]:
        """Convert vector back to case dict."""
        return {dim.name: dim.decode(vec[i]) for i, dim in enumerate(self.dimensions)}


@dataclass
class MonomialPotential:
    """
    Represents a legal element combination as a monomial potential.

    φ(x) = ∏ᵢ xᵢ^αᵢ

    Properties:
    - φ(x) = 1 iff all elements fully satisfied (xᵢ = 1)
    - φ(x) = 0 if any required element is zero
    - Partial satisfaction gives φ(x) ∈ (0, 1)
    """

    elements: List[str]  # Element names
    weights: List[float]  # Exponents αᵢ
    element_indices: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.element_indices:
            self.element_indices = {e: i for i, e in enumerate(self.elements)}
        if len(self.weights) != len(self.elements):
            raise ValueError("Weights must match elements")

    def evaluate(self, satisfaction: Dict[str, float]) -> float:
        """
        Evaluate the monomial potential.

        Args:
            satisfaction: Dict mapping element names to satisfaction degree [0, 1]

        Returns:
            Potential value in [0, 1]
        """
        result = 1.0
        for elem, weight in zip(self.elements, self.weights):
            x = satisfaction.get(elem, 0.0)
            if x == 0 and weight > 0:
                return 0.0  # Short-circuit for required elements
            result *= x**weight
        return result

    def gradient(self, satisfaction: Dict[str, float]) -> Dict[str, float]:
        """Compute gradient of potential w.r.t. each element."""
        phi = self.evaluate(satisfaction)
        if phi == 0:
            return {e: 0.0 for e in self.elements}

        return {
            elem: self.weights[i] * phi / max(satisfaction.get(elem, 1e-10), 1e-10)
            for i, elem in enumerate(self.elements)
        }

    def most_limiting_element(self, satisfaction: Dict[str, float]) -> str:
        """Find element that most limits the potential."""
        grad = self.gradient(satisfaction)
        if not grad:
            return self.elements[0] if self.elements else ""
        return max(grad.keys(), key=lambda e: grad[e])


@dataclass
class CaseDistribution:
    """
    A probability distribution over case space.

    Can be empirical (from data) or specified (prior beliefs).
    """

    case_space: CaseSpace
    _samples: Optional[List[Dict[str, Any]]] = None
    _density: Optional[Callable[[Dict[str, Any]], float]] = None

    @classmethod
    def from_samples(
        cls, case_space: CaseSpace, samples: List[Dict[str, Any]]
    ) -> "CaseDistribution":
        """Create empirical distribution from case samples."""
        return cls(case_space=case_space, _samples=samples)

    @classmethod
    def uniform(cls, case_space: CaseSpace) -> "CaseDistribution":
        """Create uniform distribution over case space."""
        return cls(case_space=case_space, _density=lambda c: 1.0)

    def measure_set(self, measurable_set: "MeasurableSet") -> float:
        """Compute probability of a measurable set."""
        if self._samples is not None:
            # Empirical estimate
            if not self._samples:
                return 0.0
            count = sum(1 for s in self._samples if measurable_set.contains(s))
            return count / len(self._samples)
        else:
            # Monte Carlo integration (simplified placeholder)
            # In a real system, we'd sample from the space
            return 0.0

    def expected_value(self, f: Callable[[Dict[str, Any]], float]) -> float:
        """Compute E[f(X)] under this distribution."""
        if self._samples:
            return float(np.mean([f(s) for s in self._samples]))
        return 0.0


@dataclass
class MeasurableSet:
    """
    A measurable subset of case space.

    Defined by a characteristic function χ: CaseSpace → {0, 1}
    """

    case_space: CaseSpace
    characteristic: Callable[[Dict[str, Any]], bool]
    description: str = ""

    def contains(self, case: Dict[str, Any]) -> bool:
        """Check if case is in this set."""
        return self.characteristic(case)

    def measure(self, distribution: CaseDistribution) -> float:
        """Compute measure of this set under given distribution."""
        return distribution.measure_set(self)

    def intersect(self, other: "MeasurableSet") -> "MeasurableSet":
        """Intersection of two measurable sets."""
        return MeasurableSet(
            case_space=self.case_space,
            characteristic=lambda c: self.contains(c) and other.contains(c),
            description=f"({self.description}) ∩ ({other.description})",
        )

    def union(self, other: "MeasurableSet") -> "MeasurableSet":
        """Union of two measurable sets."""
        return MeasurableSet(
            case_space=self.case_space,
            characteristic=lambda c: self.contains(c) or other.contains(c),
            description=f"({self.description}) ∪ ({other.description})",
        )

    def complement(self) -> "MeasurableSet":
        """Complement of this set."""
        return MeasurableSet(
            case_space=self.case_space,
            characteristic=lambda c: not self.contains(c),
            description=f"¬({self.description})",
        )


class MeasurableLegalRule(ABC):
    """
    A legal rule as a measurable function.

    R: CaseSpace → {TRUE, FALSE, UNDEFINED}

    The preimage R⁻¹({TRUE}) is measurable (decidable).
    """

    @abstractmethod
    def evaluate(self, case: Dict[str, Any]) -> MeasurableOutcome:
        """Evaluate rule on a case."""
        pass

    @abstractmethod
    def domain(self) -> MeasurableSet:
        """Return the domain where rule is defined (not UNDEFINED)."""
        pass

    @abstractmethod
    def preimage_true(self) -> MeasurableSet:
        """Return set of cases where rule evaluates to TRUE."""
        pass


class RuleConfidenceCalculator:
    """
    Calculates confidence scores using measure-theoretic framework.
    """

    def __init__(self, case_space: CaseSpace, distribution: CaseDistribution):
        self.case_space = case_space
        self.distribution = distribution

    def rule_accuracy_probability(
        self, rule: MeasurableLegalRule, ground_truth: Callable[[Dict[str, Any]], bool]
    ) -> float:
        """
        Compute P(rule agrees with ground truth).

        This is μ({x : R(x) = truth(x)})
        """
        agreement_set = MeasurableSet(
            case_space=self.case_space,
            characteristic=lambda c: (
                (rule.evaluate(c) == MeasurableOutcome.TRUE) == ground_truth(c)
            ),
            description="Agreement Set",
        )
        return self.distribution.measure_set(agreement_set)

    def rule_coverage(self, rule: MeasurableLegalRule) -> float:
        """Compute P(rule is defined) = μ(domain(R))."""
        return self.distribution.measure_set(rule.domain())

    def conditional_accuracy(
        self,
        rule: MeasurableLegalRule,
        ground_truth: Callable[[Dict[str, Any]], bool],
        condition: MeasurableSet,
    ) -> float:
        """Compute P(rule correct | condition)."""
        # P(correct ∩ condition) / P(condition)
        p_condition = self.distribution.measure_set(condition)
        if p_condition == 0:
            return 0.0

        agreement_set = MeasurableSet(
            case_space=self.case_space,
            characteristic=lambda c: (
                (rule.evaluate(c) == MeasurableOutcome.TRUE) == ground_truth(c)
            ),
            description="Agreement Set",
        )

        intersection = agreement_set.intersect(condition)
        p_intersection = self.distribution.measure_set(intersection)

        return p_intersection / p_condition
