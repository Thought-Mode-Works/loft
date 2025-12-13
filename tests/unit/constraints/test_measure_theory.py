"""
Unit tests for Measure-Theoretic Representation (Phase 7).
"""

from loft.constraints.measure_theory import (
    MonomialPotential,
    CaseSpace,
    CaseDimension,
    MeasurableSet,
    CaseDistribution,
)


def test_monomial_potential_full_satisfaction() -> None:
    """Fully satisfied elements give potential = 1."""
    potential = MonomialPotential(
        elements=["offer", "acceptance", "consideration"], weights=[1.0, 1.0, 1.0]
    )
    result = potential.evaluate({"offer": 1.0, "acceptance": 1.0, "consideration": 1.0})
    assert result == 1.0


def test_monomial_potential_zero_element() -> None:
    """Zero element gives potential = 0."""
    potential = MonomialPotential(
        elements=["offer", "acceptance", "consideration"], weights=[1.0, 1.0, 1.0]
    )
    result = potential.evaluate({"offer": 1.0, "acceptance": 0.0, "consideration": 1.0})
    assert result == 0.0


def test_monomial_potential_partial() -> None:
    """Partial satisfaction gives potential in (0, 1)."""
    potential = MonomialPotential(elements=["a", "b"], weights=[1.0, 1.0])
    result = potential.evaluate({"a": 0.5, "b": 0.5})
    assert result == 0.25  # 0.5 * 0.5


def test_measurable_set_intersection() -> None:
    """Set intersection works correctly."""
    space = CaseSpace([CaseDimension("x", "numeric", bounds=(0, 10))])
    # Note: lambda c: c["x"] > 3 assumes c["x"] is already decoded or raw value?
    # MeasurableSet.characteristic takes a case dict. CaseSpace defines dimensions.
    # Typically we work with raw values in case dicts.
    set_a = MeasurableSet(space, lambda c: float(c["x"]) > 3)
    set_b = MeasurableSet(space, lambda c: float(c["x"]) < 7)
    intersection = set_a.intersect(set_b)

    assert intersection.contains({"x": 5})  # In both
    assert not intersection.contains({"x": 2})  # Only in B
    assert not intersection.contains({"x": 8})  # Only in A


def test_empirical_distribution_measure() -> None:
    """Empirical distribution computes correct measures."""
    space = CaseSpace([CaseDimension("valid", "boolean")])
    samples = [{"valid": True}] * 7 + [{"valid": False}] * 3
    dist = CaseDistribution.from_samples(space, samples)

    valid_set = MeasurableSet(space, lambda c: bool(c["valid"]))
    measure = dist.measure_set(valid_set)
    assert abs(measure - 0.7) < 0.01


def test_gradient_identifies_limiting_element() -> None:
    """Gradient correctly identifies most limiting element."""
    potential = MonomialPotential(
        elements=["a", "b", "c"], weights=[1.0, 2.0, 1.0]  # b has higher weight
    )
    satisfaction = {"a": 0.9, "b": 0.5, "c": 0.8}
    limiting = potential.most_limiting_element(satisfaction)
    # Gradient:
    # a: 1 * phi / 0.9
    # b: 2 * phi / 0.5 = 4 * phi
    # c: 1 * phi / 0.8
    # So b should be largest
    assert limiting == "b"  # Most room for improvement with high weight
