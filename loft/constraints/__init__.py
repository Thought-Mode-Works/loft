"""
Phase 7: Geometric Constraints & Invariance Module.

This module implements formal mathematical constraints ensuring legal and logical
principles are preserved through the neuro-symbolic system.

Components:
- equivariance: O(d)-equivariance for content-neutrality
- symmetry: Party symmetry invariance
- temporal: Temporal consistency
- measure_theory: Measure-theoretic representation
- ring_structure: Ring algebraic structure for rule composition
- constitutional: Constitutional layer verification (future)
"""

from loft.constraints.equivariance import (
    TransformationType,
    EquivarianceConstraint,
    PartyPermutationEquivariance,
    AmountScalingEquivariance,
    ContentSubstitutionEquivariance,
    EquivarianceVerifier,
    EquivarianceReport,
    EquivarianceViolation,
)
from loft.constraints.symmetry import (
    SymmetryType,
    PartySymmetryConstraint,
    SymmetryViolation,
    PartySymmetryTester,
    SymmetryTestReport,
)
from loft.constraints.temporal import (
    TemporalTransformType,
    TemporalField,
    TemporalViolation,
    TemporalConsistencyReport,
    TemporalConsistencyTester,
)
from loft.constraints.measure_theory import (
    MeasurableOutcome,
    CaseSpace,
    CaseDimension,
    MonomialPotential,
    CaseDistribution,
    MeasurableSet,
    MeasurableLegalRule,
    RuleConfidenceCalculator,
)
from loft.constraints.ring_structure import (
    RingElement,
    BooleanRule,
    ConfidenceRule,
    RuleComposition,
    HomomorphismViolation,
    HomomorphismVerificationReport,
    RingHomomorphism,
    RingVerificationReport,
    RingPropertyVerifier,
    ComposedRule,
)

__all__ = [
    # Enums
    "TransformationType",
    "SymmetryType",
    "TemporalTransformType",
    "MeasurableOutcome",
    # Base classes
    "EquivarianceConstraint",
    # Constraint implementations
    "PartyPermutationEquivariance",
    "AmountScalingEquivariance",
    "ContentSubstitutionEquivariance",
    "PartySymmetryConstraint",
    # Verification
    "EquivarianceVerifier",
    "EquivarianceReport",
    "EquivarianceViolation",
    "SymmetryViolation",
    "PartySymmetryTester",
    "SymmetryTestReport",
    # Temporal
    "TemporalField",
    "TemporalViolation",
    "TemporalConsistencyReport",
    "TemporalConsistencyTester",
    # Measure Theory
    "CaseSpace",
    "CaseDimension",
    "MonomialPotential",
    "CaseDistribution",
    "MeasurableSet",
    "MeasurableLegalRule",
    "RuleConfidenceCalculator",
    # Ring Structure
    "RingElement",
    "BooleanRule",
    "ConfidenceRule",
    "RuleComposition",
    "HomomorphismViolation",
    "HomomorphismVerificationReport",
    "RingHomomorphism",
    "RingVerificationReport",
    "RingPropertyVerifier",
    "ComposedRule",
]
