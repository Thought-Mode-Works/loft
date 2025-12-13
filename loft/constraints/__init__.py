"""
Phase 7: Geometric Constraints & Invariance Module.

This module implements formal mathematical constraints ensuring legal and logical
principles are preserved through the neuro-symbolic system.

Components:
- equivariance: O(d)-equivariance for content-neutrality
- symmetry: Party symmetry invariance (future)
- temporal: Temporal consistency (future)
- measure_theory: Measure-theoretic representation (future)
- ring_structure: Ring algebraic structure for rule composition (future)
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

__all__ = [
    # Enums
    "TransformationType",
    "SymmetryType",
    # Base classes
    "EquivarianceConstraint",
    # Constraint implementations
    "PartyPermutationEquivariance",
    "AmountScalingEquivariance",
    "ContentSubstitutionEquivariance",
    "PartySymmetryConstraint",  # Added symmetry constraint
    # Verification
    "EquivarianceVerifier",
    "EquivarianceReport",
    "EquivarianceViolation",
    "SymmetryViolation",  # Added symmetry violation
    "PartySymmetryTester",  # Added symmetry tester
    "SymmetryTestReport",  # Added symmetry report
]
