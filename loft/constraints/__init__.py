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

__all__ = [
    # Enums
    "TransformationType",
    # Base classes
    "EquivarianceConstraint",
    # Constraint implementations
    "PartyPermutationEquivariance",
    "AmountScalingEquivariance",
    "ContentSubstitutionEquivariance",
    # Verification
    "EquivarianceVerifier",
    "EquivarianceReport",
    "EquivarianceViolation",
]
