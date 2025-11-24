"""
Dialectical reasoning components for Phase 4.

This module implements dialectical validation where rules are refined through
criticism and synthesis rather than simple binary acceptance/rejection.
"""

from loft.dialectical.critique_schemas import (
    Contradiction,
    CritiqueReport,
    CritiqueSeverity,
    EdgeCase,
)

__all__ = [
    "CritiqueReport",
    "EdgeCase",
    "Contradiction",
    "CritiqueSeverity",
]
