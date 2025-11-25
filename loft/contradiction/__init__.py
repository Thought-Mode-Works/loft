"""
Contradiction Management System (Phase 4.4).

Tracks, manages, and resolves contradictions between rules and competing
interpretations of legal/policy principles.
"""

from loft.contradiction.contradiction_schemas import (
    ContradictionType,
    ContradictionSeverity,
    ResolutionStrategy,
    ContradictionReport,
    RuleInterpretation,
    ResolutionResult,
    ContextClassification,
)
from loft.contradiction.contradiction_manager import ContradictionManager
from loft.contradiction.context_classifier import ContextClassifier
from loft.contradiction.contradiction_store import ContradictionStore

__all__ = [
    "ContradictionType",
    "ContradictionSeverity",
    "ResolutionStrategy",
    "ContradictionReport",
    "RuleInterpretation",
    "ResolutionResult",
    "ContextClassification",
    "ContradictionManager",
    "ContextClassifier",
    "ContradictionStore",
]
