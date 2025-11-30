"""
Meta: Reflexive Orchestration

This module implements meta-reasoning that enables the system to reason
about its own reasoning processes.

Components:
- ReasoningObserver: Observes and records reasoning patterns
- MetaReasoner: Second-order reasoning about reasoning processes
- Schemas: Data classes for patterns, observations, improvements
"""

from loft.meta.schemas import (
    Bottleneck,
    BottleneckReport,
    FailureDiagnosis,
    Improvement,
    ImprovementPriority,
    ImprovementType,
    ObservationSummary,
    PatternType,
    ReasoningChain,
    ReasoningPattern,
    ReasoningStep,
    ReasoningStepType,
)
from loft.meta.observer import (
    MetaReasoner,
    ReasoningObserver,
)

__all__ = [
    # Observer
    "ReasoningObserver",
    "MetaReasoner",
    # Schemas - Core types
    "ReasoningStep",
    "ReasoningStepType",
    "ReasoningChain",
    "ReasoningPattern",
    "PatternType",
    # Schemas - Analysis types
    "Bottleneck",
    "BottleneckReport",
    "FailureDiagnosis",
    "Improvement",
    "ImprovementType",
    "ImprovementPriority",
    "ObservationSummary",
]
