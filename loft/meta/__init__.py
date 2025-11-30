"""
Meta: Reflexive Orchestration

This module implements meta-reasoning that enables the system to reason
about its own reasoning processes.

Components:
- ReasoningObserver: Observes and records reasoning patterns
- MetaReasoner: Second-order reasoning about reasoning processes
- StrategyEvaluator: Evaluates and compares reasoning strategies
- StrategySelector: Dynamically selects optimal strategies
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
from loft.meta.strategy import (
    AnalogicalStrategy,
    BalancingTestStrategy,
    CausalChainStrategy,
    ChecklistStrategy,
    ComparisonReport,
    DialecticalStrategy,
    ReasoningStrategy,
    RuleBasedStrategy,
    SelectionExplanation,
    SimpleCase,
    StrategyCharacteristics,
    StrategyEvaluator,
    StrategyMetrics,
    StrategySelector,
    StrategyType,
    create_evaluator,
    create_selector,
    get_default_strategies,
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
    # Strategy - Core types
    "ReasoningStrategy",
    "StrategyType",
    "StrategyCharacteristics",
    "StrategyMetrics",
    "ComparisonReport",
    "SelectionExplanation",
    "SimpleCase",
    # Strategy - Implementations
    "ChecklistStrategy",
    "CausalChainStrategy",
    "BalancingTestStrategy",
    "RuleBasedStrategy",
    "DialecticalStrategy",
    "AnalogicalStrategy",
    # Strategy - Evaluation
    "StrategyEvaluator",
    "StrategySelector",
    # Strategy - Factory functions
    "get_default_strategies",
    "create_evaluator",
    "create_selector",
]
