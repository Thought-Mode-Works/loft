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
from loft.meta.prompt_optimizer import (
    ABTestConfig,
    ABTestResult,
    EffectivenessReport,
    ImprovementCandidate,
    PromptABTester,
    PromptCategory,
    PromptMetrics,
    PromptOptimizer,
    PromptVersion,
    TestStatus,
    create_ab_tester,
    create_prompt_optimizer,
)
from loft.meta.failure_analyzer import (
    ErrorCategory,
    FailureAnalysisReport,
    FailureAnalyzer,
    FailurePattern,
    PredictionError,
    Recommendation,
    RecommendationCategory,
    RecommendationEngine,
    RootCause,
    RootCauseAnalysis,
    RootCauseType,
    create_failure_analyzer,
    create_recommendation_engine,
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
    # Prompt Optimizer - Core types
    "PromptVersion",
    "PromptMetrics",
    "PromptCategory",
    "TestStatus",
    # Prompt Optimizer - Reports
    "EffectivenessReport",
    "ABTestConfig",
    "ABTestResult",
    "ImprovementCandidate",
    # Prompt Optimizer - Classes
    "PromptOptimizer",
    "PromptABTester",
    # Prompt Optimizer - Factory functions
    "create_prompt_optimizer",
    "create_ab_tester",
    # Failure Analyzer - Core types
    "ErrorCategory",
    "RootCauseType",
    "RecommendationCategory",
    # Failure Analyzer - Data classes
    "PredictionError",
    "RootCause",
    "RootCauseAnalysis",
    "FailurePattern",
    "Recommendation",
    "FailureAnalysisReport",
    # Failure Analyzer - Classes
    "FailureAnalyzer",
    "RecommendationEngine",
    # Failure Analyzer - Factory functions
    "create_failure_analyzer",
    "create_recommendation_engine",
]
