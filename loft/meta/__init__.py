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
    CounterfactualAnalysis,
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
    AggregateEffectivenessReport,
    EffectivenessReport,
    ImprovementCandidate,
    ImprovementSuggestion,
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
    create_prediction_error_from_chain,
    create_recommendation_engine,
    extract_failure_patterns,
)
from loft.meta.self_improvement import (
    ActionType,
    AutonomousImprover,
    CycleEvaluation,
    CycleResults,
    CycleStatus,
    GoalStatus,
    ImprovementAction,
    ImprovementCycle,
    ImprovementGoal,
    MetricType,
    MetricValue,
    ProgressReport,
    SafetyConfig,
    SelfImprovementTracker,
    create_default_goals,
    create_improvement_goal,
    create_improver,
    create_tracker,
)
from loft.meta.event_bus import (
    ComponentType,
    EventType,
    MetaReasoningEvent,
    MetaReasoningEventBus,
    Subscription,
    create_event,
    create_event_bus,
    get_global_event_bus,
    reset_global_event_bus,
)
from loft.meta.adapters import (
    EventDrivenIntegration,
    FailureToImprovementAdapter,
    ObserverToFailureAdapter,
    create_event_driven_integration,
    create_failure_to_improvement_adapter,
    create_observer_to_failure_adapter,
)
from loft.meta.dashboard import (
    AlertSeverity,
    DashboardExporter,
    DashboardGenerator,
    FailureSummary,
    HealthStatus,
    ImprovementSummary,
    MetaReasoningAlert,
    MetaReasoningDashboard,
    ObserverSummary,
    PromptSummary,
    StrategySummary,
    TrendData,
    TrendDirection,
    TrendReport,
    create_dashboard_generator,
)
from loft.meta.action_handlers import (
    HandlerResult,
    PromptRefinementHandler,
    StrategyAdjustmentHandler,
    create_prompt_refinement_handler,
    create_prompt_refinement_rollback_handler,
    create_strategy_adjustment_handler,
    create_strategy_adjustment_rollback_handler,
    register_real_handlers,
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
    "CounterfactualAnalysis",
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
    "AggregateEffectivenessReport",
    "ABTestConfig",
    "ABTestResult",
    "ImprovementCandidate",
    "ImprovementSuggestion",
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
    "create_prediction_error_from_chain",
    "create_recommendation_engine",
    "extract_failure_patterns",
    # Self-Improvement - Core types
    "GoalStatus",
    "CycleStatus",
    "ActionType",
    "MetricType",
    # Self-Improvement - Data classes
    "MetricValue",
    "ImprovementGoal",
    "ImprovementAction",
    "CycleResults",
    "ImprovementCycle",
    "ProgressReport",
    "CycleEvaluation",
    "SafetyConfig",
    # Self-Improvement - Classes
    "SelfImprovementTracker",
    "AutonomousImprover",
    # Self-Improvement - Factory functions
    "create_improvement_goal",
    "create_tracker",
    "create_improver",
    "create_default_goals",
    # Event Bus - Core types
    "EventType",
    "ComponentType",
    "MetaReasoningEvent",
    "Subscription",
    # Event Bus - Classes
    "MetaReasoningEventBus",
    # Event Bus - Factory functions
    "create_event",
    "create_event_bus",
    "get_global_event_bus",
    "reset_global_event_bus",
    # Adapters - Classes
    "ObserverToFailureAdapter",
    "FailureToImprovementAdapter",
    "EventDrivenIntegration",
    # Adapters - Factory functions
    "create_observer_to_failure_adapter",
    "create_failure_to_improvement_adapter",
    "create_event_driven_integration",
    # Dashboard - Core types
    "HealthStatus",
    "TrendDirection",
    "AlertSeverity",
    # Dashboard - Data classes
    "MetaReasoningAlert",
    "ObserverSummary",
    "StrategySummary",
    "PromptSummary",
    "FailureSummary",
    "ImprovementSummary",
    "TrendData",
    "TrendReport",
    "MetaReasoningDashboard",
    # Dashboard - Classes
    "DashboardGenerator",
    "DashboardExporter",
    # Dashboard - Factory functions
    "create_dashboard_generator",
    # Action Handlers - Data classes
    "HandlerResult",
    # Action Handlers - Classes
    "PromptRefinementHandler",
    "StrategyAdjustmentHandler",
    # Action Handlers - Factory functions
    "create_prompt_refinement_handler",
    "create_prompt_refinement_rollback_handler",
    "create_strategy_adjustment_handler",
    "create_strategy_adjustment_rollback_handler",
    "register_real_handlers",
]
