"""
Neural ensemble components for Phase 6 heterogeneous LLM architecture.

This package contains specialized LLMs for different aspects of ASP rule generation:
- LogicGeneratorLLM: Fine-tuned/optimized for formal ASP logic generation
- CriticLLM: Specialized for edge case and contradiction detection
- TranslatorLLM: Symbolic <-> natural language bidirectional conversion
- MetaReasonerLLM: Reasoning about reasoning (meta-level analysis)
- EnsembleOrchestrator: Coordinates specialized LLMs with voting and routing

Architecture enhancements (based on multi-agent review PR #194):
- Strategy Pattern for flexible optimization approaches
- Abstract base class (LogicGenerator) for extensibility
- Exponential backoff retry logic for resilience
- LRU caching for repeated queries
- Multi-LLM voting mechanisms (Issue #192)
- Disagreement resolution strategies (Issue #192)
"""

from .logic_generator import (
    # Main classes
    LogicGeneratorLLM,
    LogicGeneratorConfig,
    ASPGenerationResult,
    BenchmarkResult,
    # Abstract base class
    LogicGenerator,
    # Strategy Pattern components
    OptimizationStrategy,
    OptimizationStrategyType,
    PromptOptimizationStrategy,
    FewShotLearningStrategy,
    ChainOfThoughtStrategy,
    SelfConsistencyStrategy,
    create_strategy,
    # Exceptions
    ASPGenerationError,
)

from .critic import (
    # Main class
    CriticLLM,
    CriticConfig,
    # Abstract base class
    Critic,
    # Data classes
    EdgeCase,
    Contradiction,
    GeneralizationAssessment,
    CriticResult,
    # Strategy Pattern components
    CriticStrategy,
    CriticStrategyType,
    AdversarialStrategy,
    CooperativeStrategy,
    SystematicStrategy,
    DialecticalStrategy,
    create_critic_strategy,
    # Exceptions
    CriticAnalysisError,
)

from .translator import (
    # Main class
    TranslatorLLM,
    TranslatorConfig,
    # Abstract base class
    Translator,
    # Data classes
    TranslationResult,
    RoundtripResult,
    TranslatorBenchmarkResult,
    # Strategy Pattern components
    TranslationStrategy,
    TranslationStrategyType,
    LiteralStrategy,
    ContextualStrategy,
    LegalDomainStrategy,
    PedagogicalStrategy,
    create_translation_strategy,
    # Exceptions
    TranslationError,
)

from .meta_reasoner import (
    # Main class
    MetaReasonerLLM,
    MetaReasonerConfig,
    # Abstract base class
    MetaReasoner,
    # Data classes
    FailureRecord,
    Insight,
    StrategyChange,
    ImprovementReport,
    MetaReasoningResult,
    # Enums
    MetaReasoningStrategyType,
    InsightType,
    FailureCategory,
    # Strategy Pattern components
    MetaReasoningStrategy,
    AnalyticalStrategy,
    ReflectiveStrategy,
    ComparativeStrategy,
    DiagnosticStrategy,
    create_meta_reasoning_strategy,
    # Exceptions
    MetaReasoningError,
)

from .orchestrator import (
    # Main class
    EnsembleOrchestrator,
    OrchestratorConfig,
    # Abstract base class
    EnsembleOrchestratorBase,
    # Data classes
    ModelResponse,
    VotingResult,
    DisagreementRecord,
    ModelPerformanceMetrics,
    OrchestrationResult,
    # Enums
    TaskType,
    VotingStrategyType,
    DisagreementStrategyType,
    ModelStatus,
    # Voting Strategy Pattern components
    VotingStrategy,
    UnanimousVotingStrategy,
    MajorityVotingStrategy,
    WeightedVotingStrategy,
    DialecticalVotingStrategy,
    create_voting_strategy,
    # Disagreement Resolution components
    DisagreementResolver,
    DeferToCriticResolver,
    DeferToConfidenceResolver,
    SynthesizeResolver,
    EscalateResolver,
    ConservativeResolver,
    create_disagreement_resolver,
    # Exceptions
    OrchestratorError,
    TaskRoutingError,
    VotingError,
    DisagreementResolutionError,
    FallbackExhaustedError,
)

__all__ = [
    # Logic Generator
    "LogicGeneratorLLM",
    "LogicGeneratorConfig",
    "ASPGenerationResult",
    "BenchmarkResult",
    "LogicGenerator",
    "OptimizationStrategy",
    "OptimizationStrategyType",
    "PromptOptimizationStrategy",
    "FewShotLearningStrategy",
    "ChainOfThoughtStrategy",
    "SelfConsistencyStrategy",
    "create_strategy",
    "ASPGenerationError",
    # Critic LLM
    "CriticLLM",
    "CriticConfig",
    "Critic",
    "EdgeCase",
    "Contradiction",
    "GeneralizationAssessment",
    "CriticResult",
    "CriticStrategy",
    "CriticStrategyType",
    "AdversarialStrategy",
    "CooperativeStrategy",
    "SystematicStrategy",
    "DialecticalStrategy",
    "create_critic_strategy",
    "CriticAnalysisError",
    # Translator LLM
    "TranslatorLLM",
    "TranslatorConfig",
    "Translator",
    "TranslationResult",
    "RoundtripResult",
    "TranslatorBenchmarkResult",
    "TranslationStrategy",
    "TranslationStrategyType",
    "LiteralStrategy",
    "ContextualStrategy",
    "LegalDomainStrategy",
    "PedagogicalStrategy",
    "create_translation_strategy",
    "TranslationError",
    # Meta-Reasoner LLM
    "MetaReasonerLLM",
    "MetaReasonerConfig",
    "MetaReasoner",
    "FailureRecord",
    "Insight",
    "StrategyChange",
    "ImprovementReport",
    "MetaReasoningResult",
    "MetaReasoningStrategyType",
    "InsightType",
    "FailureCategory",
    "MetaReasoningStrategy",
    "AnalyticalStrategy",
    "ReflectiveStrategy",
    "ComparativeStrategy",
    "DiagnosticStrategy",
    "create_meta_reasoning_strategy",
    "MetaReasoningError",
    # Ensemble Orchestrator
    "EnsembleOrchestrator",
    "OrchestratorConfig",
    "EnsembleOrchestratorBase",
    "ModelResponse",
    "VotingResult",
    "DisagreementRecord",
    "ModelPerformanceMetrics",
    "OrchestrationResult",
    "TaskType",
    "VotingStrategyType",
    "DisagreementStrategyType",
    "ModelStatus",
    "VotingStrategy",
    "UnanimousVotingStrategy",
    "MajorityVotingStrategy",
    "WeightedVotingStrategy",
    "DialecticalVotingStrategy",
    "create_voting_strategy",
    "DisagreementResolver",
    "DeferToCriticResolver",
    "DeferToConfidenceResolver",
    "SynthesizeResolver",
    "EscalateResolver",
    "ConservativeResolver",
    "create_disagreement_resolver",
    "OrchestratorError",
    "TaskRoutingError",
    "VotingError",
    "DisagreementResolutionError",
    "FallbackExhaustedError",
]
