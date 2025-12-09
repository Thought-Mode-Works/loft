"""
Neural ensemble components for Phase 6 heterogeneous LLM architecture.

This package contains specialized LLMs for different aspects of ASP rule generation:
- LogicGeneratorLLM: Fine-tuned/optimized for formal ASP logic generation
- CriticLLM: Specialized for edge case and contradiction detection
- TranslatorLLM: Symbolic <-> natural language conversion (future)
- MetaReasonerLLM: Reasoning about reasoning (future)

Architecture enhancements (based on multi-agent review PR #194):
- Strategy Pattern for flexible optimization approaches
- Abstract base class (LogicGenerator) for extensibility
- Exponential backoff retry logic for resilience
- LRU caching for repeated queries
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
]
