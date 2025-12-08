"""
Neural ensemble components for Phase 6 heterogeneous LLM architecture.

This package contains specialized LLMs for different aspects of ASP rule generation:
- LogicGeneratorLLM: Fine-tuned/optimized for formal ASP logic generation
- CriticLLM: Specialized for edge case and contradiction detection (future)
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

__all__ = [
    # Main classes
    "LogicGeneratorLLM",
    "LogicGeneratorConfig",
    "ASPGenerationResult",
    "BenchmarkResult",
    # Abstract base class
    "LogicGenerator",
    # Strategy Pattern components
    "OptimizationStrategy",
    "OptimizationStrategyType",
    "PromptOptimizationStrategy",
    "FewShotLearningStrategy",
    "ChainOfThoughtStrategy",
    "SelfConsistencyStrategy",
    "create_strategy",
    # Exceptions
    "ASPGenerationError",
]
