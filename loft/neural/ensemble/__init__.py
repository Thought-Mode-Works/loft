"""
Neural ensemble components for Phase 6 heterogeneous LLM architecture.

This package contains specialized LLMs for different aspects of ASP rule generation:
- LogicGeneratorLLM: Fine-tuned/optimized for formal ASP logic generation
- CriticLLM: Specialized for edge case and contradiction detection (future)
- TranslatorLLM: Symbolic <-> natural language conversion (future)
- MetaReasonerLLM: Reasoning about reasoning (future)
"""

from .logic_generator import (
    LogicGeneratorLLM,
    LogicGeneratorConfig,
    ASPGenerationResult,
    BenchmarkResult,
)

__all__ = [
    "LogicGeneratorLLM",
    "LogicGeneratorConfig",
    "ASPGenerationResult",
    "BenchmarkResult",
]
