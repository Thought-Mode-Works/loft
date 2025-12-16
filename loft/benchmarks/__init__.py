"""
Benchmark framework for LOFT infrastructure.

Issue #257: Baseline Validation Benchmarks
"""

from loft.benchmarks.baseline_metrics import (
    BaselineMetrics,
    EndToEndMetrics,
    MetaReasoningMetrics,
    PersistenceMetrics,
    RuleGenerationMetrics,
    TranslationMetrics,
    ValidationMetrics,
)
from loft.benchmarks.benchmark_suite import BenchmarkConfig, BenchmarkSuite
from loft.benchmarks.comparison import (
    ComparisonReport,
    MetricChange,
    compare_baselines,
)

__all__ = [
    # Metrics
    "BaselineMetrics",
    "RuleGenerationMetrics",
    "ValidationMetrics",
    "MetaReasoningMetrics",
    "TranslationMetrics",
    "EndToEndMetrics",
    "PersistenceMetrics",
    # Suite
    "BenchmarkConfig",
    "BenchmarkSuite",
    # Comparison
    "ComparisonReport",
    "MetricChange",
    "compare_baselines",
]
