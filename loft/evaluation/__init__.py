"""
Legal QA evaluation framework.

Provides tools for evaluating legal question answering performance
using benchmark test suites.

Issue #277: Legal Question Test Suite
"""

from loft.evaluation.benchmark import BenchmarkSuite, BenchmarkLoader
from loft.evaluation.metrics import PerformanceMetrics, MetricsCalculator
from loft.evaluation.runner import EvaluationRunner

__all__ = [
    "BenchmarkSuite",
    "BenchmarkLoader",
    "PerformanceMetrics",
    "MetricsCalculator",
    "EvaluationRunner",
]
