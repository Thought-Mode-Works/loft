"""
Batch learning module for autonomous learning cycles.

Provides infrastructure for processing large batches of test cases
through the complete learning pipeline with checkpointing, progress
tracking, and metrics collection.

Example usage:
    >>> from loft.batch import BatchLearningHarness, BatchConfig
    >>>
    >>> # Create harness with configuration
    >>> config = BatchConfig(
    ...     checkpoint_interval=10,
    ...     validation_threshold=0.8,
    ... )
    >>> harness = BatchLearningHarness(config=config)
    >>>
    >>> # Define case processor
    >>> def process_case(case, rules):
    ...     # Your processing logic here
    ...     return CaseResult(...)
    >>>
    >>> # Run batch
    >>> result = harness.run_batch(
    ...     test_cases=cases,
    ...     process_case_fn=process_case,
    ... )
    >>>
    >>> # Check results
    >>> print(f"Processed: {result.success_count} / {len(result.case_results)}")
    >>> print(f"Rules accepted: {result.total_rules_accepted}")
"""

from .schemas import (
    BatchCheckpoint,
    BatchConfig,
    BatchMetrics,
    BatchProgress,
    BatchResult,
    BatchStatus,
    CaseResult,
    CaseStatus,
)

from .harness import (
    BatchLearningHarness,
    create_simple_case_processor,
)

from .full_pipeline import (
    FullPipelineProcessor,
    KnowledgeGap,
    ProcessingMetrics,
    create_full_pipeline_processor,
)

__all__ = [
    # Core harness
    "BatchLearningHarness",
    "create_simple_case_processor",
    # Full pipeline (Issue #253)
    "FullPipelineProcessor",
    "KnowledgeGap",
    "ProcessingMetrics",
    "create_full_pipeline_processor",
    # Schemas
    "BatchCheckpoint",
    "BatchConfig",
    "BatchMetrics",
    "BatchProgress",
    "BatchResult",
    "BatchStatus",
    "CaseResult",
    "CaseStatus",
]
