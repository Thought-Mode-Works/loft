"""
Validation: Multi-stage Verification

This module provides comprehensive validation infrastructure.
"""

from loft.validation.asp_validators import (
    ASPSyntaxValidator,
    ASPSemanticValidator,
    validate_asp_program,
)
from loft.validation.metrics import (
    ValidationMetrics,
    MetricsDelta,
    MetricsTracker,
    compute_accuracy,
    compute_confidence_calibration_error,
)
from loft.validation.fidelity import (
    FidelityValidator,
    FidelityTestResult,
    compute_translation_fidelity,
)
from loft.validation.test_case_validator import (
    TestCase,
    TestResult,
    TestCaseValidator,
    create_test_suite,
)

# Phase 2.2: Multi-Stage Validation Pipeline
from loft.validation.validation_schemas import (
    ValidationResult,
    TestCase as EmpiricalTestCase,
    FailureCase,
    EmpiricalValidationResult,
    ConsensusValidationResult,
    ValidationReport,
)
from loft.validation.semantic_validator import SemanticValidator
from loft.validation.empirical_validator import EmpiricalValidator
from loft.validation.consensus_validator import ConsensusValidator
from loft.validation.validation_pipeline import ValidationPipeline

__all__ = [
    # ASP Validators
    "ASPSyntaxValidator",
    "ASPSemanticValidator",
    "validate_asp_program",
    # Metrics
    "ValidationMetrics",
    "MetricsDelta",
    "MetricsTracker",
    "compute_accuracy",
    "compute_confidence_calibration_error",
    # Fidelity
    "FidelityValidator",
    "FidelityTestResult",
    "compute_translation_fidelity",
    # Test Cases
    "TestCase",
    "TestResult",
    "TestCaseValidator",
    "create_test_suite",
    # Phase 2.2: Validation Pipeline
    "ValidationResult",
    "EmpiricalTestCase",
    "FailureCase",
    "EmpiricalValidationResult",
    "ConsensusValidationResult",
    "ValidationReport",
    "SemanticValidator",
    "EmpiricalValidator",
    "ConsensusValidator",
    "ValidationPipeline",
]
