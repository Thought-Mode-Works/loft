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
    ValidationCase,
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

# Backward compatibility aliases
TestCase = ValidationCase  # For old code importing TestCase
TestCaseData = ValidationCase  # For code that already migrated to TestCaseData

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
    "ValidationCase",
    "TestCase",  # Backward compatibility alias
    "TestCaseData",  # Backward compatibility alias
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
