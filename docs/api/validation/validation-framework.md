# Validation Framework API Reference

The validation module provides comprehensive multi-stage validation for ASP rules.

## Module Overview

```python
from loft.validation import (
    # Core Validators
    ASPSyntaxValidator,
    ASPSemanticValidator,
    TestCaseValidator,
    ValidationPipeline,

    # Data Classes
    ValidationCase,
    TestResult,
    ValidationReport,

    # Utilities
    validate_asp_program,
    create_test_suite,
)
```

## Core Classes

### ValidationPipeline

Main orchestrator for multi-stage validation.

```python
class ValidationPipeline:
    """
    Multi-stage validation pipeline for ASP rules.

    Stages:
    1. Syntactic validation (ASP parser)
    2. Semantic validation (consistency check)
    3. Empirical validation (test cases)
    4. Consensus validation (multi-agent, optional)
    """

    def __init__(
        self,
        syntax_validator: Optional[ASPSyntaxValidator] = None,
        semantic_validator: Optional[ASPSemanticValidator] = None,
        empirical_validator: Optional[TestCaseValidator] = None,
        consensus_validator: Optional[ConsensusValidator] = None,
    ):
        """
        Initialize validation pipeline.

        Args:
            syntax_validator: Validator for ASP syntax
            semantic_validator: Validator for logical consistency
            empirical_validator: Validator for test case execution
            consensus_validator: Validator for multi-agent agreement (optional)
        """

    def validate(
        self,
        asp_program: str,
        existing_rules: Optional[List[str]] = None,
        test_cases: Optional[List[ValidationCase]] = None,
        threshold: float = 0.75,
    ) -> ValidationReport:
        """
        Run complete validation pipeline.

        Args:
            asp_program: ASP rule(s) to validate
            existing_rules: Current rule base for consistency checking
            test_cases: Test cases for empirical validation
            threshold: Minimum accuracy threshold (0.0-1.0)

        Returns:
            ValidationReport with results from all stages

        Example:
            >>> pipeline = ValidationPipeline()
            >>> report = pipeline.validate(
            ...     "enforceable(C) :- contract(C), has_writing(C).",
            ...     existing_rules=["contract(c1). has_writing(c1)."],
            ...     test_cases=[test1, test2],
            ...     threshold=0.80
            ... )
            >>> print(f"Passed: {report.passed}, Stage: {report.stage}")
        """
```

### ASPSyntaxValidator

Validates ASP syntax using Clingo parser.

```python
class ASPSyntaxValidator:
    """Validates ASP program syntax."""

    def validate(self, asp_program: str) -> ValidationResult:
        """
        Check if ASP program has valid syntax.

        Args:
            asp_program: ASP program text

        Returns:
            ValidationResult with passed=True if syntax is valid

        Raises:
            None (returns ValidationResult with error message)

        Example:
            >>> validator = ASPSyntaxValidator()
            >>> result = validator.validate("contract(X) :- offer(X), acceptance(X).")
            >>> assert result.passed

            >>> result = validator.validate("contract(X) :- offer(X) acceptance(X).")  # Missing comma
            >>> assert not result.passed
            >>> print(result.error_message)
```

### ASPSemanticValidator

Validates logical consistency using SAT solving.

```python
class ASPSemanticValidator:
    """Validates semantic consistency of ASP programs."""

    def validate(
        self,
        asp_program: str,
        existing_rules: List[str],
    ) -> ValidationResult:
        """
        Check if ASP program is consistent with existing rules.

        Args:
            asp_program: New ASP rule(s) to check
            existing_rules: Current rule base

        Returns:
            ValidationResult with:
            - passed=True if combined program is satisfiable
            - error_message if unsatisfiable or contradictory

        Example:
            >>> validator = ASPSemanticValidator()
            >>> existing = ["person(alice). person(bob)."]
            >>> new_rule = "happy(X) :- person(X), not sad(X)."
            >>> result = validator.validate(new_rule, existing)
            >>> assert result.passed

            >>> bad_rule = ":- person(X)."  # Makes all persons false!
            >>> result = validator.validate(bad_rule, existing)
            >>> assert not result.passed
        """
```

### TestCaseValidator

Validates rules against empirical test cases.

```python
class TestCaseValidator:
    """Validates ASP programs against test cases."""

    def validate_test_case(
        self,
        asp_program: str,
        test_case: ValidationCase
    ) -> TestResult:
        """
        Run ASP program with test case facts and check results.

        Args:
            asp_program: ASP program (rules)
            test_case: Test case with facts and expected results

        Returns:
            TestResult with:
            - passed: True if actual matches expected
            - actual_results: Predicate truth values
            - mismatches: List of mismatched predicates
            - explanation: Human-readable summary

        Example:
            >>> validator = TestCaseValidator()
            >>> program = "enforceable(C) :- contract(C), has_writing(C)."
            >>> test = ValidationCase(
            ...     case_id="sof_001",
            ...     description="Contract with writing",
            ...     asp_facts="contract(c1). has_writing(c1).",
            ...     expected_results={"enforceable": True}
            ... )
            >>> result = validator.validate_test_case(program, test)
            >>> assert result.passed
        """

    def batch_validate(
        self,
        asp_program: str,
        test_cases: List[ValidationCase]
    ) -> Dict[str, Any]:
        """
        Run all test cases and compute accuracy metrics.

        Args:
            asp_program: ASP program to test
            test_cases: List of test cases

        Returns:
            Dictionary with:
            - accuracy: float (0.0-1.0)
            - passed: int (number passed)
            - failed: int (number failed)
            - total: int (total test cases)
            - failed_cases: List[(case_id, TestResult)]
            - results: List[TestResult] (all results)

        Example:
            >>> validator = TestCaseValidator()
            >>> program = "a :- b."
            >>> tests = [
            ...     ValidationCase("t1", "test 1", "b.", {"a": True}),
            ...     ValidationCase("t2", "test 2", "c.", {"a": False})
            ... ]
            >>> stats = validator.batch_validate(program, tests)
            >>> print(f"Accuracy: {stats['accuracy']:.2%}")
            Accuracy: 100.00%
        """
```

## Data Classes

### ValidationCase

Represents a test case for empirical validation.

```python
@dataclass
class ValidationCase:
    """
    Test case for validating ASP programs.

    Attributes:
        case_id: Unique identifier for test case
        description: Human-readable description
        asp_facts: Facts to add for this test (ASP syntax)
        expected_results: Expected predicate truth values
        reasoning_chain: Optional expected inference steps
        confidence_level: "high", "medium", or "low"
    """
    case_id: str
    description: str
    asp_facts: str
    expected_results: Dict[str, bool]
    reasoning_chain: List[str] = field(default_factory=list)
    confidence_level: str = "high"
```

### TestResult

Result from running a single test case.

```python
@dataclass
class TestResult:
    """
    Result from running a single test case.

    Attributes:
        test_case: The test case that was run
        passed: True if test passed
        actual_results: Actual predicate truth values
        mismatches: List of predicates that didn't match expected
        explanation: Human-readable explanation
        error: Error message if test crashed (None if successful)
    """
    test_case: ValidationCase
    passed: bool
    actual_results: Dict[str, bool]
    mismatches: List[str]
    explanation: str
    error: Optional[str] = None
```

### ValidationReport

Comprehensive report from validation pipeline.

```python
@dataclass
class ValidationReport:
    """
    Complete validation report from all stages.

    Attributes:
        passed: Overall result (True = all stages passed)
        stage: Last stage completed ("syntax", "semantic", "empirical", "consensus", or "all")
        syntax_result: Result from syntactic validation
        semantic_result: Result from semantic validation
        empirical_result: Result from empirical validation
        consensus_result: Result from consensus validation (optional)
        confidence_score: Overall confidence (0.0-1.0)
        timestamp: When validation was performed
    """
    passed: bool
    stage: str
    syntax_result: Optional[ValidationResult] = None
    semantic_result: Optional[ValidationResult] = None
    empirical_result: Optional[Dict[str, Any]] = None
    consensus_result: Optional[ConsensusValidationResult] = None
    confidence_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
```

## Utility Functions

### validate_asp_program

Convenience function for comprehensive validation.

```python
def validate_asp_program(asp_text: str) -> Dict[str, Any]:
    """
    Comprehensive validation of an ASP program.

    Runs all validation checks and returns a summary.

    Args:
        asp_text: ASP program text to validate

    Returns:
        Dictionary with validation results:
        {
            "syntax_valid": bool,
            "syntax_error": Optional[str],
            "is_consistent": bool,
            "consistency_msg": str,
            "answer_set_count": int,
            "contradictions": List[str],
            "overall_valid": bool
        }

    Example:
        >>> result = validate_asp_program("contract(X) :- offer(X).")
        >>> assert result["overall_valid"]
        >>> assert result["syntax_valid"]

        >>> result = validate_asp_program("contract(X) :- offer(X) acceptance(X).")
        >>> assert not result["overall_valid"]
        >>> print(result["syntax_error"])
        Syntax error: expected ',' or '.' at line 1
    """
```

### create_test_suite

Create test suite from dictionary data.

```python
def create_test_suite(test_cases_data: List[Dict[str, Any]]) -> List[ValidationCase]:
    """
    Create test suite from dictionary data.

    Args:
        test_cases_data: List of test case dictionaries

    Returns:
        List of ValidationCase instances

    Example:
        >>> suite = create_test_suite([
        ...     {
        ...         "case_id": "test_1",
        ...         "description": "Basic test",
        ...         "asp_facts": "fact(a).",
        ...         "expected_results": {"fact": True}
        ...     }
        ... ])
        >>> assert len(suite) == 1
        >>> assert suite[0].case_id == "test_1"
    """
```

## Usage Patterns

### Basic Validation

```python
from loft.validation import validate_asp_program

# Quick comprehensive validation
asp_rule = "enforceable(C) :- contract(C), has_writing(C)."
result = validate_asp_program(asp_rule)
if result["overall_valid"]:
    print("Rule is valid!")
    print(f"Syntax valid: {result['syntax_valid']}")
    print(f"Consistent: {result['is_consistent']}")
else:
    print(f"Validation failed: {result['syntax_error'] or result['consistency_msg']}")
```

### Full Pipeline Validation

```python
from loft.validation import ValidationPipeline, ValidationCase, create_test_suite

# Create test cases
test_data = [
    {
        "case_id": "sof_001",
        "description": "Contract with writing",
        "asp_facts": "contract(c1). has_writing(c1).",
        "expected_results": {"enforceable": True}
    },
    {
        "case_id": "sof_002",
        "description": "Contract without writing",
        "asp_facts": "contract(c2).",
        "expected_results": {"enforceable": False}
    }
]
test_suite = create_test_suite(test_data)

# Run full validation
pipeline = ValidationPipeline()
asp_program = """
enforceable(C) :- contract(C), has_writing(C).
enforceable(C) :- contract(C), not statute_of_frauds_applies(C).
"""

report = pipeline.validate(
    asp_program,
    existing_rules=[],
    test_cases=test_suite,
    threshold=0.80
)

if report.passed:
    print(f"Validation passed! Confidence: {report.confidence_score:.2%}")
else:
    print(f"Validation failed at stage: {report.stage}")
```

### Custom Validation Flow

```python
from loft.validation import ASPSyntaxValidator, ASPSemanticValidator, TestCaseValidator

# Step-by-step validation with custom logic
syntax_validator = ASPSyntaxValidator()
semantic_validator = ASPSemanticValidator()
test_validator = TestCaseValidator()

asp_rule = "liable(X) :- negligent(X), caused_harm(X)."

# 1. Check syntax
syntax_result = syntax_validator.validate(asp_rule)
if not syntax_result.passed:
    print(f"Syntax error: {syntax_result.error_message}")
    exit(1)

# 2. Check consistency
existing_rules = ["person(alice). negligent(alice)."]
semantic_result = semantic_validator.validate(asp_rule, existing_rules)
if not semantic_result.passed:
    print(f"Consistency error: {semantic_result.error_message}")
    exit(1)

# 3. Run test cases
test_cases = [...]
stats = test_validator.batch_validate(asp_rule, test_cases)
if stats["accuracy"] < 0.75:
    print(f"Low accuracy: {stats['accuracy']:.2%}")
    print(f"Failed cases: {[fc[0] for fc in stats['failed_cases']]}")
    exit(1)

print("All validations passed!")
```

## Integration with Other Modules

### With Symbolic Core

```python
from loft.symbolic import ASPCore
from loft.validation import ValidationPipeline

core = ASPCore()
pipeline = ValidationPipeline()

# Validate before adding to core
new_rule = "enforceable(C) :- contract(C), has_writing(C)."
report = pipeline.validate(
    new_rule,
    existing_rules=core.get_all_rules(),
    test_cases=load_test_suite("contracts")
)

if report.passed:
    core.add_rule(new_rule)
else:
    logger.warning(f"Rejected rule: {report.stage} failed")
```

### With Version Control

```python
from loft.core import CoreState
from loft.validation import ValidationPipeline

state = CoreState()
pipeline = ValidationPipeline()

# Validate before committing
new_rule = "..."
report = pipeline.validate(new_rule, ...)

if report.passed:
    state.add_rule(new_rule)
    state.commit(f"Add rule with confidence={report.confidence_score:.2f}")
else:
    logger.info(f"Rule rejected at {report.stage} stage")
```

## Performance Considerations

- **Syntactic validation**: ~10ms (fast, use liberally)
- **Semantic validation**: ~50-200ms (depends on rule base size)
- **Empirical validation**: ~100-500ms (depends on test suite size)
- **Consensus validation**: ~5-10s (LLM queries, use sparingly)

**Optimization tips**:
- Cache validation results for unchanged rules
- Run empirical tests in parallel
- Use smaller test suites for operational layer
- Reserve consensus for strategic layer only

## Error Handling

All validators return `ValidationResult` objects rather than raising exceptions:

```python
result = validator.validate(asp_program)
if not result.passed:
    print(f"Error: {result.error_message}")
    # Handle gracefully
```

This enables graceful degradation and batch validation.

## See Also

- [Architecture: Validation Pipeline](../../architecture/system-overview.md#3-validation-pipeline)
- [ADR-003: Validation Framework Approach](../../adr/ADR-003-validation-framework-approach.md)
- [Examples: validate_asp_program.py](../../examples/validate_asp_program.py)
- [Source: loft/validation/](../../../loft/validation/)

---

**Last Updated**: 2025-11-27
