"""
Empirical validator for LLM-generated ASP rules.

This module validates rules by testing them against labeled test cases,
measuring accuracy and performance improvement/degradation.
"""

from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import sys
import os
from loguru import logger
import clingo

from loft.validation.validation_schemas import (
    TestCase,
    FailureCase,
    EmpiricalValidationResult,
)


@contextmanager
def suppress_clingo_warnings():
    """
    Context manager to suppress Clingo informational warnings.

    Clingo outputs warnings like "atom does not occur in any rule head"
    when validating individual rules without full context. These are
    expected and not meaningful for our validation purposes.
    """
    # Save original stderr
    old_stderr = sys.stderr
    try:
        # Redirect stderr to devnull
        sys.stderr = open(os.devnull, "w")
        yield
    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = old_stderr


class EmpiricalValidator:
    """
    Validates rules empirically using labeled test cases.

    Measures actual performance on test data to determine if a rule
    improves reasoning accuracy.
    """

    def __init__(self, min_accuracy_threshold: float = 0.7):
        """
        Initialize empirical validator.

        Args:
            min_accuracy_threshold: Minimum accuracy required for validation (0.0-1.0)
        """
        self.min_accuracy_threshold = min_accuracy_threshold
        logger.debug(
            f"Initialized EmpiricalValidator with threshold={min_accuracy_threshold}"
        )

    def validate_rule(
        self,
        rule_text: str,
        test_cases: List[TestCase],
        baseline_rules: Optional[str] = None,
    ) -> EmpiricalValidationResult:
        """
        Validate rule against labeled test cases.

        Args:
            rule_text: The ASP rule to validate
            test_cases: List of labeled test cases
            baseline_rules: Baseline ASP rules (without the new rule)

        Returns:
            EmpiricalValidationResult with accuracy metrics

        Example:
            >>> validator = EmpiricalValidator()
            >>> test_cases = [
            ...     TestCase(
            ...         case_id="tc1",
            ...         description="Valid contract",
            ...         facts="contract(c1). has_writing(c1).",
            ...         query="enforceable",
            ...         expected=True
            ...     )
            ... ]
            >>> result = validator.validate_rule(
            ...     "enforceable(C) :- contract(C), has_writing(C).",
            ...     test_cases
            ... )
            >>> assert result.is_valid
        """
        # Evaluate with new rule
        with_rule_results = self._evaluate_test_cases(
            test_cases, rule_text, baseline_rules
        )

        # Evaluate baseline (without new rule)
        baseline_results = (
            self._evaluate_test_cases(test_cases, None, baseline_rules)
            if baseline_rules
            else {"passed": 0, "failed": len(test_cases), "failures": []}
        )

        # Calculate metrics
        total = len(test_cases)
        passed = with_rule_results["passed"]
        failed = with_rule_results["failed"]

        accuracy = passed / total if total > 0 else 0.0
        baseline_accuracy = (
            baseline_results["passed"] / total if total > 0 and baseline_rules else 0.0
        )
        improvement = accuracy - baseline_accuracy

        # Identify improvements and regressions
        improvements = []
        failures = []

        for tc in test_cases:
            with_rule_correct = self._is_test_case_correct(
                tc, rule_text, baseline_rules
            )
            baseline_correct = (
                self._is_test_case_correct(tc, None, baseline_rules)
                if baseline_rules
                else False
            )

            if with_rule_correct and not baseline_correct:
                # Improvement!
                improvements.append(tc)
            elif not with_rule_correct:
                # Failure
                actual = self._get_test_case_result(tc, rule_text, baseline_rules)
                baseline = (
                    self._get_test_case_result(tc, None, baseline_rules)
                    if baseline_rules
                    else None
                )

                failure_type = "incorrect"
                if baseline_correct and not with_rule_correct:
                    failure_type = "regression"

                failures.append(
                    FailureCase(
                        test_case=tc,
                        expected=tc.expected,
                        actual=actual,
                        baseline=baseline,
                        failure_type=failure_type,
                    )
                )

        # Determine if valid (meets threshold)
        is_valid = accuracy >= self.min_accuracy_threshold

        return EmpiricalValidationResult(
            accuracy=accuracy,
            baseline_accuracy=baseline_accuracy,
            improvement=improvement,
            test_cases_passed=passed,
            test_cases_failed=failed,
            total_test_cases=total,
            failures=failures,
            improvements=improvements,
            is_valid=is_valid,
        )

    def _evaluate_test_cases(
        self,
        test_cases: List[TestCase],
        rule_text: Optional[str],
        baseline_rules: Optional[str],
    ) -> Dict[str, Any]:
        """
        Evaluate all test cases with given rules.

        Args:
            test_cases: Test cases to evaluate
            rule_text: New rule (None for baseline)
            baseline_rules: Base rules

        Returns:
            Dict with passed/failed counts and failures
        """
        passed = 0
        failed = 0
        failures = []

        for tc in test_cases:
            is_correct = self._is_test_case_correct(tc, rule_text, baseline_rules)

            if is_correct:
                passed += 1
            else:
                failed += 1
                failures.append(tc)

        return {"passed": passed, "failed": failed, "failures": failures}

    def _is_test_case_correct(
        self,
        test_case: TestCase,
        rule_text: Optional[str],
        baseline_rules: Optional[str],
    ) -> bool:
        """
        Check if test case produces expected result.

        Args:
            test_case: Test case to check
            rule_text: New rule (None for baseline)
            baseline_rules: Base rules

        Returns:
            True if result matches expected
        """
        actual = self._get_test_case_result(test_case, rule_text, baseline_rules)
        return actual == test_case.expected

    def _get_test_case_result(
        self,
        test_case: TestCase,
        rule_text: Optional[str],
        baseline_rules: Optional[str],
    ) -> Any:
        """
        Get actual result for a test case.

        Args:
            test_case: Test case to evaluate
            rule_text: New rule (None for baseline)
            baseline_rules: Base rules

        Returns:
            Actual result from running the ASP program
        """
        # Build complete program
        program_parts = []

        if baseline_rules:
            program_parts.append(baseline_rules)

        if rule_text:
            program_parts.append(rule_text)

        program_parts.append(test_case.facts)

        program = "\n".join(program_parts)

        # Run and query (suppress Clingo informational warnings)
        try:
            with suppress_clingo_warnings():
                ctl = clingo.Control()
                ctl.add("base", [], program)
                ctl.ground([("base", [])])

                # Query for the predicate
                results = []

                def on_model(model):
                    for atom in model.symbols(shown=True):
                        if atom.name == test_case.query:
                            results.append(atom)

                ctl.solve(on_model=on_model)

            # Interpret result based on expected type
            if isinstance(test_case.expected, bool):
                # Boolean query: is predicate derived?
                return len(results) > 0

            elif isinstance(test_case.expected, int):
                # Count query: how many derivations?
                return len(results)

            elif isinstance(test_case.expected, list):
                # List query: which specific atoms?
                return [str(atom) for atom in results]

            else:
                # Default: return results as strings
                return [str(atom) for atom in results]

        except Exception as e:
            logger.error(f"Error evaluating test case {test_case.case_id}: {e}")
            # Return a value that won't match expected
            return None

    def validate_batch(
        self,
        rules: List[str],
        test_cases: List[TestCase],
        baseline_rules: Optional[str] = None,
    ) -> List[EmpiricalValidationResult]:
        """
        Validate multiple rules against test cases.

        Args:
            rules: List of rules to validate
            test_cases: Test cases to use
            baseline_rules: Baseline rules

        Returns:
            List of EmpiricalValidationResult objects
        """
        results = []

        for rule in rules:
            result = self.validate_rule(rule, test_cases, baseline_rules)
            results.append(result)

        return results

    def create_test_case_from_dict(self, data: Dict[str, Any]) -> TestCase:
        """
        Create TestCase from dictionary.

        Args:
            data: Dictionary with test case data

        Returns:
            TestCase object

        Example:
            >>> validator = EmpiricalValidator()
            >>> tc = validator.create_test_case_from_dict({
            ...     "case_id": "tc1",
            ...     "description": "Test",
            ...     "facts": "a.",
            ...     "query": "a",
            ...     "expected": True
            ... })
            >>> assert tc.case_id == "tc1"
        """
        return TestCase(
            case_id=data["case_id"],
            description=data["description"],
            facts=data["facts"],
            query=data["query"],
            expected=data["expected"],
            category=data.get("category", "general"),
        )

    def get_validation_stats(
        self, results: List[EmpiricalValidationResult]
    ) -> Dict[str, Any]:
        """
        Get aggregate statistics across multiple validation results.

        Args:
            results: List of validation results

        Returns:
            Dict with aggregate stats

        Example:
            >>> validator = EmpiricalValidator()
            >>> # ... run validations ...
            >>> stats = validator.get_validation_stats(results)
            >>> assert "mean_accuracy" in stats
        """
        if not results:
            return {
                "mean_accuracy": 0.0,
                "mean_improvement": 0.0,
                "total_rules": 0,
                "rules_passed": 0,
            }

        mean_accuracy = sum(r.accuracy for r in results) / len(results)
        mean_improvement = sum(r.improvement for r in results) / len(results)
        rules_passed = sum(1 for r in results if r.is_valid)

        return {
            "mean_accuracy": mean_accuracy,
            "mean_improvement": mean_improvement,
            "total_rules": len(results),
            "rules_passed": rules_passed,
            "pass_rate": rules_passed / len(results) if results else 0.0,
        }
