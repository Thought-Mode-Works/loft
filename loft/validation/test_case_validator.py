"""
Empirical validation through test case execution.

This module runs ASP programs against test cases and measures accuracy.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import clingo
from loguru import logger


@dataclass
class TestCase:
    """
    A test case for validating ASP programs.

    Example:
        >>> test = TestCase(
        ...     case_id="test_1",
        ...     description="Contract with writing",
        ...     asp_facts="contract(c1). has_writing(c1).",
        ...     expected_results={"enforceable": True}
        ... )
    """

    case_id: str
    description: str
    asp_facts: str  # Facts to add for this test case
    expected_results: Dict[str, bool]  # predicate_name -> expected truth value
    reasoning_chain: List[str] = field(default_factory=list)  # Expected inference steps (optional)
    confidence_level: str = "high"  # "high", "medium", "low"


@dataclass
class TestResult:
    """Result from running a single test case."""

    test_case: TestCase
    passed: bool
    actual_results: Dict[str, bool]
    mismatches: List[str]
    explanation: str
    error: Optional[str] = None


class TestCaseValidator:
    """Validates ASP programs against test cases."""

    def validate_test_case(self, asp_program: str, test_case: TestCase) -> TestResult:
        """
        Run ASP program with test case facts and check results.

        Args:
            asp_program: ASP program text (rules)
            test_case: TestCase with facts and expected results

        Returns:
            TestResult with pass/fail and details

        Example:
            >>> validator = TestCaseValidator()
            >>> program = "enforceable(C) :- contract(C), has_writing(C)."
            >>> test = TestCase(
            ...     case_id="test_1",
            ...     description="Contract with writing",
            ...     asp_facts="contract(c1). has_writing(c1).",
            ...     expected_results={"enforceable": True}
            ... )
            >>> result = validator.validate_test_case(program, test)
            >>> assert result.passed
        """
        try:
            ctl = clingo.Control()

            # Add the ASP program (rules)
            ctl.add("base", [], asp_program)

            # Add test case facts
            ctl.add("base", [], test_case.asp_facts)

            # Ground the program
            ctl.ground([("base", [])])

            # Collect results
            actual_results = {}

            def on_model(model: clingo.Model) -> None:
                # For each expected predicate, check if it appears in answer set
                for pred_name in test_case.expected_results.keys():
                    # Check if any atom with this predicate name exists
                    atoms = [atom for atom in model.symbols(shown=True) if atom.name == pred_name]
                    actual_results[pred_name] = len(atoms) > 0

            # Solve
            result = ctl.solve(on_model=on_model)

            if not result.satisfiable:
                # No answer sets - predicates are false
                for pred_name in test_case.expected_results.keys():
                    actual_results[pred_name] = False

            # Compare expected vs actual
            mismatches = [
                pred
                for pred, expected in test_case.expected_results.items()
                if actual_results.get(pred, False) != expected
            ]

            passed = len(mismatches) == 0

            explanation = self._generate_explanation(test_case, actual_results, passed)

            logger.debug(f"Test {test_case.case_id}: {'PASS' if passed else 'FAIL'}")

            return TestResult(
                test_case=test_case,
                passed=passed,
                actual_results=actual_results,
                mismatches=mismatches,
                explanation=explanation,
            )

        except Exception as e:
            error_msg = f"Error running test case: {str(e)}"
            logger.error(f"Test {test_case.case_id}: ERROR - {error_msg}")

            return TestResult(
                test_case=test_case,
                passed=False,
                actual_results={},
                mismatches=list(test_case.expected_results.keys()),
                explanation=error_msg,
                error=error_msg,
            )

    def _generate_explanation(
        self,
        test_case: TestCase,
        actual_results: Dict[str, bool],
        passed: bool,
    ) -> str:
        """Generate human-readable explanation of test result."""
        lines = [
            f"Test: {test_case.case_id} - {test_case.description}",
            f"Result: {'PASS' if passed else 'FAIL'}",
            "",
            "Expected vs Actual:",
        ]

        for pred, expected in test_case.expected_results.items():
            actual = actual_results.get(pred, False)
            match = "✓" if actual == expected else "✗"
            lines.append(f"  {match} {pred}: expected={expected}, actual={actual}")

        return "\n".join(lines)

    def batch_validate(self, asp_program: str, test_cases: List[TestCase]) -> Dict[str, Any]:
        """
        Run all test cases and compute metrics.

        Args:
            asp_program: ASP program to test
            test_cases: List of TestCase instances

        Returns:
            Dictionary with validation statistics:
            {
                "accuracy": float,
                "passed": int,
                "failed": int,
                "total": int,
                "failed_cases": List[Tuple[str, TestResult]],
                "results": List[TestResult]
            }

        Example:
            >>> validator = TestCaseValidator()
            >>> program = "a :- b."
            >>> tests = [
            ...     TestCase("t1", "test 1", "b.", {"a": True}),
            ...     TestCase("t2", "test 2", "c.", {"a": False})
            ... ]
            >>> stats = validator.batch_validate(program, tests)
            >>> assert stats["accuracy"] == 1.0  # Both tests pass
        """
        results = []

        for test_case in test_cases:
            result = self.validate_test_case(asp_program, test_case)
            results.append(result)

        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        accuracy = passed_count / len(results) if results else 0.0

        failed_cases = [(r.test_case.case_id, r) for r in results if not r.passed]

        stats = {
            "accuracy": accuracy,
            "passed": passed_count,
            "failed": failed_count,
            "total": len(results),
            "failed_cases": failed_cases,
            "results": results,
        }

        logger.info(
            f"Batch validation: {passed_count}/{len(results)} passed " f"(accuracy={accuracy:.2%})"
        )

        if failed_cases:
            logger.warning(f"Failed test cases: {[fc[0] for fc in failed_cases]}")

        return stats

    def validate_with_explanation(
        self, asp_program: str, test_case: TestCase
    ) -> Tuple[TestResult, str]:
        """
        Validate test case and return detailed explanation.

        Args:
            asp_program: ASP program
            test_case: Test case to run

        Returns:
            Tuple of (TestResult, detailed_explanation)
        """
        result = self.validate_test_case(asp_program, test_case)

        detailed_explanation = f"""
{result.explanation}

Test Case Facts:
{test_case.asp_facts}

Expected Reasoning:
{chr(10).join('  - ' + step for step in test_case.reasoning_chain) if test_case.reasoning_chain else '  (not specified)'}

Actual Results:
{chr(10).join(f'  {pred}: {value}' for pred, value in result.actual_results.items())}
        """.strip()

        return (result, detailed_explanation)


def create_test_suite(test_cases_data: List[Dict[str, Any]]) -> List[TestCase]:
    """
    Create a test suite from dictionary data.

    Args:
        test_cases_data: List of test case dictionaries

    Returns:
        List of TestCase instances

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
    """
    test_suite = []

    for data in test_cases_data:
        test_case = TestCase(
            case_id=data["case_id"],
            description=data["description"],
            asp_facts=data["asp_facts"],
            expected_results=data["expected_results"],
            reasoning_chain=data.get("reasoning_chain", []),
            confidence_level=data.get("confidence_level", "high"),
        )
        test_suite.append(test_case)

    logger.info(f"Created test suite with {len(test_suite)} test cases")
    return test_suite
