"""
Unit tests for test case validator.

Tests empirical validation through test case execution.
"""

from loft.validation import TestCaseData, TestCaseValidator, create_test_suite


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_create_test_case(self) -> None:
        """Test creating a test case."""
        test = TestCaseData(
            case_id="test_1",
            description="Simple test",
            asp_facts="fact(a).",
            expected_results={"fact": True},
        )

        assert test.case_id == "test_1"
        assert test.description == "Simple test"
        assert test.asp_facts == "fact(a)."
        assert test.expected_results == {"fact": True}
        assert test.reasoning_chain == []
        assert test.confidence_level == "high"

    def test_test_case_with_reasoning(self) -> None:
        """Test case with reasoning chain."""
        test = TestCaseData(
            case_id="test_2",
            description="Test with reasoning",
            asp_facts="a. b.",
            expected_results={"c": True},
            reasoning_chain=["First a", "Then b", "Therefore c"],
        )

        assert len(test.reasoning_chain) == 3


class TestTestCaseValidator:
    """Tests for TestCaseValidator."""

    def test_validate_passing_test_case(self) -> None:
        """Test validation of a passing test case."""
        validator = TestCaseValidator()

        program = "result(X) :- fact(X)."
        test = TestCaseData(
            case_id="test_pass",
            description="Should pass",
            asp_facts="fact(a).",
            expected_results={"result": True},
        )

        result = validator.validate_test_case(program, test)

        assert result.passed
        assert result.actual_results["result"] is True
        assert len(result.mismatches) == 0
        assert result.error is None

    def test_validate_failing_test_case(self) -> None:
        """Test validation of a failing test case."""
        validator = TestCaseValidator()

        program = "result(X) :- fact(X)."
        test = TestCaseData(
            case_id="test_fail",
            description="Should fail",
            asp_facts="other(a).",  # No fact(a)
            expected_results={"result": True},  # Expects result but won't get it
        )

        result = validator.validate_test_case(program, test)

        assert not result.passed
        assert result.actual_results["result"] is False
        assert "result" in result.mismatches

    def test_validate_multiple_predicates(self) -> None:
        """Test validation with multiple expected predicates."""
        validator = TestCaseValidator()

        program = """
        a :- input.
        b :- input.
        c :- a, b.
        """

        test = TestCaseData(
            case_id="test_multi",
            description="Multiple predicates",
            asp_facts="input.",
            expected_results={"a": True, "b": True, "c": True},
        )

        result = validator.validate_test_case(program, test)

        assert result.passed
        assert all(result.actual_results[pred] for pred in ["a", "b", "c"])

    def test_validate_contract_law_example(self) -> None:
        """Test validation of legal reasoning example."""
        validator = TestCaseValidator()

        program = """
        contract(C) :- contract_fact(C).
        has_writing(C) :- writing_fact(C).
        enforceable(C) :- contract(C), has_writing(C).
        """

        test = TestCaseData(
            case_id="contract_test",
            description="Enforceable contract with writing",
            asp_facts="contract_fact(c1). writing_fact(c1).",
            expected_results={"enforceable": True},
        )

        result = validator.validate_test_case(program, test)

        assert result.passed
        assert result.actual_results["enforceable"] is True

    def test_batch_validate(self) -> None:
        """Test batch validation of multiple test cases."""
        validator = TestCaseValidator()

        program = "result(X) :- input(X)."

        tests = [
            TestCaseData("t1", "Test 1", "input(a).", {"result": True}),
            TestCaseData("t2", "Test 2", "input(b).", {"result": True}),
            TestCaseData("t3", "Test 3", "other(c).", {"result": False}),
        ]

        stats = validator.batch_validate(program, tests)

        assert stats["total"] == 3
        assert stats["passed"] == 3
        assert stats["failed"] == 0
        assert stats["accuracy"] == 1.0

    def test_batch_validate_with_failures(self) -> None:
        """Test batch validation with some failures."""
        validator = TestCaseValidator()

        program = "result(a)."  # Only result(a) is true

        tests = [
            TestCaseData("t1", "Test 1", "", {"result": True}),  # Should pass
            TestCaseData(
                "t2", "Test 2", "", {"result": False}
            ),  # Should fail - expects false but gets true
        ]

        stats = validator.batch_validate(program, tests)

        assert stats["total"] == 2
        assert stats["passed"] == 1  # Only t1 passes
        assert stats["failed"] == 1
        assert stats["accuracy"] == 0.5
        assert len(stats["failed_cases"]) == 1

    def test_validate_with_explanation(self) -> None:
        """Test validation with detailed explanation."""
        validator = TestCaseValidator()

        program = "a :- b."
        test = TestCaseData(
            case_id="explain_test",
            description="Test with explanation",
            asp_facts="b.",
            expected_results={"a": True},
            reasoning_chain=["b is true", "Therefore a is true"],
        )

        result, explanation = validator.validate_with_explanation(program, test)

        assert result.passed
        assert "explain_test" in explanation
        assert "b." in explanation
        assert len(explanation) > 0

    def test_error_handling(self) -> None:
        """Test that validator handles errors gracefully."""
        validator = TestCaseValidator()

        # Invalid ASP program
        program = "this is invalid"
        test = TestCaseData("error_test", "Error test", "a.", {"result": True})

        result = validator.validate_test_case(program, test)

        assert not result.passed
        assert result.error is not None


class TestCreateTestSuite:
    """Tests for create_test_suite helper function."""

    def test_create_suite_from_data(self) -> None:
        """Test creating test suite from dictionary data."""
        data = [
            {
                "case_id": "test_1",
                "description": "First test",
                "asp_facts": "a.",
                "expected_results": {"a": True},
            },
            {
                "case_id": "test_2",
                "description": "Second test",
                "asp_facts": "b.",
                "expected_results": {"b": True},
                "reasoning_chain": ["b is fact"],
                "confidence_level": "medium",
            },
        ]

        suite = create_test_suite(data)

        assert len(suite) == 2
        assert all(isinstance(test, TestCaseData) for test in suite)
        assert suite[0].case_id == "test_1"
        assert suite[1].confidence_level == "medium"
        assert len(suite[1].reasoning_chain) == 1

    def test_create_empty_suite(self) -> None:
        """Test creating empty test suite."""
        suite = create_test_suite([])
        assert len(suite) == 0


class TestTestCaseValidatorIntegration:
    """Integration tests for test case validator."""

    def test_statute_of_frauds_test_cases(self) -> None:
        """Test validation with statute of frauds rules."""
        validator = TestCaseValidator()

        # Simplified statute of frauds rules
        program = """
        contract(C) :- contract_fact(C).
        land_sale(C) :- land_sale_fact(C).
        has_writing(C) :- writing_fact(C).

        within_statute(C) :- land_sale(C).
        satisfies_statute_of_frauds(C) :- within_statute(C), has_writing(C).
        enforceable(C) :- satisfies_statute_of_frauds(C).
        """

        test_cases = [
            TestCaseData(
                case_id="sof_pass",
                description="Land sale with writing - enforceable",
                asp_facts="contract_fact(c1). land_sale_fact(c1). writing_fact(c1).",
                expected_results={"enforceable": True},
            ),
            TestCaseData(
                case_id="sof_fail",
                description="Land sale without writing - not enforceable",
                asp_facts="contract_fact(c2). land_sale_fact(c2).",
                expected_results={"enforceable": False},
            ),
        ]

        stats = validator.batch_validate(program, test_cases)

        assert stats["accuracy"] == 1.0
        assert stats["passed"] == 2
        assert stats["failed"] == 0

    def test_negation_as_failure_test_cases(self) -> None:
        """Test validation with negation-as-failure."""
        validator = TestCaseValidator()

        program = """
        enforceable(C) :- contract(C), not unenforceable(C).
        unenforceable(C) :- contract(C), missing_requirement(C).
        """

        tests = [
            TestCaseData(
                "naf_1",
                "Enforceable by default",
                "contract(c1).",
                {"enforceable": True},
            ),
            TestCaseData(
                "naf_2",
                "Unenforceable due to missing requirement",
                "contract(c2). missing_requirement(c2).",
                {"enforceable": False, "unenforceable": True},
            ),
        ]

        stats = validator.batch_validate(program, tests)
        assert stats["accuracy"] == 1.0
