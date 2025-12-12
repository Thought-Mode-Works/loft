"""
Unit tests for empirical validator.

Tests empirical validation of LLM-generated ASP rules using test cases.
"""

import pytest
from loft.validation.empirical_validator import EmpiricalValidator
from loft.validation.validation_schemas import TestCase, EmpiricalValidationResult


class TestEmpiricalValidator:
    """Tests for EmpiricalValidator class."""

    @pytest.fixture
    def validator(self):
        """Create an empirical validator for testing."""
        return EmpiricalValidator(min_accuracy_threshold=0.7)

    @pytest.fixture
    def sample_test_cases(self):
        """Create sample test cases."""
        return [
            TestCase(
                case_id="tc1",
                description="Contract with writing is enforceable",
                facts="contract(c1). has_writing(c1).",
                query="enforceable",
                expected=True,
                category="positive",
            ),
            TestCase(
                case_id="tc2",
                description="Contract without writing is not enforceable",
                facts="contract(c2).",
                query="enforceable",
                expected=False,
                category="negative",
            ),
            TestCase(
                case_id="tc3",
                description="Void contract is not enforceable",
                facts="contract(c3). has_writing(c3). void(c3).",
                query="enforceable",
                expected=False,
                category="negative",
            ),
        ]

    def test_validate_perfect_rule(self, validator, sample_test_cases):
        """Test validation of a perfect rule (100% accuracy)."""
        rule = """
        enforceable(C) :- contract(C), has_writing(C), not void(C).
        """

        result = validator.validate_rule(rule, sample_test_cases)

        assert isinstance(result, EmpiricalValidationResult)
        assert result.accuracy == 1.0
        assert result.test_cases_passed == 3
        assert result.test_cases_failed == 0
        assert result.is_valid
        assert len(result.failures) == 0

    def test_validate_imperfect_rule(self, validator, sample_test_cases):
        """Test validation of an imperfect rule."""
        # This rule doesn't check for void, so tc3 will fail
        rule = "enforceable(C) :- contract(C), has_writing(C)."

        result = validator.validate_rule(rule, sample_test_cases)

        assert isinstance(result, EmpiricalValidationResult)
        assert result.accuracy < 1.0
        assert result.test_cases_failed > 0
        assert len(result.failures) > 0

    def test_validate_rule_with_baseline(self, validator, sample_test_cases):
        """Test validation with baseline comparison."""
        baseline = "contract(C) :- contract_fact(C)."
        rule = "enforceable(C) :- contract(C), has_writing(C), not void(C)."

        result = validator.validate_rule(rule, sample_test_cases, baseline)

        assert isinstance(result, EmpiricalValidationResult)
        assert result.baseline_accuracy >= 0.0
        assert result.improvement >= 0.0  # Should improve from baseline

    def test_validate_rule_accuracy_threshold(self, validator):
        """Test that accuracy threshold is enforced."""
        # Create test cases where rule gets 60% accuracy (below 70% threshold)
        test_cases = [
            TestCase(
                case_id="tc1", description="desc", facts="a.", query="a", expected=True
            ),
            TestCase(
                case_id="tc2", description="desc", facts="b.", query="a", expected=False
            ),
            TestCase(
                case_id="tc3", description="desc", facts="c.", query="a", expected=False
            ),
            TestCase(
                case_id="tc4", description="desc", facts="d.", query="a", expected=False
            ),
            TestCase(
                case_id="tc5", description="desc", facts="e.", query="a", expected=False
            ),
        ]

        # Rule that only gets tc1 and tc2 correct (40%)
        rule = "a."

        result = validator.validate_rule(rule, test_cases)

        # Should fail because accuracy < 70%
        assert not result.is_valid

    def test_validate_rule_improvements(self, validator):
        """Test tracking of improvements over baseline."""
        baseline = ""  # No baseline rules
        rule = "enforceable(C) :- contract(C)."

        test_cases = [
            TestCase(
                case_id="tc1",
                description="desc",
                facts="contract(c1).",
                query="enforceable",
                expected=True,
            ),
            TestCase(
                case_id="tc2",
                description="desc",
                facts="contract(c2).",
                query="enforceable",
                expected=True,
            ),
        ]

        result = validator.validate_rule(rule, test_cases, baseline)

        # Should show improvements
        assert len(result.improvements) > 0
        assert all(isinstance(tc, TestCase) for tc in result.improvements)

    def test_validate_rule_regressions(self, validator):
        """Test detection of regressions."""
        # Baseline that derives enforceable when there's a contract
        baseline = "contract(C) :- contract_fact(C)."

        # Test case expects enforceable to be true
        test_cases = [
            TestCase(
                case_id="tc1",
                description="desc",
                facts="contract_fact(c1).",
                query="enforceable",
                expected=True,
            ),
        ]

        # This rule will NOT derive enforceable(c1) even though baseline has contract(c1)
        # because has_writing(c1) is not defined
        rule = "enforceable(C) :- contract(C), has_writing(C)."

        validator.validate_rule(rule, test_cases, baseline)

        # Should detect failure (baseline won't help because baseline doesn't derive enforceable)
        # Actually this won't show regression because baseline also doesn't derive enforceable
        # Let me change to a better example

        # Better: baseline derives enforceable, new rule doesn't
        baseline2 = "enforceable(C) :- contract(C). contract(C) :- contract_fact(C)."
        rule2 = ":- enforceable(C)."  # Constraint that prevents enforceable

        result2 = validator.validate_rule(rule2, test_cases, baseline2)

        # Should have a failure
        assert len(result2.failures) > 0

    def test_validate_batch(self, validator, sample_test_cases):
        """Test batch validation of multiple rules."""
        rules = [
            "enforceable(C) :- contract(C), has_writing(C), not void(C).",
            "enforceable(C) :- contract(C).",
            "enforceable(C) :- has_writing(C).",
        ]

        results = validator.validate_batch(rules, sample_test_cases)

        assert len(results) == 3
        assert all(isinstance(r, EmpiricalValidationResult) for r in results)
        # First rule should be best
        assert results[0].accuracy >= results[1].accuracy

    def test_create_test_case_from_dict(self, validator):
        """Test creating TestCase from dictionary."""
        data = {
            "case_id": "tc1",
            "description": "Test description",
            "facts": "fact(a).",
            "query": "fact",
            "expected": True,
            "category": "positive",
        }

        tc = validator.create_test_case_from_dict(data)

        assert isinstance(tc, TestCase)
        assert tc.case_id == "tc1"
        assert tc.description == "Test description"
        assert tc.query == "fact"
        assert tc.expected is True

    def test_get_validation_stats(self, validator):
        """Test aggregate statistics calculation."""
        results = [
            EmpiricalValidationResult(
                accuracy=0.8,
                baseline_accuracy=0.5,
                improvement=0.3,
                test_cases_passed=8,
                test_cases_failed=2,
                total_test_cases=10,
                failures=[],
                improvements=[],
                is_valid=True,
            ),
            EmpiricalValidationResult(
                accuracy=0.9,
                baseline_accuracy=0.6,
                improvement=0.3,
                test_cases_passed=9,
                test_cases_failed=1,
                total_test_cases=10,
                failures=[],
                improvements=[],
                is_valid=True,
            ),
        ]

        stats = validator.get_validation_stats(results)

        assert abs(stats["mean_accuracy"] - 0.85) < 0.001  # Floating point tolerance
        assert abs(stats["mean_improvement"] - 0.3) < 0.001
        assert stats["total_rules"] == 2
        assert stats["rules_passed"] == 2
        assert stats["pass_rate"] == 1.0

    def test_boolean_query_type(self, validator):
        """Test handling of boolean query results."""
        test_cases = [
            TestCase(
                case_id="tc1", description="desc", facts="a.", query="a", expected=True
            ),
            TestCase(
                case_id="tc2", description="desc", facts="b.", query="a", expected=False
            ),
        ]

        rule = "a."

        result = validator.validate_rule(rule, test_cases)

        # Should correctly interpret boolean results
        assert result.test_cases_passed == 1
        assert result.test_cases_failed == 1

    def test_count_query_type(self, validator):
        """Test handling of count query results."""
        test_cases = [
            TestCase(
                case_id="tc1",
                description="desc",
                facts="a. a. a.",
                query="a",
                expected=3,
            ),
        ]

        rule = "a."

        # Note: Clingo will deduplicate facts, so we get 1 not 3
        result = validator.validate_rule(rule, test_cases)

        # This will fail because Clingo deduplicates
        assert result.test_cases_failed == 1

    def test_validation_result_summary(self, validator, sample_test_cases):
        """Test that EmpiricalValidationResult summary works."""
        rule = "enforceable(C) :- contract(C), has_writing(C), not void(C)."

        result = validator.validate_rule(rule, sample_test_cases)

        summary = result.summary()
        assert "Empirical Validation" in summary
        assert "Accuracy" in summary
        assert "PASS" in summary or "FAIL" in summary

    def test_empty_test_cases(self, validator):
        """Test validation with no test cases."""
        result = validator.validate_rule("a.", [])

        assert result.accuracy == 0.0
        assert result.total_test_cases == 0
        # With empty test cases and 0% accuracy, it will fail the 70% threshold
        assert not result.is_valid

    def test_failure_case_details(self, validator):
        """Test that failure cases contain detailed information."""
        rule = "enforceable(C) :- contract(C)."  # Missing writing check

        test_cases = [
            TestCase(
                case_id="tc1",
                description="desc",
                facts="contract(c1).",
                query="enforceable",
                expected=False,  # Should not be enforceable without writing
            ),
        ]

        result = validator.validate_rule(rule, test_cases)

        assert len(result.failures) > 0
        failure = result.failures[0]
        assert failure.test_case.case_id == "tc1"
        assert failure.expected is False
        assert failure.actual is True
