"""
Unit tests for partial candidate acceptance in gap-filling responses.

Tests for issue #164: Excessive API calls due to retry mechanism with structured output validation.

These tests verify that:
1. Valid candidates pass through unchanged
2. Invalid candidates are filtered (not rejected entirely)
3. Metrics are tracked correctly
4. Recommended index is adjusted after filtering
5. All-invalid scenarios raise appropriate errors
"""

import pytest

from loft.neural.rule_schemas import (
    CandidateValidationMetrics,
    GapFillingResponse,
    GeneratedRule,
    LenientGapFillingResponse,
    LenientGeneratedRule,
    LenientRuleCandidate,
    RuleCandidate,
    ValidationFailureRecord,
    get_validation_metrics,
    parse_gap_filling_response_lenient,
    reset_validation_metrics,
)


def make_valid_rule_data(asp_rule: str = "valid(X) :- fact(X).") -> dict:
    """Create valid rule data for testing."""
    return {
        "asp_rule": asp_rule,
        "confidence": 0.9,
        "reasoning": "Test reasoning",
        "predicates_used": ["fact"],
        "new_predicates": ["valid"],
        "alternative_formulations": [],
        "source_type": "gap_fill",
        "source_text": "Test source",
        "citation": None,
        "jurisdiction": None,
    }


def make_candidate_data(asp_rule: str = "valid(X) :- fact(X).") -> dict:
    """Create valid candidate data for testing."""
    return {
        "rule": make_valid_rule_data(asp_rule),
        "applicability_score": 0.85,
        "test_cases_passed": 5,
        "test_cases_failed": 0,
        "complexity_score": 0.3,
    }


def make_gap_filling_response_data(candidates: list[dict], recommended_index: int = 0) -> dict:
    """Create gap-filling response data for testing."""
    return {
        "gap_description": "Test gap",
        "missing_predicate": "test_pred",
        "candidates": candidates,
        "recommended_index": recommended_index,
        "requires_validation": True,
        "test_cases_needed": ["case1", "case2"],
        "confidence": 0.8,
    }


class TestCandidateValidationMetrics:
    """Tests for CandidateValidationMetrics dataclass."""

    def test_record_valid_increments_counts(self):
        """Test that recording valid candidates increments counts."""
        metrics = CandidateValidationMetrics()

        metrics.record_valid()
        metrics.record_valid()

        assert metrics.total_candidates_received == 2
        assert metrics.valid_candidates_accepted == 2
        assert metrics.invalid_candidates_filtered == 0

    def test_record_invalid_increments_counts_and_records(self):
        """Test that recording invalid candidates tracks properly."""
        metrics = CandidateValidationMetrics()

        metrics.record_invalid(0, "Syntax error", "bad_rule(")
        metrics.record_invalid(2, "Missing period", "another_bad")

        assert metrics.total_candidates_received == 2
        assert metrics.valid_candidates_accepted == 0
        assert metrics.invalid_candidates_filtered == 2
        assert len(metrics.failure_records) == 2
        assert metrics.failure_records[0].candidate_index == 0
        assert metrics.failure_records[1].candidate_index == 2

    def test_get_acceptance_rate_with_mixed_results(self):
        """Test acceptance rate calculation."""
        metrics = CandidateValidationMetrics()

        metrics.record_valid()
        metrics.record_valid()
        metrics.record_invalid(2, "error", "bad")

        assert metrics.get_acceptance_rate() == pytest.approx(2 / 3)

    def test_get_acceptance_rate_with_no_candidates(self):
        """Test acceptance rate returns 1.0 when no candidates."""
        metrics = CandidateValidationMetrics()

        assert metrics.get_acceptance_rate() == 1.0

    def test_get_summary_includes_all_fields(self):
        """Test that summary includes all expected fields."""
        metrics = CandidateValidationMetrics()
        metrics.record_valid()
        metrics.record_invalid(1, "error message", "bad_rule")

        summary = metrics.get_summary()

        assert "total_candidates" in summary
        assert "valid_accepted" in summary
        assert "invalid_filtered" in summary
        assert "acceptance_rate" in summary
        assert "recent_failures" in summary
        assert summary["total_candidates"] == 2
        assert summary["valid_accepted"] == 1
        assert summary["invalid_filtered"] == 1


class TestValidationFailureRecord:
    """Tests for ValidationFailureRecord dataclass."""

    def test_creates_timestamp_automatically(self):
        """Test that timestamp is set on creation."""
        record = ValidationFailureRecord(
            candidate_index=0,
            error_message="Test error",
            asp_rule_attempted="bad_rule(",
        )

        assert record.timestamp != ""
        assert "T" in record.timestamp  # ISO format includes 'T'


class TestLenientGapFillingResponse:
    """Tests for LenientGapFillingResponse partial acceptance."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_validation_metrics()

    def test_all_valid_candidates_pass_through(self):
        """Test that all valid candidates are preserved."""
        candidates = [
            make_candidate_data("rule_a(X) :- pred(X)."),
            make_candidate_data("rule_b(X) :- other(X)."),
        ]
        data = make_gap_filling_response_data(candidates, recommended_index=1)

        lenient = LenientGapFillingResponse.model_validate(data)
        strict = lenient.to_strict_response()

        assert len(strict.candidates) == 2
        assert strict.recommended_index == 1

        metrics = get_validation_metrics()
        assert metrics.valid_candidates_accepted == 2
        assert metrics.invalid_candidates_filtered == 0

    def test_invalid_candidate_filtered_valid_preserved(self):
        """Test that invalid candidates are filtered while valid ones preserved."""
        candidates = [
            make_candidate_data("valid_rule(X) :- fact(X)."),
            make_candidate_data("invalid_rule("),  # Invalid: missing closing paren
            make_candidate_data("another_valid(Y) :- other(Y)."),
        ]
        data = make_gap_filling_response_data(candidates, recommended_index=0)

        lenient = LenientGapFillingResponse.model_validate(data)
        strict = lenient.to_strict_response()

        # Should have 2 valid candidates
        assert len(strict.candidates) == 2
        assert "valid_rule" in strict.candidates[0].rule.asp_rule
        assert "another_valid" in strict.candidates[1].rule.asp_rule

        metrics = get_validation_metrics()
        assert metrics.valid_candidates_accepted == 2
        assert metrics.invalid_candidates_filtered == 1

    def test_all_invalid_candidates_raises_error(self):
        """Test that all invalid candidates raises ValueError."""
        candidates = [
            make_candidate_data("bad_rule("),  # Invalid
            make_candidate_data("another_bad{"),  # Invalid
        ]
        data = make_gap_filling_response_data(candidates)

        lenient = LenientGapFillingResponse.model_validate(data)

        with pytest.raises(ValueError) as exc_info:
            lenient.to_strict_response()

        assert "All 2 candidates were invalid" in str(exc_info.value)

    def test_recommended_index_adjusted_when_earlier_filtered(self):
        """Test that recommended index is adjusted when earlier candidates filtered."""
        candidates = [
            make_candidate_data("invalid("),  # Index 0 - invalid
            make_candidate_data("valid_a(X) :- fact(X)."),  # Index 1 - valid -> new 0
            make_candidate_data("valid_b(Y) :- other(Y)."),  # Index 2 - valid -> new 1
        ]
        # Recommend index 2 (third candidate)
        data = make_gap_filling_response_data(candidates, recommended_index=2)

        lenient = LenientGapFillingResponse.model_validate(data)
        strict = lenient.to_strict_response()

        # Index 2 becomes index 1 after filtering index 0
        assert strict.recommended_index == 1

    def test_recommended_index_reset_when_recommended_filtered(self):
        """Test that recommended index becomes 0 when recommended candidate filtered."""
        candidates = [
            make_candidate_data("valid(X) :- fact(X)."),  # Index 0 - valid
            make_candidate_data("invalid("),  # Index 1 - invalid (was recommended)
        ]
        data = make_gap_filling_response_data(candidates, recommended_index=1)

        lenient = LenientGapFillingResponse.model_validate(data)
        strict = lenient.to_strict_response()

        # Recommended was filtered, should fall back to 0
        assert strict.recommended_index == 0

    def test_get_filtered_candidates_returns_info(self):
        """Test that filtered candidate info is accessible."""
        candidates = [
            make_candidate_data("valid(X) :- fact(X)."),
            make_candidate_data("bad_rule("),
        ]
        data = make_gap_filling_response_data(candidates)

        lenient = LenientGapFillingResponse.model_validate(data)
        lenient.to_strict_response()

        filtered = lenient.get_filtered_candidates()
        assert len(filtered) == 1
        assert filtered[0][0] == 1  # Original index
        assert "bad_rule(" in filtered[0][2]  # ASP rule


class TestParseGapFillingResponseLenient:
    """Tests for the convenience function."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_validation_metrics()

    def test_returns_strict_response_and_filtered_info(self):
        """Test that function returns both response and filtered info."""
        candidates = [
            make_candidate_data("valid(X) :- fact(X)."),
            make_candidate_data("invalid("),
        ]
        data = make_gap_filling_response_data(candidates)

        strict, filtered = parse_gap_filling_response_lenient(data)

        assert isinstance(strict, GapFillingResponse)
        assert len(strict.candidates) == 1
        assert len(filtered) == 1

    def test_raises_on_all_invalid(self):
        """Test that function raises when all candidates invalid."""
        candidates = [
            make_candidate_data("bad("),
        ]
        data = make_gap_filling_response_data(candidates)

        with pytest.raises(ValueError):
            parse_gap_filling_response_lenient(data)


class TestLenientGeneratedRule:
    """Tests for LenientGeneratedRule."""

    def test_accepts_invalid_asp_without_validation(self):
        """Test that lenient rule accepts invalid ASP during parsing."""
        data = make_valid_rule_data("completely_invalid{{{")

        # Should not raise
        lenient = LenientGeneratedRule.model_validate(data)
        assert lenient.asp_rule == "completely_invalid{{{"

    def test_to_generated_rule_validates_asp(self):
        """Test that conversion to strict validates ASP."""
        data = make_valid_rule_data("invalid_syntax(")
        lenient = LenientGeneratedRule.model_validate(data)

        with pytest.raises(ValueError):
            lenient.to_generated_rule()

    def test_valid_rule_converts_successfully(self):
        """Test that valid rules convert without error."""
        data = make_valid_rule_data("valid_rule(X) :- fact(X).")
        lenient = LenientGeneratedRule.model_validate(data)

        strict = lenient.to_generated_rule()
        assert isinstance(strict, GeneratedRule)
        assert strict.asp_rule == "valid_rule(X) :- fact(X)."


class TestLenientRuleCandidate:
    """Tests for LenientRuleCandidate."""

    def test_to_rule_candidate_validates_inner_rule(self):
        """Test that conversion validates the inner rule."""
        data = make_candidate_data("bad_rule(")

        lenient = LenientRuleCandidate.model_validate(data)

        with pytest.raises(ValueError):
            lenient.to_rule_candidate()

    def test_valid_candidate_converts_successfully(self):
        """Test that valid candidates convert without error."""
        data = make_candidate_data("good_rule(X) :- fact(X).")

        lenient = LenientRuleCandidate.model_validate(data)
        strict = lenient.to_rule_candidate()

        assert isinstance(strict, RuleCandidate)


class TestGlobalMetrics:
    """Tests for global metrics tracking."""

    def test_reset_validation_metrics_clears_all(self):
        """Test that reset clears all metrics."""
        metrics = get_validation_metrics()
        metrics.record_valid()
        metrics.record_invalid(0, "error", "bad")

        reset_validation_metrics()

        metrics = get_validation_metrics()
        assert metrics.total_candidates_received == 0
        assert metrics.valid_candidates_accepted == 0
        assert metrics.invalid_candidates_filtered == 0
        assert len(metrics.failure_records) == 0


class TestEdgeCases:
    """Tests for edge cases."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_validation_metrics()

    def test_single_valid_candidate(self):
        """Test with single valid candidate."""
        candidates = [make_candidate_data("single(X) :- fact(X).")]
        data = make_gap_filling_response_data(candidates)

        strict, filtered = parse_gap_filling_response_lenient(data)

        assert len(strict.candidates) == 1
        assert len(filtered) == 0

    def test_complex_valid_asp_rule(self):
        """Test with complex but valid ASP rule."""
        complex_rule = (
            "enforceable(C) :- contract(C), offer(C, O), acceptance(C, A), "
            "consideration(C, V), V > 0, not void(C), not voidable(C)."
        )
        candidates = [make_candidate_data(complex_rule)]
        data = make_gap_filling_response_data(candidates)

        strict, filtered = parse_gap_filling_response_lenient(data)

        assert len(strict.candidates) == 1
        assert complex_rule in strict.candidates[0].rule.asp_rule

    def test_many_candidates_with_varied_validity(self):
        """Test with many candidates, some valid some not."""
        candidates = [
            make_candidate_data("valid_1(X) :- a(X)."),  # Valid
            make_candidate_data("invalid_1("),  # Invalid
            make_candidate_data("valid_2(X) :- b(X)."),  # Valid
            make_candidate_data("invalid_2{"),  # Invalid
            make_candidate_data("valid_3(X) :- c(X)."),  # Valid
        ]
        data = make_gap_filling_response_data(candidates, recommended_index=4)

        strict, filtered = parse_gap_filling_response_lenient(data)

        assert len(strict.candidates) == 3
        assert len(filtered) == 2

        # Original index 4 (valid_3) should become index 2
        # (after filtering indices 1 and 3)
        assert strict.recommended_index == 2

        metrics = get_validation_metrics()
        assert metrics.get_acceptance_rate() == pytest.approx(0.6)
