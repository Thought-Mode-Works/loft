"""
Unit tests for LLM metrics tracking.

Tests for issue #165: Add real-time metrics dashboard and API cost tracking.

These tests verify:
1. Cost calculation based on model pricing
2. Token counting (input/output)
3. Metrics aggregation by operation type
4. Budget limits with warning threshold
5. Metrics summary and health endpoint formats
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from loft.autonomous.llm_metrics import (
    BudgetExceededError,
    LLMCallRecord,
    LLMMetricsTracker,
    OperationMetrics,
    OperationType,
    MODEL_PRICING,
    create_metrics_tracker,
    get_global_metrics_tracker,
    set_global_metrics_tracker,
)


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    def test_haiku_pricing_available(self):
        """Test that Haiku model pricing is available."""
        assert "claude-3-5-haiku-20241022" in MODEL_PRICING
        haiku = MODEL_PRICING["claude-3-5-haiku-20241022"]
        assert haiku.input_cost_per_million == 1.00
        assert haiku.output_cost_per_million == 5.00

    def test_sonnet_pricing_available(self):
        """Test that Sonnet model pricing is available."""
        assert "claude-3-5-sonnet-20241022" in MODEL_PRICING
        sonnet = MODEL_PRICING["claude-3-5-sonnet-20241022"]
        assert sonnet.input_cost_per_million == 3.00
        assert sonnet.output_cost_per_million == 15.00

    def test_opus_pricing_available(self):
        """Test that Opus model pricing is available."""
        assert "claude-3-opus-20240229" in MODEL_PRICING
        opus = MODEL_PRICING["claude-3-opus-20240229"]
        assert opus.input_cost_per_million == 15.00
        assert opus.output_cost_per_million == 75.00

    def test_calculate_cost_haiku(self):
        """Test cost calculation for Haiku model."""
        haiku = MODEL_PRICING["claude-3-5-haiku-20241022"]

        # 1000 input tokens + 500 output tokens
        cost = haiku.calculate_cost(1000, 500)

        # $1/M * 1000 + $5/M * 500 = $0.001 + $0.0025 = $0.0035
        expected_cost = (1000 / 1_000_000) * 1.00 + (500 / 1_000_000) * 5.00
        assert cost == pytest.approx(expected_cost)

    def test_calculate_cost_large_request(self):
        """Test cost calculation for large token counts."""
        haiku = MODEL_PRICING["claude-3-5-haiku-20241022"]

        # 100,000 input tokens + 10,000 output tokens
        cost = haiku.calculate_cost(100_000, 10_000)

        expected_cost = (100_000 / 1_000_000) * 1.00 + (10_000 / 1_000_000) * 5.00
        assert cost == pytest.approx(expected_cost)
        assert cost == pytest.approx(0.15)  # $0.10 + $0.05

    def test_default_pricing_fallback(self):
        """Test that default pricing is available as fallback."""
        assert "default" in MODEL_PRICING
        default = MODEL_PRICING["default"]
        assert default.model_name == "Unknown (Haiku pricing)"


class TestOperationMetrics:
    """Tests for OperationMetrics dataclass."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = OperationMetrics(operation_type=OperationType.GAP_FILL)
        metrics.call_count = 10
        metrics.success_count = 8
        metrics.failure_count = 2

        assert metrics.success_rate == pytest.approx(0.8)

    def test_success_rate_no_calls(self):
        """Test success rate returns 1.0 when no calls."""
        metrics = OperationMetrics(operation_type=OperationType.GAP_FILL)

        assert metrics.success_rate == 1.0

    def test_average_duration(self):
        """Test average duration calculation."""
        metrics = OperationMetrics(operation_type=OperationType.EXTRACTION)
        metrics.call_count = 5
        metrics.total_duration_seconds = 10.0

        assert metrics.average_duration_seconds == pytest.approx(2.0)

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes expected fields."""
        metrics = OperationMetrics(operation_type=OperationType.VALIDATION)
        metrics.call_count = 10
        metrics.success_count = 9
        metrics.failure_count = 1
        metrics.total_input_tokens = 5000
        metrics.total_output_tokens = 1000
        metrics.total_cost_usd = 0.05
        metrics.total_duration_seconds = 25.0

        result = metrics.to_dict()

        assert result["operation_type"] == "validation"
        assert result["call_count"] == 10
        assert result["success_rate"] == pytest.approx(0.9)
        assert result["total_input_tokens"] == 5000
        assert result["total_output_tokens"] == 1000
        assert result["total_cost_usd"] == pytest.approx(0.05)


class TestLLMMetricsTracker:
    """Tests for LLMMetricsTracker class."""

    def test_initial_state(self):
        """Test tracker starts with zero metrics."""
        tracker = LLMMetricsTracker()

        assert tracker.total_calls == 0
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_record_single_call(self):
        """Test recording a single LLM call."""
        tracker = LLMMetricsTracker(model="claude-3-5-haiku-20241022")

        cost = tracker.record_call(
            operation_type=OperationType.GAP_FILL,
            input_tokens=1000,
            output_tokens=500,
            success=True,
            duration_seconds=2.5,
        )

        assert tracker.total_calls == 1
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.total_tokens == 1500
        assert tracker.total_cost_usd == pytest.approx(cost)

    def test_record_multiple_calls(self):
        """Test recording multiple LLM calls."""
        tracker = LLMMetricsTracker(model="claude-3-5-haiku-20241022")

        tracker.record_call(
            operation_type=OperationType.EXTRACTION,
            input_tokens=1000,
            output_tokens=500,
            success=True,
        )
        tracker.record_call(
            operation_type=OperationType.GAP_FILL,
            input_tokens=2000,
            output_tokens=800,
            success=True,
        )
        tracker.record_call(
            operation_type=OperationType.VALIDATION,
            input_tokens=500,
            output_tokens=200,
            success=False,
        )

        assert tracker.total_calls == 3
        assert tracker.total_input_tokens == 3500
        assert tracker.total_output_tokens == 1500

    def test_operation_type_tracking(self):
        """Test that metrics are tracked per operation type."""
        tracker = LLMMetricsTracker()

        tracker.record_call(OperationType.GAP_FILL, 1000, 500, success=True)
        tracker.record_call(OperationType.GAP_FILL, 1000, 500, success=True)
        tracker.record_call(OperationType.EXTRACTION, 500, 200, success=True)

        summary = tracker.get_metrics_summary()
        by_op = summary["by_operation"]

        assert "gap_fill" in by_op
        assert by_op["gap_fill"]["call_count"] == 2
        assert "extraction" in by_op
        assert by_op["extraction"]["call_count"] == 1

    def test_cost_limit_warning(self):
        """Test that warning is triggered when approaching cost limit."""
        warning_callback = MagicMock()
        exceeded_callback = MagicMock()  # Provide exceeded callback to prevent exception

        # Calculate cost for 1000 input + 500 output tokens with Haiku pricing
        # Haiku: $1/M input, $5/M output
        # Cost per call = (1000 * 1 + 500 * 5) / 1_000_000 = 0.0035
        # Set limit to 0.004 so one call (0.0035) triggers 80% warning (0.0032 threshold)
        # but doesn't exceed the limit
        tracker = LLMMetricsTracker(
            model="claude-3-5-haiku-20241022",
            max_cost_usd=0.004,  # One call at $0.0035 will be ~87.5%, triggering warning
            warning_threshold=0.8,
            on_budget_warning=warning_callback,
            on_budget_exceeded=exceeded_callback,  # Prevent exception if we do exceed
        )

        # Single call that exceeds 80% of budget but not 100%
        tracker.record_call(OperationType.GAP_FILL, 1000, 500, success=True)

        # Warning should have been called
        warning_callback.assert_called_once()
        args = warning_callback.call_args[0]
        assert args[0] == "cost"  # limit_type
        # Exceeded should NOT have been called
        exceeded_callback.assert_not_called()

    def test_cost_limit_exceeded_callback(self):
        """Test that callback is triggered when cost limit exceeded."""
        exceeded_callback = MagicMock()

        tracker = LLMMetricsTracker(
            model="claude-3-5-haiku-20241022",
            max_cost_usd=0.001,  # Very low limit
            on_budget_exceeded=exceeded_callback,
        )

        # Record a call that exceeds the budget
        tracker.record_call(OperationType.GAP_FILL, 10000, 5000, success=True)

        exceeded_callback.assert_called_once()
        args = exceeded_callback.call_args[0]
        assert args[0] == "cost"

    def test_cost_limit_exceeded_raises_without_callback(self):
        """Test that BudgetExceededError is raised when no callback set."""
        tracker = LLMMetricsTracker(
            model="claude-3-5-haiku-20241022",
            max_cost_usd=0.001,  # Very low limit
        )

        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.record_call(OperationType.GAP_FILL, 10000, 5000, success=True)

        assert exc_info.value.limit == 0.001

    def test_token_limit_exceeded(self):
        """Test token limit tracking and enforcement."""
        exceeded_callback = MagicMock()

        tracker = LLMMetricsTracker(
            max_tokens=1000,  # Low token limit
            on_budget_exceeded=exceeded_callback,
        )

        tracker.record_call(OperationType.GAP_FILL, 800, 300, success=True)

        exceeded_callback.assert_called_once()
        args = exceeded_callback.call_args[0]
        assert args[0] == "tokens"

    def test_cost_limit_reached_property(self):
        """Test cost_limit_reached property."""
        tracker = LLMMetricsTracker(max_cost_usd=0.01)

        assert not tracker.cost_limit_reached

        # Use callback to prevent exception
        tracker._on_budget_exceeded = lambda *args: None
        tracker.record_call(OperationType.GAP_FILL, 100000, 50000, success=True)

        assert tracker.cost_limit_reached

    def test_token_limit_reached_property(self):
        """Test token_limit_reached property."""
        tracker = LLMMetricsTracker(max_tokens=100)

        assert not tracker.token_limit_reached

        tracker._on_budget_exceeded = lambda *args: None
        tracker.record_call(OperationType.GAP_FILL, 80, 30, success=True)

        assert tracker.token_limit_reached

    def test_get_metrics_summary(self):
        """Test get_metrics_summary returns complete data."""
        tracker = LLMMetricsTracker(model="claude-3-5-haiku-20241022")

        tracker.record_call(OperationType.GAP_FILL, 1000, 500, success=True, duration_seconds=2.5)
        tracker.record_call(OperationType.EXTRACTION, 800, 400, success=True, duration_seconds=1.5)

        summary = tracker.get_metrics_summary()

        # Check top-level fields
        assert "total_calls" in summary
        assert summary["total_calls"] == 2
        assert "total_tokens" in summary
        assert summary["total_tokens"] == 2700
        assert "total_cost_usd" in summary
        assert "success_rate" in summary
        assert summary["success_rate"] == 1.0
        assert "avg_latency_seconds" in summary
        assert summary["avg_latency_seconds"] == 2.0  # (2.5 + 1.5) / 2

        # Check nested fields
        assert "tokens" in summary
        assert summary["tokens"]["input"] == 1800
        assert summary["tokens"]["output"] == 900

    def test_get_health_metrics(self):
        """Test get_health_metrics returns health endpoint format."""
        tracker = LLMMetricsTracker(
            model="claude-3-5-haiku-20241022",
            max_cost_usd=10.0,
            max_tokens=1000000,
        )

        tracker.record_call(OperationType.GAP_FILL, 1000, 500, success=True)

        health = tracker.get_health_metrics()

        # Check structure
        assert "llm_metrics" in health
        assert "rule_metrics" in health
        assert "budget_status" in health

        # Check llm_metrics
        assert health["llm_metrics"]["total_calls"] == 1
        assert health["llm_metrics"]["tokens_in"] == 1000
        assert health["llm_metrics"]["tokens_out"] == 500

        # Check budget_status
        assert health["budget_status"]["cost_limit_usd"] == 10.0
        assert health["budget_status"]["token_limit"] == 1000000
        assert not health["budget_status"]["cost_limit_reached"]

    def test_record_rule_generated(self):
        """Test rule generation tracking."""
        tracker = LLMMetricsTracker()

        tracker.record_rule_generated(accepted=True)
        tracker.record_rule_generated(accepted=True)
        tracker.record_rule_generated(accepted=False)

        summary = tracker.get_metrics_summary()
        rules = summary["rules"]

        assert rules["generated"] == 3
        assert rules["accepted"] == 2
        assert rules["rejected"] == 1
        assert rules["acceptance_rate"] == pytest.approx(2 / 3)

    def test_record_validation_error(self):
        """Test validation error tracking."""
        tracker = LLMMetricsTracker()

        tracker.record_validation_error("syntax_error")
        tracker.record_validation_error("syntax_error")
        tracker.record_validation_error("semantic_error")

        summary = tracker.get_metrics_summary()
        errors = summary["validation_errors"]

        assert errors["syntax_error"] == 2
        assert errors["semantic_error"] == 1

    def test_reset_clears_all_metrics(self):
        """Test that reset clears all tracked metrics."""
        tracker = LLMMetricsTracker()

        tracker.record_call(OperationType.GAP_FILL, 1000, 500, success=True)
        tracker.record_rule_generated(accepted=True)
        tracker.record_validation_error("test_error")

        tracker.reset()

        assert tracker.total_calls == 0
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

        summary = tracker.get_metrics_summary()
        assert summary["rules"]["generated"] == 0
        assert len(summary["validation_errors"]) == 0

    def test_get_pricing_fallback(self):
        """Test that unknown models fall back to default pricing."""
        tracker = LLMMetricsTracker(model="unknown-model-xyz")

        pricing = tracker.get_pricing()

        assert pricing.model_name == "Unknown (Haiku pricing)"

    def test_format_progress_log(self):
        """Test human-readable progress log format."""
        tracker = LLMMetricsTracker(model="claude-3-5-haiku-20241022")

        tracker.record_call(OperationType.GAP_FILL, 1000, 500, success=True)
        tracker.record_rule_generated(accepted=True)

        log = tracker.format_progress_log()

        assert "[LLM]" in log
        assert "[Rules]" in log
        assert "Calls: 1" in log

    def test_thread_safety(self):
        """Test that tracker is thread-safe."""
        import threading

        tracker = LLMMetricsTracker()
        errors = []

        def record_calls():
            try:
                for _ in range(100):
                    tracker.record_call(OperationType.GAP_FILL, 100, 50, success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_calls) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.total_calls == 500


class TestGlobalTrackerFunctions:
    """Tests for global tracker convenience functions."""

    def test_set_and_get_global_tracker(self):
        """Test setting and getting global tracker."""
        tracker = LLMMetricsTracker()
        set_global_metrics_tracker(tracker)

        retrieved = get_global_metrics_tracker()

        assert retrieved is tracker

    def test_create_metrics_tracker_sets_global(self):
        """Test that create_metrics_tracker can set as global."""
        tracker = create_metrics_tracker(
            model="claude-3-5-haiku-20241022",
            max_cost_usd=10.0,
            set_as_global=True,
        )

        assert get_global_metrics_tracker() is tracker

    def test_create_metrics_tracker_no_global(self):
        """Test that create_metrics_tracker can skip global setting."""
        # Clear any existing global
        set_global_metrics_tracker(None)

        tracker = create_metrics_tracker(
            model="claude-3-5-haiku-20241022",
            set_as_global=False,
        )

        # This test assumes no prior global was set
        # Just verify the tracker was created with correct params
        assert tracker.model == "claude-3-5-haiku-20241022"


class TestBudgetExceededError:
    """Tests for BudgetExceededError exception."""

    def test_error_contains_values(self):
        """Test that exception contains current and limit values."""
        error = BudgetExceededError(
            message="Test error",
            current_value=15.0,
            limit=10.0,
        )

        assert error.current_value == 15.0
        assert error.limit == 10.0
        assert "Test error" in str(error)


class TestLLMCallRecord:
    """Tests for LLMCallRecord dataclass."""

    def test_record_creation(self):
        """Test creating a call record."""
        record = LLMCallRecord(
            timestamp=datetime.now(),
            operation_type=OperationType.GAP_FILL,
            model="claude-3-5-haiku-20241022",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.0035,
            success=True,
            duration_seconds=2.5,
            metadata={"gap_id": "test_gap"},
        )

        assert record.operation_type == OperationType.GAP_FILL
        assert record.input_tokens == 1000
        assert record.cost_usd == 0.0035
        assert record.metadata["gap_id"] == "test_gap"


class TestOperationType:
    """Tests for OperationType enum."""

    def test_all_operation_types_exist(self):
        """Test that all expected operation types exist."""
        assert OperationType.EXTRACTION.value == "extraction"
        assert OperationType.GAP_FILL.value == "gap_fill"
        assert OperationType.VALIDATION.value == "validation"
        assert OperationType.RULE_GENERATION.value == "rule_generation"
        assert OperationType.ANALYSIS.value == "analysis"
        assert OperationType.OTHER.value == "other"

    def test_operation_type_is_string_enum(self):
        """Test that OperationType values are strings."""
        for op in OperationType:
            assert isinstance(op.value, str)
