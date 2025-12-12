"""
LLM Metrics Tracking for Autonomous Test Harness.

Provides comprehensive metrics tracking for LLM API usage including:
- Token counting (input/output)
- Cost calculation based on model pricing
- Rate tracking per operation type
- Budget limits with auto-stop
- Periodic metrics logging

This module addresses issue #165: Add real-time metrics dashboard and API cost tracking.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class OperationType(str, Enum):
    """Types of LLM operations for metrics categorization."""

    EXTRACTION = "extraction"
    GAP_FILL = "gap_fill"
    VALIDATION = "validation"
    RULE_GENERATION = "rule_generation"
    ANALYSIS = "analysis"
    OTHER = "other"


@dataclass
class ModelPricing:
    """Pricing information for a Claude model (per million tokens)."""

    input_cost_per_million: float
    output_cost_per_million: float
    model_name: str

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        return input_cost + output_cost


# Claude model pricing as of January 2025
# Source: https://www.anthropic.com/pricing
MODEL_PRICING: Dict[str, ModelPricing] = {
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": ModelPricing(
        input_cost_per_million=1.00,
        output_cost_per_million=5.00,
        model_name="Claude 3.5 Haiku",
    ),
    "claude-3-5-haiku-latest": ModelPricing(
        input_cost_per_million=1.00,
        output_cost_per_million=5.00,
        model_name="Claude 3.5 Haiku",
    ),
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet-20241022": ModelPricing(
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
        model_name="Claude 3.5 Sonnet",
    ),
    "claude-3-5-sonnet-latest": ModelPricing(
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
        model_name="Claude 3.5 Sonnet",
    ),
    "claude-sonnet-4-20250514": ModelPricing(
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
        model_name="Claude Sonnet 4",
    ),
    # Claude 3 Opus
    "claude-3-opus-20240229": ModelPricing(
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
        model_name="Claude 3 Opus",
    ),
    "claude-3-opus-latest": ModelPricing(
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
        model_name="Claude 3 Opus",
    ),
    # Claude Opus 4
    "claude-opus-4-20250514": ModelPricing(
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
        model_name="Claude Opus 4",
    ),
    # Default fallback (use Haiku pricing as conservative estimate)
    "default": ModelPricing(
        input_cost_per_million=1.00,
        output_cost_per_million=5.00,
        model_name="Unknown (Haiku pricing)",
    ),
}


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""

    timestamp: datetime
    operation_type: OperationType
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    success: bool
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationMetrics:
    """Aggregated metrics for a specific operation type."""

    operation_type: OperationType
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.call_count == 0:
            return 1.0
        return self.success_count / self.call_count

    @property
    def average_duration_seconds(self) -> float:
        """Calculate average call duration."""
        if self.call_count == 0:
            return 0.0
        return self.total_duration_seconds / self.call_count

    @property
    def average_tokens_per_call(self) -> float:
        """Calculate average tokens per call."""
        if self.call_count == 0:
            return 0.0
        return (self.total_input_tokens + self.total_output_tokens) / self.call_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_type": self.operation_type.value,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "average_duration_seconds": round(self.average_duration_seconds, 2),
            "average_tokens_per_call": round(self.average_tokens_per_call, 1),
        }


class BudgetExceededError(Exception):
    """Raised when budget limits are exceeded."""

    def __init__(self, message: str, current_value: float, limit: float):
        super().__init__(message)
        self.current_value = current_value
        self.limit = limit


class LLMMetricsTracker:
    """
    Comprehensive metrics tracker for LLM API usage.

    Tracks:
    - Token counts (input/output) per call and aggregated
    - Cost calculation based on model pricing
    - Metrics breakdown by operation type
    - Budget limits with configurable actions
    - Periodic logging of metrics summaries

    Thread-safe for use in async/concurrent environments.

    Example:
        >>> tracker = LLMMetricsTracker(
        ...     model="claude-3-5-haiku-20241022",
        ...     max_cost_usd=10.0,
        ...     log_interval_seconds=300,
        ... )
        >>> tracker.record_call(
        ...     operation_type=OperationType.GAP_FILL,
        ...     input_tokens=1500,
        ...     output_tokens=500,
        ...     success=True,
        ...     duration_seconds=2.5,
        ... )
        >>> print(tracker.total_cost_usd)
        0.004  # $0.004 for this call
    """

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        max_cost_usd: Optional[float] = None,
        max_tokens: Optional[int] = None,
        warning_threshold: float = 0.8,
        log_interval_seconds: float = 300.0,
        on_budget_warning: Optional[Callable[[str, float, float], None]] = None,
        on_budget_exceeded: Optional[Callable[[str, float, float], None]] = None,
    ):
        """
        Initialize the metrics tracker.

        Args:
            model: Default model name for pricing lookup
            max_cost_usd: Maximum cost limit in USD (None = no limit)
            max_tokens: Maximum total token limit (None = no limit)
            warning_threshold: Fraction of budget at which to warn (0.8 = 80%)
            log_interval_seconds: Seconds between automatic progress logs
            on_budget_warning: Callback when approaching budget limit
            on_budget_exceeded: Callback when budget exceeded
        """
        self._model = model
        self._max_cost_usd = max_cost_usd
        self._max_tokens = max_tokens
        self._warning_threshold = warning_threshold
        self._log_interval_seconds = log_interval_seconds
        self._on_budget_warning = on_budget_warning
        self._on_budget_exceeded = on_budget_exceeded

        self._lock = threading.Lock()
        self._start_time = datetime.now()
        self._last_log_time = datetime.now()

        # Aggregate metrics
        self._total_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0
        self._total_duration_seconds = 0.0

        # Per-operation metrics
        self._operation_metrics: Dict[OperationType, OperationMetrics] = {
            op: OperationMetrics(operation_type=op) for op in OperationType
        }

        # Rule generation metrics
        self._rules_generated = 0
        self._rules_accepted = 0
        self._rules_rejected = 0

        # Validation metrics
        self._validation_errors: Dict[str, int] = {}

        # Recent call history (for debugging/analysis)
        self._recent_calls: List[LLMCallRecord] = []
        self._max_recent_calls = 100

        # Warning state
        self._cost_warning_issued = False
        self._token_warning_issued = False

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def total_calls(self) -> int:
        """Get total number of LLM calls."""
        with self._lock:
            return self._total_calls

    @property
    def total_input_tokens(self) -> int:
        """Get total input tokens used."""
        with self._lock:
            return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Get total output tokens used."""
        with self._lock:
            return self._total_output_tokens

    @property
    def total_tokens(self) -> int:
        """Get total tokens (input + output)."""
        with self._lock:
            return self._total_input_tokens + self._total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        """Get total cost in USD."""
        with self._lock:
            return self._total_cost_usd

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since tracker was created."""
        return (datetime.now() - self._start_time).total_seconds()

    @property
    def cost_limit_reached(self) -> bool:
        """Check if cost limit has been reached."""
        if self._max_cost_usd is None:
            return False
        with self._lock:
            return self._total_cost_usd >= self._max_cost_usd

    @property
    def token_limit_reached(self) -> bool:
        """Check if token limit has been reached."""
        if self._max_tokens is None:
            return False
        with self._lock:
            return (self._total_input_tokens + self._total_output_tokens) >= self._max_tokens

    def get_pricing(self, model: Optional[str] = None) -> ModelPricing:
        """Get pricing for a model.

        Args:
            model: Model name, or None to use default

        Returns:
            ModelPricing for the specified model
        """
        model_name = model or self._model
        return MODEL_PRICING.get(model_name, MODEL_PRICING["default"])

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None,
    ) -> float:
        """Calculate cost for given token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name, or None to use default

        Returns:
            Cost in USD
        """
        pricing = self.get_pricing(model)
        return pricing.calculate_cost(input_tokens, output_tokens)

    def record_call(
        self,
        operation_type: OperationType,
        input_tokens: int,
        output_tokens: int,
        success: bool = True,
        duration_seconds: float = 0.0,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Record an LLM API call.

        Args:
            operation_type: Type of operation performed
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            success: Whether the call succeeded
            duration_seconds: Call duration in seconds
            model: Model used (defaults to tracker's model)
            metadata: Additional metadata for the call

        Returns:
            Cost of this call in USD

        Raises:
            BudgetExceededError: If budget exceeded and no callback set
        """
        cost = self.calculate_cost(input_tokens, output_tokens, model)

        with self._lock:
            # Update aggregate metrics
            self._total_calls += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cost_usd += cost
            self._total_duration_seconds += duration_seconds

            # Update operation-specific metrics
            op_metrics = self._operation_metrics[operation_type]
            op_metrics.call_count += 1
            if success:
                op_metrics.success_count += 1
            else:
                op_metrics.failure_count += 1
            op_metrics.total_input_tokens += input_tokens
            op_metrics.total_output_tokens += output_tokens
            op_metrics.total_cost_usd += cost
            op_metrics.total_duration_seconds += duration_seconds

            # Record call
            record = LLMCallRecord(
                timestamp=datetime.now(),
                operation_type=operation_type,
                model=model or self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                success=success,
                duration_seconds=duration_seconds,
                metadata=metadata or {},
            )
            self._recent_calls.append(record)
            if len(self._recent_calls) > self._max_recent_calls:
                self._recent_calls = self._recent_calls[-self._max_recent_calls :]

        # Check budgets (outside lock to allow callbacks)
        self._check_budget_limits()

        # Check if we should log progress
        if self._should_log():
            self._log_progress()

        return cost

    def record_rule_generated(self, accepted: bool = True) -> None:
        """Record a rule generation result.

        Args:
            accepted: Whether the rule was accepted
        """
        with self._lock:
            self._rules_generated += 1
            if accepted:
                self._rules_accepted += 1
            else:
                self._rules_rejected += 1

    def record_validation_error(self, error_type: str) -> None:
        """Record a validation error.

        Args:
            error_type: Type of validation error
        """
        with self._lock:
            self._validation_errors[error_type] = self._validation_errors.get(error_type, 0) + 1

    def _check_budget_limits(self) -> None:
        """Check if budget limits are approached or exceeded."""
        # Check cost limit
        if self._max_cost_usd is not None:
            cost_fraction = self._total_cost_usd / self._max_cost_usd

            # Warning at threshold
            if cost_fraction >= self._warning_threshold and not self._cost_warning_issued:
                self._cost_warning_issued = True
                message = (
                    f"Approaching cost limit: ${self._total_cost_usd:.2f} / "
                    f"${self._max_cost_usd:.2f} ({cost_fraction:.0%})"
                )
                logger.warning(message)
                if self._on_budget_warning:
                    self._on_budget_warning("cost", self._total_cost_usd, self._max_cost_usd)

            # Exceeded
            if cost_fraction >= 1.0:
                message = (
                    f"Cost limit exceeded: ${self._total_cost_usd:.2f} >= ${self._max_cost_usd:.2f}"
                )
                logger.error(message)
                if self._on_budget_exceeded:
                    self._on_budget_exceeded("cost", self._total_cost_usd, self._max_cost_usd)
                else:
                    raise BudgetExceededError(message, self._total_cost_usd, self._max_cost_usd)

        # Check token limit
        if self._max_tokens is not None:
            total_tokens = self._total_input_tokens + self._total_output_tokens
            token_fraction = total_tokens / self._max_tokens

            # Warning at threshold
            if token_fraction >= self._warning_threshold and not self._token_warning_issued:
                self._token_warning_issued = True
                message = (
                    f"Approaching token limit: {total_tokens:,} / "
                    f"{self._max_tokens:,} ({token_fraction:.0%})"
                )
                logger.warning(message)
                if self._on_budget_warning:
                    self._on_budget_warning("tokens", float(total_tokens), float(self._max_tokens))

            # Exceeded
            if token_fraction >= 1.0:
                message = f"Token limit exceeded: {total_tokens:,} >= {self._max_tokens:,}"
                logger.error(message)
                if self._on_budget_exceeded:
                    self._on_budget_exceeded("tokens", float(total_tokens), float(self._max_tokens))
                else:
                    raise BudgetExceededError(message, float(total_tokens), float(self._max_tokens))

    def _should_log(self) -> bool:
        """Check if progress should be logged."""
        if self._log_interval_seconds <= 0:
            return False
        elapsed = (datetime.now() - self._last_log_time).total_seconds()
        return elapsed >= self._log_interval_seconds

    def _log_progress(self) -> None:
        """Log current progress metrics."""
        elapsed_hours = self.elapsed_seconds / 3600
        calls_per_hour = self._total_calls / elapsed_hours if elapsed_hours > 0 else 0

        rule_acceptance_rate = (
            self._rules_accepted / self._rules_generated if self._rules_generated > 0 else 0
        )

        logger.info(
            f"[LLM Metrics] Calls: {self._total_calls} ({calls_per_hour:.1f}/hr) | "
            f"Tokens: {self._total_input_tokens:,} in, {self._total_output_tokens:,} out | "
            f"Cost: ${self._total_cost_usd:.2f} | "
            f"Rules: {self._rules_accepted}/{self._rules_generated} ({rule_acceptance_rate:.0%})"
        )

        self._last_log_time = datetime.now()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dictionary with all tracked metrics
        """
        with self._lock:
            elapsed = self.elapsed_seconds
            elapsed_hours = elapsed / 3600

            # Calculate rates
            calls_per_hour = self._total_calls / elapsed_hours if elapsed_hours > 0 else 0
            cost_per_hour = self._total_cost_usd / elapsed_hours if elapsed_hours > 0 else 0

            # Calculate success rate
            total_success = sum(m.success_count for m in self._operation_metrics.values())
            success_rate = total_success / self._total_calls if self._total_calls > 0 else 1.0

            # Calculate average latency
            avg_latency = (
                self._total_duration_seconds / self._total_calls if self._total_calls > 0 else 0.0
            )

            # Rule metrics
            rule_acceptance_rate = (
                self._rules_accepted / self._rules_generated if self._rules_generated > 0 else 0
            )

            return {
                "elapsed_seconds": round(elapsed, 1),
                "elapsed_hours": round(elapsed_hours, 2),
                "model": self._model,
                "total_calls": self._total_calls,
                "calls_per_hour": round(calls_per_hour, 1),
                # Convenience fields for CLI
                "total_tokens": self._total_input_tokens + self._total_output_tokens,
                "total_cost_usd": round(self._total_cost_usd, 4),
                "success_rate": round(success_rate, 3),
                "avg_latency_seconds": round(avg_latency, 2),
                # Detailed token breakdown
                "tokens": {
                    "input": self._total_input_tokens,
                    "output": self._total_output_tokens,
                    "total": self._total_input_tokens + self._total_output_tokens,
                },
                "cost": {
                    "total_usd": round(self._total_cost_usd, 4),
                    "per_hour_usd": round(cost_per_hour, 4),
                    "limit_usd": self._max_cost_usd,
                    "remaining_usd": (
                        round(self._max_cost_usd - self._total_cost_usd, 4)
                        if self._max_cost_usd
                        else None
                    ),
                },
                "rules": {
                    "generated": self._rules_generated,
                    "accepted": self._rules_accepted,
                    "rejected": self._rules_rejected,
                    "acceptance_rate": round(rule_acceptance_rate, 3),
                },
                "validation_errors": dict(self._validation_errors),
                "by_operation": {
                    op.value: metrics.to_dict()
                    for op, metrics in self._operation_metrics.items()
                    if metrics.call_count > 0
                },
                "limits": {
                    "max_cost_usd": self._max_cost_usd,
                    "max_tokens": self._max_tokens,
                    # Compute inline to avoid deadlock (we already hold the lock)
                    "cost_limit_reached": (
                        self._max_cost_usd is not None
                        and self._total_cost_usd >= self._max_cost_usd
                    ),
                    "token_limit_reached": (
                        self._max_tokens is not None
                        and (self._total_input_tokens + self._total_output_tokens)
                        >= self._max_tokens
                    ),
                },
            }

    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Get metrics formatted for health endpoint.

        Returns:
            Dictionary suitable for /health endpoint response
        """
        summary = self.get_metrics_summary()

        return {
            "llm_metrics": {
                "total_calls": summary["total_calls"],
                "tokens_in": summary["tokens"]["input"],
                "tokens_out": summary["tokens"]["output"],
                "cost_usd": summary["cost"]["total_usd"],
                "calls_per_hour": summary["calls_per_hour"],
            },
            "rule_metrics": {
                "generated": summary["rules"]["generated"],
                "accepted": summary["rules"]["accepted"],
                "rejected": summary["rules"]["rejected"],
                "acceptance_rate": summary["rules"]["acceptance_rate"],
            },
            "budget_status": {
                "cost_limit_usd": self._max_cost_usd,
                "cost_remaining_usd": summary["cost"]["remaining_usd"],
                "cost_limit_reached": self.cost_limit_reached,
                "token_limit": self._max_tokens,
                "token_limit_reached": self.token_limit_reached,
            },
        }

    def format_progress_log(self) -> str:
        """
        Format a human-readable progress log message.

        Returns:
            Formatted progress string matching issue #165 format
        """
        summary = self.get_metrics_summary()

        lines = [
            f"[LLM] Calls: {summary['total_calls']} | "
            f"Tokens: {summary['tokens']['input']:,} input, {summary['tokens']['output']:,} output | "
            f"Cost: ${summary['cost']['total_usd']:.2f}",
            f"[Rules] Generated: {summary['rules']['generated']} | "
            f"Accepted: {summary['rules']['accepted']} ({summary['rules']['acceptance_rate']:.0%}) | "
            f"Rejected: {summary['rules']['rejected']}",
        ]

        if summary["validation_errors"]:
            error_parts = [
                f"{error_type}: {count}"
                for error_type, count in summary["validation_errors"].items()
            ]
            lines.append(f"[Validation] {' | '.join(error_parts)}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._start_time = datetime.now()
            self._last_log_time = datetime.now()
            self._total_calls = 0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_cost_usd = 0.0
            self._total_duration_seconds = 0.0
            self._rules_generated = 0
            self._rules_accepted = 0
            self._rules_rejected = 0
            self._validation_errors.clear()
            self._recent_calls.clear()
            self._cost_warning_issued = False
            self._token_warning_issued = False

            for metrics in self._operation_metrics.values():
                metrics.call_count = 0
                metrics.success_count = 0
                metrics.failure_count = 0
                metrics.total_input_tokens = 0
                metrics.total_output_tokens = 0
                metrics.total_cost_usd = 0.0
                metrics.total_duration_seconds = 0.0


# Global tracker instance for convenience
_global_tracker: Optional[LLMMetricsTracker] = None


def get_global_metrics_tracker() -> Optional[LLMMetricsTracker]:
    """Get the global metrics tracker instance."""
    return _global_tracker


def set_global_metrics_tracker(tracker: LLMMetricsTracker) -> None:
    """Set the global metrics tracker instance."""
    global _global_tracker
    _global_tracker = tracker


def create_metrics_tracker(
    model: str = "claude-3-5-haiku-20241022",
    max_cost_usd: Optional[float] = None,
    max_tokens: Optional[int] = None,
    log_interval_seconds: float = 300.0,
    set_as_global: bool = True,
) -> LLMMetricsTracker:
    """
    Factory function to create and optionally set a global metrics tracker.

    Args:
        model: Model name for pricing
        max_cost_usd: Cost limit in USD
        max_tokens: Token limit
        log_interval_seconds: Progress log interval
        set_as_global: Whether to set as global tracker

    Returns:
        Configured LLMMetricsTracker
    """
    tracker = LLMMetricsTracker(
        model=model,
        max_cost_usd=max_cost_usd,
        max_tokens=max_tokens,
        log_interval_seconds=log_interval_seconds,
    )

    if set_as_global:
        set_global_metrics_tracker(tracker)

    return tracker
