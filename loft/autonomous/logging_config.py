"""
Logging Configuration for Autonomous Test Harness.

Provides unified logging configuration that:
- Filters Clingo "info" level messages that flood logs
- Provides summarized error logging for LLM failures
- Standardizes log format across all modules
- Adds periodic progress indicators

This module addresses issue #162 - Improve logging readability for autonomous LLM test runs.
"""

import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# Standard log format used across the codebase
STANDARD_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
STANDARD_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ClingoMessageFilter(logging.Filter):
    """
    Filter that suppresses or summarizes Clingo "info" level messages.

    These messages like "atom does not occur in any rule head" are informational
    warnings that create significant noise during long-running test runs.
    Instead of logging each one, this filter:
    - Suppresses individual info messages
    - Periodically logs a summary count
    """

    # Pattern to match Clingo info messages
    CLINGO_INFO_PATTERN = re.compile(
        r"<block>:\d+:\d+-\d+: info: .*|"
        r"info: atom does not occur in any rule head|"
        r"info: operation undefined|"
        r"atom does not occur in any rule head"
    )

    def __init__(
        self,
        summary_interval_seconds: float = 60.0,
        summary_callback: Optional[Callable[[int], None]] = None,
    ):
        """
        Initialize the Clingo message filter.

        Args:
            summary_interval_seconds: How often to log summary of suppressed messages
            summary_callback: Optional callback when summary is generated
        """
        super().__init__()
        self.summary_interval_seconds = summary_interval_seconds
        self.summary_callback = summary_callback
        self.suppressed_count = 0
        self.suppressed_messages: Dict[str, int] = {}
        self.last_summary_time = datetime.now()
        self._logger = logging.getLogger(__name__)

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records, suppressing Clingo info messages.

        Args:
            record: Log record to filter

        Returns:
            True if record should be logged, False to suppress
        """
        message = record.getMessage()

        # Check if this is a Clingo info message
        if self.CLINGO_INFO_PATTERN.search(message):
            self.suppressed_count += 1

            # Track message types for summary
            message_type = self._categorize_message(message)
            self.suppressed_messages[message_type] = (
                self.suppressed_messages.get(message_type, 0) + 1
            )

            # Check if it's time to log a summary
            if self._should_log_summary():
                self._log_summary()

            return False

        return True

    def _categorize_message(self, message: str) -> str:
        """Categorize a Clingo message for summary grouping."""
        if "atom does not occur in any rule head" in message:
            return "undefined_atom_warnings"
        elif "operation undefined" in message:
            return "undefined_operation_warnings"
        else:
            return "other_info_messages"

    def _should_log_summary(self) -> bool:
        """Check if enough time has passed to log a summary."""
        elapsed = (datetime.now() - self.last_summary_time).total_seconds()
        return elapsed >= self.summary_interval_seconds

    def _log_summary(self) -> None:
        """Log summary of suppressed messages."""
        if self.suppressed_count > 0:
            summary_parts = [
                f"{count} {msg_type.replace('_', ' ')}"
                for msg_type, count in self.suppressed_messages.items()
            ]
            summary_str = ", ".join(summary_parts)

            self._logger.info(
                f"Clingo warnings suppressed (last {self.summary_interval_seconds}s): "
                f"{self.suppressed_count} total ({summary_str})"
            )

            if self.summary_callback:
                self.summary_callback(self.suppressed_count)

        self.suppressed_count = 0
        self.suppressed_messages.clear()
        self.last_summary_time = datetime.now()

    def get_final_summary(self) -> Dict[str, int]:
        """Get final summary of all suppressed messages."""
        summary = dict(self.suppressed_messages)
        summary["total"] = self.suppressed_count
        return summary


class LLMErrorSummarizer:
    """
    Summarizes LLM error logs instead of dumping full completions.

    When structured parsing fails, instead of logging hundreds of lines of
    LLM completion data, this summarizer logs a concise error summary.
    """

    # Pattern to match LLM completion dumps
    COMPLETION_DUMP_PATTERN = re.compile(
        r"<failed_attempts>|"
        r"<completion>|"
        r"Message\(id='msg_[^']+',\s*content=\[|"
        r"ToolUseBlock\(|"
        r"ContentBlock\("
    )

    def __init__(self, max_error_length: int = 200):
        """
        Initialize the error summarizer.

        Args:
            max_error_length: Maximum length of error message to include
        """
        self.max_error_length = max_error_length
        self.error_counts: Dict[str, int] = {}
        self.last_errors: List[Dict[str, Any]] = []

    def summarize_llm_error(
        self,
        error_message: str,
        attempts: int = 0,
        last_exception: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a summarized error message for LLM failures.

        Args:
            error_message: Full error message (may be very long)
            attempts: Number of retry attempts made
            last_exception: The last exception type/message
            context: Additional context (case_id, predicate, etc.)

        Returns:
            Concise summary string for logging
        """
        # Extract key information from the error
        error_type = self._categorize_error(error_message, last_exception)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Build summary
        summary_parts = []

        if context:
            if "case_id" in context:
                summary_parts.append(f"case={context['case_id']}")
            if "predicate" in context:
                summary_parts.append(f"pred={context['predicate']}")

        summary_parts.append(f"type={error_type}")

        if attempts > 0:
            summary_parts.append(f"attempts={attempts}")

        # Include truncated error if it adds value
        if last_exception:
            truncated_error = last_exception[: self.max_error_length]
            if len(last_exception) > self.max_error_length:
                truncated_error += "..."
            summary_parts.append(f"error={truncated_error}")

        summary = " | ".join(summary_parts)

        # Store for later analysis
        self.last_errors.append(
            {
                "timestamp": datetime.now().isoformat(),
                "type": error_type,
                "summary": summary,
                "context": context,
            }
        )

        # Keep only last 100 errors
        if len(self.last_errors) > 100:
            self.last_errors = self.last_errors[-100:]

        return f"LLM error: {summary}"

    def _categorize_error(
        self, error_message: str, last_exception: Optional[str]
    ) -> str:
        """Categorize the error for summary grouping."""
        combined = f"{error_message} {last_exception or ''}"
        combined_lower = combined.lower()

        if "syntax error" in combined_lower or "asp syntax" in combined_lower:
            return "asp_syntax"
        elif "unsafe variable" in combined_lower:
            return "unsafe_variable"
        elif "parsing" in combined_lower or "parse" in combined_lower:
            return "parsing_failed"
        elif "validation" in combined_lower:
            return "validation_failed"
        elif "timeout" in combined_lower:
            return "timeout"
        elif "rate limit" in combined_lower or "429" in combined:
            return "rate_limit"
        elif "truncat" in combined_lower:
            return "truncation"
        else:
            return "unknown"

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        return {
            "error_counts": dict(self.error_counts),
            "total_errors": sum(self.error_counts.values()),
            "recent_errors": self.last_errors[-10:],
        }


class ProgressIndicator:
    """
    Provides periodic progress indicators for long-running tests.

    Logs aggregated metrics at configurable intervals instead of per-case updates.
    """

    def __init__(
        self,
        log_interval_seconds: float = 300.0,  # 5 minutes default
        log_interval_cases: int = 0,  # 0 = disabled
        logger_name: str = "loft.autonomous.progress",
    ):
        """
        Initialize progress indicator.

        Args:
            log_interval_seconds: Seconds between progress logs (0 = disabled)
            log_interval_cases: Cases between progress logs (0 = disabled)
            logger_name: Logger name to use for progress output
        """
        self.log_interval_seconds = log_interval_seconds
        self.log_interval_cases = log_interval_cases
        self.logger = logging.getLogger(logger_name)

        self.start_time = datetime.now()
        self.last_log_time = datetime.now()
        self.last_log_cases = 0

        # Metrics
        self.cases_processed = 0
        self.cases_successful = 0
        self.cases_failed = 0
        self.rules_generated = 0
        self.rules_accepted = 0
        self.llm_calls = 0
        self.estimated_cost_usd = 0.0

    def update(
        self,
        cases_processed: int = 0,
        cases_successful: int = 0,
        cases_failed: int = 0,
        rules_generated: int = 0,
        rules_accepted: int = 0,
        llm_calls: int = 0,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        """
        Update progress metrics.

        Args:
            cases_processed: Total cases processed
            cases_successful: Total successful cases
            cases_failed: Total failed cases
            rules_generated: Total rules generated
            rules_accepted: Total rules accepted
            llm_calls: Total LLM API calls
            estimated_cost_usd: Estimated cost in USD
        """
        self.cases_processed = cases_processed
        self.cases_successful = cases_successful
        self.cases_failed = cases_failed
        self.rules_generated = rules_generated
        self.rules_accepted = rules_accepted
        self.llm_calls = llm_calls
        self.estimated_cost_usd = estimated_cost_usd

        if self._should_log():
            self._log_progress()

    def _should_log(self) -> bool:
        """Check if progress should be logged."""
        # Time-based check
        if self.log_interval_seconds > 0:
            elapsed = (datetime.now() - self.last_log_time).total_seconds()
            if elapsed >= self.log_interval_seconds:
                return True

        # Case-based check
        if self.log_interval_cases > 0:
            cases_since_log = self.cases_processed - self.last_log_cases
            if cases_since_log >= self.log_interval_cases:
                return True

        return False

    def _log_progress(self) -> None:
        """Log current progress."""
        elapsed = datetime.now() - self.start_time
        elapsed_hours = elapsed.total_seconds() / 3600

        # Calculate rates
        cases_per_hour = (
            self.cases_processed / elapsed_hours if elapsed_hours > 0 else 0
        )
        accuracy = (
            self.cases_successful / self.cases_processed
            if self.cases_processed > 0
            else 0
        )
        acceptance_rate = (
            self.rules_accepted / self.rules_generated
            if self.rules_generated > 0
            else 0
        )

        self.logger.info(
            f"Progress: {self.cases_processed} cases ({cases_per_hour:.1f}/hr) | "
            f"Accuracy: {accuracy:.1%} | "
            f"Rules: {self.rules_accepted}/{self.rules_generated} accepted ({acceptance_rate:.1%}) | "
            f"LLM calls: {self.llm_calls} | "
            f"Cost: ${self.estimated_cost_usd:.2f} | "
            f"Elapsed: {self._format_duration(elapsed)}"
        )

        self.last_log_time = datetime.now()
        self.last_log_cases = self.cases_processed

    def _format_duration(self, duration: timedelta) -> str:
        """Format duration as human-readable string."""
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def force_log(self) -> None:
        """Force immediate progress log."""
        self._log_progress()

    def get_summary(self) -> Dict[str, Any]:
        """Get complete progress summary."""
        elapsed = datetime.now() - self.start_time

        return {
            "elapsed_seconds": elapsed.total_seconds(),
            "elapsed_formatted": self._format_duration(elapsed),
            "cases_processed": self.cases_processed,
            "cases_successful": self.cases_successful,
            "cases_failed": self.cases_failed,
            "rules_generated": self.rules_generated,
            "rules_accepted": self.rules_accepted,
            "llm_calls": self.llm_calls,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cases_per_hour": (
                self.cases_processed / (elapsed.total_seconds() / 3600)
                if elapsed.total_seconds() > 0
                else 0
            ),
        }


def setup_autonomous_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_clingo_filter: bool = True,
    clingo_summary_interval: float = 60.0,
    progress_interval_seconds: float = 300.0,
) -> Dict[str, Any]:
    """
    Configure logging for autonomous test runs with all improvements.

    This is the main entry point for setting up logging in autonomous runs.
    It configures:
    - Standard log format across all loggers
    - Clingo message filtering (suppresses "info" warnings)
    - Progress indicators

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        enable_clingo_filter: Whether to filter Clingo info messages
        clingo_summary_interval: Seconds between Clingo warning summaries
        progress_interval_seconds: Seconds between progress logs

    Returns:
        Dictionary with logging components for further customization:
        {
            "clingo_filter": ClingoMessageFilter instance,
            "error_summarizer": LLMErrorSummarizer instance,
            "progress_indicator": ProgressIndicator instance,
        }

    Example:
        >>> components = setup_autonomous_logging(
        ...     log_level="INFO",
        ...     log_file=Path("/tmp/run.log"),
        ...     enable_clingo_filter=True,
        ... )
        >>> progress = components["progress_indicator"]
        >>> progress.update(cases_processed=10)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create handlers
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    # Configure root logger with standard format
    logging.basicConfig(
        level=level,
        format=STANDARD_LOG_FORMAT,
        datefmt=STANDARD_DATE_FORMAT,
        handlers=handlers,
        force=True,  # Reset any existing configuration
    )

    # Create components
    clingo_filter = ClingoMessageFilter(
        summary_interval_seconds=clingo_summary_interval
    )
    error_summarizer = LLMErrorSummarizer()
    progress_indicator = ProgressIndicator(
        log_interval_seconds=progress_interval_seconds
    )

    # Apply Clingo filter to relevant loggers
    if enable_clingo_filter:
        # Filter on validation and symbolic loggers
        for logger_name in [
            "loft.validation",
            "loft.symbolic",
            "loft.validation.asp_validators",
            "loft.symbolic.asp_core",
            "clingo",
        ]:
            logger = logging.getLogger(logger_name)
            logger.addFilter(clingo_filter)

    # Suppress some noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Configure loguru to use standard logging format
    # This helps unify loguru-based modules with stdlib logging
    _configure_loguru_interop(level)

    return {
        "clingo_filter": clingo_filter,
        "error_summarizer": error_summarizer,
        "progress_indicator": progress_indicator,
    }


def _configure_loguru_interop(level: int) -> None:
    """
    Configure loguru to work alongside stdlib logging.

    This ensures consistent log format between modules using loguru
    (like rule_generator.py) and those using stdlib logging.
    """
    try:
        from loguru import logger as loguru_logger

        # Remove default loguru handler
        loguru_logger.remove()

        # Add handler that forwards to stdlib logging
        # This ensures consistent format
        class InterceptHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                # Get corresponding loguru level
                try:
                    level_name = record.levelname
                except ValueError:
                    level_name = record.levelno

                # Find caller from where originated the logged message
                frame, depth = sys._getframe(6), 6
                while frame and frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1

                loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                    level_name, record.getMessage()
                )

        # Re-add loguru with format matching stdlib
        loguru_logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> - "
                "<cyan>{name}</cyan> - "
                "<level>{level}</level> - "
                "<level>{message}</level>"
            ),
            level=logging.getLevelName(level),
            colorize=True,
        )

    except ImportError:
        # loguru not installed, no interop needed
        pass


def create_log_summary(
    clingo_filter: Optional[ClingoMessageFilter] = None,
    error_summarizer: Optional[LLMErrorSummarizer] = None,
    progress_indicator: Optional[ProgressIndicator] = None,
) -> str:
    """
    Create a final log summary for end of run.

    Args:
        clingo_filter: Clingo message filter instance
        error_summarizer: LLM error summarizer instance
        progress_indicator: Progress indicator instance

    Returns:
        Formatted summary string
    """
    lines = ["=" * 60, "RUN SUMMARY", "=" * 60]

    if progress_indicator:
        summary = progress_indicator.get_summary()
        lines.extend(
            [
                "",
                "Progress:",
                f"  Elapsed: {summary['elapsed_formatted']}",
                f"  Cases: {summary['cases_processed']} ({summary['cases_per_hour']:.1f}/hr)",
                f"  Success: {summary['cases_successful']} | Failed: {summary['cases_failed']}",
                f"  Rules: {summary['rules_accepted']}/{summary['rules_generated']} accepted",
                f"  LLM calls: {summary['llm_calls']}",
                f"  Cost: ${summary['estimated_cost_usd']:.2f}",
            ]
        )

    if clingo_filter:
        summary = clingo_filter.get_final_summary()
        if summary.get("total", 0) > 0:
            lines.extend(
                [
                    "",
                    "Clingo Warnings Suppressed:",
                    f"  Total: {summary.get('total', 0)}",
                ]
            )
            for msg_type, count in summary.items():
                if msg_type != "total":
                    lines.append(f"  {msg_type}: {count}")

    if error_summarizer:
        summary = error_summarizer.get_error_summary()
        if summary.get("total_errors", 0) > 0:
            lines.extend(
                [
                    "",
                    "LLM Errors:",
                    f"  Total: {summary.get('total_errors', 0)}",
                ]
            )
            for error_type, count in summary.get("error_counts", {}).items():
                lines.append(f"  {error_type}: {count}")

    lines.append("=" * 60)

    return "\n".join(lines)
