"""
Unit tests for logging infrastructure.

Tests LOFTLogger, decorators, and analysis utilities.
"""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
from loft.logging import (
    LOFTLogger,
    get_loft_logger,
    initialize_logging,
    track_llm_call,
    track_symbolic_operation,
    track_validation,
    track_meta_reasoning,
    performance_monitor,
    LogAnalyzer,
    LogEntry,
)


class TestLOFTLogger:
    """Tests for LOFTLogger class."""

    def test_logger_initialization(self) -> None:
        """Test logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = LOFTLogger(
                log_dir=Path(tmpdir),
                level="INFO",
                enable_file_logging=False,  # Disable for testing
            )
            assert logger.log_dir == Path(tmpdir)
            assert logger.level == "INFO"

    def test_logger_creates_log_directory(self) -> None:
        """Test that logger creates log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            assert not log_dir.exists()

            _ = LOFTLogger(log_dir=log_dir, enable_file_logging=False)
            assert log_dir.exists()

    def test_get_component_logger(self) -> None:
        """Test getting a component-specific logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = LOFTLogger(log_dir=Path(tmpdir), enable_file_logging=False)
            llm_logger = logger.get_logger("llm")
            assert llm_logger is not None

    def test_sampling_always_true_when_no_rate(self) -> None:
        """Test that sampling returns True when no rate is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = LOFTLogger(log_dir=Path(tmpdir), enable_file_logging=False)
            assert logger.should_sample("llm") is True

    def test_sampling_respects_rate(self) -> None:
        """Test that sampling respects configured rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = LOFTLogger(
                log_dir=Path(tmpdir),
                enable_file_logging=False,
                sampling_rate={"llm": 0.0},  # Never sample
            )
            # With rate of 0, should always return False
            assert logger.should_sample("llm") is False


class TestGetLoftLogger:
    """Tests for get_loft_logger function."""

    def test_get_loft_logger_returns_logger(self) -> None:
        """Test that get_loft_logger returns a logger instance."""
        logger = get_loft_logger("test_component")
        assert logger is not None


class TestInitializeLogging:
    """Tests for initialize_logging function."""

    def test_initialize_logging_returns_instance(self) -> None:
        """Test that initialize_logging returns a LOFTLogger instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger_instance = initialize_logging(
                log_dir=Path(tmpdir), level="INFO", enable_file_logging=False
            )
            assert isinstance(logger_instance, LOFTLogger)


class TestTrackLLMCall:
    """Tests for track_llm_call decorator."""

    def test_track_llm_call_basic(self) -> None:
        """Test basic LLM call tracking."""

        @track_llm_call()
        def dummy_llm_call(prompt: str, model: str = "test-model") -> str:
            return "response"

        result = dummy_llm_call(prompt="test prompt", model="test-model")
        assert result == "response"

    def test_track_llm_call_captures_error(self) -> None:
        """Test that decorator captures errors."""

        @track_llm_call()
        def failing_llm_call(prompt: str, model: str = "test-model") -> str:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_llm_call(prompt="test", model="test-model")


class TestTrackSymbolicOperation:
    """Tests for track_symbolic_operation decorator."""

    def test_track_symbolic_operation_basic(self) -> None:
        """Test basic symbolic operation tracking."""

        @track_symbolic_operation("rule_add")
        def add_rule(rule: str) -> None:
            pass

        add_rule(rule="test_rule")

    def test_track_symbolic_operation_with_return(self) -> None:
        """Test symbolic operation tracking with return value."""

        @track_symbolic_operation("query")
        def query_rules() -> list:
            return ["rule1", "rule2"]

        result = query_rules()
        assert result == ["rule1", "rule2"]

    def test_track_symbolic_operation_captures_error(self) -> None:
        """Test that decorator captures errors."""

        @track_symbolic_operation("invalid")
        def failing_operation() -> None:
            raise RuntimeError("Operation failed")

        with pytest.raises(RuntimeError, match="Operation failed"):
            failing_operation()


class TestTrackValidation:
    """Tests for track_validation decorator."""

    def test_track_validation_pass(self) -> None:
        """Test validation tracking for passing validation."""

        @track_validation("syntax")
        def validate_syntax(program: str) -> bool:
            return True

        result = validate_syntax("test program")
        assert result is True

    def test_track_validation_fail(self) -> None:
        """Test validation tracking for failing validation."""

        @track_validation("semantic")
        def validate_semantic(program: str) -> bool:
            return False

        result = validate_semantic("test program")
        assert result is False


class TestTrackMetaReasoning:
    """Tests for track_meta_reasoning decorator."""

    def test_track_meta_reasoning_basic(self) -> None:
        """Test basic meta-reasoning tracking."""

        @track_meta_reasoning("self_assessment")
        def assess_performance() -> float:
            return 0.95

        result = assess_performance()
        assert result == 0.95

    def test_track_meta_reasoning_with_none_return(self) -> None:
        """Test meta-reasoning tracking with None return."""

        @track_meta_reasoning("strategy_change")
        def change_strategy() -> None:
            pass

        result = change_strategy()
        assert result is None


class TestPerformanceMonitor:
    """Tests for performance_monitor decorator."""

    def test_performance_monitor_fast_function(self) -> None:
        """Test performance monitoring for fast function."""

        @performance_monitor(threshold_ms=1000)
        def fast_function() -> str:
            return "done"

        result = fast_function()
        assert result == "done"

    def test_performance_monitor_slow_function(self) -> None:
        """Test performance monitoring for slow function."""
        import time

        @performance_monitor(threshold_ms=10)  # Very low threshold
        def slow_function() -> str:
            time.sleep(0.02)  # 20ms
            return "done"

        result = slow_function()
        assert result == "done"


class TestLogAnalyzer:
    """Tests for LogAnalyzer class."""

    def test_analyzer_initialization(self) -> None:
        """Test analyzer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = LogAnalyzer(log_dir=Path(tmpdir))
            assert analyzer.log_dir == Path(tmpdir)

    def test_parse_log_line_valid(self) -> None:
        """Test parsing a valid log line."""
        analyzer = LogAnalyzer()
        line = "2024-01-01 12:00:00.123 | INFO     | system | module:function:42 | Test message"
        entry = analyzer.parse_log_line(line)

        assert entry is not None
        assert entry.level == "INFO"
        assert entry.component == "system"
        assert entry.function == "function"
        assert entry.line == 42
        assert entry.message == "Test message"

    def test_parse_log_line_invalid(self) -> None:
        """Test parsing an invalid log line."""
        analyzer = LogAnalyzer()
        line = "Invalid log line"
        entry = analyzer.parse_log_line(line)
        assert entry is None

    def test_log_entry_matches_filters(self) -> None:
        """Test LogEntry.matches with various filters."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="INFO",
            component="llm",
            function="query",
            line=42,
            message="Test message",
            extras={},
        )

        assert entry.matches(level="INFO") is True
        assert entry.matches(level="DEBUG") is False
        assert entry.matches(component="llm") is True
        assert entry.matches(component="validation") is False
        assert entry.matches(message_contains="Test") is True
        assert entry.matches(message_contains="Missing") is False

    def test_log_entry_time_filters(self) -> None:
        """Test LogEntry time-based filters."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="INFO",
            component="system",
            function="test",
            line=1,
            message="Test",
            extras={},
        )

        assert entry.matches(after=datetime(2024, 1, 1, 11, 0, 0)) is True
        assert entry.matches(after=datetime(2024, 1, 1, 13, 0, 0)) is False
        assert entry.matches(before=datetime(2024, 1, 1, 13, 0, 0)) is True
        assert entry.matches(before=datetime(2024, 1, 1, 11, 0, 0)) is False

    def test_aggregate_by_component(self) -> None:
        """Test aggregating log entries by component."""
        analyzer = LogAnalyzer()
        entries = [
            LogEntry(
                datetime.now(),
                "INFO",
                "llm",
                "test",
                1,
                "msg1",
                {},
            ),
            LogEntry(
                datetime.now(),
                "INFO",
                "llm",
                "test",
                1,
                "msg2",
                {},
            ),
            LogEntry(
                datetime.now(),
                "INFO",
                "validation",
                "test",
                1,
                "msg3",
                {},
            ),
        ]

        counts = analyzer.aggregate_by_component(entries)
        assert counts["llm"] == 2
        assert counts["validation"] == 1

    def test_aggregate_by_level(self) -> None:
        """Test aggregating log entries by level."""
        analyzer = LogAnalyzer()
        entries = [
            LogEntry(datetime.now(), "INFO", "sys", "test", 1, "msg1", {}),
            LogEntry(datetime.now(), "ERROR", "sys", "test", 1, "msg2", {}),
            LogEntry(datetime.now(), "ERROR", "sys", "test", 1, "msg3", {}),
        ]

        counts = analyzer.aggregate_by_level(entries)
        assert counts["INFO"] == 1
        assert counts["ERROR"] == 2

    def test_get_metrics_over_time_empty(self) -> None:
        """Test metrics over time with empty entries."""
        analyzer = LogAnalyzer()
        metrics = analyzer.get_metrics_over_time([])
        assert metrics == {}

    def test_get_metrics_over_time(self) -> None:
        """Test metrics over time with entries."""
        analyzer = LogAnalyzer()
        now = datetime.now()
        entries = [
            LogEntry(now, "INFO", "sys", "test", 1, "msg1", {}),
            LogEntry(now + timedelta(minutes=30), "INFO", "sys", "test", 1, "msg2", {}),
            LogEntry(
                now + timedelta(hours=1, minutes=30),
                "INFO",
                "sys",
                "test",
                1,
                "msg3",
                {},
            ),
        ]

        metrics = analyzer.get_metrics_over_time(entries, interval=timedelta(hours=1))
        assert len(metrics) > 0  # Should have at least one bucket

    def test_generate_report_empty(self) -> None:
        """Test generating report with no entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = LogAnalyzer(log_dir=Path(tmpdir))
            report = analyzer.generate_report()
            assert "No log entries found" in report


class TestLogConfigEnhancements:
    """Tests for enhanced LogConfig."""

    def test_log_config_has_new_fields(self) -> None:
        """Test that LogConfig has new fields."""
        from loft.config import LogConfig

        config = LogConfig()
        assert hasattr(config, "log_dir")
        assert hasattr(config, "enable_file_logging")
        assert hasattr(config, "enable_console_logging")
        assert hasattr(config, "sampling_rate_llm")
        assert hasattr(config, "sampling_rate_symbolic")

    def test_log_config_defaults(self) -> None:
        """Test LogConfig default values."""
        from loft.config import LogConfig

        config = LogConfig()
        assert config.log_dir == "logs"
        assert config.enable_file_logging is True
        assert config.enable_console_logging is True
        assert config.sampling_rate_llm == 1.0
        assert config.sampling_rate_symbolic == 1.0

    def test_log_config_sampling_rate_validation(self) -> None:
        """Test that sampling rates are validated."""
        from loft.config import LogConfig
        from pydantic import ValidationError

        # Valid sampling rates
        config = LogConfig(sampling_rate_llm=0.5)
        assert config.sampling_rate_llm == 0.5

        # Invalid sampling rate (> 1.0)
        with pytest.raises(ValidationError):
            LogConfig(sampling_rate_llm=1.5)

        # Invalid sampling rate (< 0.0)
        with pytest.raises(ValidationError):
            LogConfig(sampling_rate_llm=-0.1)
