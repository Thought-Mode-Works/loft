"""
Comprehensive tests for log analysis utilities.

Tests the LogEntry, LogAnalyzer, and search_logs_cli functions.
Aims to reach 75%+ coverage for loft/logging/analysis.py.
"""

import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from loft.logging.analysis import LogEntry, LogAnalyzer, search_logs_cli


class TestLogEntry:
    """Tests for LogEntry dataclass."""

    def test_log_entry_creation(self):
        """Test creating a LogEntry."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        entry = LogEntry(
            timestamp=timestamp,
            level="INFO",
            component="llm",
            function="generate",
            line=42,
            message="Test message",
            extras={},
        )

        assert entry.timestamp == timestamp
        assert entry.level == "INFO"
        assert entry.component == "llm"
        assert entry.function == "generate"
        assert entry.line == 42
        assert entry.message == "Test message"
        assert entry.extras == {}

    def test_log_entry_with_extras(self):
        """Test LogEntry with extra fields."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="ERROR",
            component="validation",
            function="check",
            line=100,
            message="Validation failed",
            extras={"error_code": "E001", "retry_count": 3},
        )

        assert entry.extras["error_code"] == "E001"
        assert entry.extras["retry_count"] == 3

    def test_matches_level_filter(self):
        """Test matching by level."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="INFO",
            component="llm",
            function="test",
            line=1,
            message="Test",
            extras={},
        )

        assert entry.matches(level="INFO")
        assert not entry.matches(level="ERROR")

    def test_matches_component_filter(self):
        """Test matching by component."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="INFO",
            component="llm",
            function="test",
            line=1,
            message="Test",
            extras={},
        )

        assert entry.matches(component="llm")
        assert not entry.matches(component="validation")

    def test_matches_function_filter(self):
        """Test matching by function."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="INFO",
            component="llm",
            function="generate",
            line=1,
            message="Test",
            extras={},
        )

        assert entry.matches(function="generate")
        assert not entry.matches(function="validate")

    def test_matches_message_contains_filter(self):
        """Test matching by message content."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="INFO",
            component="llm",
            function="test",
            line=1,
            message="Processing request 12345",
            extras={},
        )

        assert entry.matches(message_contains="request")
        assert entry.matches(message_contains="12345")
        assert not entry.matches(message_contains="error")

    def test_matches_after_filter(self):
        """Test matching by time (after)."""
        entry_time = datetime(2024, 1, 1, 12, 0, 0)
        entry = LogEntry(
            timestamp=entry_time,
            level="INFO",
            component="llm",
            function="test",
            line=1,
            message="Test",
            extras={},
        )

        # After a time before the entry
        assert entry.matches(after=datetime(2024, 1, 1, 11, 0, 0))
        # After a time after the entry
        assert not entry.matches(after=datetime(2024, 1, 1, 13, 0, 0))

    def test_matches_before_filter(self):
        """Test matching by time (before)."""
        entry_time = datetime(2024, 1, 1, 12, 0, 0)
        entry = LogEntry(
            timestamp=entry_time,
            level="INFO",
            component="llm",
            function="test",
            line=1,
            message="Test",
            extras={},
        )

        # Before a time after the entry
        assert entry.matches(before=datetime(2024, 1, 1, 13, 0, 0))
        # Before a time before the entry
        assert not entry.matches(before=datetime(2024, 1, 1, 11, 0, 0))

    def test_matches_multiple_filters(self):
        """Test matching with multiple filters."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="ERROR",
            component="validation",
            function="check",
            line=1,
            message="Validation failed for rule 123",
            extras={},
        )

        # All filters match
        assert entry.matches(
            level="ERROR",
            component="validation",
            message_contains="failed",
        )

        # One filter doesn't match
        assert not entry.matches(
            level="ERROR",
            component="llm",  # Wrong component
            message_contains="failed",
        )

    def test_matches_no_filters(self):
        """Test matching with no filters (should always match)."""
        entry = LogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="INFO",
            component="llm",
            function="test",
            line=1,
            message="Test",
            extras={},
        )

        assert entry.matches()


class TestLogAnalyzer:
    """Tests for LogAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test creating a LogAnalyzer."""
        analyzer = LogAnalyzer()
        assert analyzer.log_dir == Path("logs")

    def test_analyzer_with_custom_log_dir(self):
        """Test LogAnalyzer with custom log directory."""
        custom_dir = Path("/tmp/custom_logs")
        analyzer = LogAnalyzer(log_dir=custom_dir)
        assert analyzer.log_dir == custom_dir

    def test_parse_log_line_valid(self):
        """Test parsing a valid log line."""
        analyzer = LogAnalyzer()
        line = "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Test message"
        entry = analyzer.parse_log_line(line)

        assert entry is not None
        assert entry.timestamp == datetime(2024, 1, 1, 12, 0, 0, 123000)
        assert entry.level == "INFO"
        assert entry.component == "llm"
        assert entry.function == "function"
        assert entry.line == 42
        assert entry.message == "Test message"

    def test_parse_log_line_different_levels(self):
        """Test parsing log lines with different levels."""
        analyzer = LogAnalyzer()

        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            line = f"2024-01-01 12:00:00.123 | {level}     | llm | module:function:42 | Test"
            entry = analyzer.parse_log_line(line)
            assert entry is not None
            assert entry.level == level

    def test_parse_log_line_invalid_format(self):
        """Test parsing invalid log line."""
        analyzer = LogAnalyzer()
        line = "This is not a valid log line"
        entry = analyzer.parse_log_line(line)

        assert entry is None

    def test_parse_log_line_invalid_timestamp(self):
        """Test parsing log line with invalid timestamp."""
        analyzer = LogAnalyzer()
        line = "invalid-timestamp | INFO | llm | module:function:42 | Test"
        entry = analyzer.parse_log_line(line)

        assert entry is None

    def test_parse_log_line_with_complex_message(self):
        """Test parsing log line with complex message."""
        analyzer = LogAnalyzer()
        line = "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Processing: key=value, status=ok"
        entry = analyzer.parse_log_line(line)

        assert entry is not None
        assert "key=value" in entry.message
        assert "status=ok" in entry.message

    def test_search_logs_file_not_found(self):
        """Test searching logs when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = LogAnalyzer(log_dir=Path(tmpdir))
            entries = analyzer.search_logs(log_file="nonexistent.log")

            assert entries == []

    def test_search_logs_no_filters(self):
        """Test searching logs without filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "test.log"

            # Create test log file
            log_file.write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Message 1\n"
                "2024-01-01 12:01:00.123 | ERROR    | validation | module:function:43 | Message 2\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.search_logs(log_file="test.log")

            assert len(entries) == 2

    def test_search_logs_with_level_filter(self):
        """Test searching logs with level filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "test.log"

            log_file.write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Message 1\n"
                "2024-01-01 12:01:00.123 | ERROR    | llm | module:function:43 | Message 2\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.search_logs(log_file="test.log", level="ERROR")

            assert len(entries) == 1
            assert entries[0].level == "ERROR"

    def test_search_logs_with_component_filter(self):
        """Test searching logs with component filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "test.log"

            log_file.write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Message 1\n"
                "2024-01-01 12:01:00.123 | INFO     | validation | module:function:43 | Message 2\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.search_logs(log_file="test.log", component="llm")

            assert len(entries) == 1
            assert entries[0].component == "llm"

    def test_search_logs_with_time_filter(self):
        """Test searching logs with time filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "test.log"

            log_file.write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Message 1\n"
                "2024-01-01 13:00:00.123 | INFO     | llm | module:function:43 | Message 2\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.search_logs(
                log_file="test.log",
                after=datetime(2024, 1, 1, 12, 30, 0),
            )

            assert len(entries) == 1
            assert entries[0].message == "Message 2"

    def test_get_llm_interactions_no_filters(self):
        """Test getting LLM interactions without filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "llm_interactions.log"

            log_file.write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | LLM call 1\n"
                "2024-01-01 13:00:00.123 | INFO     | llm | module:function:43 | LLM call 2\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.get_llm_interactions()

            assert len(entries) == 2
            assert all(e.component == "llm" for e in entries)

    def test_get_llm_interactions_with_time_range(self):
        """Test getting LLM interactions with time range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "llm_interactions.log"

            log_file.write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | LLM call 1\n"
                "2024-01-01 13:00:00.123 | INFO     | llm | module:function:43 | LLM call 2\n"
                "2024-01-01 14:00:00.123 | INFO     | llm | module:function:44 | LLM call 3\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.get_llm_interactions(
                after=datetime(2024, 1, 1, 12, 30, 0),
                before=datetime(2024, 1, 1, 13, 30, 0),
            )

            assert len(entries) == 1
            assert entries[0].message == "LLM call 2"

    def test_get_validation_failures(self):
        """Test getting validation failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "validation.log"

            log_file.write_text(
                "2024-01-01 12:00:00.123 | INFO     | validation | module:function:42 | PASS: test 1\n"
                "2024-01-01 12:01:00.123 | ERROR    | validation | module:function:43 | FAIL: test 2\n"
                "2024-01-01 12:02:00.123 | ERROR    | validation | module:function:44 | FAIL: test 3\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.get_validation_failures()

            assert len(entries) == 2
            assert all("FAIL" in e.message for e in entries)

    def test_get_validation_failures_with_time_filter(self):
        """Test getting validation failures with time filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "validation.log"

            log_file.write_text(
                "2024-01-01 12:00:00.123 | ERROR    | validation | module:function:42 | FAIL: test 1\n"
                "2024-01-01 13:00:00.123 | ERROR    | validation | module:function:43 | FAIL: test 2\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.get_validation_failures(after=datetime(2024, 1, 1, 12, 30, 0))

            assert len(entries) == 1
            assert entries[0].message == "FAIL: test 2"

    def test_get_errors(self):
        """Test getting errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "errors.log"

            log_file.write_text(
                "2024-01-01 12:00:00.123 | ERROR    | system | module:function:42 | Error 1\n"
                "2024-01-01 12:01:00.123 | ERROR    | system | module:function:43 | Error 2\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.get_errors()

            assert len(entries) == 2

    def test_get_errors_with_time_filter(self):
        """Test getting errors with time filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "errors.log"

            log_file.write_text(
                "2024-01-01 12:00:00.123 | ERROR    | system | module:function:42 | Error 1\n"
                "2024-01-01 13:00:00.123 | ERROR    | system | module:function:43 | Error 2\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.get_errors(after=datetime(2024, 1, 1, 12, 30, 0))

            assert len(entries) == 1

    def test_aggregate_by_component(self):
        """Test aggregating log entries by component."""
        analyzer = LogAnalyzer()

        entries = [
            LogEntry(datetime.now(), "INFO", "llm", "func", 1, "msg", {}),
            LogEntry(datetime.now(), "INFO", "llm", "func", 1, "msg", {}),
            LogEntry(datetime.now(), "INFO", "validation", "func", 1, "msg", {}),
        ]

        counts = analyzer.aggregate_by_component(entries)

        assert counts["llm"] == 2
        assert counts["validation"] == 1

    def test_aggregate_by_level(self):
        """Test aggregating log entries by level."""
        analyzer = LogAnalyzer()

        entries = [
            LogEntry(datetime.now(), "INFO", "llm", "func", 1, "msg", {}),
            LogEntry(datetime.now(), "ERROR", "llm", "func", 1, "msg", {}),
            LogEntry(datetime.now(), "ERROR", "llm", "func", 1, "msg", {}),
        ]

        counts = analyzer.aggregate_by_level(entries)

        assert counts["INFO"] == 1
        assert counts["ERROR"] == 2

    def test_get_metrics_over_time_empty(self):
        """Test metrics over time with empty entries."""
        analyzer = LogAnalyzer()
        metrics = analyzer.get_metrics_over_time([])

        assert metrics == {}

    def test_get_metrics_over_time_single_entry(self):
        """Test metrics over time with single entry."""
        analyzer = LogAnalyzer()
        timestamp = datetime(2024, 1, 1, 12, 0, 0)

        entries = [
            LogEntry(timestamp, "INFO", "llm", "func", 1, "msg", {}),
        ]

        metrics = analyzer.get_metrics_over_time(entries, interval=timedelta(hours=1))

        assert len(metrics) >= 1
        assert timestamp in metrics

    def test_get_metrics_over_time_multiple_entries(self):
        """Test metrics over time with multiple entries."""
        analyzer = LogAnalyzer()

        entries = [
            LogEntry(datetime(2024, 1, 1, 12, 0, 0), "INFO", "llm", "func", 1, "msg", {}),
            LogEntry(datetime(2024, 1, 1, 12, 30, 0), "INFO", "llm", "func", 1, "msg", {}),
            LogEntry(datetime(2024, 1, 1, 13, 0, 0), "INFO", "llm", "func", 1, "msg", {}),
        ]

        metrics = analyzer.get_metrics_over_time(entries, interval=timedelta(hours=1))

        # Should have buckets for time ranges
        assert len(metrics) > 0

    def test_generate_report_no_entries(self):
        """Test generating report with no entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = LogAnalyzer(log_dir=Path(tmpdir))
            report = analyzer.generate_report()

            assert "No log entries found" in report

    def test_generate_report_with_entries(self):
        """Test generating report with entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            # Create multiple log files
            (log_dir / "loft.log").write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Message\n"
            )
            (log_dir / "llm_interactions.log").write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | LLM call\n"
            )
            (log_dir / "validation.log").write_text(
                "2024-01-01 12:00:00.123 | ERROR    | validation | module:function:42 | FAIL: test\n"
            )
            (log_dir / "errors.log").write_text(
                "2024-01-01 12:00:00.123 | ERROR    | system | module:function:42 | Error\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            report = analyzer.generate_report()

            assert "LOFT System Log Report" in report
            assert "Total Log Entries:" in report
            assert "Log Levels:" in report
            assert "Components:" in report
            assert "LLM Interactions:" in report
            assert "Validation Failures:" in report
            assert "Errors:" in report

    def test_generate_report_with_time_range(self):
        """Test generating report with time range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            (log_dir / "loft.log").write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Message\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            report = analyzer.generate_report(
                after=datetime(2024, 1, 1, 11, 0, 0),
                before=datetime(2024, 1, 1, 13, 0, 0),
            )

            assert "Time Range:" in report

    def test_trace_request(self):
        """Test tracing a specific request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            # Create multiple log files with same request ID
            (log_dir / "llm.log").write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Request 12345 started\n"
            )
            (log_dir / "validation.log").write_text(
                "2024-01-01 12:01:00.123 | INFO     | validation | module:function:43 | Request 12345 validated\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.trace_request(12345)

            assert len(entries) == 2
            # Should be sorted by timestamp
            assert entries[0].timestamp < entries[1].timestamp

    def test_trace_request_not_found(self):
        """Test tracing a request that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            (log_dir / "test.log").write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Some message\n"
            )

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.trace_request(99999)

            assert len(entries) == 0


class TestSearchLogsCli:
    """Tests for search_logs_cli function."""

    def test_search_logs_cli_no_filters(self, capsys):
        """Test CLI search with no filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            (log_dir / "loft.log").write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Test message\n"
            )

            search_logs_cli(log_dir=str(log_dir))

            captured = capsys.readouterr()
            assert "Found" in captured.out
            assert "Test message" in captured.out

    def test_search_logs_cli_with_component_filter(self, capsys):
        """Test CLI search with component filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            (log_dir / "loft.log").write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | LLM message\n"
                "2024-01-01 12:01:00.123 | INFO     | validation | module:function:43 | Val message\n"
            )

            search_logs_cli(log_dir=str(log_dir), component="llm")

            captured = capsys.readouterr()
            assert "LLM message" in captured.out
            assert "Val message" not in captured.out

    def test_search_logs_cli_with_level_filter(self, capsys):
        """Test CLI search with level filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            (log_dir / "loft.log").write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Info message\n"
                "2024-01-01 12:01:00.123 | ERROR    | llm | module:function:43 | Error message\n"
            )

            search_logs_cli(log_dir=str(log_dir), level="ERROR")

            captured = capsys.readouterr()
            assert "Error message" in captured.out
            assert "Info message" not in captured.out

    def test_search_logs_cli_with_time_filter(self, capsys):
        """Test CLI search with time filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            (log_dir / "loft.log").write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Message 1\n"
                "2024-01-01 13:00:00.123 | INFO     | llm | module:function:43 | Message 2\n"
            )

            search_logs_cli(log_dir=str(log_dir), after="2024-01-01T12:30:00")

            captured = capsys.readouterr()
            assert "Message 2" in captured.out
            assert "Message 1" not in captured.out

    def test_search_logs_cli_with_message_filter(self, capsys):
        """Test CLI search with message filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            (log_dir / "loft.log").write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Error occurred\n"
                "2024-01-01 12:01:00.123 | INFO     | llm | module:function:43 | Success\n"
            )

            search_logs_cli(log_dir=str(log_dir), message_contains="Error")

            captured = capsys.readouterr()
            assert "Error occurred" in captured.out
            assert "Success" not in captured.out

    def test_search_logs_cli_multiple_filters(self, capsys):
        """Test CLI search with multiple filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            (log_dir / "loft.log").write_text(
                "2024-01-01 12:00:00.123 | ERROR    | llm | module:function:42 | LLM error\n"
                "2024-01-01 12:01:00.123 | ERROR    | validation | module:function:43 | Val error\n"
                "2024-01-01 12:02:00.123 | INFO     | llm | module:function:44 | LLM info\n"
            )

            search_logs_cli(
                log_dir=str(log_dir),
                component="llm",
                level="ERROR",
            )

            captured = capsys.readouterr()
            assert "LLM error" in captured.out
            assert "Val error" not in captured.out
            assert "LLM info" not in captured.out

    def test_search_logs_cli_no_results(self, capsys):
        """Test CLI search with no matching results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            (log_dir / "loft.log").write_text(
                "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Test\n"
            )

            search_logs_cli(log_dir=str(log_dir), component="nonexistent")

            captured = capsys.readouterr()
            assert "Found 0" in captured.out


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_log_line_with_extra_pipes(self):
        """Test parsing log line with pipes in the message."""
        analyzer = LogAnalyzer()
        line = (
            "2024-01-01 12:00:00.123 | INFO     | llm | module:function:42 | Message | with | pipes"
        )
        entry = analyzer.parse_log_line(line)

        # Should parse correctly, message includes the extra pipes
        assert entry is not None
        assert "with" in entry.message
        assert "pipes" in entry.message

    def test_search_logs_empty_file(self):
        """Test searching an empty log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "empty.log"
            log_file.write_text("")

            analyzer = LogAnalyzer(log_dir=log_dir)
            entries = analyzer.search_logs(log_file="empty.log")

            assert entries == []

    def test_aggregate_empty_entries(self):
        """Test aggregating empty entry list."""
        analyzer = LogAnalyzer()

        component_counts = analyzer.aggregate_by_component([])
        level_counts = analyzer.aggregate_by_level([])

        assert component_counts == {}
        assert level_counts == {}

    def test_trace_request_empty_directory(self):
        """Test tracing request in empty log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = LogAnalyzer(log_dir=Path(tmpdir))
            entries = analyzer.trace_request(12345)

            assert entries == []

    def test_parse_log_line_different_spacing(self):
        """Test parsing log lines with different spacing."""
        analyzer = LogAnalyzer()

        # Different spacing in level field
        line = "2024-01-01 12:00:00.123 | DEBUG    | llm | module:function:42 | Test"
        entry = analyzer.parse_log_line(line)
        assert entry is not None
        assert entry.level == "DEBUG"

    def test_get_metrics_over_time_custom_interval(self):
        """Test metrics over time with custom interval."""
        analyzer = LogAnalyzer()

        entries = [
            LogEntry(datetime(2024, 1, 1, 12, 0, 0), "INFO", "llm", "func", 1, "msg", {}),
            LogEntry(datetime(2024, 1, 1, 12, 10, 0), "INFO", "llm", "func", 1, "msg", {}),
            LogEntry(datetime(2024, 1, 1, 12, 20, 0), "INFO", "llm", "func", 1, "msg", {}),
        ]

        # Use 15-minute intervals
        metrics = analyzer.get_metrics_over_time(entries, interval=timedelta(minutes=15))

        assert len(metrics) > 0
