"""
Unit tests for the logging configuration module.

Tests the logging improvements for issue #162:
- Clingo message filtering
- LLM error summarization
- Progress indicators
- Log format standardization
"""

import logging
from datetime import timedelta
from unittest.mock import MagicMock, patch

from loft.autonomous.logging_config import (
    ClingoMessageFilter,
    LLMErrorSummarizer,
    ProgressIndicator,
    create_log_summary,
    setup_autonomous_logging,
)


class TestClingoMessageFilter:
    """Tests for Clingo message filtering."""

    def test_filter_suppresses_clingo_info_messages(self):
        """Test that Clingo info messages are suppressed."""
        filter_instance = ClingoMessageFilter(summary_interval_seconds=3600)

        # Create mock log record with Clingo info message
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="<block>:1:39-56: info: atom does not occur in any rule head: party(X)",
            args=(),
            exc_info=None,
        )

        # Should be filtered (return False)
        assert filter_instance.filter(record) is False
        assert filter_instance.suppressed_count == 1

    def test_filter_allows_non_clingo_messages(self):
        """Test that non-Clingo messages pass through."""
        filter_instance = ClingoMessageFilter(summary_interval_seconds=3600)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Processing case contract_001",
            args=(),
            exc_info=None,
        )

        # Should pass through (return True)
        assert filter_instance.filter(record) is True
        assert filter_instance.suppressed_count == 0

    def test_filter_categorizes_messages(self):
        """Test that messages are properly categorized."""
        filter_instance = ClingoMessageFilter(summary_interval_seconds=3600)

        # Test atom warning
        record1 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="atom does not occur in any rule head: test(X)",
            args=(),
            exc_info=None,
        )
        filter_instance.filter(record1)

        assert "undefined_atom_warnings" in filter_instance.suppressed_messages
        assert filter_instance.suppressed_messages["undefined_atom_warnings"] == 1

    def test_get_final_summary(self):
        """Test final summary generation."""
        filter_instance = ClingoMessageFilter(summary_interval_seconds=3600)

        # Suppress a few messages
        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=f"atom does not occur in any rule head: pred{i}(X)",
                args=(),
                exc_info=None,
            )
            filter_instance.filter(record)

        summary = filter_instance.get_final_summary()
        assert summary["total"] == 5
        assert "undefined_atom_warnings" in summary


class TestLLMErrorSummarizer:
    """Tests for LLM error summarization."""

    def test_summarize_llm_error_creates_concise_summary(self):
        """Test that error summaries are concise."""
        summarizer = LLMErrorSummarizer(max_error_length=100)

        summary = summarizer.summarize_llm_error(
            error_message="Failed to parse structured output",
            attempts=3,
            last_exception="ValidationError: asp_rule field contains incomplete syntax",
            context={"case_id": "contract_001", "predicate": "enforceable(X)"},
        )

        # Summary should be reasonably short
        assert len(summary) < 300
        assert "case=contract_001" in summary
        assert "attempts=3" in summary

    def test_categorizes_errors_correctly(self):
        """Test error categorization."""
        summarizer = LLMErrorSummarizer()

        # Test ASP syntax error
        summarizer.summarize_llm_error(
            error_message="ASP syntax validation failed",
            last_exception="Syntax error in rule",
        )
        assert summarizer.error_counts.get("asp_syntax", 0) == 1

        # Test unsafe variable error
        summarizer.summarize_llm_error(
            error_message="Rule contains unsafe variable",
            last_exception="unsafe variable X",
        )
        assert summarizer.error_counts.get("unsafe_variable", 0) == 1

    def test_get_error_summary(self):
        """Test error summary aggregation."""
        summarizer = LLMErrorSummarizer()

        # Add several errors
        for i in range(3):
            summarizer.summarize_llm_error(
                error_message=f"Error {i}",
                last_exception="syntax error",
            )

        summary = summarizer.get_error_summary()
        assert summary["total_errors"] == 3
        assert len(summary["recent_errors"]) <= 10

    def test_truncates_long_errors(self):
        """Test that long error messages are truncated."""
        summarizer = LLMErrorSummarizer(max_error_length=50)

        long_error = "A" * 200  # 200 character error
        summary = summarizer.summarize_llm_error(
            error_message="test",
            last_exception=long_error,
        )

        # Should contain truncated error with "..."
        assert "..." in summary


class TestProgressIndicator:
    """Tests for progress indicator."""

    def test_update_tracks_metrics(self):
        """Test that metrics are tracked correctly."""
        indicator = ProgressIndicator(
            log_interval_seconds=3600,  # Long interval to prevent auto-logging
        )

        indicator.update(
            cases_processed=10,
            cases_successful=8,
            cases_failed=2,
            rules_generated=20,
            rules_accepted=15,
            llm_calls=30,
            estimated_cost_usd=0.50,
        )

        assert indicator.cases_processed == 10
        assert indicator.cases_successful == 8
        assert indicator.rules_accepted == 15
        assert indicator.estimated_cost_usd == 0.50

    def test_get_summary_returns_complete_data(self):
        """Test that summary includes all metrics."""
        indicator = ProgressIndicator(log_interval_seconds=3600)

        indicator.update(
            cases_processed=100,
            cases_successful=90,
            cases_failed=10,
            rules_generated=200,
            rules_accepted=150,
            llm_calls=300,
            estimated_cost_usd=5.00,
        )

        summary = indicator.get_summary()

        assert "cases_processed" in summary
        assert "cases_successful" in summary
        assert "rules_accepted" in summary
        assert "llm_calls" in summary
        assert "cases_per_hour" in summary
        assert "elapsed_formatted" in summary

    def test_format_duration(self):
        """Test duration formatting."""
        indicator = ProgressIndicator()

        # Test hours
        duration = timedelta(hours=2, minutes=30)
        formatted = indicator._format_duration(duration)
        assert "2h" in formatted

        # Test minutes
        duration = timedelta(minutes=45, seconds=30)
        formatted = indicator._format_duration(duration)
        assert "45m" in formatted

        # Test seconds
        duration = timedelta(seconds=30)
        formatted = indicator._format_duration(duration)
        assert "30s" in formatted


class TestSetupAutonomousLogging:
    """Tests for the main logging setup function."""

    def test_returns_required_components(self):
        """Test that setup returns all required components."""
        with patch("logging.basicConfig"):
            components = setup_autonomous_logging(
                log_level="INFO",
                enable_clingo_filter=True,
            )

        assert "clingo_filter" in components
        assert "error_summarizer" in components
        assert "progress_indicator" in components

        assert isinstance(components["clingo_filter"], ClingoMessageFilter)
        assert isinstance(components["error_summarizer"], LLMErrorSummarizer)
        assert isinstance(components["progress_indicator"], ProgressIndicator)

    def test_respects_clingo_filter_setting(self):
        """Test that clingo filter can be disabled."""
        with patch("logging.basicConfig"):
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                # With filter enabled
                setup_autonomous_logging(
                    log_level="INFO",
                    enable_clingo_filter=True,
                )
                # Filter should be added
                assert mock_logger.addFilter.called


class TestCreateLogSummary:
    """Tests for log summary creation."""

    def test_creates_summary_with_all_components(self):
        """Test that summary includes all component data."""
        clingo_filter = ClingoMessageFilter()
        clingo_filter.suppressed_count = 100
        clingo_filter.suppressed_messages = {"undefined_atom_warnings": 100}

        error_summarizer = LLMErrorSummarizer()
        error_summarizer.error_counts = {"asp_syntax": 10, "parsing_failed": 5}

        progress_indicator = ProgressIndicator()
        progress_indicator.cases_processed = 50
        progress_indicator.rules_accepted = 25

        summary = create_log_summary(
            clingo_filter=clingo_filter,
            error_summarizer=error_summarizer,
            progress_indicator=progress_indicator,
        )

        assert "RUN SUMMARY" in summary
        assert "Clingo Warnings Suppressed" in summary
        assert "LLM Errors" in summary
        assert "Progress" in summary

    def test_handles_none_components(self):
        """Test that None components are handled gracefully."""
        summary = create_log_summary(
            clingo_filter=None,
            error_summarizer=None,
            progress_indicator=None,
        )

        assert "RUN SUMMARY" in summary
        # Should not crash with None components

    def test_handles_zero_counts(self):
        """Test that zero counts don't show sections."""
        clingo_filter = ClingoMessageFilter()
        clingo_filter.suppressed_count = 0

        error_summarizer = LLMErrorSummarizer()
        # No errors added

        summary = create_log_summary(
            clingo_filter=clingo_filter,
            error_summarizer=error_summarizer,
            progress_indicator=None,
        )

        # Sections with zero counts should be omitted
        assert "Clingo Warnings Suppressed" not in summary
        assert "LLM Errors" not in summary
