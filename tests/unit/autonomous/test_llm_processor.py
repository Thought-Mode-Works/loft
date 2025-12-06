"""
Unit tests for LLMCaseProcessor.

Tests the LLM case processor functionality including lazy initialization,
error handling, and edge cases (issue #166 feedback).
"""

import pytest
from unittest.mock import patch, MagicMock
from loft.autonomous.llm_processor import LLMCaseProcessor, create_llm_processor
from loft.batch.schemas import CaseStatus


class TestLLMCaseProcessorInit:
    """Test LLMCaseProcessor initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        processor = LLMCaseProcessor()

        assert processor.model == "claude-3-5-haiku-20241022"
        assert processor._initialized is False
        assert processor._llm is None
        assert processor._rule_generator is None
        assert processor._validation_pipeline is None

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        processor = LLMCaseProcessor(model="claude-3-opus-20240229")

        assert processor.model == "claude-3-opus-20240229"

    def test_init_with_custom_template(self):
        """Test initialization with custom extraction template."""
        custom_template = "Custom template: {case_text} {domain}"
        processor = LLMCaseProcessor(extraction_prompt_template=custom_template)

        assert processor._extraction_template == custom_template


class TestLLMCaseProcessorMetrics:
    """Test metrics tracking functionality."""

    def test_initial_metrics(self):
        """Test initial metrics are zero."""
        processor = LLMCaseProcessor()
        metrics = processor.get_metrics()

        assert metrics["total_llm_calls"] == 0
        assert metrics["total_tokens_used"] == 0
        assert metrics["total_cost_usd"] == 0.0
        assert metrics["cases_processed"] == 0
        assert metrics["avg_processing_time_ms"] == 0.0
        assert metrics["model"] == "claude-3-5-haiku-20241022"


class TestLLMCaseProcessorProcessCase:
    """Test process_case method edge cases."""

    def test_process_case_with_empty_text_skipped(self):
        """Test that cases with no text are skipped without initialization."""
        processor = LLMCaseProcessor()
        # Note: Empty text check happens BEFORE initialization,
        # so no mocking needed for this test case

        # Case with no text or facts
        case = {"id": "test_001", "domain": "contracts"}
        result = processor.process_case(case, [])

        assert result.status == CaseStatus.SKIPPED
        assert "No case text or facts" in (result.error_message or "")

    def test_process_case_with_empty_facts_skipped(self):
        """Test that cases with empty text and facts are skipped."""
        processor = LLMCaseProcessor()
        # Note: Empty text check happens BEFORE initialization

        case = {"id": "test_002", "text": "", "facts": "", "domain": "contracts"}
        result = processor.process_case(case, [])

        assert result.status == CaseStatus.SKIPPED

    def test_create_process_fn_returns_callable(self):
        """Test that create_process_fn returns the process_case method."""
        processor = LLMCaseProcessor()
        process_fn = processor.create_process_fn()

        assert callable(process_fn)
        assert process_fn == processor.process_case


class TestLLMCaseProcessorInitialization:
    """Test lazy initialization and error handling behavior."""

    def test_initialization_requires_api_key(self):
        """Test that initialization fails without API key."""
        processor = LLMCaseProcessor()

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                processor.initialize()

    def test_initialization_only_runs_once(self):
        """Test that initialization is idempotent."""
        processor = LLMCaseProcessor()
        processor._initialized = True

        # This should not raise even without API key
        processor.initialize()

        assert processor._initialized is True

    def test_ensure_initialized_raises_on_missing_components(self):
        """Test that _ensure_initialized raises RuntimeError when components are None."""
        processor = LLMCaseProcessor()
        processor._initialized = True
        # Leave components as None

        with pytest.raises(RuntimeError, match="not properly initialized"):
            processor._ensure_initialized()

    def test_ensure_initialized_with_all_components(self):
        """Test that _ensure_initialized passes when all components are set."""
        processor = LLMCaseProcessor()
        processor._initialized = True
        # Mock all required components
        processor._llm = MagicMock()
        processor._rule_generator = MagicMock()
        processor._validation_pipeline = MagicMock()
        processor._extract_predicates = MagicMock(return_value=[])

        # Should not raise
        processor._ensure_initialized()


class TestPredicateExtraction:
    """Test predicate extraction with error handling."""

    def test_extract_dataset_predicates_returns_empty_for_no_asp_predicates(self):
        """Test that empty list is returned when no asp_predicates."""
        processor = LLMCaseProcessor()
        extraction = {"facts": ["fact1", "fact2"]}

        result = processor._extract_dataset_predicates(extraction, "test_case")

        assert result == []

    def test_extract_dataset_predicates_returns_empty_for_no_facts(self):
        """Test graceful handling when asp_predicates set but no facts."""
        processor = LLMCaseProcessor()
        extraction = {"asp_predicates": ["pred(X)"], "facts": []}

        result = processor._extract_dataset_predicates(extraction, "test_case")

        assert result == []

    def test_extract_dataset_predicates_handles_exception(self):
        """Test that exceptions during extraction are caught and logged."""
        processor = LLMCaseProcessor()
        processor._extract_predicates = MagicMock(
            side_effect=Exception("Extraction failed")
        )
        extraction = {"asp_predicates": ["pred(X)"], "facts": ["fact1"]}

        result = processor._extract_dataset_predicates(extraction, "test_case")

        # Should return empty list on error, not raise
        assert result == []

    def test_extract_dataset_predicates_success(self):
        """Test successful predicate extraction."""
        processor = LLMCaseProcessor()
        processor._extract_predicates = MagicMock(
            return_value=["contract", "breach", "damages"]
        )
        extraction = {"asp_predicates": ["pred(X)"], "facts": ["contract(a, b)."]}

        result = processor._extract_dataset_predicates(extraction, "test_case")

        assert result == ["contract", "breach", "damages"]
        processor._extract_predicates.assert_called_once()


class TestFactoryFunction:
    """Test the factory function."""

    def test_create_llm_processor_with_defaults(self):
        """Test factory function with default parameters."""
        processor = create_llm_processor()

        assert isinstance(processor, LLMCaseProcessor)
        assert processor.model == "claude-3-5-haiku-20241022"

    def test_create_llm_processor_with_custom_model(self):
        """Test factory function with custom model."""
        processor = create_llm_processor(model="claude-3-sonnet-20240229")

        assert processor.model == "claude-3-sonnet-20240229"
