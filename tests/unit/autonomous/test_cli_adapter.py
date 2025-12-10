"""Tests for the LLMCaseProcessorAdapter in the autonomous CLI.

These tests verify the adapter that bridges the interface gap between
AutonomousTestRunner and LLMCaseProcessor (issue #185).
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# Define minimal mock types for testing without importing heavy dependencies
class MockStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


@dataclass
class MockCaseResult:
    """Mock CaseResult for testing."""

    case_id: str
    status: MockStatus = MockStatus.SUCCESS
    prediction_correct: Optional[bool] = True
    rules_generated: int = 0
    rules_accepted: int = 0
    rules_rejected: int = 0
    generated_rule_ids: list = None
    processing_time_ms: float = 100.0
    error_message: Optional[str] = None
    confidence: float = 0.9

    def __post_init__(self):
        if self.generated_rule_ids is None:
            self.generated_rule_ids = []


class TestLLMCaseProcessorAdapter:
    """Tests for the LLMCaseProcessorAdapter class."""

    def test_adapter_import(self):
        """Test that the adapter can be imported from the CLI module."""
        from loft.autonomous.cli import LLMCaseProcessorAdapter

        adapter = LLMCaseProcessorAdapter(model="claude-3-5-haiku-20241022")
        assert adapter is not None
        assert adapter._model == "claude-3-5-haiku-20241022"
        assert adapter._processor is None  # Lazy initialization
        assert adapter._accumulated_rules == []

    def test_adapter_default_model(self):
        """Test that the adapter uses a sensible default model."""
        from loft.autonomous.cli import LLMCaseProcessorAdapter

        adapter = LLMCaseProcessorAdapter()
        assert adapter._model == "claude-3-5-haiku-20241022"

    @patch("loft.autonomous.llm_processor.LLMCaseProcessor")
    def test_adapter_lazy_initialization(self, mock_processor_class):
        """Test that the adapter initializes the processor lazily."""
        from loft.autonomous.cli import LLMCaseProcessorAdapter

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        adapter = LLMCaseProcessorAdapter(model="test-model")

        # Processor should not be initialized yet
        assert adapter._processor is None
        mock_processor_class.assert_not_called()

        # Initialize explicitly
        adapter.initialize()

        # Now it should be initialized
        assert adapter._processor is not None
        mock_processor_class.assert_called_once_with(model="test-model")
        mock_processor.initialize.assert_called_once()

    @patch("loft.autonomous.llm_processor.LLMCaseProcessor")
    def test_adapter_idempotent_initialization(self, mock_processor_class):
        """Test that calling initialize() multiple times is idempotent."""
        from loft.autonomous.cli import LLMCaseProcessorAdapter

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        adapter = LLMCaseProcessorAdapter()
        adapter.initialize()
        adapter.initialize()
        adapter.initialize()

        # Should only be called once
        mock_processor_class.assert_called_once()
        mock_processor.initialize.assert_called_once()

    @patch("loft.autonomous.llm_processor.LLMCaseProcessor")
    def test_adapter_process_case(self, mock_processor_class):
        """Test that process_case correctly adapts the interface."""
        from loft.autonomous.cli import LLMCaseProcessorAdapter

        # Setup mock processor
        mock_processor = MagicMock()
        mock_result = MockCaseResult(
            case_id="test_case_1",
            status=MockStatus.SUCCESS,
            prediction_correct=True,
            rules_generated=2,
            rules_accepted=1,
            rules_rejected=1,
            generated_rule_ids=["rule_1", "rule_2"],
            processing_time_ms=150.5,
            error_message=None,
            confidence=0.85,
        )
        mock_processor.process_case.return_value = mock_result
        mock_processor_class.return_value = mock_processor

        adapter = LLMCaseProcessorAdapter()
        case_data = {
            "id": "test_case_1",
            "domain": "contracts",
            "facts": "test facts",
        }

        result = adapter.process_case(case_data)

        # Verify result is a dict with expected fields
        assert isinstance(result, dict)
        assert result["case_id"] == "test_case_1"
        assert result["correct"] is True
        assert result["domain"] == "contracts"
        assert result["status"] == "success"
        assert result["rules_generated"] == 2
        assert result["rules_accepted"] == 1
        assert result["rules_rejected"] == 1
        assert result["processing_time_ms"] == 150.5
        assert result["error_message"] is None
        assert result["confidence"] == 0.85

        # Verify accumulated rules are tracked
        assert adapter._accumulated_rules == ["rule_1", "rule_2"]

    @patch("loft.autonomous.llm_processor.LLMCaseProcessor")
    def test_adapter_accumulates_rules_across_cases(self, mock_processor_class):
        """Test that rules accumulate across multiple case processing calls."""
        from loft.autonomous.cli import LLMCaseProcessorAdapter

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        adapter = LLMCaseProcessorAdapter()

        # Process first case
        mock_processor.process_case.return_value = MockCaseResult(
            case_id="case_1",
            generated_rule_ids=["rule_1", "rule_2"],
        )
        adapter.process_case({"id": "case_1", "domain": "contracts"})

        # Verify accumulated rules after first case
        assert adapter._accumulated_rules == ["rule_1", "rule_2"]

        # Process second case
        mock_processor.process_case.return_value = MockCaseResult(
            case_id="case_2",
            generated_rule_ids=["rule_3"],
        )
        adapter.process_case({"id": "case_2", "domain": "torts"})

        # Verify accumulated rules after second case
        assert adapter._accumulated_rules == ["rule_1", "rule_2", "rule_3"]

        # Verify calls were made
        calls = mock_processor.process_case.call_args_list
        assert len(calls) == 2

        # Verify the case data was passed to each call
        assert calls[0][0][0] == {"id": "case_1", "domain": "contracts"}
        assert calls[1][0][0] == {"id": "case_2", "domain": "torts"}

    @patch("loft.autonomous.llm_processor.LLMCaseProcessor")
    def test_adapter_handles_failed_case(self, mock_processor_class):
        """Test that adapter correctly handles failed case processing."""
        from loft.autonomous.cli import LLMCaseProcessorAdapter

        mock_processor = MagicMock()
        mock_result = MockCaseResult(
            case_id="failed_case",
            status=MockStatus.FAILURE,
            prediction_correct=False,
            error_message="ASP syntax error",
            confidence=0.3,
        )
        mock_processor.process_case.return_value = mock_result
        mock_processor_class.return_value = mock_processor

        adapter = LLMCaseProcessorAdapter()
        result = adapter.process_case({"id": "failed_case", "domain": "unknown"})

        assert result["correct"] is False
        assert result["status"] == "failure"
        assert result["error_message"] == "ASP syntax error"
        assert result["confidence"] == 0.3

    @patch("loft.autonomous.llm_processor.LLMCaseProcessor")
    def test_adapter_get_processor(self, mock_processor_class):
        """Test that get_processor returns the underlying processor."""
        from loft.autonomous.cli import LLMCaseProcessorAdapter

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        adapter = LLMCaseProcessorAdapter()

        # Before initialization
        assert adapter.get_processor() is None

        # After initialization
        adapter.initialize()
        assert adapter.get_processor() is mock_processor

    @patch("loft.autonomous.llm_processor.LLMCaseProcessor")
    def test_adapter_get_failure_patterns(self, mock_processor_class):
        """Test that get_failure_patterns delegates to the processor."""
        from loft.autonomous.cli import LLMCaseProcessorAdapter

        mock_processor = MagicMock()
        mock_processor.get_failure_patterns.return_value = {
            "syntax_error": 3,
            "timeout": 1,
        }
        mock_processor_class.return_value = mock_processor

        adapter = LLMCaseProcessorAdapter()

        # Before initialization
        assert adapter.get_failure_patterns() == {}

        # After initialization
        adapter.initialize()
        patterns = adapter.get_failure_patterns()
        assert patterns == {"syntax_error": 3, "timeout": 1}
        mock_processor.get_failure_patterns.assert_called_once()

    @patch("loft.autonomous.llm_processor.LLMCaseProcessor")
    def test_adapter_clear_failure_tracking(self, mock_processor_class):
        """Test that clear_failure_tracking delegates to the processor."""
        from loft.autonomous.cli import LLMCaseProcessorAdapter

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        adapter = LLMCaseProcessorAdapter()

        # Before initialization - should not error
        adapter.clear_failure_tracking()

        # After initialization
        adapter.initialize()
        adapter.clear_failure_tracking()
        mock_processor.clear_failure_tracking.assert_called_once()

    @patch("loft.autonomous.llm_processor.LLMCaseProcessor")
    def test_adapter_handles_none_prediction_correct(self, mock_processor_class):
        """Test that adapter handles None prediction_correct value."""
        from loft.autonomous.cli import LLMCaseProcessorAdapter

        mock_processor = MagicMock()
        mock_result = MockCaseResult(
            case_id="uncertain_case",
            prediction_correct=None,  # Uncertain result
        )
        mock_processor.process_case.return_value = mock_result
        mock_processor_class.return_value = mock_processor

        adapter = LLMCaseProcessorAdapter()
        result = adapter.process_case({"id": "uncertain_case"})

        # None should default to True (optimistic assumption)
        assert result["correct"] is True

    @patch("loft.autonomous.llm_processor.LLMCaseProcessor")
    def test_adapter_extracts_domain_from_case(self, mock_processor_class):
        """Test that adapter correctly extracts domain from case data."""
        from loft.autonomous.cli import LLMCaseProcessorAdapter

        mock_processor = MagicMock()
        mock_processor.process_case.return_value = MockCaseResult(case_id="test")
        mock_processor_class.return_value = mock_processor

        adapter = LLMCaseProcessorAdapter()

        # With domain
        result = adapter.process_case({"id": "test", "domain": "torts"})
        assert result["domain"] == "torts"

        # Without domain
        result = adapter.process_case({"id": "test"})
        assert result["domain"] == "unknown"


class TestCLIOptions:
    """Tests for CLI option handling."""

    def test_enable_llm_option_exists(self):
        """Test that --enable-llm option exists in the CLI."""
        from loft.autonomous.cli import start

        # Check that the command has the enable_llm parameter
        params = {p.name for p in start.params}
        assert "enable_llm" in params

    def test_skip_api_check_option_exists(self):
        """Test that --skip-api-check option exists in the CLI."""
        from loft.autonomous.cli import start

        params = {p.name for p in start.params}
        assert "skip_api_check" in params

    def test_enable_llm_is_flag(self):
        """Test that --enable-llm is a boolean flag."""
        from loft.autonomous.cli import start

        for param in start.params:
            if param.name == "enable_llm":
                assert param.is_flag is True
                break
        else:
            pytest.fail("enable_llm parameter not found")

    def test_skip_api_check_is_flag(self):
        """Test that --skip-api-check is a boolean flag."""
        from loft.autonomous.cli import start

        for param in start.params:
            if param.name == "skip_api_check":
                assert param.is_flag is True
                break
        else:
            pytest.fail("skip_api_check parameter not found")

    def test_enable_ensemble_option_exists(self):
        """Test that --enable-ensemble option exists in the CLI (issue #207)."""
        from loft.autonomous.cli import start

        params = {p.name for p in start.params}
        assert "enable_ensemble" in params

    def test_enable_ensemble_is_flag(self):
        """Test that --enable-ensemble is a boolean flag."""
        from loft.autonomous.cli import start

        for param in start.params:
            if param.name == "enable_ensemble":
                assert param.is_flag is True
                break
        else:
            pytest.fail("enable_ensemble parameter not found")

    def test_enable_ensemble_requires_enable_llm(self):
        """Test that --enable-ensemble without --enable-llm raises UsageError."""
        from click.testing import CliRunner
        from loft.autonomous.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "start",
                "--dataset",
                "datasets/contracts/",
                "--enable-ensemble",  # Without --enable-llm
                "--duration",
                "1m",
            ],
        )

        # Should fail with usage error
        assert result.exit_code != 0
        assert "--enable-ensemble requires --enable-llm" in result.output


class TestEnsembleDiagnostics:
    """Tests for the EnsembleDiagnostics class (issue #207)."""

    def test_diagnostics_import(self):
        """Test that EnsembleDiagnostics can be imported."""
        from loft.autonomous.cli import EnsembleDiagnostics

        diagnostics = EnsembleDiagnostics()
        assert diagnostics is not None
        assert diagnostics.voting_results == []
        assert diagnostics.disagreement_records == []
        assert diagnostics.consensus_rate == 0.0
        assert diagnostics.total_tasks_processed == 0

    def test_diagnostics_record_voting_result(self):
        """Test recording voting results."""
        from loft.autonomous.cli import EnsembleDiagnostics

        diagnostics = EnsembleDiagnostics()
        diagnostics.record_voting_result(
            decision="accept_rule",
            strategy="majority",
            vote_count=3,
            confidence=0.85,
            was_unanimous=True,
        )

        assert len(diagnostics.voting_results) == 1
        assert diagnostics.voting_results[0].decision == "accept_rule"
        assert diagnostics.voting_results[0].strategy == "majority"
        assert diagnostics.total_tasks_processed == 1

    def test_diagnostics_record_disagreement(self):
        """Test recording disagreement resolutions."""
        from loft.autonomous.cli import EnsembleDiagnostics

        diagnostics = EnsembleDiagnostics()
        diagnostics.record_disagreement(
            resolution_type="defer_to_critic",
            original_responses=3,
            final_decision="reject",
        )

        assert len(diagnostics.disagreement_records) == 1
        assert diagnostics.disagreement_records[0].resolution_type == "defer_to_critic"
        assert diagnostics.critic_intervention_count == 1

    def test_diagnostics_record_escalation(self):
        """Test that escalations are counted."""
        from loft.autonomous.cli import EnsembleDiagnostics

        diagnostics = EnsembleDiagnostics()
        diagnostics.record_disagreement(
            resolution_type="escalate",
            original_responses=2,
            final_decision="human_review",
        )

        assert diagnostics.escalation_count == 1

    def test_diagnostics_update_model_performance(self):
        """Test updating model performance metrics."""
        from loft.autonomous.cli import EnsembleDiagnostics

        diagnostics = EnsembleDiagnostics()
        diagnostics.update_model_performance(
            model_name="logic_generator",
            success=True,
            confidence=0.9,
            latency_ms=150.0,
        )

        assert "logic_generator" in diagnostics.model_performance
        perf = diagnostics.model_performance["logic_generator"]
        assert perf.total_requests == 1
        assert perf.successful_requests == 1
        assert perf.average_confidence == 0.9
        assert perf.average_latency_ms == 150.0

    def test_diagnostics_to_dict(self):
        """Test converting diagnostics to dictionary."""
        from loft.autonomous.cli import EnsembleDiagnostics

        diagnostics = EnsembleDiagnostics()
        diagnostics.record_voting_result(
            decision="accept",
            strategy="unanimous",
            vote_count=2,
            confidence=0.95,
            was_unanimous=True,
        )

        result = diagnostics.to_dict()

        assert "consensus_rate" in result
        assert "escalation_count" in result
        assert "critic_intervention_count" in result
        assert "total_tasks_processed" in result
        assert "voting_strategy_distribution" in result
        assert "disagreement_resolutions" in result
        assert "model_accuracy_by_task" in result
        assert "elapsed_seconds" in result


class TestEnsembleCaseProcessorAdapter:
    """Tests for the EnsembleCaseProcessorAdapter class (issue #207)."""

    def test_adapter_import(self):
        """Test that the adapter can be imported from the CLI module."""
        from loft.autonomous.cli import EnsembleCaseProcessorAdapter

        adapter = EnsembleCaseProcessorAdapter(model="claude-3-5-haiku-20241022")
        assert adapter is not None
        assert adapter._model == "claude-3-5-haiku-20241022"
        assert adapter._orchestrator is None  # Lazy initialization
        assert adapter._accumulated_rules == []

    def test_adapter_default_model(self):
        """Test that the adapter uses a sensible default model."""
        from loft.autonomous.cli import EnsembleCaseProcessorAdapter

        adapter = EnsembleCaseProcessorAdapter()
        assert adapter._model == "claude-3-5-haiku-20241022"

    def test_adapter_get_diagnostics(self):
        """Test that get_diagnostics returns the diagnostics object."""
        from loft.autonomous.cli import (
            EnsembleCaseProcessorAdapter,
            EnsembleDiagnostics,
        )

        adapter = EnsembleCaseProcessorAdapter()
        diagnostics = adapter.get_diagnostics()

        assert isinstance(diagnostics, EnsembleDiagnostics)

    def test_adapter_get_diagnostics_dict(self):
        """Test that get_diagnostics_dict returns a dictionary."""
        from loft.autonomous.cli import EnsembleCaseProcessorAdapter

        adapter = EnsembleCaseProcessorAdapter()
        result = adapter.get_diagnostics_dict()

        assert isinstance(result, dict)
        assert "consensus_rate" in result
        assert "total_tasks_processed" in result

    def test_adapter_get_accumulated_rules(self):
        """Test that get_accumulated_rules returns a copy of rules list."""
        from loft.autonomous.cli import EnsembleCaseProcessorAdapter

        adapter = EnsembleCaseProcessorAdapter()
        rules = adapter.get_accumulated_rules()

        assert isinstance(rules, list)
        assert rules == []

    def test_adapter_get_orchestrator_before_init(self):
        """Test that get_orchestrator returns None before initialization."""
        from loft.autonomous.cli import EnsembleCaseProcessorAdapter

        adapter = EnsembleCaseProcessorAdapter()
        assert adapter.get_orchestrator() is None

    def test_adapter_get_performance_metrics_before_init(self):
        """Test that get_performance_metrics returns empty dict before init."""
        from loft.autonomous.cli import EnsembleCaseProcessorAdapter

        adapter = EnsembleCaseProcessorAdapter()
        metrics = adapter.get_performance_metrics()

        assert metrics == {}


class TestModelPerformance:
    """Tests for the ModelPerformance dataclass (issue #207)."""

    def test_model_performance_import(self):
        """Test that ModelPerformance can be imported."""
        from loft.autonomous.cli import ModelPerformance

        perf = ModelPerformance(model_name="test_model")
        assert perf.model_name == "test_model"
        assert perf.total_requests == 0
        assert perf.successful_requests == 0

    def test_model_performance_success_rate(self):
        """Test success rate calculation."""
        from loft.autonomous.cli import ModelPerformance

        perf = ModelPerformance(
            model_name="test",
            total_requests=10,
            successful_requests=8,
        )
        assert perf.success_rate == 0.8

    def test_model_performance_success_rate_zero_requests(self):
        """Test success rate with zero requests."""
        from loft.autonomous.cli import ModelPerformance

        perf = ModelPerformance(model_name="test")
        assert perf.success_rate == 0.0


class TestVotingRecord:
    """Tests for the VotingRecord dataclass (issue #207)."""

    def test_voting_record_import(self):
        """Test that VotingRecord can be imported."""
        from loft.autonomous.cli import VotingRecord

        record = VotingRecord(
            decision="accept",
            strategy="majority",
            vote_count=3,
            confidence=0.85,
        )
        assert record.decision == "accept"
        assert record.strategy == "majority"
        assert record.vote_count == 3
        assert record.confidence == 0.85


class TestDisagreementResolution:
    """Tests for the DisagreementResolution dataclass (issue #207)."""

    def test_disagreement_resolution_import(self):
        """Test that DisagreementResolution can be imported."""
        from loft.autonomous.cli import DisagreementResolution

        resolution = DisagreementResolution(
            resolution_type="defer_to_critic",
            original_responses=4,
            final_decision="reject",
        )
        assert resolution.resolution_type == "defer_to_critic"
        assert resolution.original_responses == 4
        assert resolution.final_decision == "reject"
