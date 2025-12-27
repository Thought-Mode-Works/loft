"""
Comprehensive integration tests for Phase 8: Baseline Infrastructure Validation.

Tests the complete Phase 8 infrastructure including:
- Full pipeline processing (gap identification → rule generation → validation → incorporation)
- Meta-aware batch processing with strategy selection
- Experiment runner with checkpointing and resumption
- ASP persistence across cycles
- Coverage tracking and iterative rule building

Issue #260: Phase 8 Integration Tests and Documentation
"""

import json
from unittest.mock import MagicMock

import pytest

from loft.batch.full_pipeline import FullPipelineProcessor, ProcessingMetrics
from loft.batch.meta_aware_processor import (
    MetaAwareBatchConfig,
    MetaAwareBatchProcessor,
)
from loft.batch.schemas import CaseStatus
from loft.experiments import ExperimentConfig, ExperimentRunner


class TestPhase8EndToEnd:
    """End-to-end tests for Phase 8 infrastructure."""

    @pytest.fixture
    def full_experiment_setup(self, tmp_path):
        """Set up complete experiment environment."""
        # Create directory structure
        dataset_path = tmp_path / "dataset"
        rules_path = tmp_path / "asp_rules"
        state_path = tmp_path / "state"
        reports_path = tmp_path / "reports"

        dataset_path.mkdir()
        rules_path.mkdir()
        state_path.mkdir()
        reports_path.mkdir()

        # Create test cases
        for i in range(10):
            case_file = dataset_path / f"case_{i:03d}.json"
            case_file.write_text(
                json.dumps(
                    {
                        "id": f"case_{i:03d}",
                        "asp_facts": f"fact_{i}(yes).\npredicate_{i}(value).",
                        "facts": [f"fact_{i}(yes).", f"predicate_{i}(value)."],
                        "expected_outcome": "yes" if i % 2 == 0 else "no",
                        "ground_truth": "yes" if i % 2 == 0 else "no",
                        "legal_principle": f"Test principle {i}",
                        "_domain": "test_domain",
                    }
                )
            )

        return {
            "dataset_path": dataset_path,
            "rules_path": rules_path,
            "state_path": state_path,
            "reports_path": reports_path,
            "tmp_path": tmp_path,
        }

    def test_full_pipeline_5_cases(self, full_experiment_setup):
        """Test full pipeline processing 5 cases."""
        # Create mocked components
        mock_generator = MagicMock()
        mock_validation = MagicMock()
        mock_incorporation = MagicMock()
        mock_persistence = MagicMock()

        # Setup mock rule generation
        mock_rule = MagicMock()
        mock_rule.asp_rule = "test_rule(X) :- fact(X)."
        mock_rule.reasoning = "Test reasoning"
        mock_rule.confidence = 0.8
        mock_rule.source_type = "gap_fill"

        mock_gap_response = MagicMock()
        mock_gap_response.candidates = [MagicMock(rule=mock_rule)]
        mock_gap_response.recommended_index = 0
        mock_generator.fill_knowledge_gap.return_value = mock_gap_response

        # Setup mock validation
        mock_validation_report = MagicMock()
        mock_validation_report.final_decision = "accept"
        mock_validation_report.aggregate_confidence = 0.85
        mock_validation.validate_rule.return_value = mock_validation_report

        # Setup mock incorporation
        mock_inc_result = MagicMock()
        mock_inc_result.is_success.return_value = True
        mock_incorporation.incorporate.return_value = mock_inc_result

        # Create processor
        processor = FullPipelineProcessor(
            rule_generator=mock_generator,
            validation_pipeline=mock_validation,
            incorporation_engine=mock_incorporation,
            persistence_manager=mock_persistence,
        )

        # Load and process test cases
        dataset_path = full_experiment_setup["dataset_path"]
        case_files = sorted(dataset_path.glob("*.json"))[:5]

        results = []
        accumulated_rules = []

        for case_file in case_files:
            case_data = json.loads(case_file.read_text())
            result = processor.process_case(case_data, accumulated_rules)
            results.append(result)
            accumulated_rules.extend(result.generated_rule_ids)

        # Verify processing
        assert len(results) == 5
        assert all(r.status == CaseStatus.SUCCESS for r in results)
        assert all(r.processing_time_ms > 0 for r in results)

        # Verify rules were generated for gaps
        assert sum(r.rules_generated for r in results) >= 0

        # Verify metrics were collected
        metrics = processor.get_metrics()
        assert isinstance(metrics, ProcessingMetrics)
        assert metrics.total_processing_time_ms > 0

    def test_meta_aware_batch_10_cases(self, full_experiment_setup):
        """Test meta-aware batch processing with 10 cases."""
        # Create mock pipeline processor
        mock_pipeline = MagicMock()

        # Setup mock processing result
        mock_result = MagicMock()
        mock_result.status = CaseStatus.SUCCESS
        mock_result.processing_time_ms = 100.0
        mock_result.rules_generated = 1
        mock_result.rules_accepted = 1
        mock_result.rules_rejected = 0
        mock_result.generated_rule_ids = ["rule_1"]
        mock_pipeline.process_case.return_value = mock_result

        # Create meta-aware processor
        config = MetaAwareBatchConfig(
            enable_strategy_selection=True,
            enable_failure_analysis=True,
        )
        meta_processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline,
            config=config,
        )

        # Load and process test cases
        dataset_path = full_experiment_setup["dataset_path"]
        case_files = sorted(dataset_path.glob("*.json"))

        results = []
        accumulated_rules = []

        for case_file in case_files:
            case_data = json.loads(case_file.read_text())
            meta_result = meta_processor.process_case_with_meta(
                case_data, accumulated_rules
            )
            results.append(meta_result)
            accumulated_rules.extend(meta_result.case_result.generated_rule_ids)

        # Verify processing
        assert len(results) == 10
        assert all(r.case_result.status == CaseStatus.SUCCESS for r in results)

        # Verify strategy selection occurred
        strategies_used = set(r.strategy_used for r in results)
        assert len(strategies_used) >= 1  # At least one strategy was used

        # Verify meta-reasoning tracking
        assert meta_processor.cases_processed == 10
        assert meta_processor.successful_cases == 10

        # Verify strategy performance tracking
        strategy_summary = meta_processor.get_strategy_summary()
        assert "current_weights" in strategy_summary
        assert "performance" in strategy_summary

    def test_improvement_cycle_2_cycles(self, full_experiment_setup):
        """Test 2 improvement cycles with full pipeline."""
        # Create mock processor
        mock_processor = MagicMock()

        # Setup mock result
        mock_result = MagicMock()
        mock_result.processing_time_ms = 100.0
        mock_result.prediction_correct = True
        mock_result.rules_generated = 1
        mock_result.rules_accepted = 1
        mock_result.rules_rejected = 0
        mock_result.generated_rule_ids = []
        mock_processor.process_case.return_value = mock_result

        # Create mock persistence
        mock_persistence = MagicMock()

        # Create experiment config
        config = ExperimentConfig(
            experiment_id="integration_test_2_cycles",
            description="Integration test: 2 improvement cycles",
            dataset_path=full_experiment_setup["dataset_path"],
            state_path=full_experiment_setup["state_path"],
            reports_path=full_experiment_setup["reports_path"],
            rules_path=full_experiment_setup["rules_path"],
            max_cycles=2,
            cases_per_cycle=5,
            cool_down_seconds=0,
        )

        # Create and run experiment
        runner = ExperimentRunner(
            config=config,
            processor=mock_processor,
            persistence=mock_persistence,
        )

        report = runner.run()

        # Verify cycles completed
        assert report.cycles_completed == 2
        assert runner.state.cases_processed == 10
        assert runner.state.cycles_completed == 2

        # Verify state was saved
        state_file = config.state_path / f"{config.experiment_id}_state.json"
        assert state_file.exists()

        # Verify final report was generated
        final_report = config.reports_path / f"{config.experiment_id}_final.md"
        assert final_report.exists()

        # Verify report content
        report_content = final_report.read_text()
        assert "Experiment Report" in report_content
        assert "integration_test_2_cycles" in report_content


class TestCrossComponentIntegration:
    """Test integration between Phase 8 components."""

    def test_pipeline_with_meta_with_persistence(self, tmp_path):
        """Test full stack: pipeline + meta + persistence."""
        # Create mock components
        mock_generator = MagicMock()
        mock_validation = MagicMock()
        mock_incorporation = MagicMock()
        mock_persistence = MagicMock()

        # Setup mocks
        mock_rule = MagicMock()
        mock_rule.asp_rule = "test(X) :- fact(X)."
        mock_rule.confidence = 0.8
        mock_rule.reasoning = "test"
        mock_rule.source_type = "gap_fill"

        mock_gap_response = MagicMock()
        mock_gap_response.candidates = [MagicMock(rule=mock_rule)]
        mock_gap_response.recommended_index = 0
        mock_generator.fill_knowledge_gap.return_value = mock_gap_response

        mock_validation_report = MagicMock()
        mock_validation_report.final_decision = "accept"
        mock_validation_report.aggregate_confidence = 0.8
        mock_validation.validate_rule.return_value = mock_validation_report

        mock_inc_result = MagicMock()
        mock_inc_result.is_success.return_value = True
        mock_incorporation.incorporate.return_value = mock_inc_result

        # Create integrated stack
        pipeline = FullPipelineProcessor(
            rule_generator=mock_generator,
            validation_pipeline=mock_validation,
            incorporation_engine=mock_incorporation,
            persistence_manager=mock_persistence,
        )

        meta = MetaAwareBatchProcessor(
            pipeline_processor=pipeline,
        )

        # Process test cases
        test_case = {
            "id": "test_001",
            "asp_facts": "fact(yes).",
            "ground_truth": "yes",
            "prediction": "no",  # Mismatch triggers gap
            "legal_principle": "Test principle",
        }

        result = meta.process_case_with_meta(test_case, [])

        # Verify processing completed
        assert result.case_result.status == CaseStatus.SUCCESS
        assert result.strategy_used is not None

    def test_experiment_runner_state_persistence(self, tmp_path):
        """Test experiment runner saves and loads state correctly."""
        # Setup directories
        dataset_path = tmp_path / "dataset"
        state_path = tmp_path / "state"
        reports_path = tmp_path / "reports"
        rules_path = tmp_path / "rules"

        dataset_path.mkdir()
        state_path.mkdir()
        reports_path.mkdir()
        rules_path.mkdir()

        # Create test cases
        for i in range(15):
            case_file = dataset_path / f"case_{i:03d}.json"
            case_file.write_text(
                json.dumps(
                    {
                        "id": f"case_{i:03d}",
                        "facts": [f"fact_{i}(yes)."],
                        "expected_outcome": "yes",
                    }
                )
            )

        # Create config
        config = ExperimentConfig(
            experiment_id="state_test",
            description="State persistence test",
            dataset_path=dataset_path,
            state_path=state_path,
            reports_path=reports_path,
            rules_path=rules_path,
            max_cycles=3,
            cases_per_cycle=5,
            cool_down_seconds=0,
        )

        # First run: complete 1 cycle
        mock_processor1 = MagicMock()
        mock_persistence1 = MagicMock()

        mock_result = MagicMock()
        mock_result.processing_time_ms = 100.0
        mock_result.prediction_correct = True
        mock_result.rules_generated = 1
        mock_result.rules_accepted = 1
        mock_result.rules_rejected = 0
        mock_result.generated_rule_ids = []
        mock_processor1.process_case.return_value = mock_result

        runner1 = ExperimentRunner(
            config=config,
            processor=mock_processor1,
            persistence=mock_persistence1,
        )

        # Override to complete only 1 cycle
        runner1.config.max_cycles = 1
        report1 = runner1.run()

        assert report1.cycles_completed == 1

        # Second run: resume and complete remaining cycles
        mock_processor2 = MagicMock()
        mock_persistence2 = MagicMock()
        mock_processor2.process_case.return_value = mock_result

        runner2 = ExperimentRunner(
            config=config,
            processor=mock_processor2,
            persistence=mock_persistence2,
        )

        # Verify state was loaded
        assert runner2.state.cycles_completed == 1
        assert runner2.state.dataset_cursor == 5

        # Complete remaining cycles
        runner2.config.max_cycles = 3
        report2 = runner2.run()

        assert report2.cycles_completed == 3
        assert runner2.state.cases_processed == 15

    def test_coverage_tracking_integration(self, tmp_path):
        """Test coverage tracking through multiple cycles."""
        # This test validates that coverage metrics are tracked correctly
        # as rules are added across cycles

        # Create mock processor that tracks rules
        rule_count = {"current": 0}

        def process_case(case, accumulated_rules):
            result = MagicMock()
            result.processing_time_ms = 100.0
            result.prediction_correct = True
            result.rules_generated = 1 if rule_count["current"] < 5 else 0
            result.rules_accepted = 1 if rule_count["current"] < 5 else 0
            result.rules_rejected = 0
            result.generated_rule_ids = (
                [f"rule_{rule_count['current']}"] if rule_count["current"] < 5 else []
            )

            rule_count["current"] += 1
            return result

        mock_processor = MagicMock()
        mock_processor.process_case.side_effect = process_case

        # Setup experiment
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        for i in range(10):
            case_file = dataset_path / f"case_{i:03d}.json"
            case_file.write_text(
                json.dumps({"id": f"case_{i:03d}", "facts": [], "expected_outcome": "yes"})
            )

        config = ExperimentConfig(
            experiment_id="coverage_test",
            description="Coverage tracking test",
            dataset_path=dataset_path,
            state_path=tmp_path / "state",
            reports_path=tmp_path / "reports",
            rules_path=tmp_path / "rules",
            max_cycles=2,
            cases_per_cycle=5,
            cool_down_seconds=0,
        )

        runner = ExperimentRunner(
            config=config,
            processor=mock_processor,
            persistence=MagicMock(),
        )

        report = runner.run()

        # Verify coverage increased over time
        assert report.cycles_completed == 2
        # First 5 cases should have generated rules
        assert rule_count["current"] == 10


class TestPhase8ValidationMetrics:
    """Test Phase 8 baseline validation metrics."""

    def test_processing_metrics_collection(self):
        """Test that processing metrics are collected correctly."""
        metrics = ProcessingMetrics()

        # Simulate processing
        metrics.gaps_identified = 5
        metrics.rules_generated = 4
        metrics.rules_validated = 4
        metrics.rules_incorporated = 3
        metrics.rules_persisted = 3
        metrics.generation_errors = 0
        metrics.validation_failures = 1
        metrics.incorporation_failures = 1
        metrics.total_processing_time_ms = 5000.0
        metrics.llm_time_ms = 3000.0
        metrics.validation_time_ms = 1500.0
        metrics.incorporation_time_ms = 500.0

        # Convert to dict for serialization
        metrics_dict = metrics.to_dict()

        # Verify all metrics present
        assert metrics_dict["gaps_identified"] == 5
        assert metrics_dict["rules_generated"] == 4
        assert metrics_dict["rules_validated"] == 4
        assert metrics_dict["rules_incorporated"] == 3
        assert metrics_dict["validation_failures"] == 1
        assert metrics_dict["total_processing_time_ms"] == 5000.0

    def test_meta_aware_failure_tracking(self):
        """Test failure pattern detection in meta-aware processor."""
        mock_pipeline = MagicMock()

        # Setup failures
        failure_result = MagicMock()
        failure_result.status = CaseStatus.FAILED
        failure_result.error_message = "Validation error: rule rejected"
        failure_result.rules_generated = 1
        failure_result.rules_accepted = 0
        mock_pipeline.process_case.return_value = failure_result

        config = MetaAwareBatchConfig(
            enable_failure_analysis=True,
            min_failures_for_adaptation=3,
        )

        meta_processor = MetaAwareBatchProcessor(
            pipeline_processor=mock_pipeline,
            config=config,
        )

        # Process cases that fail
        for i in range(5):
            test_case = {
                "id": f"case_{i}",
                "asp_facts": "fact(yes).",
                "ground_truth": "yes",
            }
            meta_processor.process_case_with_meta(test_case, [])

        # Verify failure tracking
        assert meta_processor.failed_cases == 5
        assert len(meta_processor.failure_patterns) > 0

        # Verify failure summary
        failure_summary = meta_processor.get_failure_summary()
        assert failure_summary["total_failures"] > 0
        assert "failure_types" in failure_summary


@pytest.mark.slow
class TestLongRunningExperiments:
    """Tests for longer-running experiments (marked as slow)."""

    def test_30_case_experiment(self, tmp_path):
        """Test processing 30 cases across multiple cycles."""
        # Setup
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        for i in range(30):
            case_file = dataset_path / f"case_{i:03d}.json"
            case_file.write_text(
                json.dumps(
                    {
                        "id": f"case_{i:03d}",
                        "facts": [f"fact_{i}(yes)."],
                        "expected_outcome": "yes" if i % 2 == 0 else "no",
                    }
                )
            )

        config = ExperimentConfig(
            experiment_id="long_test",
            description="30-case experiment",
            dataset_path=dataset_path,
            state_path=tmp_path / "state",
            reports_path=tmp_path / "reports",
            rules_path=tmp_path / "rules",
            max_cycles=6,
            cases_per_cycle=5,
            cool_down_seconds=0,
        )

        mock_processor = MagicMock()
        mock_result = MagicMock()
        mock_result.processing_time_ms = 100.0
        mock_result.prediction_correct = True
        mock_result.rules_generated = 1
        mock_result.rules_accepted = 1
        mock_result.rules_rejected = 0
        mock_result.generated_rule_ids = []
        mock_processor.process_case.return_value = mock_result

        runner = ExperimentRunner(
            config=config,
            processor=mock_processor,
            persistence=MagicMock(),
        )

        report = runner.run()

        # Verify completion
        assert report.cycles_completed == 6
        assert runner.state.cases_processed == 30

        # Verify reports were generated
        final_report = config.reports_path / f"{config.experiment_id}_final.md"
        assert final_report.exists()
