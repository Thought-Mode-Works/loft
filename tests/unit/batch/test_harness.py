"""Tests for BatchLearningHarness."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from loft.batch.harness import BatchLearningHarness, create_simple_case_processor
from loft.batch.schemas import (
    BatchConfig,
    BatchProgress,
    BatchStatus,
    CaseResult,
    CaseStatus,
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_cases():
    """Create sample test cases."""
    return [
        {
            "id": f"case_{i}",
            "facts": f"Test facts for case {i}",
            "ground_truth": "enforceable" if i % 2 == 0 else "unenforceable",
        }
        for i in range(10)
    ]


def create_mock_processor(
    success_rate: float = 0.8,
    rules_per_case: int = 1,
):
    """Create a mock case processor for testing."""

    def processor(case: Dict[str, Any], accumulated_rules: List[str]) -> CaseResult:
        case_id = case.get("id", "unknown")
        import random

        # Determine if this case succeeds
        success = random.random() < success_rate

        if success:
            return CaseResult(
                case_id=case_id,
                status=CaseStatus.SUCCESS,
                processed_at=datetime.now(),
                processing_time_ms=50.0,
                rules_generated=rules_per_case,
                rules_accepted=rules_per_case,
                prediction_correct=True,
                confidence=0.85,
                generated_rule_ids=[
                    f"rule_{case_id}_{i}" for i in range(rules_per_case)
                ],
            )
        else:
            return CaseResult(
                case_id=case_id,
                status=CaseStatus.FAILED,
                processed_at=datetime.now(),
                processing_time_ms=10.0,
                error_message="Mock failure",
            )

    return processor


class TestBatchLearningHarness:
    """Tests for BatchLearningHarness class."""

    def test_init_default_config(self, temp_output_dir):
        """Test initialization with default config."""
        harness = BatchLearningHarness(output_dir=temp_output_dir)

        assert harness.config is not None
        assert harness.output_dir == temp_output_dir

    def test_init_custom_config(self, temp_output_dir):
        """Test initialization with custom config."""
        config = BatchConfig(
            max_cases=50,
            checkpoint_interval=5,
        )
        harness = BatchLearningHarness(config=config, output_dir=temp_output_dir)

        assert harness.config.max_cases == 50
        assert harness.config.checkpoint_interval == 5

    def test_run_batch_empty(self, temp_output_dir):
        """Test running batch with no cases."""
        harness = BatchLearningHarness(output_dir=temp_output_dir)
        processor = create_mock_processor()

        result = harness.run_batch(
            test_cases=[],
            process_case_fn=processor,
        )

        assert result.status == BatchStatus.COMPLETED
        assert len(result.case_results) == 0

    def test_run_batch_success(self, temp_output_dir, sample_cases):
        """Test successful batch processing."""
        harness = BatchLearningHarness(output_dir=temp_output_dir)
        processor = create_mock_processor(success_rate=1.0)

        result = harness.run_batch(
            test_cases=sample_cases,
            process_case_fn=processor,
        )

        assert result.status == BatchStatus.COMPLETED
        assert len(result.case_results) == 10
        assert result.success_count == 10
        assert result.failure_count == 0

    def test_run_batch_with_failures(self, temp_output_dir, sample_cases):
        """Test batch with some failures."""
        config = BatchConfig(continue_on_error=True)
        harness = BatchLearningHarness(config=config, output_dir=temp_output_dir)
        processor = create_mock_processor(success_rate=0.5)

        result = harness.run_batch(
            test_cases=sample_cases,
            process_case_fn=processor,
        )

        assert result.status == BatchStatus.COMPLETED
        assert len(result.case_results) == 10
        # With 50% success rate, we expect roughly half to fail
        assert result.failure_count > 0

    def test_run_batch_max_cases(self, temp_output_dir, sample_cases):
        """Test max_cases limit."""
        config = BatchConfig(max_cases=5)
        harness = BatchLearningHarness(config=config, output_dir=temp_output_dir)
        processor = create_mock_processor(success_rate=1.0)

        result = harness.run_batch(
            test_cases=sample_cases,
            process_case_fn=processor,
        )

        assert len(result.case_results) == 5

    def test_run_batch_max_rules(self, temp_output_dir):
        """Test max_total_rules limit."""
        config = BatchConfig(max_total_rules=5)
        harness = BatchLearningHarness(config=config, output_dir=temp_output_dir)
        # Each case generates 2 rules
        processor = create_mock_processor(success_rate=1.0, rules_per_case=2)

        cases = [{"id": f"case_{i}"} for i in range(10)]
        result = harness.run_batch(
            test_cases=cases,
            process_case_fn=processor,
        )

        # Should stop after ~3 cases (6 rules would exceed limit)
        assert len(result.accumulated_rule_ids) <= 6

    def test_run_batch_creates_checkpoint(self, temp_output_dir, sample_cases):
        """Test checkpoint creation."""
        config = BatchConfig(checkpoint_interval=3)
        harness = BatchLearningHarness(config=config, output_dir=temp_output_dir)
        processor = create_mock_processor(success_rate=1.0)

        result = harness.run_batch(
            test_cases=sample_cases,
            process_case_fn=processor,
        )

        # Should create checkpoints at cases 3, 6, 9
        checkpoint_dir = temp_output_dir / result.batch_id / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.json"))
            assert len(checkpoints) >= 1

    def test_run_batch_custom_batch_id(self, temp_output_dir, sample_cases):
        """Test custom batch ID."""
        harness = BatchLearningHarness(output_dir=temp_output_dir)
        processor = create_mock_processor(success_rate=1.0)

        result = harness.run_batch(
            test_cases=sample_cases,
            process_case_fn=processor,
            batch_id="custom_batch_123",
        )

        assert result.batch_id == "custom_batch_123"

    def test_run_batch_accumulates_rules(self, temp_output_dir, sample_cases):
        """Test rule accumulation across cases."""
        harness = BatchLearningHarness(output_dir=temp_output_dir)
        processor = create_mock_processor(success_rate=1.0, rules_per_case=2)

        result = harness.run_batch(
            test_cases=sample_cases,
            process_case_fn=processor,
        )

        assert result.total_rules_accepted == 20  # 10 cases * 2 rules

    def test_run_batch_saves_result(self, temp_output_dir, sample_cases):
        """Test result saving."""
        harness = BatchLearningHarness(output_dir=temp_output_dir)
        processor = create_mock_processor(success_rate=1.0)

        result = harness.run_batch(
            test_cases=sample_cases,
            process_case_fn=processor,
        )

        result_path = temp_output_dir / result.batch_id / "result.json"
        assert result_path.exists()

        summary_path = temp_output_dir / result.batch_id / "summary.json"
        assert summary_path.exists()

    def test_progress_callback(self, temp_output_dir, sample_cases):
        """Test progress callback."""
        harness = BatchLearningHarness(output_dir=temp_output_dir)
        processor = create_mock_processor(success_rate=1.0)

        progress_updates = []

        def on_progress(progress: BatchProgress):
            progress_updates.append(progress.processed_cases)

        harness.set_callbacks(on_progress=on_progress)

        harness.run_batch(
            test_cases=sample_cases,
            process_case_fn=processor,
        )

        # Should have received updates for each case
        assert len(progress_updates) == 10
        assert progress_updates[-1] == 10

    def test_case_complete_callback(self, temp_output_dir, sample_cases):
        """Test case complete callback."""
        harness = BatchLearningHarness(output_dir=temp_output_dir)
        processor = create_mock_processor(success_rate=1.0)

        case_results = []

        def on_case_complete(result: CaseResult):
            case_results.append(result.case_id)

        harness.set_callbacks(on_case_complete=on_case_complete)

        harness.run_batch(
            test_cases=sample_cases,
            process_case_fn=processor,
        )

        assert len(case_results) == 10

    def test_get_progress_during_run(self, temp_output_dir, sample_cases):
        """Test getting progress during batch run."""
        harness = BatchLearningHarness(output_dir=temp_output_dir)
        processor = create_mock_processor(success_rate=1.0)

        progress_snapshots = []

        def on_progress(progress: BatchProgress):
            # Capture a copy of progress at each callback
            progress_snapshots.append((progress.processed_cases, progress.status))

        harness.set_callbacks(on_progress=on_progress)

        harness.run_batch(
            test_cases=sample_cases,
            process_case_fn=processor,
        )

        # Should have received progress updates during execution
        assert len(progress_snapshots) == 10
        # All intermediate callbacks should show RUNNING status
        for i, (processed, status) in enumerate(progress_snapshots[:-1]):
            assert status == BatchStatus.RUNNING, f"Progress {i} should be RUNNING"
        # Verify progress incremented properly
        assert progress_snapshots[0][0] == 1  # First update after first case
        assert progress_snapshots[-1][0] == 10  # Last update after all cases

    def test_load_result(self, temp_output_dir, sample_cases):
        """Test loading saved result."""
        harness = BatchLearningHarness(output_dir=temp_output_dir)
        processor = create_mock_processor(success_rate=1.0)

        original = harness.run_batch(
            test_cases=sample_cases,
            process_case_fn=processor,
        )

        result_path = temp_output_dir / original.batch_id / "result.json"
        loaded = BatchLearningHarness.load_result(str(result_path))

        assert loaded.batch_id == original.batch_id
        assert len(loaded.case_results) == len(original.case_results)

    def test_list_batches(self, temp_output_dir, sample_cases):
        """Test listing batch runs."""
        harness = BatchLearningHarness(output_dir=temp_output_dir)
        processor = create_mock_processor(success_rate=1.0)

        # Run two batches
        harness.run_batch(
            test_cases=sample_cases[:5],
            process_case_fn=processor,
            batch_id="batch_001",
        )
        harness.run_batch(
            test_cases=sample_cases[5:],
            process_case_fn=processor,
            batch_id="batch_002",
        )

        batches = BatchLearningHarness.list_batches(str(temp_output_dir))

        assert len(batches) == 2
        batch_ids = [b["batch_id"] for b in batches]
        assert "batch_001" in batch_ids
        assert "batch_002" in batch_ids

    def test_consecutive_errors_limit(self, temp_output_dir):
        """Test max consecutive errors limit."""
        config = BatchConfig(
            max_consecutive_errors=3,
            continue_on_error=True,
        )
        harness = BatchLearningHarness(config=config, output_dir=temp_output_dir)

        # Processor that always fails
        def failing_processor(case, rules):
            return CaseResult(
                case_id=case.get("id"),
                status=CaseStatus.FAILED,
                processed_at=datetime.now(),
                processing_time_ms=10.0,
                error_message="Always fails",
            )

        cases = [{"id": f"case_{i}"} for i in range(10)]
        result = harness.run_batch(
            test_cases=cases,
            process_case_fn=failing_processor,
        )

        # Should stop after 3 consecutive errors
        assert result.status == BatchStatus.FAILED
        assert len(result.case_results) == 3


class TestCreateSimpleCaseProcessor:
    """Tests for create_simple_case_processor helper."""

    def test_create_processor(self):
        """Test creating a simple processor."""

        def rule_gen(gap, predicates):
            return [{"id": "rule_1", "asp_rule": "test_rule."}]

        def validator(rule):
            return True

        def predictor(case, rules):
            return ("enforceable", 0.8)

        processor = create_simple_case_processor(
            rule_generator_fn=rule_gen,
            validator_fn=validator,
            predictor_fn=predictor,
        )

        case = {"id": "test", "ground_truth": "enforceable"}
        result = processor(case, [])

        assert result.status == CaseStatus.SUCCESS
        assert result.prediction_correct is True

    def test_processor_generates_rules_on_wrong_prediction(self):
        """Test rule generation when prediction is wrong."""

        def rule_gen(gap, predicates):
            return [
                {"id": "rule_1", "asp_rule": "rule1."},
                {"id": "rule_2", "asp_rule": "rule2."},
            ]

        def validator(rule):
            return True

        def predictor(case, rules):
            return ("unenforceable", 0.7)  # Wrong prediction

        processor = create_simple_case_processor(
            rule_generator_fn=rule_gen,
            validator_fn=validator,
            predictor_fn=predictor,
        )

        case = {"id": "test", "ground_truth": "enforceable"}
        result = processor(case, [])

        assert result.prediction_correct is False
        assert result.rules_generated == 2
        assert result.rules_accepted == 2

    def test_processor_handles_validation_failure(self):
        """Test handling validation failures."""

        def rule_gen(gap, predicates):
            return [{"id": "rule_1", "asp_rule": "invalid"}]

        def validator(rule):
            return False  # All rules invalid

        def predictor(case, rules):
            return ("wrong", 0.5)

        processor = create_simple_case_processor(
            rule_generator_fn=rule_gen,
            validator_fn=validator,
            predictor_fn=predictor,
        )

        case = {"id": "test", "ground_truth": "correct"}
        result = processor(case, [])

        assert result.rules_generated == 1
        assert result.rules_accepted == 0
        assert result.rules_rejected == 1

    def test_processor_handles_exception(self):
        """Test handling exceptions in processor."""

        def rule_gen(gap, predicates):
            raise ValueError("Test error")

        def validator(rule):
            return True

        def predictor(case, rules):
            return ("wrong", 0.5)

        processor = create_simple_case_processor(
            rule_generator_fn=rule_gen,
            validator_fn=validator,
            predictor_fn=predictor,
        )

        case = {"id": "test", "ground_truth": "correct"}
        result = processor(case, [])

        assert result.status == CaseStatus.FAILED
        assert "Test error" in result.error_message
