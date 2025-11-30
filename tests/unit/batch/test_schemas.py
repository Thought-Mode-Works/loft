"""Tests for batch learning schemas."""

from datetime import datetime, timedelta

from loft.batch.schemas import (
    BatchCheckpoint,
    BatchConfig,
    BatchMetrics,
    BatchProgress,
    BatchResult,
    BatchStatus,
    CaseResult,
    CaseStatus,
)


class TestCaseResult:
    """Tests for CaseResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful case result."""
        result = CaseResult(
            case_id="test_001",
            status=CaseStatus.SUCCESS,
            processed_at=datetime.now(),
            processing_time_ms=150.5,
            rules_generated=2,
            rules_accepted=1,
            rules_rejected=1,
            prediction_correct=True,
            confidence=0.85,
        )

        assert result.case_id == "test_001"
        assert result.status == CaseStatus.SUCCESS
        assert result.rules_generated == 2
        assert result.prediction_correct is True

    def test_create_failed_result(self):
        """Test creating a failed case result."""
        result = CaseResult(
            case_id="test_002",
            status=CaseStatus.FAILED,
            processed_at=datetime.now(),
            processing_time_ms=50.0,
            error_message="Test error",
        )

        assert result.status == CaseStatus.FAILED
        assert result.error_message == "Test error"

    def test_case_result_roundtrip(self):
        """Test serialization roundtrip."""
        original = CaseResult(
            case_id="roundtrip_test",
            status=CaseStatus.SUCCESS,
            processed_at=datetime.now(),
            processing_time_ms=100.0,
            rules_generated=3,
            rules_accepted=2,
            rules_rejected=1,
            prediction_correct=False,
            confidence=0.75,
            generated_rule_ids=["rule_1", "rule_2"],
            metadata={"domain": "contracts"},
        )

        data = original.to_dict()
        restored = CaseResult.from_dict(data)

        assert restored.case_id == original.case_id
        assert restored.status == original.status
        assert restored.rules_generated == original.rules_generated
        assert restored.generated_rule_ids == original.generated_rule_ids
        assert restored.metadata == original.metadata


class TestBatchProgress:
    """Tests for BatchProgress dataclass."""

    def test_create_progress(self):
        """Test creating batch progress."""
        progress = BatchProgress(
            batch_id="batch_001",
            total_cases=100,
            processed_cases=50,
            successful_cases=45,
            failed_cases=5,
        )

        assert progress.batch_id == "batch_001"
        assert progress.total_cases == 100

    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        progress = BatchProgress(
            batch_id="test",
            total_cases=100,
            processed_cases=25,
        )

        assert progress.completion_percentage == 25.0

    def test_completion_percentage_empty(self):
        """Test completion percentage with no cases."""
        progress = BatchProgress(
            batch_id="test",
            total_cases=0,
        )

        assert progress.completion_percentage == 0.0

    def test_current_accuracy(self):
        """Test accuracy calculation."""
        progress = BatchProgress(
            batch_id="test",
            total_cases=100,
            correct_predictions=80,
            incorrect_predictions=20,
        )

        assert progress.current_accuracy == 0.8

    def test_current_accuracy_no_predictions(self):
        """Test accuracy with no predictions."""
        progress = BatchProgress(
            batch_id="test",
            total_cases=100,
        )

        assert progress.current_accuracy == 0.0

    def test_acceptance_rate(self):
        """Test rule acceptance rate."""
        progress = BatchProgress(
            batch_id="test",
            total_cases=100,
            total_rules_generated=50,
            total_rules_accepted=40,
        )

        assert progress.acceptance_rate == 0.8

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        start = datetime.now() - timedelta(seconds=60)
        progress = BatchProgress(
            batch_id="test",
            total_cases=100,
            started_at=start,
            last_update=datetime.now(),
        )

        assert progress.elapsed_time_seconds >= 59  # Allow for small timing variance

    def test_progress_roundtrip(self):
        """Test serialization roundtrip."""
        original = BatchProgress(
            batch_id="roundtrip",
            total_cases=200,
            processed_cases=100,
            successful_cases=90,
            failed_cases=10,
            total_rules_generated=50,
            total_rules_accepted=40,
            correct_predictions=85,
            incorrect_predictions=15,
            started_at=datetime.now(),
            status=BatchStatus.RUNNING,
        )

        data = original.to_dict()
        restored = BatchProgress.from_dict(data)

        assert restored.batch_id == original.batch_id
        assert restored.total_cases == original.total_cases
        assert restored.processed_cases == original.processed_cases
        assert restored.status == original.status


class TestBatchCheckpoint:
    """Tests for BatchCheckpoint dataclass."""

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        checkpoint = BatchCheckpoint(
            batch_id="batch_001",
            checkpoint_id="checkpoint_001",
            created_at=datetime.now(),
            processed_case_ids=["case_1", "case_2"],
            pending_case_ids=["case_3", "case_4"],
        )

        assert checkpoint.batch_id == "batch_001"
        assert len(checkpoint.processed_case_ids) == 2
        assert len(checkpoint.pending_case_ids) == 2

    def test_checkpoint_with_results(self):
        """Test checkpoint with case results."""
        results = [
            CaseResult(
                case_id="case_1",
                status=CaseStatus.SUCCESS,
                processed_at=datetime.now(),
                processing_time_ms=100.0,
            )
        ]

        checkpoint = BatchCheckpoint(
            batch_id="batch_001",
            checkpoint_id="checkpoint_001",
            created_at=datetime.now(),
            case_results=results,
        )

        assert len(checkpoint.case_results) == 1
        assert checkpoint.case_results[0].case_id == "case_1"

    def test_checkpoint_roundtrip(self):
        """Test serialization roundtrip."""
        results = [
            CaseResult(
                case_id="case_1",
                status=CaseStatus.SUCCESS,
                processed_at=datetime.now(),
                processing_time_ms=100.0,
            )
        ]
        progress = BatchProgress(
            batch_id="batch_001",
            total_cases=100,
            processed_cases=10,
        )

        original = BatchCheckpoint(
            batch_id="batch_001",
            checkpoint_id="checkpoint_001",
            created_at=datetime.now(),
            processed_case_ids=["case_1"],
            pending_case_ids=["case_2", "case_3"],
            case_results=results,
            accumulated_rule_ids=["rule_1"],
            progress=progress,
            config={"max_cases": 100},
        )

        data = original.to_dict()
        restored = BatchCheckpoint.from_dict(data)

        assert restored.batch_id == original.batch_id
        assert restored.checkpoint_id == original.checkpoint_id
        assert len(restored.case_results) == 1
        assert restored.progress.total_cases == 100


class TestBatchMetrics:
    """Tests for BatchMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating batch metrics."""
        metrics = BatchMetrics(
            batch_id="batch_001",
            started_at=datetime.now(),
            cases_processed=100,
            rules_generated=50,
            rules_accepted=40,
        )

        assert metrics.batch_id == "batch_001"
        assert metrics.cases_processed == 100
        assert metrics.rules_accepted == 40

    def test_metrics_with_performance(self):
        """Test metrics with performance data."""
        metrics = BatchMetrics(
            batch_id="batch_001",
            started_at=datetime.now(),
            total_processing_time_ms=60000.0,
            avg_case_time_ms=600.0,
            peak_memory_mb=512.0,
        )

        assert metrics.total_processing_time_ms == 60000.0
        assert metrics.peak_memory_mb == 512.0

    def test_metrics_roundtrip(self):
        """Test serialization roundtrip."""
        original = BatchMetrics(
            batch_id="roundtrip",
            started_at=datetime.now(),
            completed_at=datetime.now(),
            cases_processed=200,
            rules_generated=100,
            rules_accepted=80,
            rules_rejected=20,
            accuracy_before=0.5,
            accuracy_after=0.75,
            accuracy_improvement=0.25,
            domain_metrics={"contracts": {"accuracy": 0.8}},
            error_types={"timeout": 5},
        )

        data = original.to_dict()
        restored = BatchMetrics.from_dict(data)

        assert restored.batch_id == original.batch_id
        assert restored.cases_processed == original.cases_processed
        assert restored.accuracy_improvement == original.accuracy_improvement
        assert restored.domain_metrics == original.domain_metrics


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_create_result(self):
        """Test creating batch result."""
        result = BatchResult(
            batch_id="batch_001",
            status=BatchStatus.COMPLETED,
            started_at=datetime.now(),
        )

        assert result.batch_id == "batch_001"
        assert result.status == BatchStatus.COMPLETED

    def test_success_count(self):
        """Test success count calculation."""
        results = [
            CaseResult(
                case_id=f"case_{i}",
                status=CaseStatus.SUCCESS if i < 8 else CaseStatus.FAILED,
                processed_at=datetime.now(),
                processing_time_ms=100.0,
            )
            for i in range(10)
        ]

        result = BatchResult(
            batch_id="batch_001",
            status=BatchStatus.COMPLETED,
            started_at=datetime.now(),
            case_results=results,
        )

        assert result.success_count == 8
        assert result.failure_count == 2

    def test_total_rules_accepted(self):
        """Test total rules accepted calculation."""
        results = [
            CaseResult(
                case_id=f"case_{i}",
                status=CaseStatus.SUCCESS,
                processed_at=datetime.now(),
                processing_time_ms=100.0,
                rules_accepted=2,
            )
            for i in range(5)
        ]

        result = BatchResult(
            batch_id="batch_001",
            status=BatchStatus.COMPLETED,
            started_at=datetime.now(),
            case_results=results,
        )

        assert result.total_rules_accepted == 10

    def test_result_roundtrip(self):
        """Test serialization roundtrip."""
        case_results = [
            CaseResult(
                case_id="case_1",
                status=CaseStatus.SUCCESS,
                processed_at=datetime.now(),
                processing_time_ms=100.0,
                rules_accepted=2,
            )
        ]
        metrics = BatchMetrics(
            batch_id="batch_001",
            started_at=datetime.now(),
            cases_processed=1,
        )

        original = BatchResult(
            batch_id="batch_001",
            status=BatchStatus.COMPLETED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            case_results=case_results,
            accumulated_rule_ids=["rule_1", "rule_2"],
            metrics=metrics,
            config={"max_cases": 100},
        )

        data = original.to_dict()
        restored = BatchResult.from_dict(data)

        assert restored.batch_id == original.batch_id
        assert restored.status == original.status
        assert len(restored.case_results) == 1
        assert restored.accumulated_rule_ids == original.accumulated_rule_ids


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BatchConfig()

        assert config.max_rules_per_case == 3
        assert config.max_total_rules == 200
        assert config.validation_threshold == 0.8
        assert config.checkpoint_interval == 10
        assert config.continue_on_error is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = BatchConfig(
            max_cases=50,
            max_rules_per_case=5,
            validation_threshold=0.9,
            use_dialectical=False,
        )

        assert config.max_cases == 50
        assert config.max_rules_per_case == 5
        assert config.use_dialectical is False

    def test_config_roundtrip(self):
        """Test serialization roundtrip."""
        original = BatchConfig(
            max_cases=100,
            checkpoint_interval=5,
            validation_threshold=0.7,
            transfer_source_domains=["contracts", "torts"],
        )

        data = original.to_dict()
        restored = BatchConfig.from_dict(data)

        assert restored.max_cases == original.max_cases
        assert restored.checkpoint_interval == original.checkpoint_interval
        assert restored.transfer_source_domains == original.transfer_source_domains
