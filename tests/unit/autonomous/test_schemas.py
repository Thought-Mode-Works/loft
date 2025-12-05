"""Unit tests for autonomous test harness schemas."""

from datetime import datetime, timedelta

import pytest

from loft.autonomous.schemas import (
    CycleResult,
    CycleStatus,
    MetaReasoningState,
    RunCheckpoint,
    RunMetrics,
    RunProgress,
    RunResult,
    RunState,
    RunStatus,
)


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_status_values(self):
        """Test all status values are defined."""
        assert RunStatus.PENDING == "pending"
        assert RunStatus.RUNNING == "running"
        assert RunStatus.COMPLETED == "completed"
        assert RunStatus.FAILED == "failed"
        assert RunStatus.CANCELLED == "cancelled"


class TestCycleStatus:
    """Tests for CycleStatus enum."""

    def test_status_values(self):
        """Test all status values are defined."""
        assert CycleStatus.PENDING == "pending"
        assert CycleStatus.ANALYZING == "analyzing"
        assert CycleStatus.COMPLETED == "completed"
        assert CycleStatus.FAILED == "failed"


class TestCycleResult:
    """Tests for CycleResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = CycleResult(cycle_number=1)

        assert result.cycle_number == 1
        assert result.status == CycleStatus.PENDING
        assert result.cases_processed == 0
        assert result.accuracy_before == 0.0
        assert result.accuracy_after == 0.0

    def test_accuracy_delta(self):
        """Test accuracy delta calculation."""
        result = CycleResult(
            cycle_number=1,
            accuracy_before=0.7,
            accuracy_after=0.8,
        )

        assert result.accuracy_delta == pytest.approx(0.1)

    def test_duration_seconds(self):
        """Test duration calculation."""
        start = datetime.now()
        end = start + timedelta(seconds=120)

        result = CycleResult(
            cycle_number=1,
            started_at=start,
            completed_at=end,
        )

        assert result.duration_seconds == pytest.approx(120.0)

    def test_duration_seconds_incomplete(self):
        """Test duration returns None when incomplete."""
        result = CycleResult(cycle_number=1, started_at=datetime.now())
        assert result.duration_seconds is None

    def test_to_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = CycleResult(
            cycle_number=1,
            status=CycleStatus.COMPLETED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            cases_processed=50,
            improvements_applied=3,
            accuracy_before=0.7,
            accuracy_after=0.75,
            failure_patterns=["pattern1", "pattern2"],
        )

        data = original.to_dict()
        restored = CycleResult.from_dict(data)

        assert restored.cycle_number == original.cycle_number
        assert restored.status == original.status
        assert restored.cases_processed == original.cases_processed
        assert restored.accuracy_before == pytest.approx(original.accuracy_before)


class TestRunProgress:
    """Tests for RunProgress dataclass."""

    def test_default_values(self):
        """Test default values."""
        progress = RunProgress()

        assert progress.total_cases == 0
        assert progress.cases_processed == 0
        assert progress.current_accuracy == 0.0

    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        progress = RunProgress(total_cases=100, cases_processed=50)
        assert progress.completion_percentage == pytest.approx(50.0)

    def test_completion_percentage_zero_total(self):
        """Test completion percentage with zero total cases."""
        progress = RunProgress(total_cases=0, cases_processed=0)
        assert progress.completion_percentage == pytest.approx(0.0)

    def test_current_accuracy(self):
        """Test current accuracy calculation."""
        progress = RunProgress(
            cases_processed=100,
            cases_successful=85,
            cases_failed=15,
        )
        assert progress.current_accuracy == pytest.approx(0.85)

    def test_cases_per_hour(self):
        """Test cases per hour calculation."""
        progress = RunProgress(
            cases_processed=100,
            elapsed_seconds=3600,  # 1 hour
        )
        assert progress.cases_per_hour == pytest.approx(100.0)

    def test_to_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = RunProgress(
            total_cases=200,
            cases_processed=100,
            cases_successful=85,
            cases_failed=15,
            elapsed_seconds=3600,
        )

        data = original.to_dict()
        restored = RunProgress.from_dict(data)

        assert restored.total_cases == original.total_cases
        assert restored.cases_processed == original.cases_processed


class TestRunMetrics:
    """Tests for RunMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        metrics = RunMetrics()

        assert metrics.overall_accuracy == 0.0
        assert metrics.rules_generated_total == 0
        assert metrics.improvement_cycles_completed == 0

    def test_to_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = RunMetrics(
            overall_accuracy=0.85,
            accuracy_by_domain={"contracts": 0.9, "torts": 0.8},
            rules_generated_total=50,
            improvement_cycles_completed=5,
        )

        data = original.to_dict()
        restored = RunMetrics.from_dict(data)

        assert restored.overall_accuracy == pytest.approx(original.overall_accuracy)
        assert restored.accuracy_by_domain == original.accuracy_by_domain


class TestRunState:
    """Tests for RunState dataclass."""

    def test_default_values(self):
        """Test default values."""
        state = RunState(run_id="test_run")

        assert state.run_id == "test_run"
        assert state.status == RunStatus.PENDING
        assert state.shutdown_requested is False

    def test_to_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = RunState(
            run_id="test_run",
            status=RunStatus.RUNNING,
            started_at=datetime.now(),
            progress=RunProgress(total_cases=100, cases_processed=50),
        )

        data = original.to_dict()
        restored = RunState.from_dict(data)

        assert restored.run_id == original.run_id
        assert restored.status == original.status


class TestMetaReasoningState:
    """Tests for MetaReasoningState dataclass."""

    def test_default_values(self):
        """Test default values."""
        state = MetaReasoningState()

        assert state.improver_state == {}
        assert state.optimizer_state == {}

    def test_to_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = MetaReasoningState(
            improver_state={"key": "value"},
            optimizer_state={"prompts": ["p1", "p2"]},
        )

        data = original.to_dict()
        restored = MetaReasoningState.from_dict(data)

        assert restored.improver_state == original.improver_state
        assert restored.optimizer_state == original.optimizer_state


class TestRunCheckpoint:
    """Tests for RunCheckpoint dataclass."""

    def test_to_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = RunCheckpoint(
            checkpoint_number=5,
            created_at=datetime.now(),
            run_id="test_run",
            run_state=RunState(run_id="test_run", status=RunStatus.RUNNING),
            config_snapshot={"max_duration_hours": 4.0},
            meta_reasoning_state=MetaReasoningState(),
            cycle_results=[CycleResult(cycle_number=1, status=CycleStatus.COMPLETED)],
            accumulated_rules=[{"rule": "test"}],
        )

        data = original.to_dict()
        restored = RunCheckpoint.from_dict(data)

        assert restored.checkpoint_number == original.checkpoint_number
        assert restored.run_id == original.run_id
        assert len(restored.cycle_results) == 1


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_duration_hours(self):
        """Test duration_hours property."""
        result = RunResult(
            run_id="test",
            status=RunStatus.COMPLETED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_duration_seconds=7200,  # 2 hours
            config_used={},
            final_metrics=RunMetrics(),
        )

        assert result.duration_hours == pytest.approx(2.0)

    def test_was_successful(self):
        """Test was_successful property."""
        completed = RunResult(
            run_id="test",
            status=RunStatus.COMPLETED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_duration_seconds=0,
            config_used={},
            final_metrics=RunMetrics(),
        )
        assert completed.was_successful is True

        failed = RunResult(
            run_id="test",
            status=RunStatus.FAILED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_duration_seconds=0,
            config_used={},
            final_metrics=RunMetrics(),
        )
        assert failed.was_successful is False

    def test_to_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = RunResult(
            run_id="test_run",
            status=RunStatus.COMPLETED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_duration_seconds=3600,
            config_used={"max_duration_hours": 4.0},
            final_metrics=RunMetrics(overall_accuracy=0.85),
            cycle_results=[CycleResult(cycle_number=1)],
        )

        data = original.to_dict()
        restored = RunResult.from_dict(data)

        assert restored.run_id == original.run_id
        assert restored.status == original.status
        assert restored.was_successful == original.was_successful
