"""Unit tests for autonomous test harness persistence."""

from datetime import datetime
from pathlib import Path

import pytest

from loft.autonomous.persistence import PersistenceManager, create_persistence_manager
from loft.autonomous.schemas import (
    MetaReasoningState,
    RunCheckpoint,
    RunMetrics,
    RunProgress,
    RunResult,
    RunState,
    RunStatus,
)


@pytest.fixture
def temp_run_dir(tmp_path):
    """Create temporary run directory."""
    run_dir = tmp_path / "test_run"
    return run_dir


@pytest.fixture
def persistence_manager(temp_run_dir):
    """Create persistence manager for testing."""
    return PersistenceManager(temp_run_dir)


class TestPersistenceManager:
    """Tests for PersistenceManager."""

    def test_init_creates_directories(self, persistence_manager, temp_run_dir):
        """Test initialization creates directory structure."""
        assert temp_run_dir.exists()
        assert (temp_run_dir / "checkpoints").exists()
        assert (temp_run_dir / "metrics").exists()
        assert (temp_run_dir / "rules").exists()
        assert (temp_run_dir / "reports").exists()

    def test_save_and_load_config(self, persistence_manager):
        """Test config save and load."""
        config = {"max_duration_hours": 4.0, "llm_model": "haiku"}

        persistence_manager.save_config(config)
        loaded = persistence_manager.load_config()

        assert loaded == config

    def test_load_config_not_found(self, persistence_manager):
        """Test loading nonexistent config returns None."""
        # Don't save anything first
        persistence_manager._run_dir = Path("/nonexistent")
        loaded = persistence_manager.load_config()
        assert loaded is None

    def test_save_and_load_state(self, persistence_manager):
        """Test state save and load."""
        state = RunState(
            run_id="test_run",
            status=RunStatus.RUNNING,
            progress=RunProgress(total_cases=100, cases_processed=50),
        )

        persistence_manager.save_state(state)
        loaded = persistence_manager.load_state()

        assert loaded.run_id == state.run_id
        assert loaded.status == state.status
        assert loaded.progress.total_cases == 100

    def test_save_and_load_checkpoint(self, persistence_manager):
        """Test checkpoint save and load."""
        checkpoint = RunCheckpoint(
            checkpoint_number=1,
            created_at=datetime.now(),
            run_id="test_run",
            run_state=RunState(run_id="test_run"),
            config_snapshot={"key": "value"},
            meta_reasoning_state=MetaReasoningState(),
        )

        path = persistence_manager.save_checkpoint(checkpoint)

        assert path.exists()
        assert "checkpoint_0001.json" in str(path)

        loaded = persistence_manager.load_checkpoint(path)
        assert loaded.checkpoint_number == 1
        assert loaded.run_id == "test_run"

    def test_load_latest_checkpoint(self, persistence_manager):
        """Test loading latest checkpoint via symlink."""
        for i in range(3):
            checkpoint = RunCheckpoint(
                checkpoint_number=i + 1,
                created_at=datetime.now(),
                run_id="test_run",
                run_state=RunState(run_id="test_run"),
                config_snapshot={},
                meta_reasoning_state=MetaReasoningState(),
            )
            persistence_manager.save_checkpoint(checkpoint)

        loaded = persistence_manager.load_checkpoint()
        assert loaded.checkpoint_number == 3

    def test_checkpoint_rotation(self, persistence_manager):
        """Test old checkpoints are rotated."""
        persistence_manager._max_checkpoints = 3

        for i in range(5):
            checkpoint = RunCheckpoint(
                checkpoint_number=i + 1,
                created_at=datetime.now(),
                run_id="test_run",
                run_state=RunState(run_id="test_run"),
                config_snapshot={},
                meta_reasoning_state=MetaReasoningState(),
            )
            persistence_manager.save_checkpoint(checkpoint)

        checkpoints = persistence_manager.list_checkpoints()
        assert len(checkpoints) == 3

    def test_append_timeline_event(self, persistence_manager):
        """Test timeline event appending."""
        persistence_manager.append_timeline_event({"type": "start"})
        persistence_manager.append_timeline_event({"type": "progress", "count": 50})
        persistence_manager.append_timeline_event({"type": "complete"})

        events = persistence_manager.read_timeline()

        assert len(events) == 3
        assert events[0]["type"] == "start"
        assert events[1]["count"] == 50
        assert all("timestamp" in e for e in events)

    def test_save_and_load_metrics(self, persistence_manager):
        """Test metrics save and load."""
        metrics = RunMetrics(
            overall_accuracy=0.85,
            accuracy_by_domain={"contracts": 0.9},
            rules_generated_total=50,
        )

        persistence_manager.save_metrics(metrics)
        loaded = persistence_manager.load_metrics()

        assert loaded.overall_accuracy == pytest.approx(0.85)
        assert loaded.rules_generated_total == 50

    def test_save_and_load_rules(self, persistence_manager):
        """Test rules save and load."""
        rules = [
            {"id": "rule1", "content": "test rule 1"},
            {"id": "rule2", "content": "test rule 2"},
        ]

        persistence_manager.save_rules(rules)
        loaded = persistence_manager.load_rules()

        assert len(loaded) == 2
        assert loaded[0]["id"] == "rule1"

    def test_save_result_creates_json_and_markdown(self, persistence_manager):
        """Test result save creates both formats."""
        result = RunResult(
            run_id="test_run",
            status=RunStatus.COMPLETED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_duration_seconds=3600,
            config_used={},
            final_metrics=RunMetrics(overall_accuracy=0.85),
        )

        path = persistence_manager.save_result(result)

        assert path.exists()
        assert (persistence_manager.run_dir / "reports" / "final_report.md").exists()

    def test_load_result(self, persistence_manager):
        """Test result load."""
        result = RunResult(
            run_id="test_run",
            status=RunStatus.COMPLETED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_duration_seconds=3600,
            config_used={},
            final_metrics=RunMetrics(overall_accuracy=0.85),
        )

        persistence_manager.save_result(result)
        loaded = persistence_manager.load_result()

        assert loaded.run_id == "test_run"
        assert loaded.status == RunStatus.COMPLETED

    def test_get_disk_usage(self, persistence_manager):
        """Test disk usage calculation."""
        # Create files in subdirectories that get_disk_usage measures
        persistence_manager.save_metrics(RunMetrics(overall_accuracy=0.5))
        persistence_manager.save_rules([{"id": "test_rule"}])

        usage = persistence_manager.get_disk_usage()

        assert "total" in usage
        assert "metrics" in usage
        assert "rules" in usage
        assert usage["total"] > 0


class TestCreatePersistenceManager:
    """Tests for factory function."""

    def test_creates_manager_with_correct_path(self, tmp_path):
        """Test factory creates manager with correct path."""
        manager = create_persistence_manager(tmp_path, "my_run")

        assert manager.run_dir == tmp_path / "my_run"
        assert manager.run_dir.exists()
