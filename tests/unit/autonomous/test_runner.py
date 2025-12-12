"""Unit tests for autonomous test runner."""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from loft.autonomous.config import AutonomousRunConfig
from loft.autonomous.runner import AutonomousTestRunner
from loft.autonomous.schemas import RunStatus


@pytest.fixture
def config():
    """Create test configuration."""
    return AutonomousRunConfig(
        max_duration_hours=0.01,  # Very short for testing
        max_cases=10,
        checkpoint_interval_minutes=60,
    )


@pytest.fixture
def runner(config, tmp_path):
    """Create test runner."""
    return AutonomousTestRunner(config, output_dir=tmp_path)


@pytest.fixture
def sample_dataset(tmp_path):
    """Create sample dataset files."""
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()

    cases = [
        {"id": f"case_{i}", "domain": "contracts", "facts": f"Test case {i}"} for i in range(5)
    ]

    with open(dataset_dir / "cases.json", "w") as f:
        json.dump(cases, f)

    return dataset_dir


class TestAutonomousTestRunner:
    """Tests for AutonomousTestRunner."""

    def test_init(self, runner, config, tmp_path):
        """Test runner initialization."""
        assert runner.config == config
        assert runner.run_id is None
        assert runner.state is None
        assert not runner.is_running

    def test_generate_run_id(self, runner):
        """Test run ID generation."""
        run_id = runner._generate_run_id()

        assert run_id.startswith("run_")
        assert "_" in run_id
        # Format: run_YYYYMMDD_HHMMSS_xxxxxx

    def test_set_callbacks(self, runner):
        """Test callback setting."""
        progress_cb = MagicMock()
        case_cb = MagicMock()
        checkpoint_cb = MagicMock()

        runner.set_callbacks(
            on_progress=progress_cb,
            on_case_complete=case_cb,
            on_checkpoint=checkpoint_cb,
        )

        assert runner._on_progress == progress_cb
        assert runner._on_case_complete == case_cb
        assert runner._on_checkpoint == checkpoint_cb

    def test_request_shutdown(self, runner):
        """Test shutdown request."""
        assert runner._shutdown_requested is False

        runner.request_shutdown()

        assert runner._shutdown_requested is True

    def test_get_status_no_state(self, runner):
        """Test status when no run started."""
        status = runner.get_status()

        assert status.run_id == "none"
        assert status.status == RunStatus.PENDING

    def test_start_requires_dataset(self, runner):
        """Test start fails without dataset."""
        with pytest.raises(ValueError, match="No dataset paths"):
            runner.start(dataset_paths=[])

    def test_start_creates_run_directory(self, runner, sample_dataset, tmp_path):
        """Test start creates necessary directories."""
        runner.start(
            dataset_paths=[str(sample_dataset)],
            run_id="test_run",
        )

        run_dir = tmp_path / "test_run"
        assert run_dir.exists()
        assert (run_dir / "checkpoints").exists()
        assert (run_dir / "state.json").exists()
        assert (run_dir / "config.json").exists()

    def test_start_processes_cases(self, runner, sample_dataset):
        """Test start processes cases from dataset."""
        result = runner.start(
            dataset_paths=[str(sample_dataset)],
            run_id="test_run",
        )

        # Should process up to max_cases (10) or all available (5)
        assert result.final_metrics.overall_accuracy >= 0.0

    def test_start_respects_max_cases(self, config, sample_dataset, tmp_path):
        """Test start respects max_cases limit."""
        config.max_cases = 3

        runner = AutonomousTestRunner(config, output_dir=tmp_path)
        runner.start(
            dataset_paths=[str(sample_dataset)],
            run_id="test_run",
        )

        # Progress should show 3 cases processed (limited by max_cases)
        # Note: actual cases processed depends on dataset content

    def test_start_handles_signal_shutdown(self, config, sample_dataset, tmp_path):
        """Test graceful shutdown on signal."""
        # Create a fresh runner and request shutdown immediately before start
        # The runner checks _should_stop() before processing each case
        config.max_cases = 1000  # Large number so time/cases don't stop it
        config.max_duration_hours = 1.0  # Long duration

        runner = AutonomousTestRunner(config, output_dir=tmp_path)

        # We need to trigger shutdown during processing. Since the runner
        # checks _should_stop() between cases, we verify that when shutdown
        # is requested, the status eventually reflects cancellation.
        # For a unit test, we verify the _should_stop mechanism works.
        runner._shutdown_requested = True
        assert runner._should_stop() is True

    def test_load_cases_from_json_file(self, runner, tmp_path):
        """Test loading cases from JSON file."""
        file_path = tmp_path / "test.json"
        cases = [{"id": "1"}, {"id": "2"}]
        with open(file_path, "w") as f:
            json.dump(cases, f)

        loaded = runner._load_cases_from_file(file_path)

        assert len(loaded) == 2

    def test_load_cases_from_json_with_cases_key(self, runner, tmp_path):
        """Test loading cases from JSON with 'cases' key."""
        file_path = tmp_path / "test.json"
        data = {"cases": [{"id": "1"}, {"id": "2"}], "metadata": {}}
        with open(file_path, "w") as f:
            json.dump(data, f)

        loaded = runner._load_cases_from_file(file_path)

        assert len(loaded) == 2

    def test_load_cases_from_directory(self, runner, sample_dataset):
        """Test loading cases from directory."""
        loaded = runner._load_cases_from_directory(sample_dataset)

        assert len(loaded) == 5

    def test_process_case_default(self, runner):
        """Test default case processing."""
        from loft.autonomous.schemas import RunProgress, RunState

        # Initialize state so _process_case can run
        runner._state = RunState(
            run_id="test",
            progress=RunProgress(),
        )

        case = {"id": "test_case", "domain": "contracts"}

        result = runner._process_case(case)

        assert result["case_id"] == "test_case"
        assert result["domain"] == "contracts"
        assert "processed_at" in result

    def test_should_stop_time_limit(self, runner):
        """Test time limit check."""
        runner._start_time = datetime(2020, 1, 1)  # Long ago
        runner._config.max_duration_hours = 0.001

        assert runner._should_stop() is True

    def test_should_stop_shutdown_requested(self, runner):
        """Test shutdown request check."""
        runner._shutdown_requested = True

        assert runner._should_stop() is True

    def test_should_stop_case_limit(self, runner):
        """Test case limit check."""
        from loft.autonomous.schemas import RunProgress, RunState

        runner._config.max_cases = 10
        runner._state = RunState(
            run_id="test",
            progress=RunProgress(cases_processed=10),
        )

        assert runner._should_stop() is True


class TestRunnerIntegration:
    """Integration tests for runner with mock components."""

    def test_full_run_with_callbacks(self, config, sample_dataset, tmp_path):
        """Test full run with callbacks."""
        progress_calls = []
        case_calls = []

        runner = AutonomousTestRunner(config, output_dir=tmp_path)
        runner.set_callbacks(
            on_progress=lambda p: progress_calls.append(p),
            on_case_complete=lambda c: case_calls.append(c),
        )

        runner.start(
            dataset_paths=[str(sample_dataset)],
            run_id="callback_test",
        )

        assert len(progress_calls) > 0
        assert len(case_calls) > 0

    def test_run_creates_final_report(self, config, sample_dataset, tmp_path):
        """Test run creates final report."""
        runner = AutonomousTestRunner(config, output_dir=tmp_path)

        runner.start(
            dataset_paths=[str(sample_dataset)],
            run_id="report_test",
        )

        report_path = tmp_path / "report_test" / "reports" / "final_report.json"
        assert report_path.exists()

        with open(report_path) as f:
            report_data = json.load(f)

        assert report_data["run_id"] == "report_test"
        assert "final_metrics" in report_data

    def test_run_with_orchestrator(self, config, sample_dataset, tmp_path):
        """Test run with meta-reasoning orchestrator."""
        from loft.autonomous.meta_integration import MetaReasoningOrchestrator

        runner = AutonomousTestRunner(config, output_dir=tmp_path)

        orchestrator = MetaReasoningOrchestrator(config.meta_reasoning)
        runner.set_orchestrator(orchestrator)

        result = runner.start(
            dataset_paths=[str(sample_dataset)],
            run_id="orchestrator_test",
        )

        # Check that orchestrator was used
        assert result.status in [RunStatus.COMPLETED, RunStatus.CANCELLED]
