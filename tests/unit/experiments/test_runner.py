"""
Unit tests for experiment runner.

Issue #256: Long-Running Experiment Runner
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from loft.experiments import (
    CumulativeMetrics,
    ExperimentConfig,
    ExperimentRunner,
    ExperimentState,
    parse_duration,
)


class TestParseDuration:
    """Tests for duration parsing."""

    def test_parse_minutes(self):
        """Test parsing minutes."""
        assert parse_duration("30m") == 1800
        assert parse_duration("5m") == 300

    def test_parse_hours(self):
        """Test parsing hours."""
        assert parse_duration("2h") == 7200
        assert parse_duration("4h") == 14400

    def test_parse_seconds(self):
        """Test parsing seconds."""
        assert parse_duration("90s") == 90

    def test_parse_no_unit(self):
        """Test parsing number without unit."""
        assert parse_duration("100") == 100


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExperimentConfig(
            experiment_id="test_001", description="Test experiment"
        )

        assert config.experiment_id == "test_001"
        assert config.max_duration_seconds == 4 * 60 * 60  # 4 hours
        assert config.max_cycles == 100
        assert config.cases_per_cycle == 20

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = ExperimentConfig(
            experiment_id="test_001",
            description="Test",
            dataset_path="datasets/test/",
            state_path="data/test/",
        )

        assert isinstance(config.dataset_path, Path)
        assert isinstance(config.state_path, Path)


class TestCumulativeMetrics:
    """Tests for CumulativeMetrics."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = CumulativeMetrics()

        assert metrics.total_cases_processed == 0
        assert metrics.total_rules_generated == 0
        assert metrics.accuracy_by_cycle == []

    def test_update_from_cycle(self):
        """Test updating metrics from cycle."""
        metrics = CumulativeMetrics()

        cycle_metrics = {
            "cases_processed": 10,
            "rules_generated": 5,
            "rules_incorporated": 3,
            "rules_rejected": 2,
            "predictions_correct": 7,
            "predictions_incorrect": 2,
            "predictions_unknown": 1,
            "accuracy": 0.78,
            "coverage": 0.90,
            "llm_calls": 15,
            "llm_cost_usd": 0.05,
        }

        metrics.update_from_cycle(cycle_metrics)

        assert metrics.total_cases_processed == 10
        assert metrics.total_rules_generated == 5
        assert metrics.total_rules_incorporated == 3
        assert metrics.total_predictions_correct == 7
        assert len(metrics.accuracy_by_cycle) == 1
        assert metrics.accuracy_by_cycle[0] == 0.78

    def test_current_accuracy(self):
        """Test current accuracy calculation."""
        metrics = CumulativeMetrics()
        metrics.total_predictions_correct = 8
        metrics.total_predictions_incorrect = 2

        assert metrics.current_accuracy == 0.8

    def test_current_coverage(self):
        """Test current coverage calculation."""
        metrics = CumulativeMetrics()
        metrics.total_predictions_correct = 7
        metrics.total_predictions_incorrect = 2
        metrics.total_predictions_unknown = 1

        # Coverage = (correct + incorrect) / total
        # = 9 / 10 = 0.9
        assert metrics.current_coverage == 0.9

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        metrics = CumulativeMetrics()
        metrics.total_cases_processed = 10
        metrics.accuracy_by_cycle = [0.5, 0.7, 0.8]

        data = metrics.to_dict()
        restored = CumulativeMetrics.from_dict(data)

        assert restored.total_cases_processed == 10
        assert restored.accuracy_by_cycle == [0.5, 0.7, 0.8]


class TestExperimentState:
    """Tests for ExperimentState."""

    def test_initialization(self):
        """Test state initialization."""
        state = ExperimentState(
            experiment_id="test_001",
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
        )

        assert state.experiment_id == "test_001"
        assert state.cycles_completed == 0
        assert state.cases_processed == 0

    def test_save_and_load(self):
        """Test state persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "test_state.json"

            # Create and save state
            state = ExperimentState(
                experiment_id="test_001",
                started_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                cycles_completed=5,
                cases_processed=100,
            )
            state.save(state_file)

            # Load state
            loaded = ExperimentState.load(state_file)

            assert loaded.experiment_id == "test_001"
            assert loaded.cycles_completed == 5
            assert loaded.cases_processed == 100

    def test_load_or_create_existing(self):
        """Test load_or_create with existing state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "test_state.json"

            # Create state
            state = ExperimentState(
                experiment_id="test_001",
                started_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                cycles_completed=3,
            )
            state.save(state_file)

            # Load or create should load
            loaded = ExperimentState.load_or_create(state_file, "test_001", "Test")

            assert loaded.cycles_completed == 3

    def test_load_or_create_new(self):
        """Test load_or_create with new state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "new_state.json"

            # Load or create should create
            state = ExperimentState.load_or_create(state_file, "test_002", "New test")

            assert state.experiment_id == "test_002"
            assert state.cycles_completed == 0

    def test_all_goals_achieved(self):
        """Test goal achievement checking."""
        state = ExperimentState(
            experiment_id="test_001",
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
        )

        # Set metrics to meet goals
        state.cumulative_metrics.total_predictions_correct = 85
        state.cumulative_metrics.total_predictions_incorrect = 15
        state.rules_incorporated = 100

        # Check goals
        achieved = state.all_goals_achieved(
            target_accuracy=0.85,  # 85/100 = 0.85
            target_coverage=0.80,
            target_rule_count=100,
        )

        assert achieved is True
        assert state.goals_achieved["accuracy"] is True
        assert state.goals_achieved["rule_count"] is True


class TestExperimentRunner:
    """Tests for ExperimentRunner."""

    def test_initialization(self):
        """Test runner initialization."""
        config = ExperimentConfig(experiment_id="test_001", description="Test")

        mock_processor = MagicMock()
        mock_persistence = MagicMock()

        runner = ExperimentRunner(
            config=config,
            processor=mock_processor,
            persistence=mock_persistence,
        )

        assert runner.config == config
        assert runner.processor == mock_processor
        assert runner.persistence == mock_persistence
        assert runner.running is False

    def test_should_continue_duration_limit(self):
        """Test continuation check with duration limit."""
        config = ExperimentConfig(
            experiment_id="test_001",
            description="Test",
            max_duration_seconds=10,  # 10 seconds
        )

        mock_processor = MagicMock()
        mock_persistence = MagicMock()

        runner = ExperimentRunner(
            config=config,
            processor=mock_processor,
            persistence=mock_persistence,
        )

        runner.running = True
        runner.start_time = 0.0

        # Mock time to exceed duration
        with patch("time.time", return_value=15.0):
            assert runner._should_continue() is False

    def test_should_continue_cycle_limit(self):
        """Test continuation check with cycle limit."""
        config = ExperimentConfig(
            experiment_id="test_001",
            description="Test",
            max_cycles=5,
        )

        mock_processor = MagicMock()
        mock_persistence = MagicMock()

        runner = ExperimentRunner(
            config=config,
            processor=mock_processor,
            persistence=mock_persistence,
        )

        runner.running = True
        runner.start_time = 0.0
        runner.state.cycles_completed = 5

        with patch("time.time", return_value=1.0):
            assert runner._should_continue() is False

    def test_load_next_batch(self):
        """Test loading next batch of cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)

            # Create mock case files
            for i in range(5):
                case_file = dataset_path / f"case_{i:03d}.json"
                case_file.write_text(json.dumps({"id": f"case_{i}", "facts": []}))

            config = ExperimentConfig(
                experiment_id="test_001",
                description="Test",
                dataset_path=dataset_path,
                cases_per_cycle=3,
            )

            mock_processor = MagicMock()
            mock_persistence = MagicMock()

            runner = ExperimentRunner(
                config=config,
                processor=mock_processor,
                persistence=mock_persistence,
            )

            # Load first batch
            batch = runner._load_next_batch()
            assert len(batch) == 3
            assert runner.state.dataset_cursor == 3

            # Load second batch
            batch = runner._load_next_batch()
            assert len(batch) == 2  # Only 2 remaining
            assert runner.state.dataset_cursor == 5


class TestInterimReport:
    """Tests for interim reports."""

    def test_to_markdown(self):
        """Test markdown report generation."""
        from loft.experiments.experiment_runner import InterimReport

        metrics = CumulativeMetrics()
        metrics.total_cases_processed = 50
        metrics.total_rules_incorporated = 10
        # Set underlying values to get 75% accuracy
        metrics.total_predictions_correct = 75
        metrics.total_predictions_incorrect = 25
        metrics.accuracy_by_cycle = [0.6, 0.7, 0.75]

        report = InterimReport(
            experiment_id="test_001",
            elapsed_time_seconds=3600,  # 1 hour
            cycles_completed=3,
            cumulative_metrics=metrics,
            current_rule_count=10,
        )

        markdown = report.to_markdown()

        assert "test_001" in markdown
        assert "50" in markdown  # cases processed
        assert "Cycle 1: 60.00%" in markdown
        assert "Cycle 3: 75.00%" in markdown
