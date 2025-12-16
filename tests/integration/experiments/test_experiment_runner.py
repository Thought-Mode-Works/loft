"""
Integration tests for experiment runner.

Issue #256: Long-Running Experiment Runner
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from loft.experiments import ExperimentConfig, ExperimentRunner


class TestExperimentRunnerIntegration:
    """Integration tests for ExperimentRunner."""

    def test_two_cycle_experiment(self):
        """Test running a 2-cycle experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            dataset_path = tmppath / "dataset"
            dataset_path.mkdir()

            # Create test cases
            for i in range(10):
                case_file = dataset_path / f"case_{i:03d}.json"
                case_file.write_text(
                    json.dumps(
                        {
                            "id": f"case_{i}",
                            "facts": [f"fact_{i}(yes)."],
                            "expected": "yes",
                        }
                    )
                )

            # Create config
            config = ExperimentConfig(
                experiment_id="integration_test",
                description="Integration test",
                dataset_path=dataset_path,
                state_path=tmppath / "state",
                reports_path=tmppath / "reports",
                rules_path=tmppath / "rules",
                max_cycles=2,
                cases_per_cycle=5,
                cool_down_seconds=0,  # No cool-down for testing
            )

            # Create mocks
            mock_processor = MagicMock()
            mock_persistence = MagicMock()

            # Mock processor to return success
            mock_result = MagicMock()
            mock_result.processing_time_ms = 100.0
            mock_result.prediction_correct = True
            mock_result.rules_generated = 1
            mock_result.rules_accepted = 1
            mock_result.rules_rejected = 0
            mock_processor.process_case.return_value = mock_result

            # Create runner
            runner = ExperimentRunner(
                config=config,
                processor=mock_processor,
                persistence=mock_persistence,
            )

            # Run experiment
            report = runner.run()

            # Verify results
            assert report.cycles_completed == 2
            assert runner.state.cases_processed == 10
            assert runner.state.cycles_completed == 2

            # Verify state was saved
            state_file = config.state_path / f"{config.experiment_id}_state.json"
            assert state_file.exists()

            # Verify final report was generated
            final_report = config.reports_path / f"{config.experiment_id}_final.md"
            assert final_report.exists()

    def test_resume_experiment(self):
        """Test resuming an interrupted experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            dataset_path = tmppath / "dataset"
            dataset_path.mkdir()

            # Create test cases
            for i in range(15):
                case_file = dataset_path / f"case_{i:03d}.json"
                case_file.write_text(
                    json.dumps({"id": f"case_{i}", "facts": [], "expected": "yes"})
                )

            # Create config
            config = ExperimentConfig(
                experiment_id="resume_test",
                description="Resume test",
                dataset_path=dataset_path,
                state_path=tmppath / "state",
                reports_path=tmppath / "reports",
                rules_path=tmppath / "rules",
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
            mock_processor1.process_case.return_value = mock_result

            runner1 = ExperimentRunner(
                config=config,
                processor=mock_processor1,
                persistence=mock_persistence1,
            )

            # Override max_cycles for first run
            runner1.config.max_cycles = 1
            report1 = runner1.run()

            assert report1.cycles_completed == 1

            # Second run: resume and complete remaining cycles
            mock_processor2 = MagicMock()
            mock_persistence2 = MagicMock()
            mock_processor2.process_case.return_value = mock_result

            # Create new runner (will load existing state)
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
