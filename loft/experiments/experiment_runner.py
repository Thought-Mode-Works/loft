"""
Experiment runner for long-running autonomous learning experiments.

Issue #256: Long-Running Experiment Runner
"""

import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loft.experiments.config import ExperimentConfig
from loft.experiments.state import CumulativeMetrics, ExperimentState

logger = logging.getLogger(__name__)


class GracefulShutdown(Exception):
    """Exception raised to signal graceful shutdown."""

    pass


@dataclass
class InterimReport:
    """Interim progress report generated periodically."""

    experiment_id: str
    elapsed_time_seconds: float
    cycles_completed: int
    cumulative_metrics: CumulativeMetrics
    current_rule_count: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_markdown(self) -> str:
        """Generate markdown report."""
        metrics = self.cumulative_metrics

        report = f"""# Interim Report: {self.experiment_id}

**Generated**: {self.timestamp}
**Elapsed Time**: {self.elapsed_time_seconds / 3600:.2f} hours
**Cycles Completed**: {self.cycles_completed}

## Progress

- **Cases Processed**: {metrics.total_cases_processed}
- **Rules Incorporated**: {self.current_rule_count}
- **Accuracy**: {metrics.current_accuracy:.2%}
- **Coverage**: {metrics.current_coverage:.2%}

## Performance

- **Rules Generated**: {metrics.total_rules_generated}
- **Rules Accepted**: {metrics.total_rules_incorporated} ({metrics.total_rules_incorporated / max(metrics.total_rules_generated, 1):.1%})
- **Rules Rejected**: {metrics.total_rules_rejected}

## Predictions

- **Correct**: {metrics.total_predictions_correct}
- **Incorrect**: {metrics.total_predictions_incorrect}
- **Unknown**: {metrics.total_predictions_unknown}

## LLM Usage

- **Total Calls**: {metrics.total_llm_calls}
- **Total Cost**: ${metrics.total_llm_cost_usd:.4f}
- **Avg Processing Time**: {metrics.avg_case_processing_time_ms:.1f}ms per case

## Accuracy Trend

"""
        for i, acc in enumerate(metrics.accuracy_by_cycle, 1):
            report += f"- Cycle {i}: {acc:.2%}\n"

        return report

    def save(self, path: Path):
        """Save report to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_markdown())
        logger.info(f"Saved interim report to {path}")


@dataclass
class ExperimentReport:
    """Final experiment report."""

    experiment_id: str
    total_duration_seconds: float
    cycles_completed: int
    state: ExperimentState
    final_metrics: CumulativeMetrics
    goals_achieved: Dict[str, bool]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_markdown(self) -> str:
        """Generate markdown final report."""
        metrics = self.final_metrics

        report = f"""# Final Experiment Report: {self.experiment_id}

**Completed**: {self.timestamp}
**Total Duration**: {self.total_duration_seconds / 3600:.2f} hours
**Total Cycles**: {self.cycles_completed}

## Goals Achievement

- **Target Accuracy**: {'✓' if self.goals_achieved.get('accuracy') else '✗'}
- **Target Coverage**: {'✓' if self.goals_achieved.get('coverage') else '✗'}
- **Target Rule Count**: {'✓' if self.goals_achieved.get('rule_count') else '✗'}

## Final Metrics

- **Cases Processed**: {metrics.total_cases_processed}
- **Rules Incorporated**: {self.state.rules_incorporated}
- **Final Accuracy**: {metrics.current_accuracy:.2%}
- **Final Coverage**: {metrics.current_coverage:.2%}

## Learning Progress

### Accuracy by Cycle

"""
        for i, acc in enumerate(metrics.accuracy_by_cycle, 1):
            report += f"- Cycle {i}: {acc:.2%}\n"

        report += f"""

### Rule Generation

- **Total Generated**: {metrics.total_rules_generated}
- **Total Incorporated**: {metrics.total_rules_incorporated}
- **Acceptance Rate**: {metrics.total_rules_incorporated / max(metrics.total_rules_generated, 1):.1%}

### Resource Usage

- **Total LLM Calls**: {metrics.total_llm_calls}
- **Total Cost**: ${metrics.total_llm_cost_usd:.4f}
- **Avg Processing Time**: {metrics.avg_case_processing_time_ms:.1f}ms per case

"""
        return report

    def save(self, path: Path):
        """Save final report to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_markdown())
        logger.info(f"Saved final report to {path}")


class ExperimentRunner:
    """Orchestrates long-running learning experiments."""

    def __init__(
        self,
        config: ExperimentConfig,
        processor: Any,  # FullPipelineProcessor or MetaAwareBatchProcessor
        persistence: Any,  # ASPPersistenceManager
    ):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
            processor: Case processor (with or without meta-awareness)
            persistence: ASP persistence manager
        """
        self.config = config
        self.processor = processor
        self.persistence = persistence

        # State management
        state_file = config.state_path / f"{config.experiment_id}_state.json"
        self.state = ExperimentState.load_or_create(
            state_file, config.experiment_id, config.description
        )
        self.state_file = state_file

        self.start_time: Optional[float] = None
        self.running = False
        self.last_report_time: Optional[float] = None

        # Signal handlers
        self._original_sigint = None
        self._original_sigterm = None

    def run(self) -> ExperimentReport:
        """
        Run the experiment until completion or duration limit.

        Returns:
            Final experiment report
        """
        self.running = True
        self.start_time = time.time()
        self.last_report_time = self.start_time

        logger.info(f"Starting experiment: {self.config.experiment_id}")
        logger.info(f"Max duration: {self.config.max_duration_seconds / 3600:.1f}h")
        logger.info(f"Max cycles: {self.config.max_cycles}")

        # Register signal handlers
        self._register_signal_handlers()

        try:
            while self._should_continue():
                cycle_start = time.time()

                # Run improvement cycle
                logger.info(
                    f"Starting cycle {self.state.cycles_completed + 1}/{self.config.max_cycles}"
                )
                cycle_metrics = self._run_cycle()

                # Update state
                self.state.cycles_completed += 1
                self.state.cases_processed += cycle_metrics.get("cases_processed", 0)
                self.state.rules_generated += cycle_metrics.get("rules_generated", 0)
                self.state.rules_incorporated += cycle_metrics.get(
                    "rules_incorporated", 0
                )
                self.state.cumulative_metrics.update_from_cycle(cycle_metrics)

                # Save state
                self.state.save(self.state_file)

                # Generate periodic report
                if self._should_report():
                    self._generate_interim_report()

                # Cool-down between cycles
                cycle_duration = time.time() - cycle_start
                logger.info(f"Cycle completed in {cycle_duration:.1f}s")
                if self.config.cool_down_seconds > 0:
                    logger.info(f"Cool-down for {self.config.cool_down_seconds}s...")
                    time.sleep(self.config.cool_down_seconds)

        except GracefulShutdown:
            logger.info("Graceful shutdown initiated")
            self._save_final_state()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self._save_final_state()
        finally:
            # Restore original signal handlers
            self._restore_signal_handlers()

        return self._generate_final_report()

    def _should_continue(self) -> bool:
        """Check if experiment should continue."""
        if not self.running:
            return False

        # Check duration limit
        elapsed = time.time() - self.start_time
        if elapsed > self.config.max_duration_seconds:
            logger.info("Max duration reached")
            return False

        # Check cycle limit
        if self.state.cycles_completed >= self.config.max_cycles:
            logger.info("Max cycles reached")
            return False

        # Check if goals achieved
        if self.state.all_goals_achieved(
            self.config.target_accuracy,
            self.config.target_coverage,
            self.config.target_rule_count,
        ):
            logger.info("All goals achieved!")
            return False

        return True

    def _run_cycle(self) -> Dict[str, Any]:
        """
        Run a single improvement cycle.

        Returns:
            Cycle metrics dictionary
        """
        cycle_id = f"cycle_{self.state.cycles_completed + 1:04d}"

        # Load next batch of cases
        cases = self._load_next_batch()

        logger.info(f"Processing {len(cases)} cases in {cycle_id}")

        # Track metrics
        metrics = {
            "cases_processed": 0,
            "rules_generated": 0,
            "rules_incorporated": 0,
            "rules_rejected": 0,
            "predictions_correct": 0,
            "predictions_incorrect": 0,
            "predictions_unknown": 0,
            "llm_calls": 0,
            "llm_cost_usd": 0.0,
            "processing_times": [],
        }

        # Process each case
        for case in cases:
            result = self.processor.process_case(case, accumulated_rules=[])

            metrics["cases_processed"] += 1
            if hasattr(result, "processing_time_ms"):
                metrics["processing_times"].append(result.processing_time_ms)

            # Track predictions
            if hasattr(result, "prediction_correct"):
                if result.prediction_correct:
                    metrics["predictions_correct"] += 1
                else:
                    metrics["predictions_incorrect"] += 1
            else:
                metrics["predictions_unknown"] += 1

            # Track rules
            if hasattr(result, "rules_generated"):
                metrics["rules_generated"] += result.rules_generated
            if hasattr(result, "rules_accepted"):
                metrics["rules_incorporated"] += result.rules_accepted
            if hasattr(result, "rules_rejected"):
                metrics["rules_rejected"] += result.rules_rejected

        # Compute averages
        if metrics["processing_times"]:
            metrics["avg_processing_time_ms"] = sum(metrics["processing_times"]) / len(
                metrics["processing_times"]
            )
        else:
            metrics["avg_processing_time_ms"] = 0.0

        # Compute accuracy/coverage
        total_predictions = (
            metrics["predictions_correct"] + metrics["predictions_incorrect"]
        )
        if total_predictions > 0:
            metrics["accuracy"] = metrics["predictions_correct"] / total_predictions
        else:
            metrics["accuracy"] = 0.0

        total_with_unknown = total_predictions + metrics["predictions_unknown"]
        if total_with_unknown > 0:
            metrics["coverage"] = total_predictions / total_with_unknown
        else:
            metrics["coverage"] = 0.0

        # Persist rules after cycle
        try:
            # Save all rules
            if hasattr(self.persistence, "save_all_rules"):
                self.persistence.save_all_rules(overwrite=True)
            # Create snapshot
            if hasattr(self.persistence, "create_snapshot"):
                self.persistence.create_snapshot(cycle_id)
        except Exception as e:
            logger.warning(f"Failed to persist rules: {e}")

        return metrics

    def _load_next_batch(self) -> List[Dict[str, Any]]:
        """Load next batch of cases from dataset."""
        # Simple implementation: load from JSON files in dataset directory
        dataset_files = sorted(self.config.dataset_path.glob("*.json"))

        # Calculate which files to load
        start_idx = self.state.dataset_cursor
        end_idx = start_idx + self.config.cases_per_cycle

        batch_files = dataset_files[start_idx:end_idx]

        # Update cursor to actual position (in case we loaded fewer files than requested)
        self.state.dataset_cursor = start_idx + len(batch_files)

        # Load cases
        cases = []
        for file in batch_files:
            try:
                with open(file, "r") as f:
                    case = json.load(f)
                    cases.append(case)
            except Exception as e:
                logger.warning(f"Failed to load case from {file}: {e}")

        logger.info(
            f"Loaded {len(cases)} cases (cursor: {start_idx} -> {self.state.dataset_cursor})"
        )
        return cases

    def _should_report(self) -> bool:
        """Check if should generate interim report."""
        if self.last_report_time is None:
            return False

        elapsed_since_report = time.time() - self.last_report_time
        return elapsed_since_report >= self.config.report_interval_seconds

    def _generate_interim_report(self):
        """Generate and save interim report."""
        elapsed = time.time() - self.start_time

        # Get current rule count
        rule_count = 0
        try:
            if hasattr(self.persistence, "get_stats"):
                stats = self.persistence.get_stats()
                rule_count = stats.get("total_rules", 0)
        except Exception:
            rule_count = self.state.rules_incorporated

        report = InterimReport(
            experiment_id=self.config.experiment_id,
            elapsed_time_seconds=elapsed,
            cycles_completed=self.state.cycles_completed,
            cumulative_metrics=self.state.cumulative_metrics,
            current_rule_count=rule_count,
        )

        # Save to disk
        report_path = (
            self.config.reports_path
            / f"{self.config.experiment_id}_interim_{self.state.cycles_completed}.md"
        )
        report.save(report_path)

        # Update last report time
        self.last_report_time = time.time()

        logger.info(
            f"Generated interim report (cycles: {self.state.cycles_completed}, "
            f"accuracy: {self.state.cumulative_metrics.current_accuracy:.2%})"
        )

    def _save_final_state(self):
        """Save final state before shutdown."""
        self.state.save(self.state_file)
        logger.info("Saved final state")

    def _generate_final_report(self) -> ExperimentReport:
        """Generate final experiment report."""
        total_duration = time.time() - self.start_time

        report = ExperimentReport(
            experiment_id=self.config.experiment_id,
            total_duration_seconds=total_duration,
            cycles_completed=self.state.cycles_completed,
            state=self.state,
            final_metrics=self.state.cumulative_metrics,
            goals_achieved=self.state.goals_achieved,
        )

        # Save final report
        final_report_path = (
            self.config.reports_path / f"{self.config.experiment_id}_final.md"
        )
        report.save(final_report_path)

        return report

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_interrupt)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        raise GracefulShutdown()
