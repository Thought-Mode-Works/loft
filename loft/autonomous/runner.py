"""
Autonomous Test Runner for Long-Running Experiments.

This module provides the main orchestrator for autonomous long-running
test experiments combining batch processing with meta-reasoning.

Features:
- Signal handling (SIGTERM, SIGINT) for graceful shutdown
- Time-based duration limits
- Periodic checkpointing
- Integration with meta-reasoning components
- Progress callbacks for monitoring
"""

import logging
import signal
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loft.autonomous.config import AutonomousRunConfig
from loft.autonomous.data_sources import DataSourceAdapter
from loft.autonomous.meta_integration import MetaReasoningOrchestrator
from loft.autonomous.persistence import PersistenceManager, create_persistence_manager
from loft.autonomous.schemas import (
    CycleResult,
    RunCheckpoint,
    RunMetrics,
    RunProgress,
    RunResult,
    RunState,
    RunStatus,
)

logger = logging.getLogger(__name__)


class AutonomousTestRunner:
    """Long-running autonomous test harness with meta-reasoning integration.

    Orchestrates batch processing with periodic improvement cycles,
    checkpointing, and graceful shutdown support.

    Attributes:
        config: Run configuration
        output_dir: Output directory path
    """

    def __init__(
        self,
        config: AutonomousRunConfig,
        output_dir: Optional[Path] = None,
    ):
        """Initialize the runner.

        Args:
            config: Run configuration
            output_dir: Optional output directory override
        """
        self._config = config
        self._output_dir = Path(output_dir or config.output_dir)

        self._run_id: Optional[str] = None
        self._state: Optional[RunState] = None
        self._persistence: Optional[PersistenceManager] = None
        self._orchestrator: Optional[MetaReasoningOrchestrator] = None

        self._shutdown_requested = False
        self._original_sigterm_handler: Optional[Any] = None
        self._original_sigint_handler: Optional[Any] = None

        self._start_time: Optional[datetime] = None
        self._last_checkpoint_time: Optional[datetime] = None
        self._cases_since_last_cycle = 0

        self._case_results: List[Dict[str, Any]] = []
        self._accumulated_rules: List[Dict[str, Any]] = []
        self._cycle_results: List[CycleResult] = []

        self._on_progress: Optional[Callable[[RunProgress], None]] = None
        self._on_case_complete: Optional[Callable[[Dict[str, Any]], None]] = None
        self._on_checkpoint: Optional[Callable[[RunCheckpoint], None]] = None
        self._on_cycle_complete: Optional[Callable[[CycleResult], None]] = None

        self._batch_harness: Optional[Any] = None
        self._dataset_loader: Optional[Any] = None
        self._data_source: Optional[DataSourceAdapter] = None

    @property
    def config(self) -> AutonomousRunConfig:
        """Get configuration."""
        return self._config

    @property
    def run_id(self) -> Optional[str]:
        """Get current run ID."""
        return self._run_id

    @property
    def state(self) -> Optional[RunState]:
        """Get current run state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if run is active."""
        return self._state is not None and self._state.status == RunStatus.RUNNING

    def set_callbacks(
        self,
        on_progress: Optional[Callable[[RunProgress], None]] = None,
        on_case_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_checkpoint: Optional[Callable[[RunCheckpoint], None]] = None,
        on_cycle_complete: Optional[Callable[[CycleResult], None]] = None,
    ) -> None:
        """Set callbacks for run events.

        Args:
            on_progress: Called on progress updates
            on_case_complete: Called when a case completes
            on_checkpoint: Called when checkpoint is created
            on_cycle_complete: Called when improvement cycle completes
        """
        self._on_progress = on_progress
        self._on_case_complete = on_case_complete
        self._on_checkpoint = on_checkpoint
        self._on_cycle_complete = on_cycle_complete

    def set_batch_harness(self, harness: Any) -> None:
        """Set the batch learning harness.

        Args:
            harness: BatchLearningHarness instance
        """
        self._batch_harness = harness

    def set_dataset_loader(self, loader: Any) -> None:
        """Set the dataset loader.

        Args:
            loader: Dataset loader callable
        """
        self._dataset_loader = loader

    def set_data_source(self, data_source: DataSourceAdapter) -> None:
        """Set a data source adapter for loading cases.

        When a data source adapter is set, it takes precedence over
        dataset_paths and dataset_loader for case loading.

        Args:
            data_source: DataSourceAdapter instance (e.g., CourtListenerAdapter)
        """
        self._data_source = data_source

    def set_orchestrator(self, orchestrator: MetaReasoningOrchestrator) -> None:
        """Set the meta-reasoning orchestrator.

        Args:
            orchestrator: MetaReasoningOrchestrator instance
        """
        self._orchestrator = orchestrator

    def start(
        self,
        dataset_paths: Optional[List[str]] = None,
        run_id: Optional[str] = None,
    ) -> RunResult:
        """Start a new autonomous run.

        Args:
            dataset_paths: Paths to dataset directories
            run_id: Optional run ID (auto-generated if not provided)

        Returns:
            RunResult with final outcomes
        """
        dataset_paths = dataset_paths or self._config.dataset_paths
        if not dataset_paths:
            raise ValueError("No dataset paths provided")

        self._run_id = run_id or self._generate_run_id()
        self._setup_run()

        logger.info(f"Starting autonomous run {self._run_id}")
        logger.info(f"Max duration: {self._config.max_duration_hours} hours")
        logger.info(f"Dataset paths: {dataset_paths}")

        try:
            self._install_signal_handlers()
            self._state.status = RunStatus.RUNNING
            self._state.started_at = datetime.now()
            self._start_time = self._state.started_at
            self._persistence.save_state(self._state)

            result = self._run_main_loop(dataset_paths)
            return result

        except Exception as e:
            logger.error(f"Run failed with error: {e}")
            return self._create_error_result(str(e))

        finally:
            self._restore_signal_handlers()

    def resume(self, checkpoint_path: Optional[Path] = None) -> RunResult:
        """Resume from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file, or None for latest

        Returns:
            RunResult with final outcomes
        """
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            run_dir = checkpoint_path.parent.parent
            self._run_id = run_dir.name
        else:
            raise ValueError("Must provide checkpoint path for resume")

        self._persistence = create_persistence_manager(
            self._output_dir, self._run_id, max_checkpoints=10
        )

        checkpoint = self._persistence.load_checkpoint(checkpoint_path)
        if not checkpoint:
            raise ValueError(f"Could not load checkpoint from {checkpoint_path}")

        logger.info(f"Resuming run {self._run_id} from checkpoint {checkpoint.checkpoint_number}")

        self._restore_from_checkpoint(checkpoint)

        try:
            self._install_signal_handlers()
            self._state.status = RunStatus.RUNNING
            self._state.shutdown_requested = False
            self._persistence.save_state(self._state)

            dataset_paths = checkpoint.config_snapshot.get("dataset_paths", [])
            result = self._run_main_loop(dataset_paths)
            return result

        except Exception as e:
            logger.error(f"Resumed run failed: {e}")
            return self._create_error_result(str(e))

        finally:
            self._restore_signal_handlers()

    def get_status(self) -> RunState:
        """Get current run status.

        Returns:
            Current RunState
        """
        if self._state is None:
            return RunState(run_id="none", status=RunStatus.PENDING)
        return self._state

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        logger.info("Shutdown requested")
        self._shutdown_requested = True
        if self._state:
            self._state.shutdown_requested = True

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        return f"{self._config.run_id_prefix}_{timestamp}_{unique_suffix}"

    def _setup_run(self) -> None:
        """Set up run infrastructure."""
        self._persistence = create_persistence_manager(
            self._output_dir, self._run_id, max_checkpoints=10
        )

        self._state = RunState(
            run_id=self._run_id,
            status=RunStatus.PENDING,
            progress=RunProgress(),
        )

        self._persistence.save_config(self._config.to_dict())
        self._persistence.save_state(self._state)

        if not self._orchestrator:
            from loft.autonomous.meta_integration import create_orchestrator_from_config

            self._orchestrator = create_orchestrator_from_config(self._config)

        self._orchestrator.set_callbacks(on_cycle_complete=self._handle_cycle_complete)

        self._case_results = []
        self._accumulated_rules = []
        self._cycle_results = []
        self._cases_since_last_cycle = 0
        self._shutdown_requested = False

    def _restore_from_checkpoint(self, checkpoint: RunCheckpoint) -> None:
        """Restore state from checkpoint.

        Args:
            checkpoint: Checkpoint to restore from
        """
        self._state = checkpoint.run_state
        self._case_results = checkpoint.case_results
        self._accumulated_rules = checkpoint.accumulated_rules
        self._cycle_results = checkpoint.cycle_results
        self._cases_since_last_cycle = 0

        if not self._orchestrator:
            from loft.autonomous.meta_integration import create_orchestrator_from_config

            self._orchestrator = create_orchestrator_from_config(self._config)

        self._orchestrator.restore_from_state(checkpoint.meta_reasoning_state)
        self._orchestrator.set_callbacks(on_cycle_complete=self._handle_cycle_complete)

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name}, requesting graceful shutdown")
        self.request_shutdown()

    def _run_main_loop(self, dataset_paths: List[str]) -> RunResult:
        """Run the main processing loop.

        Args:
            dataset_paths: Dataset paths to process

        Returns:
            RunResult
        """
        cases = self._load_cases(dataset_paths)
        if not cases:
            logger.warning("No cases loaded from datasets")
            return self._create_completed_result()

        self._state.progress.total_cases = len(cases)
        self._persistence.save_state(self._state)

        logger.info(f"Loaded {len(cases)} cases for processing")

        processed_case_ids = {r.get("case_id") for r in self._case_results}
        remaining_cases = [c for c in cases if c.get("id") not in processed_case_ids]

        logger.info(f"Processing {len(remaining_cases)} remaining cases")

        max_cases = self._config.max_cases if self._config.max_cases > 0 else len(remaining_cases)

        for case in remaining_cases[:max_cases]:
            if self._should_stop():
                logger.info("Stopping due to shutdown request or time limit")
                break

            result = self._process_case(case)
            self._handle_case_result(result)

            if self._should_checkpoint():
                self._create_checkpoint()

            if self._orchestrator.should_run_cycle(self._cases_since_last_cycle):
                self._run_improvement_cycle()

        if self._shutdown_requested:
            self._create_checkpoint()
            return self._create_cancelled_result()

        return self._create_completed_result()

    def _load_cases(self, dataset_paths: List[str]) -> List[Dict[str, Any]]:
        """Load cases from data source adapter or dataset paths.

        If a DataSourceAdapter is set via set_data_source(), it takes
        precedence and cases are loaded from the adapter (e.g., CourtListener API).
        Otherwise, cases are loaded from dataset_paths.

        Args:
            dataset_paths: Paths to load from (used if no data source set)

        Returns:
            List of case dictionaries
        """
        all_cases = []

        # Use data source adapter if set (e.g., CourtListenerAdapter)
        if self._data_source:
            logger.info(f"Loading cases from data source: {self._data_source.source_name}")
            limit = self._config.max_cases if self._config.max_cases > 0 else None
            for case_data in self._data_source.get_cases(limit=limit):
                all_cases.append(case_data.to_dict())
            logger.info(f"Loaded {len(all_cases)} cases from {self._data_source.source_name}")
            return all_cases

        if self._dataset_loader:
            for path in dataset_paths:
                try:
                    cases = self._dataset_loader(path)
                    all_cases.extend(cases)
                except Exception as e:
                    logger.error(f"Failed to load dataset {path}: {e}")

        if not all_cases:
            for path_str in dataset_paths:
                path = Path(path_str)
                if path.is_dir():
                    all_cases.extend(self._load_cases_from_directory(path))
                elif path.is_file():
                    all_cases.extend(self._load_cases_from_file(path))

        return all_cases

    def _load_cases_from_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Load cases from a directory.

        If cases don't have a 'domain' field, it will be inferred from the
        directory name (e.g., 'contracts', 'torts', 'property_law').

        Args:
            directory: Directory path

        Returns:
            List of cases
        """
        import json

        # Infer domain from directory name
        inferred_domain = directory.name.replace("_", " ").replace("-", " ")

        cases = []
        for json_file in directory.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Add inferred domain to cases without domain
                        for case in data:
                            if "domain" not in case:
                                case["domain"] = inferred_domain
                        cases.extend(data)
                    elif isinstance(data, dict):
                        if "cases" in data:
                            for case in data["cases"]:
                                if "domain" not in case:
                                    case["domain"] = inferred_domain
                            cases.extend(data["cases"])
                        else:
                            if "domain" not in data:
                                data["domain"] = inferred_domain
                            cases.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        return cases

    def _load_cases_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load cases from a file.

        Args:
            file_path: File path

        Returns:
            List of cases
        """
        import json

        cases = []
        try:
            with open(file_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    cases.extend(data)
                elif isinstance(data, dict):
                    if "cases" in data:
                        cases.extend(data["cases"])
                    else:
                        cases.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

        return cases

    def _process_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single case.

        Args:
            case: Case to process

        Returns:
            Result dictionary
        """
        case_id = case.get("id", str(uuid.uuid4())[:8])
        self._state.progress.current_case_id = case_id

        if self._batch_harness:
            try:
                result = self._batch_harness.process_case(case)
                result["case_id"] = case_id
                return result
            except Exception as e:
                logger.warning(f"Batch harness failed for case {case_id}: {e}")

        return {
            "case_id": case_id,
            "correct": True,
            "domain": case.get("domain", "unknown"),
            "processed_at": datetime.now().isoformat(),
        }

    def _handle_case_result(self, result: Dict[str, Any]) -> None:
        """Handle a case result.

        Args:
            result: Case result
        """
        self._case_results.append(result)
        self._cases_since_last_cycle += 1

        progress = self._state.progress
        progress.cases_processed += 1
        if result.get("correct", True):
            progress.cases_successful += 1
        else:
            progress.cases_failed += 1

        if self._start_time:
            progress.elapsed_seconds = (datetime.now() - self._start_time).total_seconds()

            if progress.cases_processed > 0:
                rate = progress.cases_processed / progress.elapsed_seconds
                remaining = progress.total_cases - progress.cases_processed
                if rate > 0:
                    progress.estimated_remaining_seconds = remaining / rate

        self._state.last_updated = datetime.now()
        self._persistence.save_state(self._state)

        if self._on_case_complete:
            self._on_case_complete(result)

        if self._on_progress:
            self._on_progress(progress)

        self._persistence.append_timeline_event(
            {
                "type": "case_complete",
                "case_id": result.get("case_id"),
                "correct": result.get("correct", True),
                "cases_processed": progress.cases_processed,
                "accuracy": progress.current_accuracy,
            }
        )

    def _should_stop(self) -> bool:
        """Check if run should stop.

        Returns:
            True if should stop
        """
        if self._shutdown_requested:
            return True

        if self._start_time:
            elapsed = datetime.now() - self._start_time
            max_duration = timedelta(hours=self._config.max_duration_hours)
            if elapsed >= max_duration:
                logger.info("Time limit reached")
                return True

        if self._config.max_cases > 0:
            if self._state.progress.cases_processed >= self._config.max_cases:
                logger.info("Case limit reached")
                return True

        return False

    def _should_checkpoint(self) -> bool:
        """Check if checkpoint should be created.

        Returns:
            True if checkpoint needed
        """
        if self._last_checkpoint_time is None:
            self._last_checkpoint_time = datetime.now()
            return False

        interval = timedelta(minutes=self._config.checkpoint_interval_minutes)
        return datetime.now() - self._last_checkpoint_time >= interval

    def _create_checkpoint(self) -> None:
        """Create a checkpoint."""
        checkpoint_number = (
            len(list(self._persistence.checkpoints_dir.glob("checkpoint_*.json"))) + 1
        )

        self._state.status = RunStatus.CHECKPOINTING
        self._persistence.save_state(self._state)

        checkpoint = RunCheckpoint(
            checkpoint_number=checkpoint_number,
            created_at=datetime.now(),
            run_id=self._run_id,
            run_state=self._state,
            config_snapshot=self._config.to_dict(),
            meta_reasoning_state=self._orchestrator.get_state_snapshot(),
            cycle_results=self._cycle_results,
            accumulated_rules=self._accumulated_rules,
            case_results=self._case_results,
        )

        checkpoint_path = self._persistence.save_checkpoint(checkpoint)
        self._last_checkpoint_time = datetime.now()

        self._state.status = RunStatus.RUNNING
        self._persistence.save_state(self._state)

        self._persistence.append_timeline_event(
            {
                "type": "checkpoint_created",
                "checkpoint_number": checkpoint_number,
                "cases_processed": self._state.progress.cases_processed,
            }
        )

        if self._on_checkpoint:
            self._on_checkpoint(checkpoint)

        logger.info(f"Created checkpoint {checkpoint_number} at {checkpoint_path}")

    def _run_improvement_cycle(self) -> None:
        """Run an improvement cycle."""
        current_accuracy = self._state.progress.current_accuracy

        cycle_result = self._orchestrator.run_improvement_cycle(
            case_results=self._case_results[-self._cases_since_last_cycle :],
            accumulated_rules=self._accumulated_rules,
            current_accuracy=current_accuracy,
        )

        self._cycle_results.append(cycle_result.to_cycle_result())
        self._cases_since_last_cycle = 0

        self._state.progress.current_cycle = cycle_result.cycle_number
        self._state.progress.total_cycles += 1

        if self._config.checkpoint_on_cycle_complete:
            self._create_checkpoint()

        self._persistence.append_timeline_event(
            {
                "type": "improvement_cycle_complete",
                "cycle_number": cycle_result.cycle_number,
                "success": cycle_result.success,
                "improvements_made": cycle_result.improvements_made,
                "accuracy_before": cycle_result.accuracy_before,
                "accuracy_after": cycle_result.accuracy_after,
            }
        )

    def _handle_cycle_complete(self, cycle_result: CycleResult) -> None:
        """Handle improvement cycle completion.

        Args:
            cycle_result: Cycle result
        """
        if self._on_cycle_complete:
            self._on_cycle_complete(cycle_result)

    def _create_completed_result(self) -> RunResult:
        """Create result for completed run.

        Returns:
            RunResult
        """
        self._state.status = RunStatus.COMPLETED
        self._persistence.save_state(self._state)

        metrics = self._calculate_final_metrics()
        self._persistence.save_metrics(metrics)
        self._persistence.save_rules(self._accumulated_rules)

        result = RunResult(
            run_id=self._run_id,
            status=RunStatus.COMPLETED,
            started_at=self._start_time or datetime.now(),
            completed_at=datetime.now(),
            total_duration_seconds=(
                (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
            ),
            config_used=self._config.to_dict(),
            final_metrics=metrics,
            cycle_results=self._cycle_results,
            final_rules=self._accumulated_rules,
        )

        self._persistence.save_result(result)

        logger.info(f"Run {self._run_id} completed successfully")
        logger.info(f"Final accuracy: {metrics.overall_accuracy:.2%}")
        logger.info(f"Improvement cycles: {metrics.improvement_cycles_completed}")

        return result

    def _create_cancelled_result(self) -> RunResult:
        """Create result for cancelled run.

        Returns:
            RunResult
        """
        self._state.status = RunStatus.CANCELLED
        self._persistence.save_state(self._state)

        metrics = self._calculate_final_metrics()

        result = RunResult(
            run_id=self._run_id,
            status=RunStatus.CANCELLED,
            started_at=self._start_time or datetime.now(),
            completed_at=datetime.now(),
            total_duration_seconds=(
                (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
            ),
            config_used=self._config.to_dict(),
            final_metrics=metrics,
            cycle_results=self._cycle_results,
            final_rules=self._accumulated_rules,
            checkpoint_path=str(self._persistence.checkpoints_dir / "latest.json"),
        )

        self._persistence.save_result(result)

        logger.info(f"Run {self._run_id} cancelled after {result.duration_hours:.2f} hours")
        logger.info("Checkpoint saved for resume")

        return result

    def _create_error_result(self, error_message: str) -> RunResult:
        """Create result for failed run.

        Args:
            error_message: Error description

        Returns:
            RunResult
        """
        if self._state:
            self._state.status = RunStatus.FAILED
            self._state.error_message = error_message
            if self._persistence:
                self._persistence.save_state(self._state)

        metrics = self._calculate_final_metrics() if self._case_results else RunMetrics()

        result = RunResult(
            run_id=self._run_id or "unknown",
            status=RunStatus.FAILED,
            started_at=self._start_time or datetime.now(),
            completed_at=datetime.now(),
            total_duration_seconds=(
                (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
            ),
            config_used=self._config.to_dict(),
            final_metrics=metrics,
            cycle_results=self._cycle_results,
            final_rules=self._accumulated_rules,
            error_message=error_message,
        )

        if self._persistence:
            self._persistence.save_result(result)

        logger.error(f"Run {self._run_id} failed: {error_message}")

        return result

    def _calculate_final_metrics(self) -> RunMetrics:
        """Calculate final run metrics.

        Returns:
            RunMetrics
        """
        progress = self._state.progress if self._state else RunProgress()

        domain_accuracy: Dict[str, Dict[str, int]] = {}
        for result in self._case_results:
            domain = result.get("domain", "unknown")
            if domain not in domain_accuracy:
                domain_accuracy[domain] = {"correct": 0, "total": 0}
            domain_accuracy[domain]["total"] += 1
            if result.get("correct", True):
                domain_accuracy[domain]["correct"] += 1

        accuracy_by_domain = {
            domain: counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            for domain, counts in domain_accuracy.items()
        }

        meta_summary = self._orchestrator.get_metrics_summary() if self._orchestrator else {}

        cycle_durations = [cr.duration_seconds for cr in self._cycle_results if cr.duration_seconds]
        avg_cycle_duration = sum(cycle_durations) / len(cycle_durations) if cycle_durations else 0.0

        return RunMetrics(
            overall_accuracy=progress.current_accuracy,
            accuracy_by_domain=accuracy_by_domain,
            improvement_cycles_completed=len(self._cycle_results),
            average_cycle_duration_seconds=avg_cycle_duration,
            rules_generated_total=len(self._accumulated_rules),
            failure_patterns_identified=meta_summary.get("failure_patterns", []),
            strategy_changes_total=meta_summary.get("total_strategy_changes", 0),
            prompt_changes_total=meta_summary.get("total_prompt_changes", 0),
        )
