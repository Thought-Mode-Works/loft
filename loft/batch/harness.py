"""
Batch learning harness for autonomous learning cycles.

Orchestrates processing of test case batches through the complete
learning pipeline: gap identification, rule generation, validation,
and evolution tracking.
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from .schemas import (
    BatchCheckpoint,
    BatchConfig,
    BatchMetrics,
    BatchProgress,
    BatchResult,
    BatchStatus,
    CaseResult,
    CaseStatus,
)


class BatchLearningHarness:
    """
    Orchestrates autonomous learning cycles across test case batches.

    Processes test cases through the complete learning pipeline:
    1. Load and analyze test case
    2. Run ASP reasoning with current rules
    3. Identify knowledge gaps from failures
    4. Generate candidate rules
    5. Validate candidates through multi-stage pipeline
    6. Track rule evolution and incorporate accepted rules
    7. Collect metrics and checkpoint progress

    Example:
        harness = BatchLearningHarness(config=BatchConfig())

        # Process a batch of test cases
        result = harness.run_batch(
            test_cases=cases,
            process_case_fn=my_case_processor,
        )

        # Resume from checkpoint
        result = harness.resume_from_checkpoint("checkpoint_001.json")
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize batch learning harness.

        Args:
            config: Batch configuration (uses defaults if not provided)
            output_dir: Directory for results and checkpoints
        """
        self.config = config or BatchConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("data/batch_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Runtime state
        self._progress: Optional[BatchProgress] = None
        self._accumulated_rule_ids: List[str] = []
        self._case_results: List[CaseResult] = []
        self._consecutive_errors: int = 0
        self._checkpoint_counter: int = 0

        # Callbacks
        self._on_progress: Optional[Callable[[BatchProgress], None]] = None
        self._on_case_complete: Optional[Callable[[CaseResult], None]] = None
        self._on_checkpoint: Optional[Callable[[BatchCheckpoint], None]] = None

    def run_batch(
        self,
        test_cases: List[Dict[str, Any]],
        process_case_fn: Callable[[Dict[str, Any], List[str]], CaseResult],
        batch_id: Optional[str] = None,
    ) -> BatchResult:
        """
        Process a batch of test cases through the learning cycle.

        Args:
            test_cases: List of test case dictionaries with at least 'id' field
            process_case_fn: Function to process a single case.
                            Signature: (case: Dict, accumulated_rules: List[str]) -> CaseResult
            batch_id: Optional batch identifier (generated if not provided)

        Returns:
            BatchResult with all case results and metrics
        """
        batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting batch {batch_id} with {len(test_cases)} cases")

        # Apply max_cases limit if configured
        if self.config.max_cases and len(test_cases) > self.config.max_cases:
            test_cases = test_cases[: self.config.max_cases]
            logger.info(f"Limited to {self.config.max_cases} cases")

        # Initialize progress
        self._progress = BatchProgress(
            batch_id=batch_id,
            total_cases=len(test_cases),
            started_at=datetime.now(),
            status=BatchStatus.RUNNING,
        )
        self._accumulated_rule_ids = []
        self._case_results = []
        self._consecutive_errors = 0
        self._checkpoint_counter = 0

        # Track metrics
        metrics = BatchMetrics(batch_id=batch_id, started_at=datetime.now())

        try:
            # Process each case
            for i, case in enumerate(test_cases):
                case_id = case.get("id", f"case_{i}")
                self._progress.current_case_id = case_id
                self._progress.last_update = datetime.now()

                # Check for max rules limit
                if len(self._accumulated_rule_ids) >= self.config.max_total_rules:
                    logger.info(f"Reached max rules limit ({self.config.max_total_rules})")
                    break

                # Check for too many consecutive errors
                if self._consecutive_errors >= self.config.max_consecutive_errors:
                    logger.error(
                        f"Too many consecutive errors ({self._consecutive_errors}), stopping"
                    )
                    self._progress.status = BatchStatus.FAILED
                    break

                # Process the case
                result = self._process_single_case(case, case_id, process_case_fn, metrics)

                # Update progress
                self._update_progress(result)

                # Notify callback
                if self._on_progress:
                    self._on_progress(self._progress)

                # Checkpoint if needed
                if (i + 1) % self.config.checkpoint_interval == 0:
                    pending_ids = [
                        c.get("id", f"case_{j}") for j, c in enumerate(test_cases) if j > i
                    ]
                    self._create_checkpoint(pending_ids)

                # Estimate completion
                self._estimate_completion()

            # Mark complete
            if self._progress.status == BatchStatus.RUNNING:
                self._progress.status = BatchStatus.COMPLETED

        except KeyboardInterrupt:
            logger.warning("Batch processing interrupted by user")
            self._progress.status = BatchStatus.CANCELLED
            # Save final checkpoint
            self._create_checkpoint([])

        except Exception as e:
            logger.exception(f"Batch processing failed: {e}")
            self._progress.status = BatchStatus.FAILED
            metrics.total_errors += 1
            metrics.error_types["batch_failure"] = metrics.error_types.get("batch_failure", 0) + 1

        # Finalize metrics
        metrics.completed_at = datetime.now()
        metrics.cases_processed = self._progress.processed_cases
        metrics.rules_generated = self._progress.total_rules_generated
        metrics.rules_accepted = self._progress.total_rules_accepted
        metrics.rules_rejected = self._progress.total_rules_rejected

        if self._case_results:
            total_time = sum(r.processing_time_ms for r in self._case_results)
            metrics.total_processing_time_ms = total_time
            metrics.avg_case_time_ms = total_time / len(self._case_results)

        metrics.accuracy_after = self._progress.current_accuracy

        # Build result
        result = BatchResult(
            batch_id=batch_id,
            status=self._progress.status,
            started_at=self._progress.started_at,
            completed_at=datetime.now(),
            case_results=self._case_results,
            accumulated_rule_ids=self._accumulated_rule_ids,
            metrics=metrics,
            config=self.config.to_dict(),
        )

        # Save result
        self._save_result(result)

        logger.info(
            f"Batch {batch_id} completed: {result.success_count} successes, "
            f"{result.failure_count} failures, {result.total_rules_accepted} rules accepted"
        )

        return result

    def _process_single_case(
        self,
        case: Dict[str, Any],
        case_id: str,
        process_case_fn: Callable[[Dict[str, Any], List[str]], CaseResult],
        metrics: BatchMetrics,
    ) -> CaseResult:
        """Process a single test case."""
        start_time = time.time()

        try:
            # Call the processing function
            result = process_case_fn(case, self._accumulated_rule_ids.copy())

            # Update accumulated rules
            if result.generated_rule_ids:
                self._accumulated_rule_ids.extend(result.generated_rule_ids)

            # Reset consecutive errors on success
            if result.status == CaseStatus.SUCCESS:
                self._consecutive_errors = 0
            elif result.status == CaseStatus.FAILED:
                self._consecutive_errors += 1
                metrics.total_errors += 1
                error_type = result.error_message or "unknown"
                metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1

            # Callback
            if self._on_case_complete:
                self._on_case_complete(result)

            return result

        except Exception as e:
            self._consecutive_errors += 1
            metrics.total_errors += 1
            error_type = type(e).__name__
            metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1

            logger.error(f"Error processing case {case_id}: {e}")

            if not self.config.continue_on_error:
                raise

            return CaseResult(
                case_id=case_id,
                status=CaseStatus.FAILED,
                processed_at=datetime.now(),
                processing_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )

    def _update_progress(self, result: CaseResult) -> None:
        """Update progress tracking after processing a case."""
        self._case_results.append(result)
        self._progress.processed_cases += 1
        self._progress.last_update = datetime.now()

        if result.status == CaseStatus.SUCCESS:
            self._progress.successful_cases += 1
        elif result.status == CaseStatus.FAILED:
            self._progress.failed_cases += 1
        elif result.status == CaseStatus.SKIPPED:
            self._progress.skipped_cases += 1

        self._progress.total_rules_generated += result.rules_generated
        self._progress.total_rules_accepted += result.rules_accepted
        self._progress.total_rules_rejected += result.rules_rejected

        if result.prediction_correct is not None:
            if result.prediction_correct:
                self._progress.correct_predictions += 1
            else:
                self._progress.incorrect_predictions += 1

    def _estimate_completion(self) -> None:
        """Estimate completion time based on current progress."""
        if self._progress.processed_cases == 0:
            return

        elapsed = self._progress.elapsed_time_seconds
        avg_time_per_case = elapsed / self._progress.processed_cases
        remaining_cases = self._progress.total_cases - self._progress.processed_cases
        estimated_remaining = remaining_cases * avg_time_per_case

        self._progress.estimated_completion = datetime.now() + timedelta(
            seconds=estimated_remaining
        )

    def _create_checkpoint(self, pending_case_ids: List[str]) -> BatchCheckpoint:
        """Create a checkpoint for resumption."""
        self._checkpoint_counter += 1
        checkpoint_id = f"checkpoint_{self._checkpoint_counter:04d}"

        checkpoint = BatchCheckpoint(
            batch_id=self._progress.batch_id,
            checkpoint_id=checkpoint_id,
            created_at=datetime.now(),
            processed_case_ids=[r.case_id for r in self._case_results],
            pending_case_ids=pending_case_ids,
            case_results=self._case_results.copy(),
            accumulated_rule_ids=self._accumulated_rule_ids.copy(),
            progress=self._progress,
            config=self.config.to_dict(),
        )

        # Save checkpoint
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        logger.info(f"Created checkpoint {checkpoint_id} at {self._progress.processed_cases} cases")

        # Callback
        if self._on_checkpoint:
            self._on_checkpoint(checkpoint)

        return checkpoint

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get path for checkpoint file."""
        return self.output_dir / self._progress.batch_id / "checkpoints" / f"{checkpoint_id}.json"

    def resume_from_checkpoint(
        self,
        checkpoint_path: str,
        test_cases: List[Dict[str, Any]],
        process_case_fn: Callable[[Dict[str, Any], List[str]], CaseResult],
    ) -> BatchResult:
        """
        Resume batch processing from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint JSON file
            test_cases: Full list of test cases (will filter to pending)
            process_case_fn: Function to process a single case

        Returns:
            BatchResult with combined results
        """
        # Load checkpoint
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = BatchCheckpoint.from_dict(json.load(f))

        logger.info(
            f"Resuming from checkpoint {checkpoint.checkpoint_id} "
            f"({len(checkpoint.processed_case_ids)} cases already processed)"
        )

        # Restore state
        self._accumulated_rule_ids = checkpoint.accumulated_rule_ids.copy()
        self._case_results = checkpoint.case_results.copy()
        self._progress = checkpoint.progress
        self._progress.status = BatchStatus.RUNNING
        self.config = BatchConfig.from_dict(checkpoint.config)

        # Filter to pending cases
        processed_ids = set(checkpoint.processed_case_ids)
        pending_cases = [c for c in test_cases if c.get("id") not in processed_ids]

        if not pending_cases:
            logger.info("No pending cases to process")
            self._progress.status = BatchStatus.COMPLETED
            return BatchResult(
                batch_id=checkpoint.batch_id,
                status=BatchStatus.COMPLETED,
                started_at=self._progress.started_at,
                completed_at=datetime.now(),
                case_results=self._case_results,
                accumulated_rule_ids=self._accumulated_rule_ids,
                config=self.config.to_dict(),
            )

        # Continue processing
        return self.run_batch(
            test_cases=pending_cases,
            process_case_fn=process_case_fn,
            batch_id=checkpoint.batch_id,
        )

    def _save_result(self, result: BatchResult) -> None:
        """Save batch result to disk."""
        result_dir = self.output_dir / result.batch_id
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save full result
        result_path = result_dir / "result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save summary
        summary = {
            "batch_id": result.batch_id,
            "status": result.status.value,
            "started_at": result.started_at.isoformat(),
            "completed_at": (result.completed_at.isoformat() if result.completed_at else None),
            "total_cases": len(result.case_results),
            "successful_cases": result.success_count,
            "failed_cases": result.failure_count,
            "rules_accepted": result.total_rules_accepted,
            "accuracy": (result.metrics.accuracy_after if result.metrics else 0.0),
        }
        summary_path = result_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved batch result to {result_path}")

    def set_callbacks(
        self,
        on_progress: Optional[Callable[[BatchProgress], None]] = None,
        on_case_complete: Optional[Callable[[CaseResult], None]] = None,
        on_checkpoint: Optional[Callable[[BatchCheckpoint], None]] = None,
    ) -> None:
        """
        Set callback functions for batch events.

        Args:
            on_progress: Called after each case with updated progress
            on_case_complete: Called after each case with the result
            on_checkpoint: Called when a checkpoint is created
        """
        self._on_progress = on_progress
        self._on_case_complete = on_case_complete
        self._on_checkpoint = on_checkpoint

    def get_progress(self) -> Optional[BatchProgress]:
        """Get current progress (if batch is running)."""
        return self._progress

    @staticmethod
    def load_result(result_path: str) -> BatchResult:
        """Load a batch result from disk."""
        with open(result_path, "r", encoding="utf-8") as f:
            return BatchResult.from_dict(json.load(f))

    @staticmethod
    def list_batches(output_dir: str) -> List[Dict[str, Any]]:
        """List all batch runs in the output directory."""
        output_path = Path(output_dir)
        batches = []

        for batch_dir in output_path.iterdir():
            if batch_dir.is_dir():
                summary_path = batch_dir / "summary.json"
                if summary_path.exists():
                    with open(summary_path, "r", encoding="utf-8") as f:
                        batches.append(json.load(f))

        return sorted(batches, key=lambda b: b.get("started_at", ""), reverse=True)


def create_simple_case_processor(
    rule_generator_fn: Callable[[str, List[str]], List[Dict[str, Any]]],
    validator_fn: Callable[[str], bool],
    predictor_fn: Callable[[Dict[str, Any], List[str]], tuple[str, float]],
) -> Callable[[Dict[str, Any], List[str]], CaseResult]:
    """
    Create a simple case processor from component functions.

    This is a helper for creating case processors without implementing
    the full pipeline.

    Args:
        rule_generator_fn: (gap_description, existing_predicates) -> list of rule dicts
        validator_fn: (asp_rule) -> is_valid
        predictor_fn: (case, rules) -> (prediction, confidence)

    Returns:
        A case processor function compatible with BatchLearningHarness
    """

    def process_case(case: Dict[str, Any], accumulated_rules: List[str]) -> CaseResult:
        start_time = time.time()
        case_id = case.get("id", str(uuid.uuid4())[:8])

        try:
            # Make prediction with current rules
            prediction, confidence = predictor_fn(case, accumulated_rules)

            # Check if correct
            ground_truth = case.get("ground_truth", "")
            prediction_correct = prediction == ground_truth

            # If incorrect, try to generate new rules
            rules_generated = 0
            rules_accepted = 0
            rules_rejected = 0
            generated_rule_ids = []

            if not prediction_correct:
                # Generate rule candidates
                gap_description = (
                    f"Case {case_id} predicted {prediction} but should be {ground_truth}"
                )
                candidates = rule_generator_fn(gap_description, [])

                rules_generated = len(candidates)

                # Validate each candidate
                for candidate in candidates:
                    rule_text = candidate.get("asp_rule", "")
                    rule_id = candidate.get("id", str(uuid.uuid4())[:8])

                    if validator_fn(rule_text):
                        rules_accepted += 1
                        generated_rule_ids.append(rule_id)
                    else:
                        rules_rejected += 1

            return CaseResult(
                case_id=case_id,
                status=CaseStatus.SUCCESS,
                processed_at=datetime.now(),
                processing_time_ms=(time.time() - start_time) * 1000,
                rules_generated=rules_generated,
                rules_accepted=rules_accepted,
                rules_rejected=rules_rejected,
                prediction_correct=prediction_correct,
                confidence=confidence,
                generated_rule_ids=generated_rule_ids,
            )

        except Exception as e:
            return CaseResult(
                case_id=case_id,
                status=CaseStatus.FAILED,
                processed_at=datetime.now(),
                processing_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )

    return process_case
