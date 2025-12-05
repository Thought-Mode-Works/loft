"""
Persistence Layer for Autonomous Test Harness.

This module handles saving and loading of run state, checkpoints,
metrics, and results for autonomous long-running experiments.

Features:
- Atomic checkpoint writes with temp file + rename
- Automatic checkpoint rotation (keep last N)
- JSONL timeline for append-only event logging
- State file for real-time monitoring
- Report generation (JSON and Markdown)
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loft.autonomous.schemas import (
    RunCheckpoint,
    RunMetrics,
    RunResult,
    RunState,
)

logger = logging.getLogger(__name__)


class PersistenceManager:
    """Manages persistence for autonomous test runs.

    Handles saving and loading of:
    - Run state (current status for monitoring)
    - Checkpoints (full state for resume)
    - Metrics (aggregate and timeline)
    - Results (final report)

    Attributes:
        run_dir: Directory for this run's files
        max_checkpoints: Maximum checkpoints to retain
    """

    def __init__(
        self,
        run_dir: Path,
        max_checkpoints: int = 10,
    ):
        """Initialize persistence manager.

        Args:
            run_dir: Directory for run files
            max_checkpoints: Maximum checkpoints to keep
        """
        self._run_dir = Path(run_dir)
        self._max_checkpoints = max_checkpoints
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create directory structure for run."""
        directories = [
            self._run_dir,
            self._run_dir / "checkpoints",
            self._run_dir / "metrics",
            self._run_dir / "rules",
            self._run_dir / "reports",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        """Get run directory path."""
        return self._run_dir

    @property
    def state_path(self) -> Path:
        """Get state file path."""
        return self._run_dir / "state.json"

    @property
    def config_path(self) -> Path:
        """Get config file path."""
        return self._run_dir / "config.json"

    @property
    def timeline_path(self) -> Path:
        """Get timeline file path."""
        return self._run_dir / "metrics" / "timeline.jsonl"

    @property
    def checkpoints_dir(self) -> Path:
        """Get checkpoints directory."""
        return self._run_dir / "checkpoints"

    def save_config(self, config_dict: Dict[str, Any]) -> None:
        """Save run configuration.

        Args:
            config_dict: Configuration dictionary
        """
        self._write_json_atomic(self.config_path, config_dict)
        logger.debug(f"Saved config to {self.config_path}")

    def load_config(self) -> Optional[Dict[str, Any]]:
        """Load run configuration.

        Returns:
            Configuration dictionary or None if not found
        """
        if not self.config_path.exists():
            return None
        return self._read_json(self.config_path)

    def save_state(self, state: RunState) -> None:
        """Save current run state.

        This is called frequently for monitoring purposes.
        Uses atomic write to prevent corruption.

        Args:
            state: Current run state
        """
        state_dict = state.to_dict()
        state_dict["saved_at"] = datetime.now().isoformat()
        self._write_json_atomic(self.state_path, state_dict)

    def load_state(self) -> Optional[RunState]:
        """Load current run state.

        Returns:
            RunState or None if not found
        """
        if not self.state_path.exists():
            return None
        data = self._read_json(self.state_path)
        if data:
            return RunState.from_dict(data)
        return None

    def save_checkpoint(self, checkpoint: RunCheckpoint) -> Path:
        """Save a checkpoint for resume capability.

        Creates checkpoint file and symlink to latest.
        Rotates old checkpoints if over limit.

        Args:
            checkpoint: Checkpoint to save

        Returns:
            Path to saved checkpoint file
        """
        checkpoint_filename = f"checkpoint_{checkpoint.checkpoint_number:04d}.json"
        checkpoint_path = self.checkpoints_dir / checkpoint_filename

        self._write_json_atomic(checkpoint_path, checkpoint.to_dict())

        latest_link = self.checkpoints_dir / "latest.json"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_filename)

        self._rotate_checkpoints()

        logger.info(f"Saved checkpoint {checkpoint.checkpoint_number} to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Optional[RunCheckpoint]:
        """Load a checkpoint for resume.

        Args:
            checkpoint_path: Specific checkpoint to load, or None for latest

        Returns:
            RunCheckpoint or None if not found
        """
        if checkpoint_path is None:
            latest_link = self.checkpoints_dir / "latest.json"
            if not latest_link.exists():
                return None
            checkpoint_path = latest_link.resolve()

        if not checkpoint_path.exists():
            return None

        data = self._read_json(checkpoint_path)
        if data:
            return RunCheckpoint.from_dict(data)
        return None

    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints.

        Returns:
            List of checkpoint file paths, sorted by number
        """
        checkpoints = list(self.checkpoints_dir.glob("checkpoint_*.json"))
        return sorted(checkpoints)

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints beyond limit."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) > self._max_checkpoints:
            to_remove = checkpoints[: len(checkpoints) - self._max_checkpoints]
            for checkpoint_path in to_remove:
                checkpoint_path.unlink()
                logger.debug(f"Rotated old checkpoint: {checkpoint_path}")

    def append_timeline_event(self, event: Dict[str, Any]) -> None:
        """Append event to timeline log.

        Uses JSONL format for append-only logging.

        Args:
            event: Event dictionary to log
        """
        event["timestamp"] = datetime.now().isoformat()
        with open(self.timeline_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def read_timeline(self) -> List[Dict[str, Any]]:
        """Read all timeline events.

        Returns:
            List of event dictionaries
        """
        events = []
        if self.timeline_path.exists():
            with open(self.timeline_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        return events

    def save_metrics(self, metrics: RunMetrics, filename: str = "run_metrics.json") -> None:
        """Save run metrics.

        Args:
            metrics: Metrics to save
            filename: Output filename
        """
        metrics_path = self._run_dir / "metrics" / filename
        self._write_json_atomic(metrics_path, metrics.to_dict())
        logger.debug(f"Saved metrics to {metrics_path}")

    def load_metrics(self, filename: str = "run_metrics.json") -> Optional[RunMetrics]:
        """Load run metrics.

        Args:
            filename: Metrics filename

        Returns:
            RunMetrics or None if not found
        """
        metrics_path = self._run_dir / "metrics" / filename
        if not metrics_path.exists():
            return None
        data = self._read_json(metrics_path)
        if data:
            return RunMetrics.from_dict(data)
        return None

    def save_rules(self, rules: List[Dict[str, Any]], filename: str = "evolved_rules.json") -> None:
        """Save accumulated rules.

        Args:
            rules: Rules to save
            filename: Output filename
        """
        rules_path = self._run_dir / "rules" / filename
        self._write_json_atomic(rules_path, {"rules": rules, "count": len(rules)})
        logger.debug(f"Saved {len(rules)} rules to {rules_path}")

    def load_rules(self, filename: str = "evolved_rules.json") -> List[Dict[str, Any]]:
        """Load saved rules.

        Args:
            filename: Rules filename

        Returns:
            List of rule dictionaries
        """
        rules_path = self._run_dir / "rules" / filename
        if not rules_path.exists():
            return []
        data = self._read_json(rules_path)
        if data:
            return data.get("rules", [])
        return []

    def save_result(self, result: RunResult) -> Path:
        """Save final run result.

        Creates both JSON and Markdown reports.

        Args:
            result: Final result to save

        Returns:
            Path to JSON result file
        """
        json_path = self._run_dir / "reports" / "final_report.json"
        self._write_json_atomic(json_path, result.to_dict())

        md_path = self._run_dir / "reports" / "final_report.md"
        self._write_markdown_report(md_path, result)

        logger.info(f"Saved final result to {json_path}")
        return json_path

    def load_result(self) -> Optional[RunResult]:
        """Load final run result.

        Returns:
            RunResult or None if not found
        """
        json_path = self._run_dir / "reports" / "final_report.json"
        if not json_path.exists():
            return None
        data = self._read_json(json_path)
        if data:
            return RunResult.from_dict(data)
        return None

    def _write_markdown_report(self, path: Path, result: RunResult) -> None:
        """Generate markdown report from result.

        Args:
            path: Output path
            result: Run result
        """
        lines = [
            f"# Autonomous Run Report: {result.run_id}",
            "",
            "## Summary",
            "",
            f"- **Status:** {result.status.value}",
            f"- **Duration:** {result.duration_hours:.2f} hours",
            f"- **Started:** {result.started_at.isoformat()}",
            f"- **Completed:** {result.completed_at.isoformat()}",
            "",
            "## Metrics",
            "",
            f"- **Overall Accuracy:** {result.final_metrics.overall_accuracy:.2%}",
            f"- **Improvement Cycles:** {result.final_metrics.improvement_cycles_completed}",
            f"- **Rules Generated:** {result.final_metrics.rules_generated_total}",
            f"- **Rules Promoted:** {result.final_metrics.rules_promoted_total}",
            f"- **LLM Calls:** {result.final_metrics.llm_calls_total}",
            f"- **Estimated Cost:** ${result.final_metrics.llm_cost_estimate:.2f}",
            "",
        ]

        if result.final_metrics.accuracy_by_domain:
            lines.extend(
                [
                    "## Accuracy by Domain",
                    "",
                ]
            )
            for domain, accuracy in result.final_metrics.accuracy_by_domain.items():
                lines.append(f"- **{domain}:** {accuracy:.2%}")
            lines.append("")

        if result.cycle_results:
            lines.extend(
                [
                    "## Improvement Cycles",
                    "",
                    "| Cycle | Status | Cases | Accuracy Before | Accuracy After | Delta |",
                    "|-------|--------|-------|-----------------|----------------|-------|",
                ]
            )
            for cycle in result.cycle_results:
                lines.append(
                    f"| {cycle.cycle_number} | {cycle.status.value} | "
                    f"{cycle.cases_processed} | {cycle.accuracy_before:.2%} | "
                    f"{cycle.accuracy_after:.2%} | {cycle.accuracy_delta:+.2%} |"
                )
            lines.append("")

        if result.final_metrics.failure_patterns_identified:
            lines.extend(
                [
                    "## Identified Failure Patterns",
                    "",
                ]
            )
            for pattern in result.final_metrics.failure_patterns_identified:
                lines.append(f"- {pattern}")
            lines.append("")

        if result.error_message:
            lines.extend(
                [
                    "## Error",
                    "",
                    "```",
                    result.error_message,
                    "```",
                    "",
                ]
            )

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _write_json_atomic(self, path: Path, data: Dict[str, Any]) -> None:
        """Write JSON file atomically using temp file + rename.

        Args:
            path: Target path
            data: Data to write
        """
        temp_path = path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            temp_path.rename(path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read JSON file.

        Args:
            path: File path

        Returns:
            Parsed data or None on error
        """
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read {path}: {e}")
            return None

    def cleanup(self, keep_results: bool = True) -> None:
        """Clean up run directory.

        Args:
            keep_results: Whether to keep final reports
        """
        if keep_results:
            for subdir in ["checkpoints", "metrics", "rules"]:
                subdir_path = self._run_dir / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)
                    subdir_path.mkdir()
        else:
            if self._run_dir.exists():
                shutil.rmtree(self._run_dir)

    def get_disk_usage(self) -> Dict[str, int]:
        """Get disk usage for run directory.

        Returns:
            Dictionary with size in bytes for each subdirectory
        """
        usage = {}
        for subdir in ["checkpoints", "metrics", "rules", "reports"]:
            subdir_path = self._run_dir / subdir
            if subdir_path.exists():
                total_size = sum(f.stat().st_size for f in subdir_path.rglob("*") if f.is_file())
                usage[subdir] = total_size
            else:
                usage[subdir] = 0
        usage["total"] = sum(usage.values())
        return usage


def create_persistence_manager(
    output_dir: Path,
    run_id: str,
    max_checkpoints: int = 10,
) -> PersistenceManager:
    """Factory function to create a persistence manager.

    Args:
        output_dir: Base output directory
        run_id: Run identifier
        max_checkpoints: Maximum checkpoints to retain

    Returns:
        Configured PersistenceManager
    """
    run_dir = Path(output_dir) / run_id
    return PersistenceManager(run_dir, max_checkpoints)
