"""
Meta-state persistence for batch processing.

Persists meta-reasoning state between batch runs:
- Strategy weights and performance history
- Failure patterns and adaptations
- Prompt versions and effectiveness metrics
- Improvement goals and progress

Issue #255: Phase 8 meta-reasoning batch integration.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class MetaState:
    """
    Persisted meta-reasoning state.

    Captures all meta-reasoning state that should survive between
    batch runs to enable continuous learning and adaptation.

    Example:
        >>> state = MetaState.load(Path("./meta_state.json"))
        >>> state.strategy_weights["dialectical"] = 1.5
        >>> state.save(Path("./meta_state.json"))
    """

    # Strategy configuration
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    strategy_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Prompt configuration
    prompt_versions: Dict[str, str] = field(default_factory=dict)
    prompt_effectiveness: Dict[str, float] = field(default_factory=dict)

    # Failure analysis
    failure_patterns: List[Dict[str, Any]] = field(default_factory=list)
    failure_pattern_count: Dict[str, int] = field(default_factory=dict)

    # Adaptations history
    adaptations: List[Dict[str, Any]] = field(default_factory=list)

    # Improvement tracking
    improvement_goals: List[Dict[str, Any]] = field(default_factory=list)
    improvement_progress: Dict[str, float] = field(default_factory=dict)

    # Session metadata
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    total_cases_processed: int = 0
    total_cycles_completed: int = 0
    version: str = "1.0.0"

    def save(self, path: Path) -> None:
        """
        Save meta-state to disk.

        Args:
            path: Path to save state file
        """
        self.last_updated = datetime.now().isoformat()

        data = {
            "strategy_weights": self.strategy_weights,
            "strategy_performance": self.strategy_performance,
            "prompt_versions": self.prompt_versions,
            "prompt_effectiveness": self.prompt_effectiveness,
            "failure_patterns": self.failure_patterns[-100:],  # Keep last 100
            "failure_pattern_count": self.failure_pattern_count,
            "adaptations": self.adaptations[-50:],  # Keep last 50
            "improvement_goals": self.improvement_goals,
            "improvement_progress": self.improvement_progress,
            "last_updated": self.last_updated,
            "total_cases_processed": self.total_cases_processed,
            "total_cycles_completed": self.total_cycles_completed,
            "version": self.version,
        }

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug(f"Saved meta-state to {path}")

    @classmethod
    def load(cls, path: Path) -> "MetaState":
        """
        Load meta-state from disk.

        Args:
            path: Path to state file

        Returns:
            Loaded MetaState or new instance if file doesn't exist
        """
        if not path.exists():
            logger.debug(f"No meta-state file at {path}, creating new")
            return cls()

        try:
            with open(path, "r") as f:
                data = json.load(f)

            state = cls(
                strategy_weights=data.get("strategy_weights", {}),
                strategy_performance=data.get("strategy_performance", {}),
                prompt_versions=data.get("prompt_versions", {}),
                prompt_effectiveness=data.get("prompt_effectiveness", {}),
                failure_patterns=data.get("failure_patterns", []),
                failure_pattern_count=data.get("failure_pattern_count", {}),
                adaptations=data.get("adaptations", []),
                improvement_goals=data.get("improvement_goals", []),
                improvement_progress=data.get("improvement_progress", {}),
                last_updated=data.get("last_updated", datetime.now().isoformat()),
                total_cases_processed=data.get("total_cases_processed", 0),
                total_cycles_completed=data.get("total_cycles_completed", 0),
                version=data.get("version", "1.0.0"),
            )

            logger.debug(f"Loaded meta-state from {path}")
            return state

        except Exception as e:
            logger.warning(f"Failed to load meta-state from {path}: {e}")
            return cls()

    def update_from_processor(self, processor: Any) -> None:
        """
        Update state from a MetaAwareBatchProcessor.

        Args:
            processor: MetaAwareBatchProcessor instance
        """
        # Update strategy weights
        if hasattr(processor, "config") and hasattr(
            processor.config, "strategy_weights"
        ):
            self.strategy_weights = processor.config.strategy_weights.copy()

        # Update strategy performance
        if hasattr(processor, "strategy_performance"):
            for strategy, perf in processor.strategy_performance.items():
                self.strategy_performance[strategy] = perf.copy()

        # Update failure patterns
        if hasattr(processor, "failure_patterns"):
            for pattern in processor.failure_patterns:
                pattern_dict = (
                    pattern.to_dict() if hasattr(pattern, "to_dict") else pattern
                )
                self.failure_patterns.append(pattern_dict)

                # Update counts
                failure_type = pattern_dict.get("failure_type", "unknown")
                self.failure_pattern_count[failure_type] = (
                    self.failure_pattern_count.get(failure_type, 0) + 1
                )

        # Update adaptations
        if hasattr(processor, "adaptations"):
            for adaptation in processor.adaptations:
                adaptation_dict = (
                    adaptation.to_dict()
                    if hasattr(adaptation, "to_dict")
                    else adaptation
                )
                self.adaptations.append(adaptation_dict)

        # Update counts
        if hasattr(processor, "cases_processed"):
            self.total_cases_processed += processor.cases_processed

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of meta-state."""
        return {
            "version": self.version,
            "last_updated": self.last_updated,
            "total_cases_processed": self.total_cases_processed,
            "total_cycles_completed": self.total_cycles_completed,
            "strategies_tracked": len(self.strategy_weights),
            "failure_patterns_recorded": len(self.failure_patterns),
            "adaptations_made": len(self.adaptations),
            "improvement_goals": len(self.improvement_goals),
        }

    def reset_session_state(self) -> None:
        """Reset session-specific state while keeping learned state."""
        # Keep strategy weights and performance
        # Keep prompt effectiveness
        # Reset session-level counters
        self.failure_patterns = self.failure_patterns[-50:]  # Keep some history
        self.adaptations = self.adaptations[-25:]  # Keep some history


class MetaStateManager:
    """
    Manages meta-state persistence and updates.

    Provides convenient methods for loading, saving, and updating
    meta-state during batch processing.

    Example:
        >>> manager = MetaStateManager(Path("./batch_run/"))
        >>> state = manager.load_or_create()
        >>> # ... process cases ...
        >>> manager.save(state)
    """

    def __init__(self, base_dir: Path):
        """
        Initialize meta-state manager.

        Args:
            base_dir: Base directory for state storage
        """
        self.base_dir = Path(base_dir)
        self.state_file = self.base_dir / "meta_state.json"

    def load_or_create(self) -> MetaState:
        """Load existing state or create new one."""
        return MetaState.load(self.state_file)

    def save(self, state: MetaState) -> None:
        """Save state to disk."""
        state.save(self.state_file)

    def checkpoint(self, state: MetaState, checkpoint_id: str) -> Path:
        """
        Create checkpoint of current state.

        Args:
            state: Current meta-state
            checkpoint_id: Identifier for checkpoint

        Returns:
            Path to checkpoint file
        """
        checkpoint_dir = self.base_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / f"meta_state_{checkpoint_id}.json"
        state.save(checkpoint_file)

        return checkpoint_file

    def restore_checkpoint(self, checkpoint_id: str) -> Optional[MetaState]:
        """
        Restore state from checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Restored MetaState or None if not found
        """
        checkpoint_file = (
            self.base_dir / "checkpoints" / f"meta_state_{checkpoint_id}.json"
        )

        if checkpoint_file.exists():
            return MetaState.load(checkpoint_file)

        return None

    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        checkpoint_dir = self.base_dir / "checkpoints"

        if not checkpoint_dir.exists():
            return []

        checkpoints = []
        for f in checkpoint_dir.glob("meta_state_*.json"):
            # Extract checkpoint ID from filename
            checkpoint_id = f.stem.replace("meta_state_", "")
            checkpoints.append(checkpoint_id)

        return sorted(checkpoints)


def create_meta_state_manager(base_dir: Path) -> MetaStateManager:
    """Factory function to create a meta-state manager."""
    return MetaStateManager(base_dir)
