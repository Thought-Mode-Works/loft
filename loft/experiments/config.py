"""
Experiment configuration for long-running learning experiments.

Issue #256: Long-Running Experiment Runner
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for long-running experiments."""

    # Identity
    experiment_id: str
    description: str

    # Duration limits
    max_duration_seconds: int = 4 * 60 * 60  # 4 hours default
    max_cycles: int = 100

    # Batch settings
    cases_per_cycle: int = 20
    dataset_path: Path = Path("datasets/contracts/")

    # Checkpointing
    checkpoint_interval: int = 5  # Save state every N cycles
    report_interval_seconds: int = 30 * 60  # 30 minutes

    # Resource management
    cool_down_seconds: int = 10  # Pause between cycles
    max_api_calls_per_minute: int = 20

    # Paths
    state_path: Path = Path("data/experiments/")
    reports_path: Path = Path("reports/experiments/")
    rules_path: Path = Path("asp_rules/")
    meta_state_dir: Path = Path("data/meta_state/")

    # Goals
    target_accuracy: float = 0.85
    target_coverage: float = 0.80
    target_rule_count: int = 100

    # LLM settings
    model: str = "claude-3-5-haiku-20241022"
    enable_llm: bool = True
    enable_meta: bool = True

    def __post_init__(self):
        """Ensure paths are Path objects."""
        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)
        if isinstance(self.state_path, str):
            self.state_path = Path(self.state_path)
        if isinstance(self.reports_path, str):
            self.reports_path = Path(self.reports_path)
        if isinstance(self.rules_path, str):
            self.rules_path = Path(self.rules_path)
        if isinstance(self.meta_state_dir, str):
            self.meta_state_dir = Path(self.meta_state_dir)

        # Create directories
        self.state_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self.rules_path.mkdir(parents=True, exist_ok=True)
        self.meta_state_dir.mkdir(parents=True, exist_ok=True)


def parse_duration(duration_str: str) -> int:
    """
    Parse duration string to seconds.

    Args:
        duration_str: Duration string like "30m", "2h", "4h"

    Returns:
        Duration in seconds

    Examples:
        >>> parse_duration("30m")
        1800
        >>> parse_duration("2h")
        7200
    """
    duration_str = duration_str.strip().lower()

    # Extract number and unit
    if duration_str.endswith("m"):
        return int(duration_str[:-1]) * 60
    elif duration_str.endswith("h"):
        return int(duration_str[:-1]) * 60 * 60
    elif duration_str.endswith("s"):
        return int(duration_str[:-1])
    else:
        # Assume seconds if no unit
        return int(duration_str)
