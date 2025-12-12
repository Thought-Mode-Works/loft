"""
Configuration for Autonomous Test Harness.

This module provides Pydantic-based configuration for long-running
autonomous test experiments with full validation and environment
variable support.

Classes:
- NotificationConfig: Slack/webhook notification settings
- MetaReasoningConfig: Meta-reasoning component settings
- SafetyConfig: Safety limits and rollback settings
- AutonomousRunConfig: Main configuration class
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class NotificationConfig(BaseModel):
    """Configuration for notifications.

    Attributes:
        slack_webhook_url: Slack incoming webhook URL
        notify_on_start: Send notification when run starts
        notify_on_milestone: Send notification on milestones
        notify_on_cycle_complete: Send notification after each cycle
        notify_on_error: Send notification on errors
        notify_on_completion: Send notification when run completes
        milestone_interval_cases: Cases between milestone notifications
    """

    slack_webhook_url: Optional[str] = Field(
        default=None, description="Slack webhook URL for notifications"
    )
    notify_on_start: bool = Field(default=True, description="Notify when run starts")
    notify_on_milestone: bool = Field(
        default=True, description="Notify on processing milestones"
    )
    notify_on_cycle_complete: bool = Field(
        default=True, description="Notify after each improvement cycle"
    )
    notify_on_error: bool = Field(default=True, description="Notify on errors")
    notify_on_completion: bool = Field(
        default=True, description="Notify when run completes"
    )
    milestone_interval_cases: int = Field(
        default=100, description="Cases between milestone notifications"
    )

    @field_validator("slack_webhook_url", mode="before")
    @classmethod
    def validate_webhook_url(cls, value: Optional[str]) -> Optional[str]:
        """Validate webhook URL format if provided."""
        if value is None or value == "":
            return None
        if not value.startswith("https://hooks.slack.com/"):
            raise ValueError("Invalid Slack webhook URL format")
        return value


class MetaReasoningConfig(BaseModel):
    """Configuration for meta-reasoning components.

    Attributes:
        enable_autonomous_improvement: Enable the AutonomousImprover
        enable_prompt_optimization: Enable PromptOptimizer
        enable_failure_analysis: Enable FailureAnalyzer
        enable_strategy_selection: Enable StrategySelector
        improvement_cycle_interval_cases: Cases between improvement cycles
        min_cases_for_analysis: Minimum cases before running analysis
        max_improvement_actions_per_cycle: Limit on actions per cycle
        confidence_threshold: Minimum confidence for applying changes
    """

    enable_autonomous_improvement: bool = Field(
        default=True, description="Enable AutonomousImprover"
    )
    enable_prompt_optimization: bool = Field(
        default=True, description="Enable PromptOptimizer"
    )
    enable_failure_analysis: bool = Field(
        default=True, description="Enable FailureAnalyzer"
    )
    enable_strategy_selection: bool = Field(
        default=True, description="Enable StrategySelector"
    )
    improvement_cycle_interval_cases: int = Field(
        default=50, description="Cases between improvement cycles"
    )
    min_cases_for_analysis: int = Field(
        default=10, description="Minimum cases before analysis"
    )
    max_improvement_actions_per_cycle: int = Field(
        default=5, description="Max actions per improvement cycle"
    )
    confidence_threshold: float = Field(
        default=0.7, description="Minimum confidence for changes"
    )


class SafetyConfig(BaseModel):
    """Configuration for safety limits and rollback.

    Attributes:
        max_accuracy_drop: Maximum accuracy drop before rollback
        auto_rollback_on_degradation: Auto-rollback on performance drop
        min_accuracy_threshold: Minimum acceptable accuracy
        max_consecutive_failures: Max consecutive failures before pause
        require_validation_before_apply: Require validation before changes
        preserve_baseline_rules: Never modify baseline rules
    """

    max_accuracy_drop: float = Field(
        default=0.05, description="Max accuracy drop before rollback (5%)"
    )
    auto_rollback_on_degradation: bool = Field(
        default=True, description="Auto-rollback on degradation"
    )
    min_accuracy_threshold: float = Field(
        default=0.6, description="Minimum acceptable accuracy"
    )
    max_consecutive_failures: int = Field(
        default=10, description="Max consecutive failures before pause"
    )
    require_validation_before_apply: bool = Field(
        default=True, description="Require validation before applying changes"
    )
    preserve_baseline_rules: bool = Field(
        default=True, description="Never modify baseline rules"
    )


class HealthConfig(BaseModel):
    """Configuration for health endpoint.

    Attributes:
        enabled: Whether to enable health endpoint
        port: Port for health endpoint
        host: Host to bind to
    """

    enabled: bool = Field(default=True, description="Enable health endpoint")
    port: int = Field(default=8080, description="Health endpoint port")
    host: str = Field(default="0.0.0.0", description="Health endpoint host")


class AutonomousRunConfig(BaseModel):
    """Main configuration for autonomous test runs.

    This is the primary configuration class that combines all
    sub-configurations for autonomous long-running experiments.

    Attributes:
        max_duration_hours: Maximum run duration in hours
        max_cases: Maximum cases to process (0 = unlimited)
        checkpoint_interval_minutes: Minutes between checkpoints
        checkpoint_on_cycle_complete: Create checkpoint after each cycle
        llm_model: LLM model to use
        llm_temperature: LLM temperature setting
        llm_max_retries: Max retries for LLM calls
        dataset_paths: Paths to dataset directories
        output_dir: Directory for run outputs
        run_id_prefix: Prefix for generated run IDs
        notification: Notification settings
        meta_reasoning: Meta-reasoning settings
        safety: Safety settings
        health: Health endpoint settings
        batch_size: Cases to process in each batch
        parallel_workers: Number of parallel workers
        log_level: Logging level
        log_to_file: Whether to log to file
    """

    # Duration limits
    max_duration_hours: float = Field(
        default=4.0, description="Maximum run duration in hours"
    )
    max_cases: int = Field(
        default=0, description="Maximum cases to process (0 = unlimited)"
    )

    # Checkpointing
    checkpoint_interval_minutes: int = Field(
        default=15, description="Minutes between checkpoints"
    )
    checkpoint_on_cycle_complete: bool = Field(
        default=True, description="Checkpoint after each improvement cycle"
    )

    # LLM settings
    llm_model: str = Field(
        default="claude-3-5-haiku-20241022", description="LLM model to use"
    )
    llm_temperature: float = Field(default=0.3, description="LLM temperature")
    llm_max_retries: int = Field(default=3, description="Max LLM call retries")

    # Paths
    dataset_paths: List[str] = Field(
        default_factory=list, description="Dataset directory paths"
    )
    output_dir: str = Field(
        default="data/autonomous_runs", description="Output directory"
    )
    run_id_prefix: str = Field(default="run", description="Run ID prefix")

    # Sub-configurations
    notification: NotificationConfig = Field(default_factory=NotificationConfig)
    meta_reasoning: MetaReasoningConfig = Field(default_factory=MetaReasoningConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)

    # Processing settings
    batch_size: int = Field(default=10, description="Cases per batch")
    parallel_workers: int = Field(default=1, description="Parallel workers")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_to_file: bool = Field(default=True, description="Log to file")

    @field_validator("max_duration_hours")
    @classmethod
    def validate_duration(cls, value: float) -> float:
        """Validate duration is positive and reasonable."""
        if value <= 0:
            raise ValueError("max_duration_hours must be positive")
        if value > 24:
            raise ValueError("max_duration_hours cannot exceed 24 hours")
        return value

    @field_validator("checkpoint_interval_minutes")
    @classmethod
    def validate_checkpoint_interval(cls, value: int) -> int:
        """Validate checkpoint interval."""
        if value < 1:
            raise ValueError("checkpoint_interval_minutes must be at least 1")
        if value > 60:
            raise ValueError("checkpoint_interval_minutes should not exceed 60")
        return value

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Validate temperature is in valid range."""
        if value < 0 or value > 1:
            raise ValueError("llm_temperature must be between 0 and 1")
        return value

    @classmethod
    def from_env(cls) -> "AutonomousRunConfig":
        """Create configuration from environment variables.

        Environment variables:
        - AUTONOMOUS_MAX_DURATION_HOURS
        - AUTONOMOUS_MAX_CASES
        - AUTONOMOUS_CHECKPOINT_INTERVAL
        - AUTONOMOUS_LLM_MODEL
        - AUTONOMOUS_OUTPUT_DIR
        - SLACK_WEBHOOK_URL
        - AUTONOMOUS_HEALTH_PORT
        - AUTONOMOUS_LOG_LEVEL

        Returns:
            AutonomousRunConfig populated from environment
        """
        notification_config = NotificationConfig(
            slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL"),
        )

        health_config = HealthConfig(
            port=int(os.environ.get("AUTONOMOUS_HEALTH_PORT", "8080")),
        )

        return cls(
            max_duration_hours=float(
                os.environ.get("AUTONOMOUS_MAX_DURATION_HOURS", "4.0")
            ),
            max_cases=int(os.environ.get("AUTONOMOUS_MAX_CASES", "0")),
            checkpoint_interval_minutes=int(
                os.environ.get("AUTONOMOUS_CHECKPOINT_INTERVAL", "15")
            ),
            llm_model=os.environ.get(
                "AUTONOMOUS_LLM_MODEL", "claude-3-5-haiku-20241022"
            ),
            output_dir=os.environ.get("AUTONOMOUS_OUTPUT_DIR", "data/autonomous_runs"),
            log_level=os.environ.get("AUTONOMOUS_LOG_LEVEL", "INFO"),
            notification=notification_config,
            health=health_config,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "AutonomousRunConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            AutonomousRunConfig loaded from file
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return self.model_dump()

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def get_run_output_dir(self, run_id: str) -> Path:
        """Get output directory for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            Path to run output directory
        """
        return Path(self.output_dir) / run_id

    def validate_paths(self) -> List[str]:
        """Validate configured paths exist.

        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []

        for dataset_path in self.dataset_paths:
            if not Path(dataset_path).exists():
                errors.append(f"Dataset path does not exist: {dataset_path}")

        return errors
