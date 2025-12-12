"""Unit tests for autonomous test harness configuration."""

import os
from pathlib import Path

import pytest

from loft.autonomous.config import (
    AutonomousRunConfig,
    HealthConfig,
    MetaReasoningConfig,
    NotificationConfig,
    SafetyConfig,
)


class TestNotificationConfig:
    """Tests for NotificationConfig."""

    def test_default_values(self):
        """Test default values."""
        config = NotificationConfig()

        assert config.slack_webhook_url is None
        assert config.notify_on_start is True
        assert config.notify_on_completion is True
        assert config.milestone_interval_cases == 100

    def test_valid_webhook_url(self):
        """Test valid Slack webhook URL is accepted."""
        config = NotificationConfig(
            slack_webhook_url="https://hooks.slack.com/services/xxx"
        )
        assert config.slack_webhook_url == "https://hooks.slack.com/services/xxx"

    def test_invalid_webhook_url(self):
        """Test invalid webhook URL raises error."""
        with pytest.raises(ValueError, match="Invalid Slack webhook"):
            NotificationConfig(slack_webhook_url="https://example.com/webhook")

    def test_empty_webhook_url_becomes_none(self):
        """Test empty string becomes None."""
        config = NotificationConfig(slack_webhook_url="")
        assert config.slack_webhook_url is None


class TestMetaReasoningConfig:
    """Tests for MetaReasoningConfig."""

    def test_default_values(self):
        """Test default values."""
        config = MetaReasoningConfig()

        assert config.enable_autonomous_improvement is True
        assert config.enable_prompt_optimization is True
        assert config.enable_failure_analysis is True
        assert config.improvement_cycle_interval_cases == 50
        assert config.confidence_threshold == 0.7

    def test_custom_values(self):
        """Test custom values."""
        config = MetaReasoningConfig(
            enable_autonomous_improvement=False,
            improvement_cycle_interval_cases=100,
        )

        assert config.enable_autonomous_improvement is False
        assert config.improvement_cycle_interval_cases == 100


class TestSafetyConfig:
    """Tests for SafetyConfig."""

    def test_default_values(self):
        """Test default values."""
        config = SafetyConfig()

        assert config.max_accuracy_drop == 0.05
        assert config.auto_rollback_on_degradation is True
        assert config.min_accuracy_threshold == 0.6
        assert config.max_consecutive_failures == 10


class TestHealthConfig:
    """Tests for HealthConfig."""

    def test_default_values(self):
        """Test default values."""
        config = HealthConfig()

        assert config.enabled is True
        assert config.port == 8080
        assert config.host == "0.0.0.0"


class TestAutonomousRunConfig:
    """Tests for AutonomousRunConfig."""

    def test_default_values(self):
        """Test default values."""
        config = AutonomousRunConfig()

        assert config.max_duration_hours == 4.0
        assert config.max_cases == 0
        assert config.checkpoint_interval_minutes == 15
        assert config.llm_model == "claude-3-5-haiku-20241022"
        assert config.output_dir == "data/autonomous_runs"

    def test_duration_validation_positive(self):
        """Test duration must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            AutonomousRunConfig(max_duration_hours=0)

        with pytest.raises(ValueError, match="must be positive"):
            AutonomousRunConfig(max_duration_hours=-1)

    def test_duration_validation_max(self):
        """Test duration cannot exceed 24 hours."""
        with pytest.raises(ValueError, match="cannot exceed 24"):
            AutonomousRunConfig(max_duration_hours=25)

    def test_checkpoint_interval_validation(self):
        """Test checkpoint interval validation."""
        with pytest.raises(ValueError, match="at least 1"):
            AutonomousRunConfig(checkpoint_interval_minutes=0)

        with pytest.raises(ValueError, match="should not exceed 60"):
            AutonomousRunConfig(checkpoint_interval_minutes=120)

    def test_temperature_validation(self):
        """Test temperature validation."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            AutonomousRunConfig(llm_temperature=-0.1)

        with pytest.raises(ValueError, match="between 0 and 1"):
            AutonomousRunConfig(llm_temperature=1.5)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = AutonomousRunConfig(
            max_duration_hours=2.0,
            max_cases=500,
        )

        data = config.to_dict()

        assert data["max_duration_hours"] == 2.0
        assert data["max_cases"] == 500
        assert "notification" in data
        assert "meta_reasoning" in data

    def test_nested_configs(self):
        """Test nested configuration access."""
        config = AutonomousRunConfig()

        assert isinstance(config.notification, NotificationConfig)
        assert isinstance(config.meta_reasoning, MetaReasoningConfig)
        assert isinstance(config.safety, SafetyConfig)
        assert isinstance(config.health, HealthConfig)

    def test_get_run_output_dir(self):
        """Test run output directory generation."""
        config = AutonomousRunConfig(output_dir="data/runs")

        output_dir = config.get_run_output_dir("test_run_001")

        assert output_dir == Path("data/runs/test_run_001")

    def test_validate_paths_empty(self):
        """Test path validation with no paths."""
        config = AutonomousRunConfig()
        errors = config.validate_paths()
        assert errors == []

    def test_validate_paths_nonexistent(self):
        """Test path validation with nonexistent path."""
        config = AutonomousRunConfig(dataset_paths=["/nonexistent/path/to/dataset"])
        errors = config.validate_paths()
        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_from_env(self):
        """Test configuration from environment variables."""
        env_vars = {
            "AUTONOMOUS_MAX_DURATION_HOURS": "2.0",
            "AUTONOMOUS_MAX_CASES": "100",
            "AUTONOMOUS_OUTPUT_DIR": "/tmp/runs",
        }

        original_env = {}
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            config = AutonomousRunConfig.from_env()

            assert config.max_duration_hours == 2.0
            assert config.max_cases == 100
            assert config.output_dir == "/tmp/runs"
        finally:
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
