"""
Unit tests for configuration system.

These tests verify that the configuration system works correctly
and can load settings from environment variables.
"""

import pytest
from loft.config import Config, LLMConfig, ValidationConfig, ASPConfig, LogConfig


def test_config_has_defaults() -> None:
    """Test that Config initializes with sensible defaults."""
    config = Config()

    assert config.llm.provider == "anthropic"
    assert config.llm.temperature == 0.7
    assert config.llm.max_tokens == 4096

    assert config.validation.consistency_check is True
    assert config.validation.confidence_threshold_constitutional == 1.0
    assert config.validation.confidence_threshold_strategic == 0.9
    assert config.validation.confidence_threshold_tactical == 0.8
    assert config.validation.confidence_threshold_operational == 0.6

    assert config.asp.programs_dir == "programs"
    assert config.asp.max_answer_sets == 10
    assert config.asp.optimization is True

    assert config.logging.level == "INFO"


def test_llm_config_defaults() -> None:
    """Test LLMConfig default values."""
    llm_config = LLMConfig()

    assert llm_config.provider == "anthropic"
    assert llm_config.model == "claude-3-5-sonnet-20241022"
    assert llm_config.temperature == 0.7
    assert llm_config.max_tokens == 4096


def test_validation_config_defaults() -> None:
    """Test ValidationConfig default values."""
    validation_config = ValidationConfig()

    assert validation_config.consistency_check is True
    assert validation_config.confidence_threshold_constitutional == 1.0
    assert validation_config.confidence_threshold_strategic == 0.9
    assert validation_config.confidence_threshold_tactical == 0.8
    assert validation_config.confidence_threshold_operational == 0.6


def test_asp_config_defaults() -> None:
    """Test ASPConfig default values."""
    asp_config = ASPConfig()

    assert asp_config.programs_dir == "programs"
    assert asp_config.max_answer_sets == 10
    assert asp_config.optimization is True
    assert asp_config.stats is False


def test_asp_config_programs_path() -> None:
    """Test ASPConfig programs_path property."""
    asp_config = ASPConfig(programs_dir="programs")
    path = asp_config.programs_path

    assert path.is_absolute()
    assert path.name == "programs"


def test_log_config_defaults() -> None:
    """Test LogConfig default values."""
    log_config = LogConfig()

    assert log_config.level == "INFO"
    assert log_config.rotation == "100 MB"
    assert log_config.retention == "1 month"


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Config.from_env() reads environment variables correctly."""
    # Set environment variables
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2000")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD_STRATEGIC", "0.95")
    monkeypatch.setenv("ASP_PROGRAMS_DIR", "custom_programs")
    monkeypatch.setenv("ASP_MAX_ANSWER_SETS", "20")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    config = Config.from_env()

    assert config.llm.provider == "openai"
    assert config.llm.model == "gpt-4"
    assert config.llm.temperature == 0.5
    assert config.llm.max_tokens == 2000
    assert config.llm.api_key == "test-key-123"

    assert config.validation.confidence_threshold_strategic == 0.95

    assert config.asp.programs_dir == "custom_programs"
    assert config.asp.max_answer_sets == 20

    assert config.logging.level == "DEBUG"


def test_validation_config_thresholds_are_in_range() -> None:
    """Test that all confidence thresholds are between 0 and 1."""
    validation_config = ValidationConfig()

    assert 0.0 <= validation_config.confidence_threshold_constitutional <= 1.0
    assert 0.0 <= validation_config.confidence_threshold_strategic <= 1.0
    assert 0.0 <= validation_config.confidence_threshold_tactical <= 1.0
    assert 0.0 <= validation_config.confidence_threshold_operational <= 1.0


def test_llm_config_temperature_range() -> None:
    """Test that temperature must be in valid range."""
    # Valid temperatures
    LLMConfig(temperature=0.0)
    LLMConfig(temperature=1.0)
    LLMConfig(temperature=2.0)

    # Invalid temperatures should raise validation error
    with pytest.raises(Exception):  # Pydantic ValidationError
        LLMConfig(temperature=-0.1)

    with pytest.raises(Exception):  # Pydantic ValidationError
        LLMConfig(temperature=2.1)


def test_config_is_importable_from_top_level() -> None:
    """Test that config can be imported from loft package."""
    from loft import config

    assert isinstance(config, Config)


def test_stratification_thresholds_decrease() -> None:
    """Test that confidence thresholds decrease down the stratification layers.

    Constitutional (1.0) > Strategic (0.9) > Tactical (0.8) > Operational (0.6)
    This ensures higher layers require more confidence for modification.
    """
    validation_config = ValidationConfig()

    assert (
        validation_config.confidence_threshold_constitutional
        >= validation_config.confidence_threshold_strategic
    )
    assert (
        validation_config.confidence_threshold_strategic
        >= validation_config.confidence_threshold_tactical
    )
    assert (
        validation_config.confidence_threshold_tactical
        >= validation_config.confidence_threshold_operational
    )
