"""
Configuration management for LOFT.

This module provides centralized configuration for all system components:
- LLM API settings
- Validation thresholds
- ASP solver configuration
- Logging settings
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM providers and models."""

    provider: Literal["anthropic", "openai"] = Field(
        default="anthropic", description="Primary LLM provider to use"
    )
    model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Model identifier for the chosen provider",
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""),
        description="API key for LLM provider",
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperature for LLM sampling"
    )
    max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens in LLM response")


class ValidationConfig(BaseModel):
    """Configuration for validation thresholds across stratified layers."""

    consistency_check: bool = Field(
        default=True, description="Enable consistency checking for ASP programs"
    )
    confidence_threshold_constitutional: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for constitutional layer (never auto-modify)",
    )
    confidence_threshold_strategic: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Confidence threshold for strategic layer"
    )
    confidence_threshold_tactical: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence threshold for tactical layer"
    )
    confidence_threshold_operational: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Confidence threshold for operational layer"
    )


class ASPConfig(BaseModel):
    """Configuration for Answer Set Programming solver (Clingo)."""

    programs_dir: str = Field(
        default="programs", description="Directory containing ASP program files"
    )
    max_answer_sets: int = Field(
        default=10, gt=0, description="Maximum number of answer sets to compute"
    )
    optimization: bool = Field(
        default=True, description="Enable optimization in ASP solving"
    )
    stats: bool = Field(
        default=False, description="Enable statistics output from Clingo (for debugging)"
    )

    @property
    def programs_path(self) -> Path:
        """Get absolute path to programs directory."""
        return Path(self.programs_dir).resolve()


class LogConfig(BaseModel):
    """Configuration for logging system."""

    level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>",
        description="Log message format",
    )
    rotation: str = Field(default="100 MB", description="Log file rotation size")
    retention: str = Field(default="1 month", description="Log file retention period")


class Config(BaseModel):
    """Main configuration object for LOFT system."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    asp: ASPConfig = Field(default_factory=ASPConfig)
    logging: LogConfig = Field(default_factory=LogConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "anthropic"),
                model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022"),
                api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            ),
            validation=ValidationConfig(
                confidence_threshold_strategic=float(
                    os.getenv("CONFIDENCE_THRESHOLD_STRATEGIC", "0.9")
                ),
                confidence_threshold_tactical=float(
                    os.getenv("CONFIDENCE_THRESHOLD_TACTICAL", "0.8")
                ),
                confidence_threshold_operational=float(
                    os.getenv("CONFIDENCE_THRESHOLD_OPERATIONAL", "0.6")
                ),
            ),
            asp=ASPConfig(
                programs_dir=os.getenv("ASP_PROGRAMS_DIR", "programs"),
                max_answer_sets=int(os.getenv("ASP_MAX_ANSWER_SETS", "10")),
            ),
            logging=LogConfig(level=os.getenv("LOG_LEVEL", "INFO")),
        )


# Global configuration instance
# This can be imported throughout the codebase
config = Config.from_env()
