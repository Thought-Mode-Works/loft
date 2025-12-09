"""
Enhanced logging infrastructure for LOFT system.

Provides structured logging with:
- Component-specific log levels
- LLM interaction tracking
- Symbolic core monitoring
- Validation logging
- Meta-reasoning tracking
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
from datetime import datetime


class LOFTLogger:
    """
    Enhanced logger for LOFT system with component-specific features.

    Features:
    - Structured logging with context
    - Component-specific log levels
    - Log rotation and retention
    - Sampling for high-volume events
    - Privacy filtering
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        rotation: str = "100 MB",
        retention: str = "1 month",
        level: str = "INFO",
        format_string: Optional[str] = None,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        sampling_rate: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the LOFT logger.

        Args:
            log_dir: Directory for log files
            rotation: When to rotate log files
            retention: How long to keep old logs
            level: Default log level
            format_string: Custom format string
            enable_file_logging: Whether to log to files
            enable_console_logging: Whether to log to console
            sampling_rate: Sampling rates by component (e.g., {"llm": 0.1})
        """
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.rotation = rotation
        self.retention = retention
        self.level = level
        self.sampling_rate = sampling_rate or {}

        # Default format
        self.format_string = format_string or (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[component]}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        # Remove default handler
        logger.remove()

        # Add console handler if enabled
        if enable_console_logging:
            logger.add(
                sys.stderr,
                format=self.format_string,
                level=level,
                colorize=True,
            )

        # Add file handlers if enabled
        if enable_file_logging:
            self._add_file_handlers()

        # Set default context
        self.logger = logger.bind(component="system")

    def _add_file_handlers(self) -> None:
        """Add file handlers for different log levels and components."""

        # Main log file
        logger.add(
            self.log_dir / "loft.log",
            format=self.format_string,
            level=self.level,
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",
        )

        # LLM interactions log (always TRACE level for full capture)
        logger.add(
            self.log_dir / "llm_interactions.log",
            format=self.format_string,
            level="TRACE",
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",
            filter=lambda record: record["extra"].get("component") == "llm",
        )

        # Symbolic core operations log
        logger.add(
            self.log_dir / "symbolic_core.log",
            format=self.format_string,
            level="DEBUG",
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",
            filter=lambda record: record["extra"].get("component") == "symbolic_core",
        )

        # Validation log
        logger.add(
            self.log_dir / "validation.log",
            format=self.format_string,
            level="INFO",
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",
            filter=lambda record: record["extra"].get("component") == "validation",
        )

        # Meta-reasoning log
        logger.add(
            self.log_dir / "meta_reasoning.log",
            format=self.format_string,
            level="INFO",
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",
            filter=lambda record: record["extra"].get("component") == "meta_reasoning",
        )

        # Error log (ERROR and above only)
        logger.add(
            self.log_dir / "errors.log",
            format=self.format_string,
            level="ERROR",
            rotation=self.rotation,
            retention=self.retention,
            compression="zip",
        )

    def get_logger(self, component: str) -> Any:
        """
        Get a logger bound to a specific component.

        Args:
            component: Component name (e.g., "llm", "symbolic_core", "validation")

        Returns:
            Logger instance bound to the component
        """
        return logger.bind(component=component)

    def should_sample(self, component: str) -> bool:
        """
        Determine if this log entry should be sampled.

        Args:
            component: Component name

        Returns:
            True if should log, False if should skip (sampled out)
        """
        if component not in self.sampling_rate:
            return True

        import random

        return random.random() < self.sampling_rate[component]


def get_loft_logger(component: str = "system") -> Any:
    """
    Get a component-specific logger.

    This is a convenience function for getting loggers throughout the codebase.

    Args:
        component: Component name

    Returns:
        Logger instance

    Example:
        >>> log = get_loft_logger("llm")
        >>> log.info("Processing LLM request", model="claude-3", tokens=100)
    """
    return logger.bind(component=component)


def log_llm_interaction(
    logger_instance: Any, event: str, model: str, **kwargs: Any
) -> None:
    """
    Log an LLM interaction with structured data.

    Args:
        logger_instance: Logger to use
        event: Event type (e.g., "request", "response")
        model: Model name
        **kwargs: Additional context (prompt, response, tokens, etc.)
    """
    logger_instance.trace(
        f"LLM {event}",
        event=event,
        model=model,
        timestamp=datetime.utcnow().isoformat(),
        **kwargs,
    )


def log_symbolic_operation(logger_instance: Any, operation: str, **kwargs: Any) -> None:
    """
    Log a symbolic core operation.

    Args:
        logger_instance: Logger to use
        operation: Operation type (e.g., "rule_add", "rule_modify", "query")
        **kwargs: Additional context
    """
    logger_instance.debug(
        f"Symbolic operation: {operation}",
        operation=operation,
        timestamp=datetime.utcnow().isoformat(),
        **kwargs,
    )


def log_validation_result(
    logger_instance: Any, validation_type: str, result: bool, **kwargs: Any
) -> None:
    """
    Log a validation result.

    Args:
        logger_instance: Logger to use
        validation_type: Type of validation
        result: Validation result (pass/fail)
        **kwargs: Additional context
    """
    level = "info" if result else "warning"
    getattr(logger_instance, level)(
        f"Validation {validation_type}: {'PASS' if result else 'FAIL'}",
        validation_type=validation_type,
        result=result,
        timestamp=datetime.utcnow().isoformat(),
        **kwargs,
    )


def log_meta_reasoning(logger_instance: Any, event: str, **kwargs: Any) -> None:
    """
    Log a meta-reasoning event.

    Args:
        logger_instance: Logger to use
        event: Event type (e.g., "self_assessment", "strategy_change")
        **kwargs: Additional context
    """
    logger_instance.info(
        f"Meta-reasoning: {event}",
        event=event,
        timestamp=datetime.utcnow().isoformat(),
        **kwargs,
    )


# Global logger instance
_loft_logger: Optional[LOFTLogger] = None


def initialize_logging(
    log_dir: Optional[Path] = None, level: str = "INFO", **kwargs: Any
) -> LOFTLogger:
    """
    Initialize the LOFT logging system.

    This should be called once at application startup.

    Args:
        log_dir: Directory for log files
        level: Default log level
        **kwargs: Additional configuration for LOFTLogger

    Returns:
        Configured LOFTLogger instance
    """
    global _loft_logger
    _loft_logger = LOFTLogger(log_dir=log_dir, level=level, **kwargs)
    return _loft_logger


def get_logger_instance() -> Optional[LOFTLogger]:
    """Get the global logger instance."""
    return _loft_logger
