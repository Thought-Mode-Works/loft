"""
Logging infrastructure for LOFT system.

Provides structured logging, decorators for tracking, and analysis tools.
"""

from .logger import (
    LOFTLogger,
    get_loft_logger,
    initialize_logging,
    get_logger_instance,
    log_llm_interaction,
    log_symbolic_operation,
    log_validation_result,
    log_meta_reasoning,
)

from .decorators import (
    track_llm_call,
    track_symbolic_operation,
    track_validation,
    track_meta_reasoning,
    performance_monitor,
)

from .analysis import (
    LogAnalyzer,
    LogEntry,
    search_logs_cli,
)

__all__ = [
    # Logger
    "LOFTLogger",
    "get_loft_logger",
    "initialize_logging",
    "get_logger_instance",
    "log_llm_interaction",
    "log_symbolic_operation",
    "log_validation_result",
    "log_meta_reasoning",
    # Decorators
    "track_llm_call",
    "track_symbolic_operation",
    "track_validation",
    "track_meta_reasoning",
    "performance_monitor",
    # Analysis
    "LogAnalyzer",
    "LogEntry",
    "search_logs_cli",
]
