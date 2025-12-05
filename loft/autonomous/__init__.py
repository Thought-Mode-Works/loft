"""
Autonomous Test Harness for Long-Running Experiments.

This module provides infrastructure for running long-duration (4+ hours)
autonomous test experiments with meta-reasoning integration, checkpointing,
Docker deployment, and Slack notifications.

Main Components:
- AutonomousTestRunner: Main orchestrator for autonomous runs
- AutonomousRunConfig: Configuration for runs
- MetaReasoningOrchestrator: Coordinates meta-reasoning components
- PersistenceManager: Handles state and checkpoint storage
- NotificationManager: Sends Slack/webhook notifications
- HealthServer: HTTP health endpoint for Docker

Quick Start:
    from loft.autonomous import AutonomousTestRunner, AutonomousRunConfig

    config = AutonomousRunConfig(
        max_duration_hours=4.0,
        dataset_paths=["datasets/contracts/"],
    )

    runner = AutonomousTestRunner(config)
    result = runner.start()

CLI Usage:
    # Start new run
    loft-autonomous start --dataset datasets/contracts/ --duration 4h

    # Resume from checkpoint
    loft-autonomous resume --checkpoint data/autonomous_runs/run_001/checkpoints/latest.json

    # Check status
    loft-autonomous status --run-id run_001
"""

from loft.autonomous.config import (
    AutonomousRunConfig,
    HealthConfig,
    MetaReasoningConfig,
    NotificationConfig,
    SafetyConfig,
)
from loft.autonomous.health import HealthServer, HealthStatus, create_health_server
from loft.autonomous.meta_integration import (
    FailureAnalysisReport,
    ImprovementCycleResult,
    MetaReasoningOrchestrator,
    create_orchestrator_from_config,
)
from loft.autonomous.notifications import (
    NotificationManager,
    NotificationMessage,
    create_notification_manager,
)
from loft.autonomous.persistence import (
    PersistenceManager,
    create_persistence_manager,
)
from loft.autonomous.runner import AutonomousTestRunner
from loft.autonomous.schemas import (
    CycleResult,
    CycleStatus,
    MetaReasoningState,
    RunCheckpoint,
    RunMetrics,
    RunProgress,
    RunResult,
    RunState,
    RunStatus,
)

__all__ = [
    # Runner
    "AutonomousTestRunner",
    # Config
    "AutonomousRunConfig",
    "HealthConfig",
    "MetaReasoningConfig",
    "NotificationConfig",
    "SafetyConfig",
    # Schemas
    "CycleResult",
    "CycleStatus",
    "MetaReasoningState",
    "RunCheckpoint",
    "RunMetrics",
    "RunProgress",
    "RunResult",
    "RunState",
    "RunStatus",
    # Meta Integration
    "FailureAnalysisReport",
    "ImprovementCycleResult",
    "MetaReasoningOrchestrator",
    "create_orchestrator_from_config",
    # Persistence
    "PersistenceManager",
    "create_persistence_manager",
    # Notifications
    "NotificationManager",
    "NotificationMessage",
    "create_notification_manager",
    # Health
    "HealthServer",
    "HealthStatus",
    "create_health_server",
]
