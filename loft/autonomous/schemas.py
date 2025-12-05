"""
Data Schemas for Autonomous Test Harness.

This module defines the core data structures for autonomous long-running
test experiments, including run state, checkpoints, and results.

Schemas:
- RunStatus: Enum for run lifecycle states
- CycleStatus: Enum for improvement cycle states
- RunState: Current state of an autonomous run
- RunCheckpoint: Serializable checkpoint for resume capability
- RunResult: Final result of an autonomous run
- CycleResult: Result of a single improvement cycle
- RunProgress: Real-time progress tracking
- RunMetrics: Aggregate metrics for a run
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RunStatus(str, Enum):
    """Status of an autonomous run."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CycleStatus(str, Enum):
    """Status of an improvement cycle."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    IMPROVING = "improving"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CycleResult:
    """Result of a single improvement cycle.

    Attributes:
        cycle_number: Sequential cycle number
        status: Completion status
        started_at: When the cycle started
        completed_at: When the cycle ended
        cases_processed: Number of cases processed this cycle
        improvements_applied: Number of improvements applied
        accuracy_before: Accuracy before this cycle
        accuracy_after: Accuracy after this cycle
        rules_generated: Number of new rules generated
        rules_promoted: Number of rules promoted to production
        failure_patterns: Identified failure patterns
        prompt_changes: Number of prompt modifications
        strategy_changes: Number of strategy modifications
        error_message: Error message if failed
    """

    cycle_number: int
    status: CycleStatus = CycleStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cases_processed: int = 0
    improvements_applied: int = 0
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    rules_generated: int = 0
    rules_promoted: int = 0
    failure_patterns: List[str] = field(default_factory=list)
    prompt_changes: int = 0
    strategy_changes: int = 0
    error_message: Optional[str] = None

    @property
    def accuracy_delta(self) -> float:
        """Calculate accuracy change from this cycle."""
        return self.accuracy_after - self.accuracy_before

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate cycle duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "cycle_number": self.cycle_number,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "cases_processed": self.cases_processed,
            "improvements_applied": self.improvements_applied,
            "accuracy_before": self.accuracy_before,
            "accuracy_after": self.accuracy_after,
            "rules_generated": self.rules_generated,
            "rules_promoted": self.rules_promoted,
            "failure_patterns": self.failure_patterns,
            "prompt_changes": self.prompt_changes,
            "strategy_changes": self.strategy_changes,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleResult":
        """Deserialize from dictionary."""
        return cls(
            cycle_number=data["cycle_number"],
            status=CycleStatus(data["status"]),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            cases_processed=data.get("cases_processed", 0),
            improvements_applied=data.get("improvements_applied", 0),
            accuracy_before=data.get("accuracy_before", 0.0),
            accuracy_after=data.get("accuracy_after", 0.0),
            rules_generated=data.get("rules_generated", 0),
            rules_promoted=data.get("rules_promoted", 0),
            failure_patterns=data.get("failure_patterns", []),
            prompt_changes=data.get("prompt_changes", 0),
            strategy_changes=data.get("strategy_changes", 0),
            error_message=data.get("error_message"),
        )


@dataclass
class RunProgress:
    """Real-time progress tracking for an autonomous run.

    Attributes:
        total_cases: Total cases in the dataset
        cases_processed: Cases completed
        cases_successful: Cases that succeeded
        cases_failed: Cases that failed
        current_case_id: Currently processing case
        current_cycle: Current improvement cycle number
        total_cycles: Total improvement cycles completed
        elapsed_seconds: Time elapsed since start
        estimated_remaining_seconds: Estimated time remaining
    """

    total_cases: int = 0
    cases_processed: int = 0
    cases_successful: int = 0
    cases_failed: int = 0
    current_case_id: Optional[str] = None
    current_cycle: int = 0
    total_cycles: int = 0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_cases == 0:
            return 0.0
        return (self.cases_processed / self.total_cases) * 100

    @property
    def current_accuracy(self) -> float:
        """Calculate current accuracy."""
        if self.cases_processed == 0:
            return 0.0
        return self.cases_successful / self.cases_processed

    @property
    def cases_per_hour(self) -> float:
        """Calculate processing rate."""
        if self.elapsed_seconds == 0:
            return 0.0
        return (self.cases_processed / self.elapsed_seconds) * 3600

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_cases": self.total_cases,
            "cases_processed": self.cases_processed,
            "cases_successful": self.cases_successful,
            "cases_failed": self.cases_failed,
            "current_case_id": self.current_case_id,
            "current_cycle": self.current_cycle,
            "total_cycles": self.total_cycles,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "completion_percentage": self.completion_percentage,
            "current_accuracy": self.current_accuracy,
            "cases_per_hour": self.cases_per_hour,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunProgress":
        """Deserialize from dictionary."""
        return cls(
            total_cases=data.get("total_cases", 0),
            cases_processed=data.get("cases_processed", 0),
            cases_successful=data.get("cases_successful", 0),
            cases_failed=data.get("cases_failed", 0),
            current_case_id=data.get("current_case_id"),
            current_cycle=data.get("current_cycle", 0),
            total_cycles=data.get("total_cycles", 0),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
            estimated_remaining_seconds=data.get("estimated_remaining_seconds"),
        )


@dataclass
class RunMetrics:
    """Aggregate metrics for an autonomous run.

    Attributes:
        overall_accuracy: Final accuracy across all cases
        accuracy_by_domain: Accuracy broken down by domain
        accuracy_timeline: Accuracy at each checkpoint
        rules_generated_total: Total rules generated
        rules_promoted_total: Total rules promoted
        rules_rolled_back: Rules that were rolled back
        improvement_cycles_completed: Number of improvement cycles
        average_cycle_duration_seconds: Average cycle duration
        llm_calls_total: Total LLM API calls
        llm_tokens_used: Total tokens consumed
        llm_cost_estimate: Estimated API cost
        failure_patterns_identified: Unique failure patterns found
        strategy_changes_total: Total strategy modifications
        prompt_changes_total: Total prompt modifications
    """

    overall_accuracy: float = 0.0
    accuracy_by_domain: Dict[str, float] = field(default_factory=dict)
    accuracy_timeline: List[Dict[str, Any]] = field(default_factory=list)
    rules_generated_total: int = 0
    rules_promoted_total: int = 0
    rules_rolled_back: int = 0
    improvement_cycles_completed: int = 0
    average_cycle_duration_seconds: float = 0.0
    llm_calls_total: int = 0
    llm_tokens_used: int = 0
    llm_cost_estimate: float = 0.0
    failure_patterns_identified: List[str] = field(default_factory=list)
    strategy_changes_total: int = 0
    prompt_changes_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "overall_accuracy": self.overall_accuracy,
            "accuracy_by_domain": self.accuracy_by_domain,
            "accuracy_timeline": self.accuracy_timeline,
            "rules_generated_total": self.rules_generated_total,
            "rules_promoted_total": self.rules_promoted_total,
            "rules_rolled_back": self.rules_rolled_back,
            "improvement_cycles_completed": self.improvement_cycles_completed,
            "average_cycle_duration_seconds": self.average_cycle_duration_seconds,
            "llm_calls_total": self.llm_calls_total,
            "llm_tokens_used": self.llm_tokens_used,
            "llm_cost_estimate": self.llm_cost_estimate,
            "failure_patterns_identified": self.failure_patterns_identified,
            "strategy_changes_total": self.strategy_changes_total,
            "prompt_changes_total": self.prompt_changes_total,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunMetrics":
        """Deserialize from dictionary."""
        return cls(
            overall_accuracy=data.get("overall_accuracy", 0.0),
            accuracy_by_domain=data.get("accuracy_by_domain", {}),
            accuracy_timeline=data.get("accuracy_timeline", []),
            rules_generated_total=data.get("rules_generated_total", 0),
            rules_promoted_total=data.get("rules_promoted_total", 0),
            rules_rolled_back=data.get("rules_rolled_back", 0),
            improvement_cycles_completed=data.get("improvement_cycles_completed", 0),
            average_cycle_duration_seconds=data.get("average_cycle_duration_seconds", 0.0),
            llm_calls_total=data.get("llm_calls_total", 0),
            llm_tokens_used=data.get("llm_tokens_used", 0),
            llm_cost_estimate=data.get("llm_cost_estimate", 0.0),
            failure_patterns_identified=data.get("failure_patterns_identified", []),
            strategy_changes_total=data.get("strategy_changes_total", 0),
            prompt_changes_total=data.get("prompt_changes_total", 0),
        )


@dataclass
class RunState:
    """Current state of an autonomous run.

    This is the primary state object updated during execution and
    used for monitoring and health checks.

    Attributes:
        run_id: Unique identifier for this run
        status: Current run status
        started_at: When the run started
        last_updated: Last state update time
        progress: Current progress metrics
        current_cycle_result: In-progress cycle result
        error_message: Error message if failed
        shutdown_requested: Whether graceful shutdown was requested
    """

    run_id: str
    status: RunStatus = RunStatus.PENDING
    started_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    progress: RunProgress = field(default_factory=RunProgress)
    current_cycle_result: Optional[CycleResult] = None
    error_message: Optional[str] = None
    shutdown_requested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_updated": (self.last_updated.isoformat() if self.last_updated else None),
            "progress": self.progress.to_dict(),
            "current_cycle_result": (
                self.current_cycle_result.to_dict() if self.current_cycle_result else None
            ),
            "error_message": self.error_message,
            "shutdown_requested": self.shutdown_requested,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunState":
        """Deserialize from dictionary."""
        return cls(
            run_id=data["run_id"],
            status=RunStatus(data["status"]),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            last_updated=(
                datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None
            ),
            progress=RunProgress.from_dict(data.get("progress", {})),
            current_cycle_result=(
                CycleResult.from_dict(data["current_cycle_result"])
                if data.get("current_cycle_result")
                else None
            ),
            error_message=data.get("error_message"),
            shutdown_requested=data.get("shutdown_requested", False),
        )


@dataclass
class MetaReasoningState:
    """Serializable state of meta-reasoning components.

    Captures the state of all meta-reasoning components for
    checkpoint/resume capability.

    Attributes:
        improver_state: AutonomousImprover state
        optimizer_state: PromptOptimizer state
        analyzer_state: FailureAnalyzer state
        selector_state: StrategySelector state
        observer_state: ReasoningObserver state
    """

    improver_state: Dict[str, Any] = field(default_factory=dict)
    optimizer_state: Dict[str, Any] = field(default_factory=dict)
    analyzer_state: Dict[str, Any] = field(default_factory=dict)
    selector_state: Dict[str, Any] = field(default_factory=dict)
    observer_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "improver_state": self.improver_state,
            "optimizer_state": self.optimizer_state,
            "analyzer_state": self.analyzer_state,
            "selector_state": self.selector_state,
            "observer_state": self.observer_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaReasoningState":
        """Deserialize from dictionary."""
        return cls(
            improver_state=data.get("improver_state", {}),
            optimizer_state=data.get("optimizer_state", {}),
            analyzer_state=data.get("analyzer_state", {}),
            selector_state=data.get("selector_state", {}),
            observer_state=data.get("observer_state", {}),
        )


@dataclass
class RunCheckpoint:
    """Complete checkpoint for resuming an autonomous run.

    Contains all state necessary to resume a run from where it left off.

    Attributes:
        checkpoint_number: Sequential checkpoint number
        created_at: When checkpoint was created
        run_id: Run identifier
        run_state: Current run state
        config_snapshot: Configuration at checkpoint time
        meta_reasoning_state: State of meta-reasoning components
        cycle_results: All completed cycle results
        accumulated_rules: Rules generated so far
        case_results: Results for processed cases
        batch_checkpoint: Underlying batch harness checkpoint data
    """

    checkpoint_number: int
    created_at: datetime
    run_id: str
    run_state: RunState
    config_snapshot: Dict[str, Any]
    meta_reasoning_state: MetaReasoningState
    cycle_results: List[CycleResult] = field(default_factory=list)
    accumulated_rules: List[Dict[str, Any]] = field(default_factory=list)
    case_results: List[Dict[str, Any]] = field(default_factory=list)
    batch_checkpoint: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "checkpoint_number": self.checkpoint_number,
            "created_at": self.created_at.isoformat(),
            "run_id": self.run_id,
            "run_state": self.run_state.to_dict(),
            "config_snapshot": self.config_snapshot,
            "meta_reasoning_state": self.meta_reasoning_state.to_dict(),
            "cycle_results": [cr.to_dict() for cr in self.cycle_results],
            "accumulated_rules": self.accumulated_rules,
            "case_results": self.case_results,
            "batch_checkpoint": self.batch_checkpoint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunCheckpoint":
        """Deserialize from dictionary."""
        return cls(
            checkpoint_number=data["checkpoint_number"],
            created_at=datetime.fromisoformat(data["created_at"]),
            run_id=data["run_id"],
            run_state=RunState.from_dict(data["run_state"]),
            config_snapshot=data.get("config_snapshot", {}),
            meta_reasoning_state=MetaReasoningState.from_dict(data.get("meta_reasoning_state", {})),
            cycle_results=[CycleResult.from_dict(cr) for cr in data.get("cycle_results", [])],
            accumulated_rules=data.get("accumulated_rules", []),
            case_results=data.get("case_results", []),
            batch_checkpoint=data.get("batch_checkpoint"),
        )


@dataclass
class RunResult:
    """Final result of a completed autonomous run.

    Attributes:
        run_id: Unique identifier
        status: Final status
        started_at: Start time
        completed_at: End time
        total_duration_seconds: Total run duration
        config_used: Configuration used for the run
        final_metrics: Aggregate metrics
        cycle_results: All improvement cycle results
        final_rules: Rules at end of run
        error_message: Error if failed
        checkpoint_path: Path to last checkpoint
    """

    run_id: str
    status: RunStatus
    started_at: datetime
    completed_at: datetime
    total_duration_seconds: float
    config_used: Dict[str, Any]
    final_metrics: RunMetrics
    cycle_results: List[CycleResult] = field(default_factory=list)
    final_rules: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    checkpoint_path: Optional[str] = None

    @property
    def duration_hours(self) -> float:
        """Get duration in hours."""
        return self.total_duration_seconds / 3600

    @property
    def was_successful(self) -> bool:
        """Check if run completed successfully."""
        return self.status == RunStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "total_duration_seconds": self.total_duration_seconds,
            "duration_hours": self.duration_hours,
            "config_used": self.config_used,
            "final_metrics": self.final_metrics.to_dict(),
            "cycle_results": [cr.to_dict() for cr in self.cycle_results],
            "final_rules": self.final_rules,
            "error_message": self.error_message,
            "checkpoint_path": self.checkpoint_path,
            "was_successful": self.was_successful,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunResult":
        """Deserialize from dictionary."""
        return cls(
            run_id=data["run_id"],
            status=RunStatus(data["status"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            total_duration_seconds=data["total_duration_seconds"],
            config_used=data.get("config_used", {}),
            final_metrics=RunMetrics.from_dict(data.get("final_metrics", {})),
            cycle_results=[CycleResult.from_dict(cr) for cr in data.get("cycle_results", [])],
            final_rules=data.get("final_rules", []),
            error_message=data.get("error_message"),
            checkpoint_path=data.get("checkpoint_path"),
        )
