"""
Schemas for batch learning operations.

Provides dataclasses for batch processing results, checkpoints,
progress tracking, and metrics collection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class BatchStatus(Enum):
    """Status of a batch processing job."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CaseStatus(Enum):
    """Status of individual case processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CaseResult:
    """Result of processing a single test case."""

    case_id: str
    status: CaseStatus
    processed_at: datetime
    processing_time_ms: float

    # Rule generation
    rules_generated: int = 0
    rules_accepted: int = 0
    rules_rejected: int = 0

    # Accuracy
    prediction_correct: Optional[bool] = None
    confidence: float = 0.0

    # Details
    generated_rule_ids: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "case_id": self.case_id,
            "status": self.status.value,
            "processed_at": self.processed_at.isoformat(),
            "processing_time_ms": self.processing_time_ms,
            "rules_generated": self.rules_generated,
            "rules_accepted": self.rules_accepted,
            "rules_rejected": self.rules_rejected,
            "prediction_correct": self.prediction_correct,
            "confidence": self.confidence,
            "generated_rule_ids": self.generated_rule_ids,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseResult":
        """Create from dictionary."""
        return cls(
            case_id=data["case_id"],
            status=CaseStatus(data["status"]),
            processed_at=datetime.fromisoformat(data["processed_at"]),
            processing_time_ms=data["processing_time_ms"],
            rules_generated=data.get("rules_generated", 0),
            rules_accepted=data.get("rules_accepted", 0),
            rules_rejected=data.get("rules_rejected", 0),
            prediction_correct=data.get("prediction_correct"),
            confidence=data.get("confidence", 0.0),
            generated_rule_ids=data.get("generated_rule_ids", []),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BatchProgress:
    """Real-time progress tracking for batch processing."""

    batch_id: str
    total_cases: int
    processed_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    skipped_cases: int = 0

    # Rule accumulation
    total_rules_generated: int = 0
    total_rules_accepted: int = 0
    total_rules_rejected: int = 0

    # Accuracy tracking
    correct_predictions: int = 0
    incorrect_predictions: int = 0

    # Timing
    started_at: Optional[datetime] = None
    last_update: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None

    # Current state
    current_case_id: Optional[str] = None
    status: BatchStatus = BatchStatus.PENDING

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_cases == 0:
            return 0.0
        return (self.processed_cases / self.total_cases) * 100

    @property
    def current_accuracy(self) -> float:
        """Calculate current prediction accuracy."""
        total = self.correct_predictions + self.incorrect_predictions
        if total == 0:
            return 0.0
        return self.correct_predictions / total

    @property
    def acceptance_rate(self) -> float:
        """Calculate rule acceptance rate."""
        if self.total_rules_generated == 0:
            return 0.0
        return self.total_rules_accepted / self.total_rules_generated

    @property
    def elapsed_time_seconds(self) -> float:
        """Calculate elapsed time in seconds."""
        if self.started_at is None:
            return 0.0
        end = self.last_update or datetime.now()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_id": self.batch_id,
            "total_cases": self.total_cases,
            "processed_cases": self.processed_cases,
            "successful_cases": self.successful_cases,
            "failed_cases": self.failed_cases,
            "skipped_cases": self.skipped_cases,
            "total_rules_generated": self.total_rules_generated,
            "total_rules_accepted": self.total_rules_accepted,
            "total_rules_rejected": self.total_rules_rejected,
            "correct_predictions": self.correct_predictions,
            "incorrect_predictions": self.incorrect_predictions,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "estimated_completion": (
                self.estimated_completion.isoformat()
                if self.estimated_completion
                else None
            ),
            "current_case_id": self.current_case_id,
            "status": self.status.value,
            "completion_percentage": self.completion_percentage,
            "current_accuracy": self.current_accuracy,
            "acceptance_rate": self.acceptance_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchProgress":
        """Create from dictionary."""
        return cls(
            batch_id=data["batch_id"],
            total_cases=data["total_cases"],
            processed_cases=data.get("processed_cases", 0),
            successful_cases=data.get("successful_cases", 0),
            failed_cases=data.get("failed_cases", 0),
            skipped_cases=data.get("skipped_cases", 0),
            total_rules_generated=data.get("total_rules_generated", 0),
            total_rules_accepted=data.get("total_rules_accepted", 0),
            total_rules_rejected=data.get("total_rules_rejected", 0),
            correct_predictions=data.get("correct_predictions", 0),
            incorrect_predictions=data.get("incorrect_predictions", 0),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            last_update=(
                datetime.fromisoformat(data["last_update"])
                if data.get("last_update")
                else None
            ),
            estimated_completion=(
                datetime.fromisoformat(data["estimated_completion"])
                if data.get("estimated_completion")
                else None
            ),
            current_case_id=data.get("current_case_id"),
            status=BatchStatus(data.get("status", "pending")),
        )


@dataclass
class BatchCheckpoint:
    """Checkpoint for resumable batch processing."""

    batch_id: str
    checkpoint_id: str
    created_at: datetime

    # Progress state
    processed_case_ids: List[str] = field(default_factory=list)
    pending_case_ids: List[str] = field(default_factory=list)

    # Accumulated results
    case_results: List[CaseResult] = field(default_factory=list)
    accumulated_rule_ids: List[str] = field(default_factory=list)

    # Progress snapshot
    progress: Optional[BatchProgress] = None

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_id": self.batch_id,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at.isoformat(),
            "processed_case_ids": self.processed_case_ids,
            "pending_case_ids": self.pending_case_ids,
            "case_results": [r.to_dict() for r in self.case_results],
            "accumulated_rule_ids": self.accumulated_rule_ids,
            "progress": self.progress.to_dict() if self.progress else None,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchCheckpoint":
        """Create from dictionary."""
        return cls(
            batch_id=data["batch_id"],
            checkpoint_id=data["checkpoint_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            processed_case_ids=data.get("processed_case_ids", []),
            pending_case_ids=data.get("pending_case_ids", []),
            case_results=[
                CaseResult.from_dict(r) for r in data.get("case_results", [])
            ],
            accumulated_rule_ids=data.get("accumulated_rule_ids", []),
            progress=(
                BatchProgress.from_dict(data["progress"])
                if data.get("progress")
                else None
            ),
            config=data.get("config", {}),
        )


@dataclass
class BatchMetrics:
    """Comprehensive metrics for a batch processing run."""

    batch_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Volume metrics
    cases_processed: int = 0
    rules_generated: int = 0
    rules_accepted: int = 0
    rules_rejected: int = 0

    # Performance metrics
    total_processing_time_ms: float = 0.0
    avg_case_time_ms: float = 0.0
    avg_rule_generation_time_ms: float = 0.0
    peak_memory_mb: float = 0.0

    # Quality metrics
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    accuracy_improvement: float = 0.0
    consistency_score: float = 1.0
    new_contradictions: int = 0

    # Per-domain breakdown
    domain_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Rule evolution
    rules_by_evolution_method: Dict[str, int] = field(default_factory=dict)
    rules_by_stratification: Dict[str, int] = field(default_factory=dict)

    # Error tracking
    total_errors: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_id": self.batch_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "cases_processed": self.cases_processed,
            "rules_generated": self.rules_generated,
            "rules_accepted": self.rules_accepted,
            "rules_rejected": self.rules_rejected,
            "total_processing_time_ms": self.total_processing_time_ms,
            "avg_case_time_ms": self.avg_case_time_ms,
            "avg_rule_generation_time_ms": self.avg_rule_generation_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "accuracy_before": self.accuracy_before,
            "accuracy_after": self.accuracy_after,
            "accuracy_improvement": self.accuracy_improvement,
            "consistency_score": self.consistency_score,
            "new_contradictions": self.new_contradictions,
            "domain_metrics": self.domain_metrics,
            "rules_by_evolution_method": self.rules_by_evolution_method,
            "rules_by_stratification": self.rules_by_stratification,
            "total_errors": self.total_errors,
            "error_types": self.error_types,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchMetrics":
        """Create from dictionary."""
        return cls(
            batch_id=data["batch_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            cases_processed=data.get("cases_processed", 0),
            rules_generated=data.get("rules_generated", 0),
            rules_accepted=data.get("rules_accepted", 0),
            rules_rejected=data.get("rules_rejected", 0),
            total_processing_time_ms=data.get("total_processing_time_ms", 0.0),
            avg_case_time_ms=data.get("avg_case_time_ms", 0.0),
            avg_rule_generation_time_ms=data.get("avg_rule_generation_time_ms", 0.0),
            peak_memory_mb=data.get("peak_memory_mb", 0.0),
            accuracy_before=data.get("accuracy_before", 0.0),
            accuracy_after=data.get("accuracy_after", 0.0),
            accuracy_improvement=data.get("accuracy_improvement", 0.0),
            consistency_score=data.get("consistency_score", 1.0),
            new_contradictions=data.get("new_contradictions", 0),
            domain_metrics=data.get("domain_metrics", {}),
            rules_by_evolution_method=data.get("rules_by_evolution_method", {}),
            rules_by_stratification=data.get("rules_by_stratification", {}),
            total_errors=data.get("total_errors", 0),
            error_types=data.get("error_types", {}),
        )


@dataclass
class BatchResult:
    """Complete result of a batch processing run."""

    batch_id: str
    status: BatchStatus
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Results
    case_results: List[CaseResult] = field(default_factory=list)
    accumulated_rule_ids: List[str] = field(default_factory=list)

    # Summary metrics
    metrics: Optional[BatchMetrics] = None

    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)

    # Checkpoints created
    checkpoint_ids: List[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        """Count of successfully processed cases."""
        return sum(1 for r in self.case_results if r.status == CaseStatus.SUCCESS)

    @property
    def failure_count(self) -> int:
        """Count of failed cases."""
        return sum(1 for r in self.case_results if r.status == CaseStatus.FAILED)

    @property
    def total_rules_accepted(self) -> int:
        """Total rules accepted across all cases."""
        return sum(r.rules_accepted for r in self.case_results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_id": self.batch_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "case_results": [r.to_dict() for r in self.case_results],
            "accumulated_rule_ids": self.accumulated_rule_ids,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "config": self.config,
            "checkpoint_ids": self.checkpoint_ids,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_rules_accepted": self.total_rules_accepted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchResult":
        """Create from dictionary."""
        return cls(
            batch_id=data["batch_id"],
            status=BatchStatus(data["status"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            case_results=[
                CaseResult.from_dict(r) for r in data.get("case_results", [])
            ],
            accumulated_rule_ids=data.get("accumulated_rule_ids", []),
            metrics=(
                BatchMetrics.from_dict(data["metrics"]) if data.get("metrics") else None
            ),
            config=data.get("config", {}),
            checkpoint_ids=data.get("checkpoint_ids", []),
        )


@dataclass
class BatchConfig:
    """Configuration for batch learning runs."""

    # Processing limits
    max_cases: Optional[int] = None
    max_rules_per_case: int = 3
    max_total_rules: int = 200

    # Validation thresholds
    validation_threshold: float = 0.8
    min_confidence: float = 0.6

    # Checkpointing
    checkpoint_interval: int = 10
    checkpoint_dir: str = "data/batch_checkpoints"

    # Error handling
    max_errors_per_case: int = 3
    continue_on_error: bool = True
    max_consecutive_errors: int = 10

    # Performance
    enable_caching: bool = True
    parallel_validation: bool = False

    # Dialectical refinement
    use_dialectical: bool = True
    max_dialectical_rounds: int = 3
    dialectical_convergence_threshold: float = 0.85

    # Cross-domain transfer
    enable_transfer_learning: bool = True
    transfer_source_domains: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_cases": self.max_cases,
            "max_rules_per_case": self.max_rules_per_case,
            "max_total_rules": self.max_total_rules,
            "validation_threshold": self.validation_threshold,
            "min_confidence": self.min_confidence,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "max_errors_per_case": self.max_errors_per_case,
            "continue_on_error": self.continue_on_error,
            "max_consecutive_errors": self.max_consecutive_errors,
            "enable_caching": self.enable_caching,
            "parallel_validation": self.parallel_validation,
            "use_dialectical": self.use_dialectical,
            "max_dialectical_rounds": self.max_dialectical_rounds,
            "dialectical_convergence_threshold": self.dialectical_convergence_threshold,
            "enable_transfer_learning": self.enable_transfer_learning,
            "transfer_source_domains": self.transfer_source_domains,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchConfig":
        """Create from dictionary."""
        return cls(
            max_cases=data.get("max_cases"),
            max_rules_per_case=data.get("max_rules_per_case", 3),
            max_total_rules=data.get("max_total_rules", 200),
            validation_threshold=data.get("validation_threshold", 0.8),
            min_confidence=data.get("min_confidence", 0.6),
            checkpoint_interval=data.get("checkpoint_interval", 10),
            checkpoint_dir=data.get("checkpoint_dir", "data/batch_checkpoints"),
            max_errors_per_case=data.get("max_errors_per_case", 3),
            continue_on_error=data.get("continue_on_error", True),
            max_consecutive_errors=data.get("max_consecutive_errors", 10),
            enable_caching=data.get("enable_caching", True),
            parallel_validation=data.get("parallel_validation", False),
            use_dialectical=data.get("use_dialectical", True),
            max_dialectical_rounds=data.get("max_dialectical_rounds", 3),
            dialectical_convergence_threshold=data.get(
                "dialectical_convergence_threshold", 0.85
            ),
            enable_transfer_learning=data.get("enable_transfer_learning", True),
            transfer_source_domains=data.get("transfer_source_domains", []),
        )
