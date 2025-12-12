"""
Model-specific Prompt Optimizer for Neural Ensemble.

This module implements prompt optimization tailored to specific models in the
heterogeneous neural ensemble. It provides A/B testing, performance tracking,
and prompt evolution based on failure analysis.

Part of Phase 6: Heterogeneous Neural Ensemble (Issue #193).

Key capabilities:
- Model-specific prompt templates for each ensemble member
- Automated prompt A/B testing framework
- Performance tracking per model per prompt variant
- Prompt evolution based on success metrics
- Integration with meta-reasoning for optimization suggestions
"""

from __future__ import annotations

import random
import statistics
import threading
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from loguru import logger


# =============================================================================
# Custom Exceptions
# =============================================================================


class PromptOptimizationError(Exception):
    """Custom exception for prompt optimization failures."""

    def __init__(
        self, message: str, model: Optional[str] = None, task_type: Optional[str] = None
    ):
        super().__init__(message)
        self.model = model
        self.task_type = task_type


class ABTestingError(Exception):
    """Custom exception for A/B testing failures."""

    def __init__(self, message: str, test_id: Optional[str] = None):
        super().__init__(message)
        self.test_id = test_id


# =============================================================================
# Enums
# =============================================================================


class ModelType(Enum):
    """Types of models in the ensemble."""

    LOGIC_GENERATOR = "logic_generator"
    CRITIC = "critic"
    TRANSLATOR = "translator"
    META_REASONER = "meta_reasoner"
    GENERAL = "general"


class PromptTaskType(Enum):
    """Types of tasks for prompt optimization."""

    ASP_GENERATION = "asp_generation"
    EDGE_CASE_DETECTION = "edge_case_detection"
    CONTRADICTION_DETECTION = "contradiction_detection"
    TRANSLATION_NL_TO_ASP = "translation_nl_to_asp"
    TRANSLATION_ASP_TO_NL = "translation_asp_to_nl"
    FAILURE_ANALYSIS = "failure_analysis"
    STRATEGY_RECOMMENDATION = "strategy_recommendation"


class PromptVariantStatus(Enum):
    """Status of a prompt variant in testing."""

    DRAFT = "draft"
    TESTING = "testing"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class ABTestStatus(Enum):
    """Status of an A/B test."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelProfile:
    """Profile capturing model strengths and weaknesses.

    Attributes:
        model_name: Name/identifier of the model
        model_type: Type of model
        strengths: List of identified strengths
        weaknesses: List of identified weaknesses
        optimal_prompt_styles: Styles that work well with this model
        average_latency_ms: Average response latency
        success_rate: Historical success rate (0-1)
        preferred_temperature: Optimal temperature for this model
        max_tokens_effective: Effective max token limit
        context_window: Context window size
        last_updated: When profile was last updated
    """

    model_name: str
    model_type: ModelType = ModelType.GENERAL
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    optimal_prompt_styles: List[str] = field(default_factory=list)
    average_latency_ms: float = 0.0
    success_rate: float = 0.0
    preferred_temperature: float = 0.7
    max_tokens_effective: int = 4096
    context_window: int = 8192
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "optimal_prompt_styles": self.optimal_prompt_styles,
            "average_latency_ms": self.average_latency_ms,
            "success_rate": self.success_rate,
            "preferred_temperature": self.preferred_temperature,
            "max_tokens_effective": self.max_tokens_effective,
            "context_window": self.context_window,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class PromptTemplate:
    """A prompt template with metadata.

    Attributes:
        template_id: Unique identifier
        name: Human-readable name
        template: The actual prompt template string
        model_type: Type of model this is designed for
        task_type: Type of task this handles
        variables: List of variables in the template
        version: Version number
        status: Current status
        created_at: Creation timestamp
        performance_score: Aggregated performance score (0-1)
        usage_count: Number of times used
    """

    template_id: str
    name: str
    template: str
    model_type: ModelType
    task_type: PromptTaskType
    variables: List[str] = field(default_factory=list)
    version: int = 1
    status: PromptVariantStatus = PromptVariantStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    performance_score: float = 0.0
    usage_count: int = 0

    def render(self, **kwargs: Any) -> str:
        """Render the template with provided variables."""
        result = self.template
        for var, value in kwargs.items():
            placeholder = f"{{{var}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "template": self.template,
            "model_type": self.model_type.value,
            "task_type": self.task_type.value,
            "variables": self.variables,
            "version": self.version,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "performance_score": self.performance_score,
            "usage_count": self.usage_count,
        }


@dataclass
class PromptPerformanceRecord:
    """Record of prompt performance for a specific execution.

    Attributes:
        record_id: Unique identifier
        template_id: ID of the template used
        model_name: Model that executed the prompt
        task_type: Type of task
        success: Whether the task succeeded
        latency_ms: Response latency in milliseconds
        quality_score: Quality score of the output (0-1)
        error_message: Error message if failed
        timestamp: When this was recorded
        context: Additional context
    """

    record_id: str
    template_id: str
    model_name: str
    task_type: PromptTaskType
    success: bool = True
    latency_ms: float = 0.0
    quality_score: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "template_id": self.template_id,
            "model_name": self.model_name,
            "task_type": self.task_type.value,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "quality_score": self.quality_score,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


@dataclass
class ABTestResult:
    """Result of an A/B test comparison.

    Attributes:
        test_id: Unique identifier for the test
        variant_a_id: Template ID for variant A
        variant_b_id: Template ID for variant B
        model_name: Model used for testing
        task_type: Task type being tested
        variant_a_samples: Number of samples for A
        variant_b_samples: Number of samples for B
        variant_a_success_rate: Success rate for A
        variant_b_success_rate: Success rate for B
        variant_a_avg_quality: Average quality for A
        variant_b_avg_quality: Average quality for B
        variant_a_avg_latency: Average latency for A
        variant_b_avg_latency: Average latency for B
        winner: Which variant won (A, B, or None for tie)
        confidence: Statistical confidence in the result
        status: Test status
        started_at: When test started
        completed_at: When test completed
    """

    test_id: str
    variant_a_id: str
    variant_b_id: str
    model_name: str
    task_type: PromptTaskType
    variant_a_samples: int = 0
    variant_b_samples: int = 0
    variant_a_success_rate: float = 0.0
    variant_b_success_rate: float = 0.0
    variant_a_avg_quality: float = 0.0
    variant_b_avg_quality: float = 0.0
    variant_a_avg_latency: float = 0.0
    variant_b_avg_latency: float = 0.0
    winner: Optional[str] = None  # "A", "B", or None
    confidence: float = 0.0
    status: ABTestStatus = ABTestStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "test_id": self.test_id,
            "variant_a_id": self.variant_a_id,
            "variant_b_id": self.variant_b_id,
            "model_name": self.model_name,
            "task_type": self.task_type.value,
            "variant_a_samples": self.variant_a_samples,
            "variant_b_samples": self.variant_b_samples,
            "variant_a_success_rate": self.variant_a_success_rate,
            "variant_b_success_rate": self.variant_b_success_rate,
            "variant_a_avg_quality": self.variant_a_avg_quality,
            "variant_b_avg_quality": self.variant_b_avg_quality,
            "variant_a_avg_latency": self.variant_a_avg_latency,
            "variant_b_avg_latency": self.variant_b_avg_latency,
            "winner": self.winner,
            "confidence": self.confidence,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }


@dataclass
class PromptEvolutionResult:
    """Result of prompt evolution based on failures.

    Attributes:
        original_template_id: ID of the original template
        evolved_template: The new evolved template
        changes_made: List of changes made
        failure_patterns_addressed: Failure patterns that were addressed
        expected_improvement: Expected improvement description
        confidence: Confidence in the improvement
    """

    original_template_id: str
    evolved_template: PromptTemplate
    changes_made: List[str] = field(default_factory=list)
    failure_patterns_addressed: List[str] = field(default_factory=list)
    expected_improvement: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "original_template_id": self.original_template_id,
            "evolved_template": self.evolved_template.to_dict(),
            "changes_made": self.changes_made,
            "failure_patterns_addressed": self.failure_patterns_addressed,
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
        }


@dataclass
class OptimizationSuggestion:
    """A suggestion for prompt optimization from meta-reasoning.

    Attributes:
        suggestion_id: Unique identifier
        target_model: Model to optimize for
        target_task: Task type to optimize
        current_issue: Description of current issue
        suggested_change: Suggested change
        rationale: Reasoning behind the suggestion
        priority: Priority level (high, medium, low)
        confidence: Confidence in the suggestion
    """

    suggestion_id: str
    target_model: str
    target_task: PromptTaskType
    current_issue: str
    suggested_change: str
    rationale: str
    priority: str = "medium"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "suggestion_id": self.suggestion_id,
            "target_model": self.target_model,
            "target_task": self.target_task.value,
            "current_issue": self.current_issue,
            "suggested_change": self.suggested_change,
            "rationale": self.rationale,
            "priority": self.priority,
            "confidence": self.confidence,
        }


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PromptOptimizerConfig:
    """Configuration for the ModelPromptOptimizer.

    Attributes:
        min_samples_for_ab_test: Minimum samples needed for A/B test
        confidence_threshold: Minimum confidence to declare winner
        performance_decay_factor: Factor for decaying old performance data
        max_templates_per_task: Maximum templates to keep per task
        auto_deprecate_threshold: Threshold below which to auto-deprecate
        enable_auto_evolution: Whether to auto-evolve prompts
    """

    min_samples_for_ab_test: int = 30
    confidence_threshold: float = 0.95
    performance_decay_factor: float = 0.9
    max_templates_per_task: int = 5
    auto_deprecate_threshold: float = 0.3
    enable_auto_evolution: bool = True


# =============================================================================
# Abstract Base Class
# =============================================================================


class PromptOptimizer(ABC):
    """Abstract base class for prompt optimizers.

    Follows the Template Method pattern, allowing different optimization
    strategies while maintaining consistent interface.
    """

    @abstractmethod
    def get_optimized_prompt(
        self,
        task_type: PromptTaskType,
        model_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get the best-performing prompt for this model and task type.

        Args:
            task_type: Type of task
            model_name: Name of the model
            context: Additional context for template rendering

        Returns:
            The optimized prompt string
        """
        pass

    @abstractmethod
    def record_performance(
        self,
        template_id: str,
        model_name: str,
        task_type: PromptTaskType,
        success: bool,
        quality_score: float,
        latency_ms: float,
        error_message: Optional[str] = None,
    ) -> None:
        """Record the performance of a prompt execution.

        Args:
            template_id: ID of the template used
            model_name: Model that executed
            task_type: Type of task
            success: Whether it succeeded
            quality_score: Quality of output
            latency_ms: Latency in ms
            error_message: Error if failed
        """
        pass

    @abstractmethod
    def evolve_prompt(
        self,
        template_id: str,
        failure_patterns: List[Dict[str, Any]],
    ) -> PromptEvolutionResult:
        """Evolve a prompt based on failure analysis.

        Args:
            template_id: Template to evolve
            failure_patterns: Patterns of failures to address

        Returns:
            Evolution result with new template
        """
        pass


# =============================================================================
# Prompt Registry
# =============================================================================


class PromptRegistry:
    """Registry for managing prompt templates.

    Thread-safe registry for storing, retrieving, and managing
    prompt templates across the ensemble.
    """

    def __init__(self) -> None:
        """Initialize the prompt registry."""
        self._templates: Dict[str, PromptTemplate] = {}
        self._templates_by_task: Dict[PromptTaskType, List[str]] = {
            task: [] for task in PromptTaskType
        }
        self._templates_by_model: Dict[ModelType, List[str]] = {
            model: [] for model in ModelType
        }
        self._lock = threading.RLock()
        self._initialize_default_templates()

    def _initialize_default_templates(self) -> None:
        """Initialize default prompt templates for each model/task type."""
        # Logic Generator templates
        self._add_default_template(
            name="logic_generator_asp_default",
            template=(
                "Convert the following legal principle to an ASP rule.\n\n"
                "Legal Principle: {principle}\n\n"
                "Available Predicates: {predicates}\n\n"
                "Domain: {domain}\n\n"
                "Requirements:\n"
                "- Output valid ASP syntax\n"
                "- Use only the provided predicates\n"
                "- Ensure rule is logically consistent\n\n"
                "ASP Rule:"
            ),
            model_type=ModelType.LOGIC_GENERATOR,
            task_type=PromptTaskType.ASP_GENERATION,
            variables=["principle", "predicates", "domain"],
        )

        # Critic templates
        self._add_default_template(
            name="critic_edge_case_default",
            template=(
                "Analyze the following ASP rule for potential edge cases.\n\n"
                "Rule: {rule}\n\n"
                "Domain Context: {domain}\n\n"
                "Identify:\n"
                "1. Boundary conditions not handled\n"
                "2. Unusual input scenarios\n"
                "3. Potential logical gaps\n\n"
                "Edge Cases:"
            ),
            model_type=ModelType.CRITIC,
            task_type=PromptTaskType.EDGE_CASE_DETECTION,
            variables=["rule", "domain"],
        )

        self._add_default_template(
            name="critic_contradiction_default",
            template=(
                "Check for contradictions between the following ASP rules.\n\n"
                "Rules:\n{rules}\n\n"
                "Context: {context}\n\n"
                "Identify any logical contradictions or conflicts.\n\n"
                "Contradictions:"
            ),
            model_type=ModelType.CRITIC,
            task_type=PromptTaskType.CONTRADICTION_DETECTION,
            variables=["rules", "context"],
        )

        # Translator templates
        self._add_default_template(
            name="translator_nl_to_asp_default",
            template=(
                "Translate the following natural language statement to ASP.\n\n"
                "Statement: {statement}\n\n"
                "Available Predicates: {predicates}\n\n"
                "Domain: {domain}\n\n"
                "ASP Translation:"
            ),
            model_type=ModelType.TRANSLATOR,
            task_type=PromptTaskType.TRANSLATION_NL_TO_ASP,
            variables=["statement", "predicates", "domain"],
        )

        self._add_default_template(
            name="translator_asp_to_nl_default",
            template=(
                "Explain the following ASP rule in natural language.\n\n"
                "ASP Rule: {rule}\n\n"
                "Context: {context}\n\n"
                "Provide a clear, human-readable explanation.\n\n"
                "Explanation:"
            ),
            model_type=ModelType.TRANSLATOR,
            task_type=PromptTaskType.TRANSLATION_ASP_TO_NL,
            variables=["rule", "context"],
        )

        # Meta-Reasoner templates
        self._add_default_template(
            name="meta_reasoner_failure_analysis_default",
            template=(
                "Analyze the following failure patterns.\n\n"
                "Failures:\n{failures}\n\n"
                "System Context: {context}\n\n"
                "Identify:\n"
                "1. Root causes\n"
                "2. Common patterns\n"
                "3. Recommended fixes\n\n"
                "Analysis:"
            ),
            model_type=ModelType.META_REASONER,
            task_type=PromptTaskType.FAILURE_ANALYSIS,
            variables=["failures", "context"],
        )

        self._add_default_template(
            name="meta_reasoner_strategy_default",
            template=(
                "Given the current system state, recommend strategy changes.\n\n"
                "Current Performance: {performance}\n\n"
                "Recent Issues: {issues}\n\n"
                "Available Strategies: {strategies}\n\n"
                "Recommendations:"
            ),
            model_type=ModelType.META_REASONER,
            task_type=PromptTaskType.STRATEGY_RECOMMENDATION,
            variables=["performance", "issues", "strategies"],
        )

    def _add_default_template(
        self,
        name: str,
        template: str,
        model_type: ModelType,
        task_type: PromptTaskType,
        variables: List[str],
    ) -> None:
        """Add a default template."""
        template_id = f"default_{name}"
        prompt_template = PromptTemplate(
            template_id=template_id,
            name=name,
            template=template,
            model_type=model_type,
            task_type=task_type,
            variables=variables,
            status=PromptVariantStatus.ACTIVE,
        )
        self._templates[template_id] = prompt_template
        self._templates_by_task[task_type].append(template_id)
        self._templates_by_model[model_type].append(template_id)

    def register_template(self, template: PromptTemplate) -> str:
        """Register a new prompt template.

        Args:
            template: The template to register

        Returns:
            The template ID
        """
        with self._lock:
            self._templates[template.template_id] = template
            self._templates_by_task[template.task_type].append(template.template_id)
            self._templates_by_model[template.model_type].append(template.template_id)
            logger.debug(f"Registered template: {template.template_id}")
            return template.template_id

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID.

        Args:
            template_id: Template ID

        Returns:
            The template or None
        """
        with self._lock:
            return self._templates.get(template_id)

    def get_templates_for_task(
        self, task_type: PromptTaskType, status: Optional[PromptVariantStatus] = None
    ) -> List[PromptTemplate]:
        """Get all templates for a task type.

        Args:
            task_type: Task type
            status: Optional filter by status

        Returns:
            List of matching templates
        """
        with self._lock:
            template_ids = self._templates_by_task.get(task_type, [])
            templates = [
                self._templates[tid] for tid in template_ids if tid in self._templates
            ]
            if status:
                templates = [t for t in templates if t.status == status]
            return templates

    def get_templates_for_model(
        self, model_type: ModelType, status: Optional[PromptVariantStatus] = None
    ) -> List[PromptTemplate]:
        """Get all templates for a model type.

        Args:
            model_type: Model type
            status: Optional filter by status

        Returns:
            List of matching templates
        """
        with self._lock:
            template_ids = self._templates_by_model.get(model_type, [])
            templates = [
                self._templates[tid] for tid in template_ids if tid in self._templates
            ]
            if status:
                templates = [t for t in templates if t.status == status]
            return templates

    def update_template_status(
        self, template_id: str, status: PromptVariantStatus
    ) -> bool:
        """Update a template's status.

        Args:
            template_id: Template ID
            status: New status

        Returns:
            True if updated, False if not found
        """
        with self._lock:
            if template_id in self._templates:
                self._templates[template_id].status = status
                logger.debug(f"Updated template {template_id} status to {status}")
                return True
            return False

    def update_template_performance(
        self, template_id: str, performance_score: float
    ) -> bool:
        """Update a template's performance score.

        Args:
            template_id: Template ID
            performance_score: New score

        Returns:
            True if updated, False if not found
        """
        with self._lock:
            if template_id in self._templates:
                self._templates[template_id].performance_score = performance_score
                self._templates[template_id].usage_count += 1
                return True
            return False

    def get_best_template(
        self, task_type: PromptTaskType, model_type: Optional[ModelType] = None
    ) -> Optional[PromptTemplate]:
        """Get the best performing active template.

        Args:
            task_type: Task type
            model_type: Optional model type filter

        Returns:
            Best template or None
        """
        with self._lock:
            templates = self.get_templates_for_task(
                task_type, status=PromptVariantStatus.ACTIVE
            )
            if model_type:
                templates = [t for t in templates if t.model_type == model_type]
            if not templates:
                return None
            return max(templates, key=lambda t: t.performance_score)

    def get_all_templates(self) -> List[PromptTemplate]:
        """Get all registered templates.

        Returns:
            List of all templates
        """
        with self._lock:
            return list(self._templates.values())


# =============================================================================
# Performance Tracker
# =============================================================================


class PromptPerformanceTracker:
    """Tracks performance metrics for prompt templates.

    Thread-safe tracker that maintains performance history and
    computes aggregate metrics. Uses bounded deques to prevent
    unbounded memory growth.

    Attributes:
        max_history_per_template: Maximum number of records to retain per template.
    """

    def __init__(self, max_history_per_template: int = 1000) -> None:
        """Initialize the tracker.

        Args:
            max_history_per_template: Maximum records to keep per template.
                Older records are automatically discarded when limit is reached.
        """
        self._records: Dict[str, Deque[PromptPerformanceRecord]] = {}
        self._max_history = max_history_per_template
        self._lock = threading.RLock()

    def record(self, record: PromptPerformanceRecord) -> None:
        """Record a performance measurement.

        Thread-safe method that stores performance data. Uses deque with
        maxlen to automatically evict oldest records when limit is reached.

        Args:
            record: Performance record to store
        """
        with self._lock:
            if record.template_id not in self._records:
                self._records[record.template_id] = deque(maxlen=self._max_history)
            self._records[record.template_id].append(record)
            logger.debug(
                f"Recorded performance for template {record.template_id}: "
                f"success={record.success}, quality={record.quality_score:.2f}"
            )

    def get_records(
        self,
        template_id: str,
        model_name: Optional[str] = None,
        task_type: Optional[PromptTaskType] = None,
        limit: Optional[int] = None,
    ) -> List[PromptPerformanceRecord]:
        """Get performance records for a template.

        Thread-safe retrieval of performance records with optional filtering.

        Args:
            template_id: Template ID to retrieve records for
            model_name: Optional filter to only include records for this model
            task_type: Optional filter to only include records for this task type
            limit: Maximum number of records to return (from most recent)

        Returns:
            List of matching records, ordered from oldest to newest
        """
        with self._lock:
            records_deque = self._records.get(template_id)
            if records_deque is None:
                return []
            records = list(records_deque)
            if model_name:
                records = [r for r in records if r.model_name == model_name]
            if task_type:
                records = [r for r in records if r.task_type == task_type]
            if limit:
                records = records[-limit:]
            return records

    def get_aggregate_metrics(
        self,
        template_id: str,
        model_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get aggregate metrics for a template.

        Args:
            template_id: Template ID
            model_name: Optional filter by model

        Returns:
            Dictionary with aggregate metrics
        """
        records = self.get_records(template_id, model_name=model_name)
        if not records:
            return {
                "success_rate": 0.0,
                "avg_quality": 0.0,
                "avg_latency_ms": 0.0,
                "sample_count": 0,
            }

        successes = sum(1 for r in records if r.success)
        quality_scores = [r.quality_score for r in records if r.success]
        latencies = [r.latency_ms for r in records]

        return {
            "success_rate": successes / len(records),
            "avg_quality": statistics.mean(quality_scores) if quality_scores else 0.0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "sample_count": len(records),
        }

    def get_failure_patterns(
        self,
        template_id: str,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get failure patterns for a template.

        Args:
            template_id: Template ID
            model_name: Optional filter by model

        Returns:
            List of failure patterns with counts
        """
        records = self.get_records(template_id, model_name=model_name)
        failures = [r for r in records if not r.success]

        # Group by error message
        error_counts: Dict[str, int] = {}
        error_contexts: Dict[str, List[Dict[str, Any]]] = {}
        for f in failures:
            error_key = f.error_message or "unknown"
            error_counts[error_key] = error_counts.get(error_key, 0) + 1
            if error_key not in error_contexts:
                error_contexts[error_key] = []
            error_contexts[error_key].append(f.context)

        patterns = []
        for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            patterns.append(
                {
                    "error_pattern": error,
                    "count": count,
                    "frequency": count / len(records) if records else 0,
                    "sample_contexts": error_contexts[error][:3],  # Keep first 3
                }
            )
        return patterns

    def clear_records(self, template_id: Optional[str] = None) -> int:
        """Clear performance records.

        Thread-safe method to clear stored performance data.

        Args:
            template_id: Specific template to clear records for.
                If None, clears all records for all templates.

        Returns:
            Number of records that were cleared
        """
        with self._lock:
            if template_id:
                records_deque = self._records.get(template_id)
                count = len(records_deque) if records_deque else 0
                self._records.pop(template_id, None)
            else:
                count = sum(len(r) for r in self._records.values())
                self._records.clear()
            return count


# =============================================================================
# A/B Testing Framework
# =============================================================================


class ABTestingFramework:
    """Framework for running A/B tests on prompt variants.

    Manages A/B tests between prompt variants to determine
    which performs better under controlled conditions.
    """

    def __init__(
        self,
        registry: PromptRegistry,
        tracker: PromptPerformanceTracker,
        config: PromptOptimizerConfig,
    ) -> None:
        """Initialize the A/B testing framework.

        Args:
            registry: Prompt registry
            tracker: Performance tracker
            config: Optimizer configuration
        """
        self._registry = registry
        self._tracker = tracker
        self._config = config
        self._active_tests: Dict[str, ABTestResult] = {}
        self._completed_tests: List[ABTestResult] = []
        self._lock = threading.RLock()

    def create_test(
        self,
        variant_a_id: str,
        variant_b_id: str,
        model_name: str,
        task_type: PromptTaskType,
    ) -> ABTestResult:
        """Create a new A/B test.

        Args:
            variant_a_id: Template ID for variant A
            variant_b_id: Template ID for variant B
            model_name: Model to use for testing
            task_type: Task type being tested

        Returns:
            The created test result object

        Raises:
            ABTestingError: If templates not found
        """
        # Validate templates exist
        template_a = self._registry.get_template(variant_a_id)
        template_b = self._registry.get_template(variant_b_id)
        if not template_a or not template_b:
            raise ABTestingError(
                f"Templates not found: A={variant_a_id}, B={variant_b_id}"
            )

        test_id = f"ab_test_{uuid.uuid4().hex[:8]}"
        test = ABTestResult(
            test_id=test_id,
            variant_a_id=variant_a_id,
            variant_b_id=variant_b_id,
            model_name=model_name,
            task_type=task_type,
            status=ABTestStatus.PENDING,
        )

        with self._lock:
            self._active_tests[test_id] = test
        logger.info(f"Created A/B test {test_id}: {variant_a_id} vs {variant_b_id}")
        return test

    def start_test(self, test_id: str) -> bool:
        """Start an A/B test.

        Args:
            test_id: Test ID

        Returns:
            True if started, False if not found
        """
        with self._lock:
            if test_id in self._active_tests:
                self._active_tests[test_id].status = ABTestStatus.RUNNING
                self._active_tests[test_id].started_at = datetime.now()
                logger.info(f"Started A/B test {test_id}")
                return True
            return False

    def select_variant(self, test_id: str) -> Optional[str]:
        """Randomly select a variant for the next execution.

        Args:
            test_id: Test ID

        Returns:
            Template ID to use, or None if test not found
        """
        with self._lock:
            test = self._active_tests.get(test_id)
            if not test or test.status != ABTestStatus.RUNNING:
                return None

            # Simple random selection
            if random.random() < 0.5:
                return test.variant_a_id
            else:
                return test.variant_b_id

    def record_result(
        self,
        test_id: str,
        template_id: str,
        success: bool,
        quality_score: float,
        latency_ms: float,
    ) -> None:
        """Record a result for an A/B test.

        Args:
            test_id: Test ID
            template_id: Template that was used
            success: Whether it succeeded
            quality_score: Quality score
            latency_ms: Latency in ms
        """
        with self._lock:
            test = self._active_tests.get(test_id)
            if not test or test.status != ABTestStatus.RUNNING:
                return

            # Update running totals
            if template_id == test.variant_a_id:
                test.variant_a_samples += 1
                # Update running averages
                n = test.variant_a_samples
                test.variant_a_success_rate = (
                    test.variant_a_success_rate * (n - 1) + (1.0 if success else 0.0)
                ) / n
                if success:
                    test.variant_a_avg_quality = (
                        test.variant_a_avg_quality * (n - 1) + quality_score
                    ) / n
                test.variant_a_avg_latency = (
                    test.variant_a_avg_latency * (n - 1) + latency_ms
                ) / n
            elif template_id == test.variant_b_id:
                test.variant_b_samples += 1
                n = test.variant_b_samples
                test.variant_b_success_rate = (
                    test.variant_b_success_rate * (n - 1) + (1.0 if success else 0.0)
                ) / n
                if success:
                    test.variant_b_avg_quality = (
                        test.variant_b_avg_quality * (n - 1) + quality_score
                    ) / n
                test.variant_b_avg_latency = (
                    test.variant_b_avg_latency * (n - 1) + latency_ms
                ) / n

            # Check if we have enough samples to conclude
            self._check_test_completion(test)

    def _check_test_completion(self, test: ABTestResult) -> None:
        """Check if a test has enough data to conclude.

        Evaluates whether sufficient samples have been collected and computes
        statistical confidence using a simplified heuristic. Handles edge cases
        like zero denominators and invalid configurations gracefully.

        Args:
            test: Test to check for completion readiness
        """
        min_samples = self._config.min_samples_for_ab_test

        # Guard against invalid configuration
        if min_samples <= 0:
            logger.warning(
                f"Invalid min_samples_for_ab_test={min_samples}, skipping completion check"
            )
            return

        if test.variant_a_samples < min_samples or test.variant_b_samples < min_samples:
            return

        try:
            # Calculate confidence using simple z-test approximation
            # (simplified for this implementation)
            rate_diff = abs(test.variant_a_success_rate - test.variant_b_success_rate)
            quality_diff = abs(test.variant_a_avg_quality - test.variant_b_avg_quality)

            # Simple confidence heuristic with division safety
            total_samples = test.variant_a_samples + test.variant_b_samples
            denominator = min_samples * 4
            if denominator == 0:
                base_confidence = 0.0
            else:
                base_confidence = min(1.0, total_samples / denominator)

            effect_size = max(rate_diff, quality_diff)
            test.confidence = base_confidence * (0.5 + effect_size)

            # Clamp confidence to valid range
            test.confidence = max(0.0, min(1.0, test.confidence))

            if test.confidence >= self._config.confidence_threshold:
                # Determine winner based on combined score
                score_a = (
                    test.variant_a_success_rate * 0.6 + test.variant_a_avg_quality * 0.4
                )
                score_b = (
                    test.variant_b_success_rate * 0.6 + test.variant_b_avg_quality * 0.4
                )

                if score_a > score_b + 0.05:  # 5% margin
                    test.winner = "A"
                elif score_b > score_a + 0.05:
                    test.winner = "B"
                else:
                    test.winner = None  # Tie

                test.status = ABTestStatus.COMPLETED
                test.completed_at = datetime.now()
                self._completed_tests.append(test)
                del self._active_tests[test.test_id]
                logger.info(
                    f"A/B test {test.test_id} completed: winner={test.winner}, "
                    f"confidence={test.confidence:.2f}"
                )
        except (ZeroDivisionError, ValueError, TypeError) as e:
            logger.error(
                f"Error in statistical calculation for test {test.test_id}: {e}"
            )

    def get_test(self, test_id: str) -> Optional[ABTestResult]:
        """Get a test by ID.

        Args:
            test_id: Test ID

        Returns:
            Test result or None
        """
        with self._lock:
            if test_id in self._active_tests:
                return self._active_tests[test_id]
            for test in self._completed_tests:
                if test.test_id == test_id:
                    return test
            return None

    def get_active_tests(self) -> List[ABTestResult]:
        """Get all active tests.

        Returns:
            List of active tests
        """
        with self._lock:
            return list(self._active_tests.values())

    def get_completed_tests(self, limit: Optional[int] = None) -> List[ABTestResult]:
        """Get completed tests.

        Args:
            limit: Maximum to return

        Returns:
            List of completed tests
        """
        with self._lock:
            tests = self._completed_tests[:]
            if limit:
                tests = tests[-limit:]
            return tests

    def cancel_test(self, test_id: str) -> bool:
        """Cancel an active test.

        Args:
            test_id: Test ID

        Returns:
            True if cancelled, False if not found
        """
        with self._lock:
            if test_id in self._active_tests:
                test = self._active_tests[test_id]
                test.status = ABTestStatus.CANCELLED
                test.completed_at = datetime.now()
                self._completed_tests.append(test)
                del self._active_tests[test_id]
                logger.info(f"Cancelled A/B test {test_id}")
                return True
            return False


# =============================================================================
# Main ModelPromptOptimizer Class
# =============================================================================


class ModelPromptOptimizer(PromptOptimizer):
    """Optimizes prompts for specific models in the ensemble.

    This class coordinates prompt optimization across the heterogeneous
    neural ensemble, providing model-specific templates, A/B testing,
    performance tracking, and prompt evolution.

    Attributes:
        model_profiles: Profiles for each model
        registry: Prompt template registry
        tracker: Performance tracker
        ab_framework: A/B testing framework
        config: Optimizer configuration
    """

    def __init__(
        self,
        model_profiles: Optional[Dict[str, ModelProfile]] = None,
        config: Optional[PromptOptimizerConfig] = None,
    ) -> None:
        """Initialize the ModelPromptOptimizer.

        Args:
            model_profiles: Dictionary mapping model names to profiles
            config: Optimizer configuration
        """
        self._model_profiles = model_profiles or {}
        self._config = config or PromptOptimizerConfig()
        self._registry = PromptRegistry()
        self._tracker = PromptPerformanceTracker()
        self._ab_framework = ABTestingFramework(
            self._registry, self._tracker, self._config
        )
        self._optimization_suggestions: List[OptimizationSuggestion] = []
        self._lock = threading.RLock()
        logger.info("ModelPromptOptimizer initialized")

    @property
    def registry(self) -> PromptRegistry:
        """Get the prompt registry."""
        return self._registry

    @property
    def tracker(self) -> PromptPerformanceTracker:
        """Get the performance tracker."""
        return self._tracker

    @property
    def ab_framework(self) -> ABTestingFramework:
        """Get the A/B testing framework."""
        return self._ab_framework

    def register_model_profile(self, model_name: str, profile: ModelProfile) -> None:
        """Register a model profile.

        Args:
            model_name: Name of the model
            profile: Profile for the model
        """
        with self._lock:
            self._model_profiles[model_name] = profile
            logger.info(f"Registered model profile: {model_name}")

    def get_model_profile(self, model_name: str) -> Optional[ModelProfile]:
        """Get a model profile.

        Args:
            model_name: Name of the model

        Returns:
            The profile or None
        """
        return self._model_profiles.get(model_name)

    def get_optimized_prompt(
        self,
        task_type: PromptTaskType,
        model_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get the best-performing prompt for this model and task type.

        Args:
            task_type: Type of task
            model_name: Name of the model
            context: Additional context for template rendering

        Returns:
            The optimized prompt string

        Raises:
            PromptOptimizationError: If no suitable template found
        """
        context = context or {}

        # Get model profile for model type hint
        profile = self._model_profiles.get(model_name)
        model_type = profile.model_type if profile else None

        # Get best template
        template = self._registry.get_best_template(task_type, model_type)
        if not template:
            # Fall back to any active template for this task
            templates = self._registry.get_templates_for_task(
                task_type, status=PromptVariantStatus.ACTIVE
            )
            if templates:
                template = templates[0]

        if not template:
            raise PromptOptimizationError(
                f"No template found for task {task_type}",
                model=model_name,
                task_type=str(task_type),
            )

        # Render template with context
        rendered = template.render(**context)
        logger.debug(
            f"Using template {template.template_id} for {model_name}/{task_type}"
        )
        return rendered

    def get_template_for_execution(
        self,
        task_type: PromptTaskType,
        model_name: str,
    ) -> Tuple[str, Optional[str]]:
        """Get a template ID for execution, potentially from an A/B test.

        Args:
            task_type: Type of task
            model_name: Name of the model

        Returns:
            Tuple of (template_id, ab_test_id or None)
        """
        # Check for active A/B tests for this model/task
        active_tests = self._ab_framework.get_active_tests()
        for test in active_tests:
            if test.model_name == model_name and test.task_type == task_type:
                variant = self._ab_framework.select_variant(test.test_id)
                if variant:
                    return (variant, test.test_id)

        # No A/B test, use best template
        profile = self._model_profiles.get(model_name)
        model_type = profile.model_type if profile else None
        template = self._registry.get_best_template(task_type, model_type)
        if template:
            return (template.template_id, None)

        # Fall back
        templates = self._registry.get_templates_for_task(
            task_type, status=PromptVariantStatus.ACTIVE
        )
        if templates:
            return (templates[0].template_id, None)

        raise PromptOptimizationError(
            f"No template found for task {task_type}",
            model=model_name,
            task_type=str(task_type),
        )

    def record_performance(
        self,
        template_id: str,
        model_name: str,
        task_type: PromptTaskType,
        success: bool,
        quality_score: float,
        latency_ms: float,
        error_message: Optional[str] = None,
        ab_test_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record the performance of a prompt execution.

        Args:
            template_id: ID of the template used
            model_name: Model that executed
            task_type: Type of task
            success: Whether it succeeded
            quality_score: Quality of output (0-1)
            latency_ms: Latency in ms
            error_message: Error if failed
            ab_test_id: ID of A/B test if applicable
            context: Additional context
        """
        record = PromptPerformanceRecord(
            record_id=f"perf_{uuid.uuid4().hex[:12]}",
            template_id=template_id,
            model_name=model_name,
            task_type=task_type,
            success=success,
            quality_score=quality_score,
            latency_ms=latency_ms,
            error_message=error_message,
            context=context or {},
        )
        self._tracker.record(record)

        # Update template performance score
        metrics = self._tracker.get_aggregate_metrics(template_id, model_name)
        combined_score = metrics["success_rate"] * 0.6 + metrics["avg_quality"] * 0.4
        self._registry.update_template_performance(template_id, combined_score)

        # Record in A/B test if applicable
        if ab_test_id:
            self._ab_framework.record_result(
                ab_test_id, template_id, success, quality_score, latency_ms
            )

        # Check for auto-deprecation
        if (
            self._config.enable_auto_evolution
            and combined_score < self._config.auto_deprecate_threshold
            and metrics["sample_count"] >= self._config.min_samples_for_ab_test
        ):
            template = self._registry.get_template(template_id)
            if template and template.status == PromptVariantStatus.ACTIVE:
                logger.warning(
                    f"Auto-deprecating template {template_id} due to low score "
                    f"({combined_score:.2f})"
                )
                self._registry.update_template_status(
                    template_id, PromptVariantStatus.DEPRECATED
                )

    def evolve_prompt(
        self,
        template_id: str,
        failure_patterns: List[Dict[str, Any]],
    ) -> PromptEvolutionResult:
        """Evolve a prompt based on failure analysis.

        This creates a new variant of the prompt designed to address
        the identified failure patterns.

        Args:
            template_id: Template to evolve
            failure_patterns: Patterns of failures to address

        Returns:
            Evolution result with new template

        Raises:
            PromptOptimizationError: If template not found
        """
        template = self._registry.get_template(template_id)
        if not template:
            raise PromptOptimizationError(f"Template not found: {template_id}")

        changes_made: List[str] = []
        evolved_content = template.template

        # Apply evolution heuristics based on failure patterns
        for pattern in failure_patterns:
            error_type = pattern.get("error_pattern", "")

            if "syntax" in error_type.lower():
                # Add syntax hints
                evolved_content = self._add_syntax_constraints(evolved_content)
                changes_made.append("Added explicit syntax constraints")

            if "timeout" in error_type.lower():
                # Add conciseness instruction
                evolved_content = self._add_conciseness_instruction(evolved_content)
                changes_made.append("Added conciseness instruction")

            if "validation" in error_type.lower():
                # Add validation emphasis
                evolved_content = self._add_validation_emphasis(evolved_content)
                changes_made.append("Added validation emphasis")

            if "grounding" in error_type.lower() or "predicate" in error_type.lower():
                # Add predicate usage reminder
                evolved_content = self._add_predicate_reminder(evolved_content)
                changes_made.append("Added predicate usage reminder")

        # Create new template
        new_template = PromptTemplate(
            template_id=f"{template_id}_v{template.version + 1}_{uuid.uuid4().hex[:4]}",
            name=f"{template.name}_evolved",
            template=evolved_content,
            model_type=template.model_type,
            task_type=template.task_type,
            variables=template.variables,
            version=template.version + 1,
            status=PromptVariantStatus.TESTING,
        )

        # Register the new template
        self._registry.register_template(new_template)

        # Create evolution result
        result = PromptEvolutionResult(
            original_template_id=template_id,
            evolved_template=new_template,
            changes_made=changes_made,
            failure_patterns_addressed=[
                p.get("error_pattern", "") for p in failure_patterns
            ],
            expected_improvement="Addresses identified failure patterns",
            confidence=0.6,  # Base confidence
        )

        logger.info(
            f"Evolved template {template_id} -> {new_template.template_id}: "
            f"{len(changes_made)} changes"
        )
        return result

    def _add_syntax_constraints(self, prompt: str) -> str:
        """Add syntax constraint hints to a prompt."""
        constraint = (
            "\n\nIMPORTANT: Ensure strict syntax compliance. "
            "Each rule must end with a period. "
            "Variables must start with uppercase letters."
        )
        return prompt + constraint

    def _add_conciseness_instruction(self, prompt: str) -> str:
        """Add conciseness instruction to a prompt."""
        instruction = (
            "\n\nBe concise. Provide only the requested output without explanation."
        )
        return prompt + instruction

    def _add_validation_emphasis(self, prompt: str) -> str:
        """Add validation emphasis to a prompt."""
        emphasis = (
            "\n\nBefore responding, verify:\n"
            "1. Output matches expected format\n"
            "2. All requirements are met\n"
            "3. No logical contradictions"
        )
        return prompt + emphasis

    def _add_predicate_reminder(self, prompt: str) -> str:
        """Add predicate usage reminder to a prompt."""
        reminder = "\n\nOnly use predicates from the provided list. Do not introduce new predicates."
        return prompt + reminder

    def analyze_model_characteristics(self, model_name: str) -> ModelProfile:
        """Profile model strengths/weaknesses for prompt design.

        Analyzes historical performance to build a model profile.

        Args:
            model_name: Name of the model

        Returns:
            Updated model profile
        """
        profile = self._model_profiles.get(model_name)
        if not profile:
            profile = ModelProfile(model_name=model_name)

        # Collect performance data across all templates
        all_templates = self._registry.get_all_templates()
        total_success = 0
        total_count = 0
        total_latency = 0.0
        latency_count = 0

        task_performance: Dict[PromptTaskType, float] = {}

        for template in all_templates:
            metrics = self._tracker.get_aggregate_metrics(
                template.template_id, model_name
            )
            if metrics["sample_count"] > 0:
                total_success += int(metrics["success_rate"] * metrics["sample_count"])
                total_count += metrics["sample_count"]
                total_latency += metrics["avg_latency_ms"] * metrics["sample_count"]
                latency_count += metrics["sample_count"]

                # Track per-task performance
                if template.task_type not in task_performance:
                    task_performance[template.task_type] = []
                task_performance[template.task_type] = metrics["success_rate"]

        # Update profile
        if total_count > 0:
            profile.success_rate = total_success / total_count
        if latency_count > 0:
            profile.average_latency_ms = total_latency / latency_count

        # Identify strengths and weaknesses
        sorted_tasks = sorted(
            task_performance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        if sorted_tasks:
            profile.strengths = [t[0].value for t in sorted_tasks[:2] if t[1] > 0.7]
            profile.weaknesses = [t[0].value for t in sorted_tasks[-2:] if t[1] < 0.5]

        profile.last_updated = datetime.now()
        self._model_profiles[model_name] = profile
        logger.info(f"Analyzed model characteristics for {model_name}")
        return profile

    def start_ab_test(
        self,
        variant_a_id: str,
        variant_b_id: str,
        model_name: str,
        task_type: PromptTaskType,
    ) -> str:
        """Start an A/B test between two prompt variants.

        Args:
            variant_a_id: Template ID for variant A
            variant_b_id: Template ID for variant B
            model_name: Model to use
            task_type: Task type

        Returns:
            Test ID
        """
        test = self._ab_framework.create_test(
            variant_a_id, variant_b_id, model_name, task_type
        )
        self._ab_framework.start_test(test.test_id)
        return test.test_id

    def get_ab_test_results(self, test_id: str) -> Optional[ABTestResult]:
        """Get results of an A/B test.

        Args:
            test_id: Test ID

        Returns:
            Test results or None
        """
        return self._ab_framework.get_test(test_id)

    def add_optimization_suggestion(
        self,
        target_model: str,
        target_task: PromptTaskType,
        current_issue: str,
        suggested_change: str,
        rationale: str,
        priority: str = "medium",
        confidence: float = 0.5,
    ) -> str:
        """Add an optimization suggestion from meta-reasoning.

        Args:
            target_model: Model to optimize for
            target_task: Task type
            current_issue: Description of current issue
            suggested_change: Suggested change
            rationale: Reasoning
            priority: Priority level
            confidence: Confidence in suggestion

        Returns:
            Suggestion ID
        """
        suggestion = OptimizationSuggestion(
            suggestion_id=f"suggest_{uuid.uuid4().hex[:8]}",
            target_model=target_model,
            target_task=target_task,
            current_issue=current_issue,
            suggested_change=suggested_change,
            rationale=rationale,
            priority=priority,
            confidence=confidence,
        )
        with self._lock:
            self._optimization_suggestions.append(suggestion)
        logger.info(f"Added optimization suggestion: {suggestion.suggestion_id}")
        return suggestion.suggestion_id

    def get_optimization_suggestions(
        self,
        model_name: Optional[str] = None,
        task_type: Optional[PromptTaskType] = None,
        min_confidence: float = 0.0,
    ) -> List[OptimizationSuggestion]:
        """Get optimization suggestions.

        Args:
            model_name: Optional filter by model
            task_type: Optional filter by task
            min_confidence: Minimum confidence filter

        Returns:
            List of matching suggestions
        """
        with self._lock:
            suggestions = self._optimization_suggestions[:]

        if model_name:
            suggestions = [s for s in suggestions if s.target_model == model_name]
        if task_type:
            suggestions = [s for s in suggestions if s.target_task == task_type]
        suggestions = [s for s in suggestions if s.confidence >= min_confidence]

        return sorted(
            suggestions,
            key=lambda s: (
                -{"high": 2, "medium": 1, "low": 0}.get(s.priority, 0),
                -s.confidence,
            ),
        )

    def get_performance_report(
        self,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a performance report.

        Args:
            model_name: Optional filter by model

        Returns:
            Performance report dictionary
        """
        templates = self._registry.get_all_templates()
        active_templates = [
            t for t in templates if t.status == PromptVariantStatus.ACTIVE
        ]

        report: Dict[str, Any] = {
            "total_templates": len(templates),
            "active_templates": len(active_templates),
            "templates_by_task": {},
            "templates_by_model": {},
            "top_performers": [],
            "underperformers": [],
            "active_ab_tests": len(self._ab_framework.get_active_tests()),
            "completed_ab_tests": len(self._ab_framework.get_completed_tests()),
        }

        # Group by task
        for task in PromptTaskType:
            task_templates = self._registry.get_templates_for_task(task)
            if task_templates:
                report["templates_by_task"][task.value] = {
                    "count": len(task_templates),
                    "active": len(
                        [
                            t
                            for t in task_templates
                            if t.status == PromptVariantStatus.ACTIVE
                        ]
                    ),
                }

        # Group by model
        for model in ModelType:
            model_templates = self._registry.get_templates_for_model(model)
            if model_templates:
                report["templates_by_model"][model.value] = {
                    "count": len(model_templates),
                    "active": len(
                        [
                            t
                            for t in model_templates
                            if t.status == PromptVariantStatus.ACTIVE
                        ]
                    ),
                }

        # Top performers
        sorted_templates = sorted(
            active_templates,
            key=lambda t: t.performance_score,
            reverse=True,
        )
        report["top_performers"] = [
            {
                "template_id": t.template_id,
                "name": t.name,
                "score": t.performance_score,
                "usage_count": t.usage_count,
            }
            for t in sorted_templates[:5]
        ]

        # Underperformers
        report["underperformers"] = [
            {
                "template_id": t.template_id,
                "name": t.name,
                "score": t.performance_score,
                "usage_count": t.usage_count,
            }
            for t in sorted_templates[-5:]
            if t.performance_score < 0.5
        ]

        return report

    def clear_data(self) -> Dict[str, int]:
        """Clear all tracked data.

        Returns:
            Dictionary with counts of cleared items
        """
        with self._lock:
            suggestions_cleared = len(self._optimization_suggestions)
            self._optimization_suggestions.clear()

        records_cleared = self._tracker.clear_records()

        return {
            "suggestions_cleared": suggestions_cleared,
            "records_cleared": records_cleared,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_prompt_optimizer(
    model_profiles: Optional[Dict[str, ModelProfile]] = None,
    config: Optional[PromptOptimizerConfig] = None,
) -> ModelPromptOptimizer:
    """Create a ModelPromptOptimizer instance.

    Args:
        model_profiles: Optional model profiles
        config: Optional configuration

    Returns:
        Configured ModelPromptOptimizer
    """
    return ModelPromptOptimizer(model_profiles=model_profiles, config=config)


def create_default_model_profiles() -> Dict[str, ModelProfile]:
    """Create default model profiles for common LLMs.

    Returns:
        Dictionary of model name to profile
    """
    return {
        "claude-3-5-haiku-20241022": ModelProfile(
            model_name="claude-3-5-haiku-20241022",
            model_type=ModelType.GENERAL,
            strengths=["fast_response", "cost_effective", "good_instruction_following"],
            weaknesses=["complex_reasoning", "long_context"],
            optimal_prompt_styles=["concise", "structured", "explicit_instructions"],
            preferred_temperature=0.3,
            max_tokens_effective=4096,
            context_window=200000,
        ),
        "claude-3-sonnet-20240229": ModelProfile(
            model_name="claude-3-sonnet-20240229",
            model_type=ModelType.GENERAL,
            strengths=["balanced_capability", "good_reasoning", "reliable"],
            weaknesses=["slower_than_haiku"],
            optimal_prompt_styles=["detailed", "step_by_step", "examples_helpful"],
            preferred_temperature=0.5,
            max_tokens_effective=4096,
            context_window=200000,
        ),
        "claude-3-opus-20240229": ModelProfile(
            model_name="claude-3-opus-20240229",
            model_type=ModelType.META_REASONER,
            strengths=["complex_reasoning", "nuanced_analysis", "creative"],
            weaknesses=["slower", "higher_cost"],
            optimal_prompt_styles=["detailed", "open_ended", "complex_tasks"],
            preferred_temperature=0.7,
            max_tokens_effective=4096,
            context_window=200000,
        ),
        "gpt-4": ModelProfile(
            model_name="gpt-4",
            model_type=ModelType.LOGIC_GENERATOR,
            strengths=["logical_reasoning", "code_generation", "structured_output"],
            weaknesses=["occasional_hallucination"],
            optimal_prompt_styles=["precise", "examples_helpful", "system_prompts"],
            preferred_temperature=0.5,
            max_tokens_effective=4096,
            context_window=8192,
        ),
    }
