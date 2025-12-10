"""
Ensemble Orchestrator - Coordinating multiple specialized LLMs.

This module implements the orchestration layer that coordinates the Phase 6
specialized LLMs (LogicGenerator, Critic, Translator, MetaReasoner) to work
together effectively. It provides task routing, voting mechanisms, disagreement
resolution, and performance monitoring.

Part of Phase 6: Heterogeneous Neural Ensemble (Issue #192).

Key components:
- Task routing: Match tasks to appropriate specialized models
- Multi-LLM voting: Aggregate outputs using various voting strategies
- Disagreement resolution: Handle conflicting outputs from models
- Fallback handling: Graceful degradation when specialized models fail
- Performance monitoring: Track and optimize model selection over time
"""

from __future__ import annotations

import hashlib
import statistics
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict, Union

from loguru import logger

# Import specialized LLM components
from loft.neural.ensemble.logic_generator import (
    LogicGeneratorConfig,
    LogicGeneratorLLM,
)
from loft.neural.ensemble.critic import (
    CriticConfig,
    CriticLLM,
)
from loft.neural.ensemble.translator import (
    TranslatorConfig,
    TranslatorLLM,
)
from loft.neural.ensemble.meta_reasoner import (
    MetaReasonerConfig,
    MetaReasonerLLM,
)
from loft.neural.llm_interface import LLMInterface


# =============================================================================
# Custom Exceptions
# =============================================================================


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""

    pass


class TaskRoutingError(OrchestratorError):
    """Raised when task cannot be routed to any specialized model."""

    pass


class VotingError(OrchestratorError):
    """Raised when voting fails to reach a decision."""

    pass


class DisagreementResolutionError(OrchestratorError):
    """Raised when disagreement cannot be resolved."""

    pass


class FallbackExhaustedError(OrchestratorError):
    """Raised when all fallback options are exhausted."""

    pass


class AggregatedOrchestrationError(OrchestratorError):
    """Exception that aggregates multiple model errors during orchestration.

    This exception is raised when orchestration fails and provides detailed
    context about which models were tried and why they failed.

    Attributes:
        model_errors: List of ModelError instances describing each failure
        attempted_models: List of model IDs that were attempted
        task_type: The type of task that failed
        message: Human-readable summary of the failures
    """

    def __init__(
        self,
        message: str,
        model_errors: Optional[List["ModelError"]] = None,
        attempted_models: Optional[List[str]] = None,
        task_type: Optional["TaskType"] = None,
    ):
        """Initialize the aggregated orchestration error.

        Args:
            message: Human-readable error message
            model_errors: List of individual model errors
            attempted_models: List of model IDs that were attempted
            task_type: The task type that was being performed
        """
        super().__init__(message)
        self.model_errors = model_errors or []
        self.attempted_models = attempted_models or []
        self.task_type = task_type

    def __str__(self) -> str:
        """Return detailed string representation of the error."""
        parts = [self.args[0] if self.args else "Orchestration failed"]
        if self.attempted_models:
            parts.append(f"Attempted models: {', '.join(self.attempted_models)}")
        if self.model_errors:
            error_summaries = [
                f"{e.model_id}: {e.error_type} - {e.message}" for e in self.model_errors
            ]
            parts.append(f"Errors: {'; '.join(error_summaries)}")
        return " | ".join(parts)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a structured summary of all errors.

        Returns:
            Dictionary with error details suitable for logging or reporting
        """
        return {
            "message": self.args[0] if self.args else "Orchestration failed",
            "task_type": self.task_type.value if self.task_type else None,
            "attempted_models": self.attempted_models,
            "model_errors": [
                {
                    "model_id": e.model_id,
                    "error_type": e.error_type,
                    "message": e.message,
                    "timestamp": e.timestamp,
                }
                for e in self.model_errors
            ],
            "total_failures": len(self.model_errors),
        }


# =============================================================================
# Enums and Types
# =============================================================================


class TaskType(Enum):
    """Types of tasks that can be orchestrated."""

    RULE_GENERATION = "rule_generation"
    RULE_CRITICISM = "rule_criticism"
    TRANSLATION_TO_NL = "translation_to_nl"
    TRANSLATION_TO_ASP = "translation_to_asp"
    META_ANALYSIS = "meta_analysis"
    FULL_PIPELINE = "full_pipeline"  # Generate -> Critique -> Refine


class VotingStrategyType(Enum):
    """Types of voting strategies for multi-LLM consensus."""

    UNANIMOUS = "unanimous"  # All models must agree
    MAJORITY = "majority"  # >50% must agree
    WEIGHTED = "weighted"  # Weight by confidence scores
    DIALECTICAL = "dialectical"  # Thesis-antithesis-synthesis approach


class DisagreementStrategyType(Enum):
    """Types of disagreement resolution strategies."""

    DEFER_TO_CRITIC = "defer_to_critic"  # Critic has final say
    DEFER_TO_CONFIDENCE = "defer_to_confidence"  # Highest confidence wins
    SYNTHESIZE = "synthesize"  # Try to combine views
    ESCALATE = "escalate"  # Request human review
    CONSERVATIVE = "conservative"  # Default to safest option


class ModelStatus(Enum):
    """Status of a specialized model."""

    AVAILABLE = "available"
    BUSY = "busy"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


# =============================================================================
# TypedDicts for Input Data (Issue #204)
# =============================================================================


class RuleGenerationInput(TypedDict, total=False):
    """Input data structure for rule generation tasks.

    Attributes:
        principle: The legal principle or statement to convert to ASP rule
        domain: The domain context (e.g., 'legal', 'contracts', 'torts')
        predicates: List of available predicates for rule generation
    """

    principle: str
    domain: str
    predicates: List[str]


class RuleCriticismInput(TypedDict, total=False):
    """Input data structure for rule criticism tasks.

    Attributes:
        rule: The ASP rule to analyze and critique
        domain: The domain context for analysis
        existing_rules: List of existing rules for consistency checking
    """

    rule: str
    domain: str
    existing_rules: List[str]


class TranslationToNLInput(TypedDict, total=False):
    """Input data structure for ASP-to-natural-language translation.

    Attributes:
        rule: The ASP rule to translate
        domain: The domain context for translation
    """

    rule: str
    domain: str


class TranslationToASPInput(TypedDict, total=False):
    """Input data structure for natural-language-to-ASP translation.

    Attributes:
        text: The natural language text to convert to ASP
        domain: The domain context for translation
        predicates: List of available predicates for translation
    """

    text: str
    domain: str
    predicates: List[str]


class MetaAnalysisInput(TypedDict, total=False):
    """Input data structure for meta-analysis tasks.

    Attributes:
        failures: List of failure records to analyze
        insights: List of existing insights to consider
    """

    failures: List[Any]
    insights: List[Any]


class FullPipelineInput(TypedDict, total=False):
    """Input data structure for full pipeline tasks (generate -> critique -> refine).

    Attributes:
        principle: The legal principle or statement to process
        domain: The domain context
        predicates: List of available predicates
    """

    principle: str
    domain: str
    predicates: List[str]


# Type alias for all possible input data types
TaskInputData = Union[
    str,
    RuleGenerationInput,
    RuleCriticismInput,
    TranslationToNLInput,
    TranslationToASPInput,
    MetaAnalysisInput,
    FullPipelineInput,
    Dict[str, Any],  # Fallback for backwards compatibility
]

# Type alias for context dictionary
ContextDict = Dict[str, Any]

# Type alias for response metadata
ResponseMetadata = Dict[str, Any]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ModelError:
    """Represents an error from a specific model during orchestration.

    This dataclass captures detailed information about a model failure,
    enabling structured error reporting and debugging.

    Attributes:
        model_id: Identifier of the model that failed (e.g., 'logic_generator')
        error_type: Type/class of the error (e.g., 'TimeoutError', 'ValueError')
        message: Human-readable error message
        timestamp: Unix timestamp when the error occurred
        context: Optional additional context about the failure
    """

    model_id: str
    error_type: str
    message: str
    timestamp: float = field(default_factory=time.time)
    context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all error details
        """
        return {
            "model_id": self.model_id,
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "context": self.context,
        }


@dataclass
class ModelResponse:
    """Response from a specialized model.

    Attributes:
        model_type: Type of model that produced this response
        result: The actual result (varies by model type)
        confidence: Confidence score for this response (0.0-1.0)
        latency_ms: Time taken to produce response
        metadata: Additional metadata about the response
    """

    model_type: str
    result: Any
    confidence: float
    latency_ms: float
    metadata: ResponseMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence score."""
        if not 0.0 <= self.confidence <= 1.0:
            logger.warning(
                f"Invalid confidence {self.confidence}, clamping to [0.0, 1.0]"
            )
            self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class VotingResult:
    """Result of a voting process.

    Attributes:
        decision: The final decision
        strategy_used: Voting strategy that was used
        vote_counts: Count of votes for each option
        confidence: Confidence in the decision
        participating_models: Models that participated in voting
        dissenting_models: Models that disagreed with decision
        reasoning: Explanation of voting outcome
    """

    decision: Any
    strategy_used: VotingStrategyType
    vote_counts: Dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0
    participating_models: List[str] = field(default_factory=list)
    dissenting_models: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class DisagreementRecord:
    """Record of a disagreement between models.

    Attributes:
        task_type: Type of task being performed
        conflicting_responses: Responses that conflict
        resolution: How the disagreement was resolved
        resolution_strategy: Strategy used to resolve
        final_decision: The final decision after resolution
        timestamp: When the disagreement occurred
    """

    task_type: TaskType
    conflicting_responses: List[ModelResponse]
    resolution: str
    resolution_strategy: DisagreementStrategyType
    final_decision: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model.

    Attributes:
        model_type: Type of the model
        total_requests: Total number of requests
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        average_latency_ms: Average response latency
        average_confidence: Average confidence score
        last_success: Timestamp of last successful request
        last_failure: Timestamp of last failed request
        error_types: Count of different error types
    """

    model_type: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    average_confidence: float = 0.0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    error_types: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


@dataclass
class OrchestratorConfig:
    """Configuration for the Ensemble Orchestrator.

    Attributes:
        default_voting_strategy: Default voting strategy to use
        default_disagreement_strategy: Default disagreement resolution strategy
        enable_caching: Whether to cache results
        cache_ttl_seconds: Time-to-live for cached results
        max_retries: Maximum retries for failed model calls
        timeout_seconds: Timeout for individual model calls
        enable_performance_tracking: Whether to track model performance
        min_confidence_threshold: Minimum confidence to accept a response
        enable_fallback: Whether to enable fallback mechanisms
        parallel_execution: Whether to execute models in parallel
    """

    default_voting_strategy: VotingStrategyType = VotingStrategyType.WEIGHTED
    default_disagreement_strategy: DisagreementStrategyType = (
        DisagreementStrategyType.DEFER_TO_CONFIDENCE
    )
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_retries: int = 3
    timeout_seconds: float = 60.0
    enable_performance_tracking: bool = True
    min_confidence_threshold: float = 0.6
    enable_fallback: bool = True
    parallel_execution: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not 0.0 <= self.min_confidence_threshold <= 1.0:
            raise ValueError("min_confidence_threshold must be between 0.0 and 1.0")
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")


@dataclass
class OrchestrationResult:
    """Result from an orchestrated task.

    Attributes:
        task_type: Type of task that was performed
        final_result: The final result after orchestration
        model_responses: Individual responses from each model
        voting_result: Result of voting (if applicable)
        disagreements: Any disagreements that occurred
        total_latency_ms: Total time for orchestration
        from_cache: Whether result was from cache
        metadata: Additional metadata
        errors: List of errors that occurred during orchestration (Issue #201)
        failed_models: List of model IDs that failed during orchestration (Issue #201)
    """

    task_type: TaskType
    final_result: Any
    model_responses: List[ModelResponse] = field(default_factory=list)
    voting_result: Optional[VotingResult] = None
    disagreements: List[DisagreementRecord] = field(default_factory=list)
    total_latency_ms: float = 0.0
    from_cache: bool = False
    metadata: ResponseMetadata = field(default_factory=dict)
    errors: List[ModelError] = field(default_factory=list)
    failed_models: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred during orchestration.

        Returns:
            True if there were any errors, False otherwise
        """
        return len(self.errors) > 0

    @property
    def partial_success(self) -> bool:
        """Check if orchestration partially succeeded (some models worked, some failed).

        Returns:
            True if there were both successful responses and errors
        """
        return len(self.model_responses) > 0 and len(self.errors) > 0

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a structured summary of all errors.

        Returns:
            Dictionary with error details suitable for logging or reporting
        """
        return {
            "total_errors": len(self.errors),
            "failed_models": self.failed_models,
            "successful_models": [r.model_type for r in self.model_responses],
            "errors": [e.to_dict() for e in self.errors],
        }


# =============================================================================
# Voting Strategies (Strategy Pattern)
# =============================================================================


class VotingStrategy(ABC):
    """Abstract base class for voting strategies."""

    @property
    @abstractmethod
    def strategy_type(self) -> VotingStrategyType:
        """Return the strategy type enum."""
        pass

    @abstractmethod
    def vote(
        self,
        responses: List[ModelResponse],
        context: Optional[ContextDict] = None,
    ) -> VotingResult:
        """Perform voting on responses.

        Args:
            responses: List of model responses to vote on
            context: Additional context for voting

        Returns:
            VotingResult with the decision
        """
        pass


class UnanimousVotingStrategy(VotingStrategy):
    """Voting strategy requiring unanimous agreement."""

    @property
    def strategy_type(self) -> VotingStrategyType:
        return VotingStrategyType.UNANIMOUS

    def vote(
        self,
        responses: List[ModelResponse],
        context: Optional[ContextDict] = None,
    ) -> VotingResult:
        logger.debug("Performing unanimous voting")
        if not responses:
            raise VotingError("No responses to vote on")

        # Group responses by their result (simplified comparison)
        result_groups: Dict[str, List[ModelResponse]] = {}
        for response in responses:
            result_key = self._result_to_key(response.result)
            if result_key not in result_groups:
                result_groups[result_key] = []
            result_groups[result_key].append(response)

        vote_counts = {k: len(v) for k, v in result_groups.items()}
        participating = [r.model_type for r in responses]

        # Check for unanimity
        if len(result_groups) == 1:
            result_key = list(result_groups.keys())[0]
            avg_confidence = statistics.mean([r.confidence for r in responses])
            return VotingResult(
                decision=responses[0].result,
                strategy_used=self.strategy_type,
                vote_counts=vote_counts,
                confidence=avg_confidence,
                participating_models=participating,
                dissenting_models=[],
                reasoning="All models agreed on the result",
            )
        else:
            # No unanimity - return most common with low confidence
            most_common_key = max(
                result_groups.keys(), key=lambda k: len(result_groups[k])
            )
            most_common_responses = result_groups[most_common_key]
            dissenting = [
                r.model_type
                for r in responses
                if self._result_to_key(r.result) != most_common_key
            ]
            return VotingResult(
                decision=most_common_responses[0].result,
                strategy_used=self.strategy_type,
                vote_counts=vote_counts,
                confidence=0.3,  # Low confidence due to lack of unanimity
                participating_models=participating,
                dissenting_models=dissenting,
                reasoning=f"Unanimity not achieved. {len(dissenting)} models dissented.",
            )

    def _result_to_key(self, result: Any) -> str:
        """Convert result to a hashable key for comparison."""
        if isinstance(result, str):
            return result
        elif hasattr(result, "__dict__"):
            return str(sorted(result.__dict__.items()))
        else:
            return str(result)


class MajorityVotingStrategy(VotingStrategy):
    """Voting strategy requiring majority agreement (>50%)."""

    @property
    def strategy_type(self) -> VotingStrategyType:
        return VotingStrategyType.MAJORITY

    def vote(
        self,
        responses: List[ModelResponse],
        context: Optional[ContextDict] = None,
    ) -> VotingResult:
        logger.debug("Performing majority voting")
        if not responses:
            raise VotingError("No responses to vote on")

        # Group responses by their result
        result_groups: Dict[str, List[ModelResponse]] = {}
        for response in responses:
            result_key = self._result_to_key(response.result)
            if result_key not in result_groups:
                result_groups[result_key] = []
            result_groups[result_key].append(response)

        vote_counts = {k: len(v) for k, v in result_groups.items()}
        participating = [r.model_type for r in responses]
        majority_threshold = len(responses) / 2

        # Find majority
        for result_key, group_responses in result_groups.items():
            if len(group_responses) > majority_threshold:
                avg_confidence = statistics.mean(
                    [r.confidence for r in group_responses]
                )
                dissenting = [
                    r.model_type
                    for r in responses
                    if self._result_to_key(r.result) != result_key
                ]
                return VotingResult(
                    decision=group_responses[0].result,
                    strategy_used=self.strategy_type,
                    vote_counts=vote_counts,
                    confidence=avg_confidence * (len(group_responses) / len(responses)),
                    participating_models=participating,
                    dissenting_models=dissenting,
                    reasoning=(
                        f"Majority achieved with {len(group_responses)}/{len(responses)} "
                        f"votes"
                    ),
                )

        # No majority - return most common with reduced confidence
        # Use deterministic tie-breaking: vote count (desc), avg confidence (desc), result key (asc)
        def tie_breaking_key(result_key: str) -> tuple:
            group = result_groups[result_key]
            avg_conf = statistics.mean([r.confidence for r in group])
            return (
                -len(group),  # Primary: vote count (negative for descending)
                -avg_conf,  # Secondary: average confidence (negative for descending)
                result_key,  # Tertiary: lexicographic for determinism (ascending)
            )

        most_common_key = min(result_groups.keys(), key=tie_breaking_key)
        most_common = result_groups[most_common_key]
        dissenting = [
            r.model_type
            for r in responses
            if self._result_to_key(r.result) != most_common_key
        ]
        return VotingResult(
            decision=most_common[0].result,
            strategy_used=self.strategy_type,
            vote_counts=vote_counts,
            confidence=0.4,  # Lower confidence due to lack of majority
            participating_models=participating,
            dissenting_models=dissenting,
            reasoning="No majority achieved. Returning plurality result.",
        )

    def _result_to_key(self, result: Any) -> str:
        """Convert result to a hashable key for comparison."""
        if isinstance(result, str):
            return result
        elif hasattr(result, "__dict__"):
            return str(sorted(result.__dict__.items()))
        else:
            return str(result)


class WeightedVotingStrategy(VotingStrategy):
    """Voting strategy that weights votes by confidence scores."""

    @property
    def strategy_type(self) -> VotingStrategyType:
        return VotingStrategyType.WEIGHTED

    def vote(
        self,
        responses: List[ModelResponse],
        context: Optional[ContextDict] = None,
    ) -> VotingResult:
        logger.debug("Performing weighted voting")
        if not responses:
            raise VotingError("No responses to vote on")

        # Note: Confidence scores are already validated and clamped to [0.0, 1.0]
        # by ModelResponse.__post_init__, so no additional validation is needed here.

        # Calculate weighted scores for each result
        result_weights: Dict[str, float] = {}
        result_responses: Dict[str, List[ModelResponse]] = {}
        for response in responses:
            result_key = self._result_to_key(response.result)
            if result_key not in result_weights:
                result_weights[result_key] = 0.0
                result_responses[result_key] = []
            result_weights[result_key] += response.confidence
            result_responses[result_key].append(response)

        # Find highest weighted result
        best_key = max(result_weights.keys(), key=lambda k: result_weights[k])
        total_weight = sum(result_weights.values())

        participating = [r.model_type for r in responses]
        dissenting = [
            r.model_type for r in responses if self._result_to_key(r.result) != best_key
        ]

        # Calculate normalized confidence
        normalized_confidence = (
            result_weights[best_key] / total_weight if total_weight > 0 else 0.0
        )

        vote_counts = {k: len(v) for k, v in result_responses.items()}

        return VotingResult(
            decision=result_responses[best_key][0].result,
            strategy_used=self.strategy_type,
            vote_counts=vote_counts,
            confidence=normalized_confidence,
            participating_models=participating,
            dissenting_models=dissenting,
            reasoning=(
                f"Weighted voting selected result with {result_weights[best_key]:.2f} "
                f"total weight ({normalized_confidence:.1%} of total)"
            ),
        )

    def _result_to_key(self, result: Any) -> str:
        """Convert result to a hashable key for comparison."""
        if isinstance(result, str):
            return result
        elif hasattr(result, "__dict__"):
            return str(sorted(result.__dict__.items()))
        else:
            return str(result)


class DialecticalVotingStrategy(VotingStrategy):
    """Voting strategy using thesis-antithesis-synthesis approach."""

    @property
    def strategy_type(self) -> VotingStrategyType:
        return VotingStrategyType.DIALECTICAL

    def vote(
        self,
        responses: List[ModelResponse],
        context: Optional[ContextDict] = None,
    ) -> VotingResult:
        logger.debug("Performing dialectical voting")
        if not responses:
            raise VotingError("No responses to vote on")

        if len(responses) < 2:
            # Not enough for dialectical process
            return VotingResult(
                decision=responses[0].result,
                strategy_used=self.strategy_type,
                vote_counts={"single": 1},
                confidence=responses[0].confidence,
                participating_models=[responses[0].model_type],
                dissenting_models=[],
                reasoning="Single response - dialectical process not applicable",
            )

        # Sort by confidence to identify thesis (highest) and antithesis
        sorted_responses = sorted(responses, key=lambda r: r.confidence, reverse=True)
        thesis = sorted_responses[0]
        antithesis = sorted_responses[-1] if len(sorted_responses) > 1 else None

        participating = [r.model_type for r in responses]

        # If thesis and antithesis agree, strong consensus
        if antithesis and self._results_match(thesis.result, antithesis.result):
            return VotingResult(
                decision=thesis.result,
                strategy_used=self.strategy_type,
                vote_counts={"consensus": len(responses)},
                confidence=(thesis.confidence + antithesis.confidence) / 2,
                participating_models=participating,
                dissenting_models=[],
                reasoning=(
                    "Dialectical synthesis: Thesis and antithesis agree, "
                    "indicating strong consensus"
                ),
            )

        # Look for synthesis among other responses
        if len(sorted_responses) > 2:
            middle_responses = sorted_responses[1:-1]
            # Check if any middle responses can serve as synthesis
            for response in middle_responses:
                if self._could_be_synthesis(
                    thesis.result, antithesis.result, response.result
                ):
                    return VotingResult(
                        decision=response.result,
                        strategy_used=self.strategy_type,
                        vote_counts={
                            "thesis": 1,
                            "antithesis": 1,
                            "synthesis": len(middle_responses),
                        },
                        confidence=response.confidence,
                        participating_models=participating,
                        dissenting_models=[],
                        reasoning=(
                            "Dialectical synthesis found in intermediate response"
                        ),
                    )

        # Default to thesis with reduced confidence
        dissenting = [r.model_type for r in responses if r != thesis]
        return VotingResult(
            decision=thesis.result,
            strategy_used=self.strategy_type,
            vote_counts={"thesis": 1, "other": len(responses) - 1},
            confidence=thesis.confidence * 0.7,  # Reduced due to unresolved dialectic
            participating_models=participating,
            dissenting_models=dissenting,
            reasoning="Dialectical process incomplete - defaulting to highest confidence",
        )

    def _results_match(self, result1: Any, result2: Any) -> bool:
        """Check if two results match."""
        if isinstance(result1, str) and isinstance(result2, str):
            return result1 == result2
        elif hasattr(result1, "__dict__") and hasattr(result2, "__dict__"):
            return result1.__dict__ == result2.__dict__
        return result1 == result2

    def _could_be_synthesis(self, thesis: Any, antithesis: Any, candidate: Any) -> bool:
        """Check if candidate could be a synthesis of thesis and antithesis."""
        # Simplified check - in practice this would be domain-specific
        if (
            isinstance(thesis, str)
            and isinstance(antithesis, str)
            and isinstance(candidate, str)
        ):
            # Check if candidate contains elements from both
            thesis_words = set(thesis.lower().split())
            antithesis_words = set(antithesis.lower().split())
            candidate_words = set(candidate.lower().split())

            has_thesis_elements = bool(thesis_words & candidate_words)
            has_antithesis_elements = bool(antithesis_words & candidate_words)
            return has_thesis_elements and has_antithesis_elements

        return False


# Factory function for voting strategies
def create_voting_strategy(
    strategy_type: VotingStrategyType,
) -> VotingStrategy:
    """Create a voting strategy instance.

    Args:
        strategy_type: The type of voting strategy to create

    Returns:
        An instance of the requested voting strategy
    """
    strategies: Dict[VotingStrategyType, Type[VotingStrategy]] = {
        VotingStrategyType.UNANIMOUS: UnanimousVotingStrategy,
        VotingStrategyType.MAJORITY: MajorityVotingStrategy,
        VotingStrategyType.WEIGHTED: WeightedVotingStrategy,
        VotingStrategyType.DIALECTICAL: DialecticalVotingStrategy,
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown voting strategy type: {strategy_type}")

    return strategies[strategy_type]()


# =============================================================================
# Disagreement Resolution Strategies
# =============================================================================


class DisagreementResolver(ABC):
    """Abstract base class for disagreement resolution."""

    @property
    @abstractmethod
    def strategy_type(self) -> DisagreementStrategyType:
        """Return the strategy type enum."""
        pass

    @abstractmethod
    def resolve(
        self,
        responses: List[ModelResponse],
        context: Optional[ContextDict] = None,
    ) -> Tuple[Any, str]:
        """Resolve disagreement between responses.

        Args:
            responses: List of conflicting responses
            context: Additional context for resolution

        Returns:
            Tuple of (resolved_result, explanation)
        """
        pass


class DeferToCriticResolver(DisagreementResolver):
    """Resolver that defers to critic's assessment."""

    @property
    def strategy_type(self) -> DisagreementStrategyType:
        return DisagreementStrategyType.DEFER_TO_CRITIC

    def resolve(
        self,
        responses: List[ModelResponse],
        context: Optional[ContextDict] = None,
    ) -> Tuple[Any, str]:
        logger.debug("Resolving disagreement by deferring to critic")

        # Find critic response
        for response in responses:
            if response.model_type == "critic":
                return (
                    response.result,
                    "Deferred to critic's assessment as the authoritative source",
                )

        # No critic - fall back to highest confidence
        best = max(responses, key=lambda r: r.confidence)
        return (
            best.result,
            f"No critic response found. Selected highest confidence from {best.model_type}",
        )


class DeferToConfidenceResolver(DisagreementResolver):
    """Resolver that selects highest confidence response."""

    @property
    def strategy_type(self) -> DisagreementStrategyType:
        return DisagreementStrategyType.DEFER_TO_CONFIDENCE

    def resolve(
        self,
        responses: List[ModelResponse],
        context: Optional[ContextDict] = None,
    ) -> Tuple[Any, str]:
        logger.debug("Resolving disagreement by selecting highest confidence")

        best = max(responses, key=lambda r: r.confidence)
        return (
            best.result,
            f"Selected {best.model_type} response with highest confidence ({best.confidence:.2f})",
        )


class SynthesizeResolver(DisagreementResolver):
    """Resolver that tries to synthesize conflicting views."""

    @property
    def strategy_type(self) -> DisagreementStrategyType:
        return DisagreementStrategyType.SYNTHESIZE

    def resolve(
        self,
        responses: List[ModelResponse],
        context: Optional[ContextDict] = None,
    ) -> Tuple[Any, str]:
        logger.debug("Attempting to synthesize conflicting responses")

        # For string results, try to find common ground
        if all(isinstance(r.result, str) for r in responses):
            # Find common elements
            common_words = None
            for response in responses:
                words = set(response.result.lower().split())
                if common_words is None:
                    common_words = words
                else:
                    common_words &= words

            if common_words:
                # Use response with most common words
                best = max(
                    responses,
                    key=lambda r: len(set(r.result.lower().split()) & common_words),
                )
                return (
                    best.result,
                    f"Synthesized by selecting response with most common elements from {best.model_type}",
                )

        # Default to weighted average for confidence
        weighted_sum = sum(r.confidence for r in responses)
        if weighted_sum > 0:
            best = max(responses, key=lambda r: r.confidence)
            return (
                best.result,
                f"Could not synthesize - selected {best.model_type} with highest confidence",
            )

        return (responses[0].result, "Could not synthesize - returned first response")


class EscalateResolver(DisagreementResolver):
    """Resolver that escalates to human review."""

    @property
    def strategy_type(self) -> DisagreementStrategyType:
        return DisagreementStrategyType.ESCALATE

    def resolve(
        self,
        responses: List[ModelResponse],
        context: Optional[ContextDict] = None,
    ) -> Tuple[Any, str]:
        logger.warning("Escalating disagreement to human review")

        # Return best response but flag for review
        best = max(responses, key=lambda r: r.confidence)

        return (
            best.result,
            (
                f"ESCALATED: Disagreement requires human review. "
                f"Tentatively using {best.model_type} response."
            ),
        )


class ConservativeResolver(DisagreementResolver):
    """Resolver that defaults to the safest/most conservative option."""

    @property
    def strategy_type(self) -> DisagreementStrategyType:
        return DisagreementStrategyType.CONSERVATIVE

    def resolve(
        self,
        responses: List[ModelResponse],
        context: Optional[ContextDict] = None,
    ) -> Tuple[Any, str]:
        logger.debug("Resolving disagreement with conservative approach")

        # Prefer critic if available (as they tend to be conservative)
        for response in responses:
            if response.model_type == "critic":
                return (
                    response.result,
                    "Conservative resolution: deferred to critic's cautious assessment",
                )

        # Otherwise, select response with lowest confidence (most uncertain = most cautious)
        most_uncertain = min(responses, key=lambda r: r.confidence)
        return (
            most_uncertain.result,
            (
                f"Conservative resolution: selected {most_uncertain.model_type} "
                f"(most conservative/cautious response)"
            ),
        )


# Factory function for disagreement resolvers
def create_disagreement_resolver(
    strategy_type: DisagreementStrategyType,
) -> DisagreementResolver:
    """Create a disagreement resolver instance.

    Args:
        strategy_type: The type of resolver to create

    Returns:
        An instance of the requested resolver
    """
    resolvers: Dict[DisagreementStrategyType, Type[DisagreementResolver]] = {
        DisagreementStrategyType.DEFER_TO_CRITIC: DeferToCriticResolver,
        DisagreementStrategyType.DEFER_TO_CONFIDENCE: DeferToConfidenceResolver,
        DisagreementStrategyType.SYNTHESIZE: SynthesizeResolver,
        DisagreementStrategyType.ESCALATE: EscalateResolver,
        DisagreementStrategyType.CONSERVATIVE: ConservativeResolver,
    }

    if strategy_type not in resolvers:
        raise ValueError(f"Unknown disagreement strategy type: {strategy_type}")

    return resolvers[strategy_type]()


# =============================================================================
# Abstract Base Class for Orchestrator
# =============================================================================


class EnsembleOrchestratorBase(ABC):
    """Abstract base class for ensemble orchestration.

    Provides the interface for coordinating specialized LLMs. Concrete
    implementations can use different coordination strategies.
    """

    @abstractmethod
    def route_task(
        self,
        task_type: TaskType,
        input_data: TaskInputData,
        context: Optional[ContextDict] = None,
    ) -> OrchestrationResult:
        """Route a task to appropriate specialized model(s).

        Args:
            task_type: Type of task to perform
            input_data: Input data for the task
            context: Additional context

        Returns:
            OrchestrationResult with the final output
        """
        pass

    @abstractmethod
    def aggregate_responses(
        self,
        responses: List[ModelResponse],
        voting_strategy: Optional[VotingStrategyType] = None,
    ) -> VotingResult:
        """Aggregate responses from multiple models.

        Args:
            responses: List of model responses
            voting_strategy: Strategy to use for voting

        Returns:
            VotingResult with the aggregated decision
        """
        pass

    @abstractmethod
    def resolve_disagreement(
        self,
        responses: List[ModelResponse],
        strategy: Optional[DisagreementStrategyType] = None,
    ) -> Tuple[Any, str]:
        """Resolve disagreement between model responses.

        Args:
            responses: Conflicting responses
            strategy: Resolution strategy to use

        Returns:
            Tuple of (resolved_result, explanation)
        """
        pass

    @abstractmethod
    def get_model_status(self, model_type: str) -> ModelStatus:
        """Get the status of a specific model.

        Args:
            model_type: Type of model to check

        Returns:
            ModelStatus enum value
        """
        pass

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, ModelPerformanceMetrics]:
        """Get performance metrics for all models.

        Returns:
            Dictionary mapping model type to metrics
        """
        pass


# =============================================================================
# Main Orchestrator Implementation
# =============================================================================


class EnsembleOrchestrator(EnsembleOrchestratorBase):
    """Main implementation of the ensemble orchestrator.

    Coordinates LogicGenerator, Critic, Translator, and MetaReasoner LLMs
    to work together on complex tasks. Provides task routing, voting,
    disagreement resolution, and performance monitoring.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        llm_interface: Optional[LLMInterface] = None,
        logic_generator: Optional[LogicGeneratorLLM] = None,
        critic: Optional[CriticLLM] = None,
        translator: Optional[TranslatorLLM] = None,
        meta_reasoner: Optional[MetaReasonerLLM] = None,
    ):
        """Initialize the ensemble orchestrator.

        Args:
            config: Configuration for orchestration
            llm_interface: LLM interface for model calls
            logic_generator: Optional pre-configured LogicGeneratorLLM
            critic: Optional pre-configured CriticLLM
            translator: Optional pre-configured TranslatorLLM
            meta_reasoner: Optional pre-configured MetaReasonerLLM
        """
        self.config = config or OrchestratorConfig()
        self._llm_interface = llm_interface

        # Specialized models (lazy initialization)
        self._logic_generator = logic_generator
        self._critic = critic
        self._translator = translator
        self._meta_reasoner = meta_reasoner

        # Voting and resolution strategies
        self._voting_strategy = create_voting_strategy(
            self.config.default_voting_strategy
        )
        self._disagreement_resolver = create_disagreement_resolver(
            self.config.default_disagreement_strategy
        )

        # Performance tracking
        self._performance_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self._disagreement_history: List[DisagreementRecord] = []
        self._lock = threading.Lock()

        # Cache
        self._cache: Dict[str, Tuple[OrchestrationResult, float]] = {}
        self._cache_lock = threading.Lock()

        # Model status
        self._model_status: Dict[str, ModelStatus] = {}

        logger.info(
            f"EnsembleOrchestrator initialized with config: "
            f"voting={self.config.default_voting_strategy.value}, "
            f"disagreement={self.config.default_disagreement_strategy.value}"
        )

    @property
    def logic_generator(self) -> LogicGeneratorLLM:
        """Lazy initialization of LogicGenerator."""
        if self._logic_generator is None:
            if self._llm_interface is None:
                raise OrchestratorError(
                    "LLM interface required to initialize LogicGenerator"
                )
            self._logic_generator = LogicGeneratorLLM(
                config=LogicGeneratorConfig(),
                llm_interface=self._llm_interface,
            )
            self._model_status["logic_generator"] = ModelStatus.AVAILABLE
        return self._logic_generator

    @property
    def critic(self) -> CriticLLM:
        """Lazy initialization of Critic."""
        if self._critic is None:
            if self._llm_interface is None:
                raise OrchestratorError("LLM interface required to initialize Critic")
            self._critic = CriticLLM(
                config=CriticConfig(),
                llm_interface=self._llm_interface,
            )
            self._model_status["critic"] = ModelStatus.AVAILABLE
        return self._critic

    @property
    def translator(self) -> TranslatorLLM:
        """Lazy initialization of Translator."""
        if self._translator is None:
            if self._llm_interface is None:
                raise OrchestratorError(
                    "LLM interface required to initialize Translator"
                )
            self._translator = TranslatorLLM(
                config=TranslatorConfig(),
                llm_interface=self._llm_interface,
            )
            self._model_status["translator"] = ModelStatus.AVAILABLE
        return self._translator

    @property
    def meta_reasoner(self) -> MetaReasonerLLM:
        """Lazy initialization of MetaReasoner."""
        if self._meta_reasoner is None:
            if self._llm_interface is None:
                raise OrchestratorError(
                    "LLM interface required to initialize MetaReasoner"
                )
            self._meta_reasoner = MetaReasonerLLM(
                config=MetaReasonerConfig(),
                llm_interface=self._llm_interface,
            )
            self._model_status["meta_reasoner"] = ModelStatus.AVAILABLE
        return self._meta_reasoner

    def _get_cache_key(
        self,
        task_type: TaskType,
        input_data: TaskInputData,
        context: Optional[ContextDict],
    ) -> str:
        """Generate a cache key for the given task."""
        key_parts = [task_type.value, str(input_data)]
        if context:
            key_parts.append(str(sorted(context.items())))
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _check_cache(self, cache_key: str) -> Optional[OrchestrationResult]:
        """Check if result is in cache and still valid."""
        if not self.config.enable_caching:
            return None

        with self._cache_lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self.config.cache_ttl_seconds:
                    logger.debug(f"Cache hit for key {cache_key}")
                    result.from_cache = True
                    return result
                else:
                    # Expired
                    del self._cache[cache_key]
        return None

    def _store_cache(self, cache_key: str, result: OrchestrationResult) -> None:
        """Store result in cache."""
        if not self.config.enable_caching:
            return

        with self._cache_lock:
            self._cache[cache_key] = (result, time.time())

    def _update_performance_metrics(
        self,
        model_type: str,
        success: bool,
        latency_ms: float,
        confidence: float,
        error_type: Optional[str] = None,
    ) -> None:
        """Update performance metrics for a model."""
        if not self.config.enable_performance_tracking:
            return

        with self._lock:
            if model_type not in self._performance_metrics:
                self._performance_metrics[model_type] = ModelPerformanceMetrics(
                    model_type=model_type
                )

            metrics = self._performance_metrics[model_type]
            metrics.total_requests += 1

            if success:
                metrics.successful_requests += 1
                metrics.last_success = time.time()
                # Update running averages
                n = metrics.successful_requests
                metrics.average_latency_ms = (
                    metrics.average_latency_ms * (n - 1) + latency_ms
                ) / n
                metrics.average_confidence = (
                    metrics.average_confidence * (n - 1) + confidence
                ) / n
            else:
                metrics.failed_requests += 1
                metrics.last_failure = time.time()
                if error_type:
                    metrics.error_types[error_type] = (
                        metrics.error_types.get(error_type, 0) + 1
                    )

    def route_task(
        self,
        task_type: TaskType,
        input_data: TaskInputData,
        context: Optional[ContextDict] = None,
    ) -> OrchestrationResult:
        """Route a task to appropriate specialized model(s).

        Args:
            task_type: Type of task to perform
            input_data: Input data for the task (see TypedDicts for structure)
            context: Additional context

        Returns:
            OrchestrationResult with the final output

        Raises:
            TaskRoutingError: If input_data is None or empty
        """
        # Validate input_data
        if input_data is None:
            raise TaskRoutingError("input_data cannot be None")
        if isinstance(input_data, str) and not input_data.strip():
            raise TaskRoutingError("input_data cannot be empty")
        if isinstance(input_data, dict) and not input_data:
            raise TaskRoutingError("input_data cannot be an empty dictionary")
        if isinstance(input_data, list) and not input_data:
            raise TaskRoutingError("input_data cannot be an empty list")

        start_time = time.time()
        logger.info(f"Routing task: {task_type.value}")

        # Check cache
        cache_key = self._get_cache_key(task_type, input_data, context)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result

        try:
            if task_type == TaskType.RULE_GENERATION:
                result = self._route_rule_generation(input_data, context)
            elif task_type == TaskType.RULE_CRITICISM:
                result = self._route_rule_criticism(input_data, context)
            elif task_type == TaskType.TRANSLATION_TO_NL:
                result = self._route_translation_to_nl(input_data, context)
            elif task_type == TaskType.TRANSLATION_TO_ASP:
                result = self._route_translation_to_asp(input_data, context)
            elif task_type == TaskType.META_ANALYSIS:
                result = self._route_meta_analysis(input_data, context)
            elif task_type == TaskType.FULL_PIPELINE:
                result = self._route_full_pipeline(input_data, context)
            else:
                raise TaskRoutingError(f"Unknown task type: {task_type}")

            result.total_latency_ms = (time.time() - start_time) * 1000
            result.task_type = task_type

            # Cache the result
            self._store_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Task routing failed: {e}")

            # Determine primary model for error reporting (Issue #201)
            primary_model_map = {
                TaskType.RULE_GENERATION: "logic_generator",
                TaskType.RULE_CRITICISM: "critic",
                TaskType.TRANSLATION_TO_NL: "translator",
                TaskType.TRANSLATION_TO_ASP: "translator",
                TaskType.META_ANALYSIS: "meta_reasoner",
                TaskType.FULL_PIPELINE: "logic_generator",
            }
            primary_model = primary_model_map.get(task_type, "unknown")

            if self.config.enable_fallback:
                return self._handle_fallback(
                    task_type, input_data, context, e, primary_model
                )

            # Raise aggregated error with detailed context (Issue #201)
            model_error = ModelError(
                model_id=primary_model,
                error_type=type(e).__name__,
                message=str(e),
                context={"task_type": task_type.value},
            )
            raise AggregatedOrchestrationError(
                message=f"Task routing failed for {task_type.value}",
                model_errors=[model_error],
                attempted_models=[primary_model],
                task_type=task_type,
            ) from e

    def _route_rule_generation(
        self,
        input_data: TaskInputData,
        context: Optional[ContextDict],
    ) -> OrchestrationResult:
        """Route rule generation task."""
        logger.debug("Routing to LogicGenerator")
        start = time.time()

        # Extract parameters from input_data
        if isinstance(input_data, dict):
            principle = input_data.get("principle", str(input_data))
            domain = input_data.get("domain", "legal")
            predicates = input_data.get("predicates", [])
        else:
            principle = str(input_data)
            domain = context.get("domain", "legal") if context else "legal"
            predicates = context.get("predicates", []) if context else []

        try:
            result = self.logic_generator.generate_rule(
                principle=principle,
                domain=domain,
                available_predicates=predicates,
                context=context,
            )

            latency_ms = (time.time() - start) * 1000
            self._update_performance_metrics(
                "logic_generator",
                success=True,
                latency_ms=latency_ms,
                confidence=result.confidence,
            )

            response = ModelResponse(
                model_type="logic_generator",
                result=result,
                confidence=result.confidence,
                latency_ms=latency_ms,
            )

            return OrchestrationResult(
                task_type=TaskType.RULE_GENERATION,
                final_result=result,
                model_responses=[response],
            )

        except Exception as e:
            logger.error(f"Rule generation failed: {e}")
            self._update_performance_metrics(
                "logic_generator",
                success=False,
                latency_ms=(time.time() - start) * 1000,
                confidence=0.0,
                error_type=type(e).__name__,
            )
            raise

    def _route_rule_criticism(
        self,
        input_data: TaskInputData,
        context: Optional[ContextDict],
    ) -> OrchestrationResult:
        """Route rule criticism task."""
        logger.debug("Routing to Critic")
        start = time.time()

        # Extract rule from input
        if isinstance(input_data, dict):
            rule = input_data.get("rule", str(input_data))
            domain = input_data.get("domain", "legal")
            existing_rules = input_data.get("existing_rules", [])
        else:
            rule = str(input_data)
            domain = context.get("domain", "legal") if context else "legal"
            existing_rules = context.get("existing_rules", []) if context else []

        try:
            result = self.critic.analyze_rule(
                rule=rule,
                domain=domain,
                existing_rules=existing_rules,
            )

            latency_ms = (time.time() - start) * 1000
            self._update_performance_metrics(
                "critic",
                success=True,
                latency_ms=latency_ms,
                confidence=result.overall_quality_score,
            )

            response = ModelResponse(
                model_type="critic",
                result=result,
                confidence=result.overall_quality_score,
                latency_ms=latency_ms,
            )

            return OrchestrationResult(
                task_type=TaskType.RULE_CRITICISM,
                final_result=result,
                model_responses=[response],
            )

        except Exception as e:
            logger.error(f"Rule criticism failed: {e}")
            self._update_performance_metrics(
                "critic",
                success=False,
                latency_ms=(time.time() - start) * 1000,
                confidence=0.0,
                error_type=type(e).__name__,
            )
            raise

    def _route_translation_to_nl(
        self,
        input_data: TaskInputData,
        context: Optional[ContextDict],
    ) -> OrchestrationResult:
        """Route ASP to NL translation task."""
        logger.debug("Routing to Translator (ASP -> NL)")
        start = time.time()

        if isinstance(input_data, dict):
            asp_rule = input_data.get("rule", str(input_data))
            domain = input_data.get("domain", "legal")
        else:
            asp_rule = str(input_data)
            domain = context.get("domain", "legal") if context else "legal"

        try:
            result = self.translator.asp_to_natural_language(
                asp_rule=asp_rule,
                domain=domain,
                context=context,
            )

            latency_ms = (time.time() - start) * 1000
            self._update_performance_metrics(
                "translator",
                success=True,
                latency_ms=latency_ms,
                confidence=result.confidence,
            )

            response = ModelResponse(
                model_type="translator",
                result=result,
                confidence=result.confidence,
                latency_ms=latency_ms,
            )

            return OrchestrationResult(
                task_type=TaskType.TRANSLATION_TO_NL,
                final_result=result,
                model_responses=[response],
            )

        except Exception as e:
            logger.error(f"Translation to NL failed: {e}")
            self._update_performance_metrics(
                "translator",
                success=False,
                latency_ms=(time.time() - start) * 1000,
                confidence=0.0,
                error_type=type(e).__name__,
            )
            raise

    def _route_translation_to_asp(
        self,
        input_data: TaskInputData,
        context: Optional[ContextDict],
    ) -> OrchestrationResult:
        """Route NL to ASP translation task."""
        logger.debug("Routing to Translator (NL -> ASP)")
        start = time.time()

        if isinstance(input_data, dict):
            nl_text = input_data.get("text", str(input_data))
            domain = input_data.get("domain", "legal")
            predicates = input_data.get("predicates", [])
        else:
            nl_text = str(input_data)
            domain = context.get("domain", "legal") if context else "legal"
            predicates = context.get("predicates", []) if context else []

        try:
            result = self.translator.natural_language_to_asp(
                natural_language=nl_text,
                domain=domain,
                available_predicates=predicates,
                context=context,
            )

            latency_ms = (time.time() - start) * 1000
            self._update_performance_metrics(
                "translator",
                success=True,
                latency_ms=latency_ms,
                confidence=result.confidence,
            )

            response = ModelResponse(
                model_type="translator",
                result=result,
                confidence=result.confidence,
                latency_ms=latency_ms,
            )

            return OrchestrationResult(
                task_type=TaskType.TRANSLATION_TO_ASP,
                final_result=result,
                model_responses=[response],
            )

        except Exception as e:
            logger.error(f"Translation to ASP failed: {e}")
            self._update_performance_metrics(
                "translator",
                success=False,
                latency_ms=(time.time() - start) * 1000,
                confidence=0.0,
                error_type=type(e).__name__,
            )
            raise

    def _route_meta_analysis(
        self,
        input_data: TaskInputData,
        context: Optional[ContextDict],
    ) -> OrchestrationResult:
        """Route meta-analysis task."""
        logger.debug("Routing to MetaReasoner")
        start = time.time()

        if isinstance(input_data, dict):
            failures = input_data.get("failures", [])
            insights = input_data.get("insights", [])
        else:
            failures = []
            insights = []

        try:
            result = self.meta_reasoner.analyze_reasoning_patterns(
                failures=failures,
                insights=insights,
                context=context,
            )

            latency_ms = (time.time() - start) * 1000
            self._update_performance_metrics(
                "meta_reasoner",
                success=True,
                latency_ms=latency_ms,
                confidence=result.confidence if hasattr(result, "confidence") else 0.8,
            )

            response = ModelResponse(
                model_type="meta_reasoner",
                result=result,
                confidence=result.confidence if hasattr(result, "confidence") else 0.8,
                latency_ms=latency_ms,
            )

            return OrchestrationResult(
                task_type=TaskType.META_ANALYSIS,
                final_result=result,
                model_responses=[response],
            )

        except Exception as e:
            logger.error(f"Meta analysis failed: {e}")
            self._update_performance_metrics(
                "meta_reasoner",
                success=False,
                latency_ms=(time.time() - start) * 1000,
                confidence=0.0,
                error_type=type(e).__name__,
            )
            raise

    def _route_full_pipeline(
        self,
        input_data: TaskInputData,
        context: Optional[ContextDict],
    ) -> OrchestrationResult:
        """Route through full pipeline: Generate -> Critique -> Refine.

        This is the main orchestration flow that uses multiple models.
        """
        logger.info("Running full orchestration pipeline")
        start = time.time()
        all_responses: List[ModelResponse] = []
        disagreements: List[DisagreementRecord] = []

        # Step 1: Generate rule
        logger.debug("Pipeline step 1: Generating rule")
        gen_result = self._route_rule_generation(input_data, context)
        all_responses.extend(gen_result.model_responses)
        generated_rule = gen_result.final_result

        # Step 2: Critique the generated rule
        logger.debug("Pipeline step 2: Critiquing rule")
        critic_input = {
            "rule": (
                generated_rule.asp_rule
                if hasattr(generated_rule, "asp_rule")
                else str(generated_rule)
            ),
            "domain": context.get("domain", "legal") if context else "legal",
        }
        critic_result = self._route_rule_criticism(critic_input, context)
        all_responses.extend(critic_result.model_responses)
        criticism = critic_result.final_result

        # Step 3: Check if refinement is needed based on critic's assessment
        if hasattr(criticism, "recommendation"):
            if criticism.recommendation == "reject":
                # Rule rejected - try to regenerate with criticism feedback
                logger.info("Rule rejected by critic - attempting refinement")
                refined_context = context.copy() if context else {}
                refined_context["criticism"] = criticism
                refined_context["edge_cases"] = [
                    ec.description for ec in criticism.edge_cases
                ]

                # Regenerate with feedback
                refined_result = self._route_rule_generation(
                    input_data, refined_context
                )
                all_responses.extend(refined_result.model_responses)

                # Record disagreement
                disagreements.append(
                    DisagreementRecord(
                        task_type=TaskType.FULL_PIPELINE,
                        conflicting_responses=[
                            gen_result.model_responses[0],
                            critic_result.model_responses[0],
                        ],
                        resolution="regenerated_with_feedback",
                        resolution_strategy=DisagreementStrategyType.SYNTHESIZE,
                        final_decision=refined_result.final_result,
                    )
                )

                final_result = refined_result.final_result
            else:
                final_result = generated_rule
        else:
            final_result = generated_rule

        # Step 4: Vote on final result if we have multiple valid outputs
        voting_result = None
        if len(all_responses) > 1:
            voting_result = self.aggregate_responses(all_responses)

        total_latency = (time.time() - start) * 1000

        return OrchestrationResult(
            task_type=TaskType.FULL_PIPELINE,
            final_result=final_result,
            model_responses=all_responses,
            voting_result=voting_result,
            disagreements=disagreements,
            total_latency_ms=total_latency,
            metadata={
                "criticism": criticism,
                "pipeline_steps": (
                    ["generate", "critique", "refine"]
                    if disagreements
                    else ["generate", "critique"]
                ),
            },
        )

    def _handle_fallback(
        self,
        task_type: TaskType,
        input_data: TaskInputData,
        context: Optional[ContextDict],
        original_error: Exception,
        primary_model: str = "unknown",
    ) -> OrchestrationResult:
        """Handle fallback when primary routing fails.

        Tracks all attempted models and their errors for comprehensive error
        reporting (Issue #201).

        Args:
            task_type: Type of task being performed
            input_data: Input data for the task
            context: Additional context
            original_error: The original error from the primary model
            primary_model: ID of the primary model that failed

        Returns:
            OrchestrationResult with fallback result and error details

        Raises:
            AggregatedOrchestrationError: When all fallbacks are exhausted
        """
        logger.warning(f"Attempting fallback for {task_type.value}")

        # Track all errors that occurred (Issue #201)
        all_errors: List[ModelError] = []
        attempted_models: List[str] = [primary_model]

        # Record the original error
        original_model_error = ModelError(
            model_id=primary_model,
            error_type=type(original_error).__name__,
            message=str(original_error),
            context={"task_type": task_type.value, "role": "primary"},
        )
        all_errors.append(original_model_error)

        # Try alternative models based on task type
        fallback_models = self._get_fallback_models(task_type)

        for fallback_model in fallback_models:
            attempted_models.append(fallback_model)
            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                # Simplified fallback - just return a placeholder
                # In a full implementation, this would actually call the fallback model
                return OrchestrationResult(
                    task_type=task_type,
                    final_result=None,
                    errors=all_errors,
                    failed_models=[e.model_id for e in all_errors],
                    metadata={
                        "fallback": True,
                        "fallback_model": fallback_model,
                        "original_error": str(original_error),
                        "attempted_models": attempted_models,
                    },
                )
            except Exception as e:
                logger.warning(f"Fallback {fallback_model} also failed: {e}")
                fallback_error = ModelError(
                    model_id=fallback_model,
                    error_type=type(e).__name__,
                    message=str(e),
                    context={"task_type": task_type.value, "role": "fallback"},
                )
                all_errors.append(fallback_error)
                continue

        # All fallbacks exhausted - raise aggregated error with full context
        raise AggregatedOrchestrationError(
            message=f"All fallbacks exhausted for {task_type.value}",
            model_errors=all_errors,
            attempted_models=attempted_models,
            task_type=task_type,
        )

    def _get_fallback_models(self, task_type: TaskType) -> List[str]:
        """Get list of fallback models for a task type."""
        fallbacks = {
            TaskType.RULE_GENERATION: ["translator", "meta_reasoner"],
            TaskType.RULE_CRITICISM: ["meta_reasoner"],
            TaskType.TRANSLATION_TO_NL: ["logic_generator"],
            TaskType.TRANSLATION_TO_ASP: ["logic_generator"],
            TaskType.META_ANALYSIS: ["critic"],
            TaskType.FULL_PIPELINE: [],
        }
        return fallbacks.get(task_type, [])

    def aggregate_responses(
        self,
        responses: List[ModelResponse],
        voting_strategy: Optional[VotingStrategyType] = None,
    ) -> VotingResult:
        """Aggregate responses from multiple models using voting.

        Args:
            responses: List of model responses to aggregate
            voting_strategy: Strategy to use (defaults to config)

        Returns:
            VotingResult with the aggregated decision
        """
        if not responses:
            raise VotingError("No responses to aggregate")

        strategy = (
            create_voting_strategy(voting_strategy)
            if voting_strategy
            else self._voting_strategy
        )

        logger.debug(
            f"Aggregating {len(responses)} responses using {strategy.strategy_type.value}"
        )

        return strategy.vote(responses)

    def resolve_disagreement(
        self,
        responses: List[ModelResponse],
        strategy: Optional[DisagreementStrategyType] = None,
    ) -> Tuple[Any, str]:
        """Resolve disagreement between model responses.

        Args:
            responses: Conflicting responses to resolve
            strategy: Resolution strategy (defaults to config)

        Returns:
            Tuple of (resolved_result, explanation)
        """
        if not responses:
            raise DisagreementResolutionError("No responses to resolve")

        resolver = (
            create_disagreement_resolver(strategy)
            if strategy
            else self._disagreement_resolver
        )

        logger.debug(f"Resolving disagreement using {resolver.strategy_type.value}")

        result, explanation = resolver.resolve(responses)

        # Record disagreement
        with self._lock:
            self._disagreement_history.append(
                DisagreementRecord(
                    task_type=TaskType.RULE_GENERATION,  # Default, should be passed
                    conflicting_responses=responses,
                    resolution=explanation,
                    resolution_strategy=resolver.strategy_type,
                    final_decision=result,
                )
            )

        return result, explanation

    def get_model_status(self, model_type: str) -> ModelStatus:
        """Get the status of a specific model.

        Args:
            model_type: Type of model to check

        Returns:
            ModelStatus enum value
        """
        return self._model_status.get(model_type, ModelStatus.UNAVAILABLE)

    def get_performance_metrics(self) -> Dict[str, ModelPerformanceMetrics]:
        """Get performance metrics for all models.

        Returns:
            Dictionary mapping model type to metrics
        """
        with self._lock:
            return dict(self._performance_metrics)

    def get_disagreement_history(self) -> List[DisagreementRecord]:
        """Get history of disagreements.

        Returns:
            List of disagreement records
        """
        with self._lock:
            return list(self._disagreement_history)

    def optimize_routing(
        self,
        task_type: TaskType,
    ) -> Dict[str, float]:
        """Optimize routing based on historical performance.

        Analyzes performance metrics to suggest optimal model selection
        weights for a given task type.

        Args:
            task_type: Type of task to optimize

        Returns:
            Dictionary mapping model type to suggested weight
        """
        logger.debug(f"Optimizing routing for {task_type.value}")

        with self._lock:
            weights: Dict[str, float] = {}

            for model_type, metrics in self._performance_metrics.items():
                if metrics.total_requests == 0:
                    weights[model_type] = 0.5  # Default weight for untested models
                else:
                    # Weight based on success rate and confidence
                    success_weight = metrics.success_rate
                    confidence_weight = metrics.average_confidence
                    # Penalize high latency
                    latency_penalty = min(
                        1.0, 1000 / (metrics.average_latency_ms + 100)
                    )

                    weights[model_type] = (
                        success_weight * 0.4
                        + confidence_weight * 0.4
                        + latency_penalty * 0.2
                    )

            # Normalize weights
            total = sum(weights.values()) or 1.0
            return {k: v / total for k, v in weights.items()}

    def clear_cache(self) -> int:
        """Clear the result cache.

        Returns:
            Number of entries cleared
        """
        with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared {count} cached entries")
            return count

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self._lock:
            self._performance_metrics.clear()
            self._disagreement_history.clear()
            logger.info("Reset all performance metrics")
