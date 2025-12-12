"""
LLM interface for question answering with structured outputs.

Provides core data structures and base classes for interacting with
multiple LLM providers (Anthropic, OpenAI, local models).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel

if TYPE_CHECKING:
    from loft.autonomous.llm_metrics import LLMMetricsTracker, OperationType

T = TypeVar("T", bound=BaseModel)


@dataclass
class ResponseMetadata:
    """Metadata about an LLM response."""

    model: str  # Model identifier used
    tokens_input: int  # Input tokens consumed
    tokens_output: int  # Output tokens generated
    tokens_total: int  # Total tokens
    latency_ms: float  # Response latency in milliseconds
    cost_usd: float  # Estimated cost in USD
    timestamp: str  # ISO timestamp
    provider: str  # Provider name
    retries: int = 0  # Number of retries needed
    cache_hit: bool = False  # Whether response came from cache

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_total": self.tokens_total,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "retries": self.retries,
            "cache_hit": self.cache_hit,
        }


@dataclass
class LLMQuery:
    """
    Request structure for LLM queries.

    Encapsulates all information needed to make an LLM request with
    structured output support.
    """

    question: str  # The question/prompt to send
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    output_schema: Optional[Type[BaseModel]] = None  # Pydantic schema for response
    temperature: float = 0.7  # Sampling temperature (0.0-2.0)
    max_tokens: int = 4096  # Maximum tokens in response
    system_prompt: Optional[str] = None  # System prompt override
    few_shot_examples: list[str] = field(default_factory=list)  # Example responses
    cot_enabled: bool = False  # Enable chain-of-thought scaffolding

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "context": self.context,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "few_shot_examples": self.few_shot_examples,
            "cot_enabled": self.cot_enabled,
        }


@dataclass
class LLMResponse:
    """
    Response structure from LLM queries.

    Wraps both structured and unstructured outputs with metadata.
    """

    content: BaseModel  # Parsed structured output
    raw_text: str  # Full LLM response text
    confidence: float  # Model's uncertainty estimate (0.0-1.0)
    metadata: ResponseMetadata  # Response metadata
    reasoning: Optional[str] = None  # Chain-of-thought reasoning if available

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content.model_dump() if self.content else None,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "metadata": self.metadata.to_dict(),
            "reasoning": self.reasoning,
        }


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations (Anthropic, OpenAI, local) must
    implement these methods.
    """

    def __init__(self, api_key: str, model: str, **kwargs: Any):
        """
        Initialize provider.

        Args:
            api_key: API key for the provider
            model: Model identifier
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.model = model
        self.config = kwargs

    @abstractmethod
    def query(
        self,
        llm_query: LLMQuery,
        response_model: Optional[Type[T]] = None,
    ) -> LLMResponse:
        """
        Send a query to the LLM.

        Args:
            llm_query: The query to send
            response_model: Optional Pydantic model for structured output

        Returns:
            LLMResponse with parsed output

        Raises:
            LLMProviderError: On API or parsing errors
        """
        pass

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost in USD for a given token count.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        pass


class LLMInterface:
    """
    Main interface for LLM interactions.

    Manages multiple providers, caching, retries, and cost tracking.
    Optionally integrates with LLMMetricsTracker for autonomous run monitoring.
    """

    def __init__(
        self,
        provider: LLMProvider,
        enable_cache: bool = True,
        max_retries: int = 3,
        metrics_tracker: Optional["LLMMetricsTracker"] = None,
        default_operation_type: Optional["OperationType"] = None,
    ):
        """
        Initialize LLM interface.

        Args:
            provider: LLM provider instance
            enable_cache: Whether to enable response caching
            max_retries: Maximum number of retries on failures
            metrics_tracker: Optional LLMMetricsTracker for autonomous run monitoring
            default_operation_type: Default operation type for metrics tracking
        """
        self.provider = provider
        self.enable_cache = enable_cache
        self.max_retries = max_retries
        self._metrics_tracker = metrics_tracker
        self._default_operation_type = default_operation_type
        self._cache: Dict[str, LLMResponse] = {}
        self._total_cost = 0.0
        self._total_tokens = 0

    def query(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Type[T]] = None,
        operation_type: Optional["OperationType"] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Query the LLM with optional structured output.

        Args:
            question: The question to ask
            context: Additional context dict
            output_schema: Pydantic model for structured response
            operation_type: Operation type for metrics tracking (uses default if None)
            **kwargs: Additional query parameters

        Returns:
            LLMResponse with parsed output
        """
        import time

        llm_query = LLMQuery(
            question=question,
            context=context or {},
            output_schema=output_schema,
            **kwargs,
        )

        # Check cache
        cache_key = self._get_cache_key(llm_query)
        if self.enable_cache and cache_key in self._cache:
            cached_response = self._cache[cache_key]
            cached_response.metadata.cache_hit = True
            return cached_response

        # Query provider with timing
        start_time = time.time()
        success = True
        try:
            response = self.provider.query(llm_query, output_schema)
        except Exception:
            success = False
            raise
        finally:
            duration_seconds = time.time() - start_time
            # Record metrics if tracker is available
            if self._metrics_tracker is not None and success:
                # Get or use fallback for operation type
                from loft.autonomous.llm_metrics import (
                    OperationType,
                    get_global_metrics_tracker,
                )

                op_type = operation_type or self._default_operation_type or OperationType.OTHER
                self._metrics_tracker.record_call(
                    operation_type=op_type,
                    input_tokens=response.metadata.tokens_input,
                    output_tokens=response.metadata.tokens_output,
                    success=success,
                    duration_seconds=duration_seconds,
                    model=response.metadata.model,
                )
            elif self._metrics_tracker is None:
                # Try global tracker as fallback
                from loft.autonomous.llm_metrics import (
                    OperationType,
                    get_global_metrics_tracker,
                )

                global_tracker = get_global_metrics_tracker()
                if global_tracker is not None and success:
                    op_type = operation_type or self._default_operation_type or OperationType.OTHER
                    global_tracker.record_call(
                        operation_type=op_type,
                        input_tokens=response.metadata.tokens_input,
                        output_tokens=response.metadata.tokens_output,
                        success=success,
                        duration_seconds=duration_seconds,
                        model=response.metadata.model,
                    )

        # Update tracking
        self._total_cost += response.metadata.cost_usd
        self._total_tokens += response.metadata.tokens_total

        # Cache response
        if self.enable_cache:
            self._cache[cache_key] = response

        return response

    def _get_cache_key(self, query: LLMQuery) -> str:
        """Generate cache key for a query."""
        import hashlib
        import json

        # Create deterministic hash of query
        query_dict = query.to_dict()
        query_str = json.dumps(query_dict, sort_keys=True)
        return hashlib.sha256(query_str.encode()).hexdigest()

    def get_total_cost(self) -> float:
        """Get total cost of all queries."""
        return self._total_cost

    def get_total_tokens(self) -> int:
        """Get total tokens consumed."""
        return self._total_tokens

    def set_metrics_tracker(self, tracker: "LLMMetricsTracker") -> None:
        """Set the metrics tracker for this interface.

        Args:
            tracker: LLMMetricsTracker instance for recording calls
        """
        self._metrics_tracker = tracker

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
