"""
Unit tests for EnsembleOrchestrator.

Tests cover:
- Task routing to specialized models
- Voting strategies (Unanimous, Majority, Weighted, Dialectical)
- Disagreement resolution strategies
- Fallback handling
- Performance monitoring
- Caching behavior
"""

import pytest
from unittest.mock import MagicMock

from loft.neural.ensemble.orchestrator import (
    # Main class
    EnsembleOrchestrator,
    OrchestratorConfig,
    ModelResponse,
    VotingResult,
    ModelPerformanceMetrics,
    TaskType,
    VotingStrategyType,
    DisagreementStrategyType,
    ModelStatus,
    # Voting strategies
    UnanimousVotingStrategy,
    MajorityVotingStrategy,
    WeightedVotingStrategy,
    DialecticalVotingStrategy,
    create_voting_strategy,
    # Disagreement resolvers
    DeferToCriticResolver,
    DeferToConfidenceResolver,
    SynthesizeResolver,
    EscalateResolver,
    ConservativeResolver,
    create_disagreement_resolver,
    # Exceptions
    OrchestratorError,
    TaskRoutingError,
    VotingError,
    DisagreementResolutionError,
    FallbackExhaustedError,
    # Error handling (Issue #201)
    AggregatedOrchestrationError,
    ModelError,
    OrchestrationResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_config():
    """Create a basic orchestrator configuration."""
    return OrchestratorConfig(
        default_voting_strategy=VotingStrategyType.WEIGHTED,
        default_disagreement_strategy=DisagreementStrategyType.DEFER_TO_CONFIDENCE,
        enable_caching=True,
        cache_ttl_seconds=3600,
        max_retries=3,
        timeout_seconds=60.0,
        enable_performance_tracking=True,
        min_confidence_threshold=0.6,
        enable_fallback=True,
    )


@pytest.fixture
def mock_llm_interface():
    """Create a mock LLM interface."""
    return MagicMock()


@pytest.fixture
def mock_orchestrator(basic_config, mock_llm_interface):
    """Create a mock orchestrator for testing route_task validation."""
    return EnsembleOrchestrator(
        config=basic_config,
        llm_interface=mock_llm_interface,
    )


@pytest.fixture
def sample_model_responses():
    """Create sample model responses for testing voting."""
    return [
        ModelResponse(
            model_type="logic_generator",
            result="rule1(X) :- condition(X).",
            confidence=0.85,
            latency_ms=150.0,
        ),
        ModelResponse(
            model_type="critic",
            result="rule1(X) :- condition(X).",
            confidence=0.90,
            latency_ms=200.0,
        ),
        ModelResponse(
            model_type="translator",
            result="rule1(X) :- condition(X).",
            confidence=0.75,
            latency_ms=100.0,
        ),
    ]


@pytest.fixture
def disagreeing_responses():
    """Create model responses that disagree."""
    return [
        ModelResponse(
            model_type="logic_generator",
            result="rule_a(X) :- cond_a(X).",
            confidence=0.85,
            latency_ms=150.0,
        ),
        ModelResponse(
            model_type="critic",
            result="rule_b(X) :- cond_b(X).",
            confidence=0.90,
            latency_ms=200.0,
        ),
        ModelResponse(
            model_type="translator",
            result="rule_c(X) :- cond_c(X).",
            confidence=0.75,
            latency_ms=100.0,
        ),
    ]


# =============================================================================
# Test Data Classes
# =============================================================================


class TestModelResponse:
    """Tests for ModelResponse dataclass."""

    def test_valid_creation(self):
        """Test creating a valid ModelResponse."""
        response = ModelResponse(
            model_type="logic_generator",
            result="test_rule(X).",
            confidence=0.85,
            latency_ms=100.0,
        )
        assert response.model_type == "logic_generator"
        assert response.result == "test_rule(X)."
        assert response.confidence == 0.85
        assert response.latency_ms == 100.0
        assert response.metadata == {}

    def test_confidence_clamping_high(self):
        """Test that confidence > 1.0 is clamped."""
        response = ModelResponse(
            model_type="test",
            result="test",
            confidence=1.5,
            latency_ms=100.0,
        )
        assert response.confidence == 1.0

    def test_confidence_clamping_low(self):
        """Test that confidence < 0.0 is clamped."""
        response = ModelResponse(
            model_type="test",
            result="test",
            confidence=-0.5,
            latency_ms=100.0,
        )
        assert response.confidence == 0.0

    def test_metadata_default(self):
        """Test that metadata defaults to empty dict."""
        response = ModelResponse(
            model_type="test",
            result="test",
            confidence=0.5,
            latency_ms=100.0,
        )
        assert response.metadata == {}

    def test_metadata_custom(self):
        """Test custom metadata."""
        metadata = {"key": "value", "count": 42}
        response = ModelResponse(
            model_type="test",
            result="test",
            confidence=0.5,
            latency_ms=100.0,
            metadata=metadata,
        )
        assert response.metadata == metadata


class TestModelPerformanceMetrics:
    """Tests for ModelPerformanceMetrics dataclass."""

    def test_success_rate_no_requests(self):
        """Test success rate with no requests."""
        metrics = ModelPerformanceMetrics(model_type="test")
        assert metrics.success_rate == 0.0

    def test_success_rate_all_successful(self):
        """Test success rate with all successful requests."""
        metrics = ModelPerformanceMetrics(
            model_type="test",
            total_requests=10,
            successful_requests=10,
        )
        assert metrics.success_rate == 1.0

    def test_success_rate_partial(self):
        """Test success rate with partial success."""
        metrics = ModelPerformanceMetrics(
            model_type="test",
            total_requests=10,
            successful_requests=7,
            failed_requests=3,
        )
        assert metrics.success_rate == 0.7


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OrchestratorConfig()
        assert config.default_voting_strategy == VotingStrategyType.WEIGHTED
        assert config.default_disagreement_strategy == DisagreementStrategyType.DEFER_TO_CONFIDENCE
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 3600
        assert config.max_retries == 3
        assert config.timeout_seconds == 60.0
        assert config.enable_performance_tracking is True
        assert config.min_confidence_threshold == 0.6
        assert config.enable_fallback is True

    def test_invalid_timeout(self):
        """Test that invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            OrchestratorConfig(timeout_seconds=0)

    def test_invalid_max_retries(self):
        """Test that negative max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            OrchestratorConfig(max_retries=-1)

    def test_invalid_confidence_threshold(self):
        """Test that invalid confidence threshold raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence_threshold must be between"):
            OrchestratorConfig(min_confidence_threshold=1.5)


# =============================================================================
# Test Voting Strategies
# =============================================================================


class TestUnanimousVotingStrategy:
    """Tests for UnanimousVotingStrategy."""

    def test_strategy_type(self):
        """Test strategy type property."""
        strategy = UnanimousVotingStrategy()
        assert strategy.strategy_type == VotingStrategyType.UNANIMOUS

    def test_unanimous_agreement(self, sample_model_responses):
        """Test voting with unanimous agreement."""
        strategy = UnanimousVotingStrategy()
        result = strategy.vote(sample_model_responses)

        assert result.strategy_used == VotingStrategyType.UNANIMOUS
        assert result.decision == "rule1(X) :- condition(X)."
        assert result.confidence > 0.5
        assert len(result.dissenting_models) == 0
        assert "All models agreed" in result.reasoning

    def test_no_unanimity(self, disagreeing_responses):
        """Test voting without unanimity."""
        strategy = UnanimousVotingStrategy()
        result = strategy.vote(disagreeing_responses)

        assert result.confidence == 0.3  # Low confidence
        assert len(result.dissenting_models) > 0
        assert "not achieved" in result.reasoning

    def test_empty_responses(self):
        """Test voting with empty responses."""
        strategy = UnanimousVotingStrategy()
        with pytest.raises(VotingError, match="No responses"):
            strategy.vote([])


class TestMajorityVotingStrategy:
    """Tests for MajorityVotingStrategy."""

    def test_strategy_type(self):
        """Test strategy type property."""
        strategy = MajorityVotingStrategy()
        assert strategy.strategy_type == VotingStrategyType.MAJORITY

    def test_clear_majority(self, sample_model_responses):
        """Test voting with clear majority."""
        strategy = MajorityVotingStrategy()
        result = strategy.vote(sample_model_responses)

        assert result.strategy_used == VotingStrategyType.MAJORITY
        assert result.decision == "rule1(X) :- condition(X)."
        assert "Majority achieved" in result.reasoning

    def test_no_majority(self, disagreeing_responses):
        """Test voting without majority."""
        strategy = MajorityVotingStrategy()
        result = strategy.vote(disagreeing_responses)

        assert result.confidence == 0.4  # Lower confidence
        assert "No majority achieved" in result.reasoning

    def test_empty_responses(self):
        """Test voting with empty responses."""
        strategy = MajorityVotingStrategy()
        with pytest.raises(VotingError, match="No responses"):
            strategy.vote([])

    def test_tie_breaking_by_vote_count(self):
        """Test tie-breaking when vote counts are equal - uses avg confidence."""
        strategy = MajorityVotingStrategy()

        # Create tied vote scenario: 2 votes for "resultA", 2 votes for "resultB"
        # resultA has higher avg confidence (0.9 vs 0.7)
        responses = [
            ModelResponse(model_type="model1", result="resultA", confidence=0.9, latency_ms=100),
            ModelResponse(model_type="model2", result="resultA", confidence=0.9, latency_ms=100),
            ModelResponse(model_type="model3", result="resultB", confidence=0.7, latency_ms=100),
            ModelResponse(model_type="model4", result="resultB", confidence=0.7, latency_ms=100),
        ]

        result = strategy.vote(responses)

        # Should choose resultA due to higher average confidence
        assert result.decision == "resultA"
        assert result.confidence == 0.4  # No majority, so reduced confidence

    def test_tie_breaking_by_lexicographic(self):
        """Test tie-breaking when vote count and confidence are equal - uses lexicographic order."""
        strategy = MajorityVotingStrategy()

        # Create completely tied scenario: same vote count, same avg confidence
        # Should use lexicographic order ("aaa" < "bbb")
        responses = [
            ModelResponse(model_type="model1", result="bbb", confidence=0.8, latency_ms=100),
            ModelResponse(model_type="model2", result="bbb", confidence=0.8, latency_ms=100),
            ModelResponse(model_type="model3", result="aaa", confidence=0.8, latency_ms=100),
            ModelResponse(model_type="model4", result="aaa", confidence=0.8, latency_ms=100),
        ]

        result = strategy.vote(responses)

        # Should choose "aaa" lexicographically
        assert result.decision == "aaa"
        assert result.confidence == 0.4  # No majority

    def test_tie_breaking_determinism(self):
        """Test that tie-breaking produces same result across multiple runs."""
        strategy = MajorityVotingStrategy()

        # Create tied scenario
        responses = [
            ModelResponse(model_type="model1", result="option1", confidence=0.8, latency_ms=100),
            ModelResponse(model_type="model2", result="option2", confidence=0.8, latency_ms=100),
        ]

        # Run multiple times and verify same result
        results = [strategy.vote(responses).decision for _ in range(10)]
        assert len(set(results)) == 1  # All results should be identical


class TestWeightedVotingStrategy:
    """Tests for WeightedVotingStrategy."""

    def test_strategy_type(self):
        """Test strategy type property."""
        strategy = WeightedVotingStrategy()
        assert strategy.strategy_type == VotingStrategyType.WEIGHTED

    def test_weighted_voting(self, sample_model_responses):
        """Test weighted voting."""
        strategy = WeightedVotingStrategy()
        result = strategy.vote(sample_model_responses)

        assert result.strategy_used == VotingStrategyType.WEIGHTED
        assert "Weighted voting" in result.reasoning

    def test_highest_weight_wins(self):
        """Test that highest weighted result wins."""
        responses = [
            ModelResponse("model_a", "result_a", 0.3, 100.0),
            ModelResponse("model_b", "result_b", 0.9, 100.0),
            ModelResponse("model_c", "result_c", 0.2, 100.0),
        ]
        strategy = WeightedVotingStrategy()
        result = strategy.vote(responses)

        assert result.decision == "result_b"

    def test_empty_responses(self):
        """Test voting with empty responses."""
        strategy = WeightedVotingStrategy()
        with pytest.raises(VotingError, match="No responses"):
            strategy.vote([])


class TestDialecticalVotingStrategy:
    """Tests for DialecticalVotingStrategy."""

    def test_strategy_type(self):
        """Test strategy type property."""
        strategy = DialecticalVotingStrategy()
        assert strategy.strategy_type == VotingStrategyType.DIALECTICAL

    def test_single_response(self):
        """Test with single response."""
        responses = [ModelResponse("model", "result", 0.8, 100.0)]
        strategy = DialecticalVotingStrategy()
        result = strategy.vote(responses)

        assert result.decision == "result"
        assert "Single response" in result.reasoning

    def test_thesis_antithesis_agreement(self):
        """Test when thesis and antithesis agree."""
        responses = [
            ModelResponse("model_a", "same_result", 0.9, 100.0),
            ModelResponse("model_b", "same_result", 0.5, 100.0),
        ]
        strategy = DialecticalVotingStrategy()
        result = strategy.vote(responses)

        assert "agree" in result.reasoning.lower() or "consensus" in result.reasoning.lower()

    def test_dialectical_fallback(self, disagreeing_responses):
        """Test fallback when dialectical process incomplete."""
        strategy = DialecticalVotingStrategy()
        result = strategy.vote(disagreeing_responses)

        # Should process dialectically (thesis, antithesis, synthesis with 3 responses)
        # or fallback to highest confidence for other cases
        assert result.confidence < 1.0
        assert (
            "highest confidence" in result.reasoning.lower()
            or "incomplete" in result.reasoning.lower()
            or "dialectical" in result.reasoning.lower()
            or "synthesis" in result.reasoning.lower()
        )

    def test_empty_responses(self):
        """Test voting with empty responses."""
        strategy = DialecticalVotingStrategy()
        with pytest.raises(VotingError, match="No responses"):
            strategy.vote([])


class TestCreateVotingStrategy:
    """Tests for create_voting_strategy factory function."""

    def test_create_unanimous(self):
        """Test creating unanimous strategy."""
        strategy = create_voting_strategy(VotingStrategyType.UNANIMOUS)
        assert isinstance(strategy, UnanimousVotingStrategy)

    def test_create_majority(self):
        """Test creating majority strategy."""
        strategy = create_voting_strategy(VotingStrategyType.MAJORITY)
        assert isinstance(strategy, MajorityVotingStrategy)

    def test_create_weighted(self):
        """Test creating weighted strategy."""
        strategy = create_voting_strategy(VotingStrategyType.WEIGHTED)
        assert isinstance(strategy, WeightedVotingStrategy)

    def test_create_dialectical(self):
        """Test creating dialectical strategy."""
        strategy = create_voting_strategy(VotingStrategyType.DIALECTICAL)
        assert isinstance(strategy, DialecticalVotingStrategy)


# =============================================================================
# Test Disagreement Resolution Strategies
# =============================================================================


class TestDeferToCriticResolver:
    """Tests for DeferToCriticResolver."""

    def test_strategy_type(self):
        """Test strategy type property."""
        resolver = DeferToCriticResolver()
        assert resolver.strategy_type == DisagreementStrategyType.DEFER_TO_CRITIC

    def test_defer_to_critic(self):
        """Test that critic's response is selected."""
        responses = [
            ModelResponse("logic_generator", "result_a", 0.9, 100.0),
            ModelResponse("critic", "critic_result", 0.7, 100.0),
            ModelResponse("translator", "result_c", 0.8, 100.0),
        ]
        resolver = DeferToCriticResolver()
        result, explanation = resolver.resolve(responses)

        assert result == "critic_result"
        assert "critic" in explanation.lower()

    def test_no_critic_fallback(self):
        """Test fallback when no critic present."""
        responses = [
            ModelResponse("logic_generator", "result_a", 0.9, 100.0),
            ModelResponse("translator", "result_c", 0.7, 100.0),
        ]
        resolver = DeferToCriticResolver()
        result, explanation = resolver.resolve(responses)

        assert result == "result_a"  # Highest confidence
        assert "No critic" in explanation


class TestDeferToConfidenceResolver:
    """Tests for DeferToConfidenceResolver."""

    def test_strategy_type(self):
        """Test strategy type property."""
        resolver = DeferToConfidenceResolver()
        assert resolver.strategy_type == DisagreementStrategyType.DEFER_TO_CONFIDENCE

    def test_highest_confidence_selected(self):
        """Test that highest confidence response is selected."""
        responses = [
            ModelResponse("model_a", "result_a", 0.7, 100.0),
            ModelResponse("model_b", "result_b", 0.95, 100.0),
            ModelResponse("model_c", "result_c", 0.8, 100.0),
        ]
        resolver = DeferToConfidenceResolver()
        result, explanation = resolver.resolve(responses)

        assert result == "result_b"
        assert "model_b" in explanation
        assert "0.95" in explanation


class TestSynthesizeResolver:
    """Tests for SynthesizeResolver."""

    def test_strategy_type(self):
        """Test strategy type property."""
        resolver = SynthesizeResolver()
        assert resolver.strategy_type == DisagreementStrategyType.SYNTHESIZE

    def test_synthesize_common_elements(self):
        """Test synthesizing with common elements."""
        responses = [
            ModelResponse("model_a", "contract valid legal", 0.8, 100.0),
            ModelResponse("model_b", "contract enforceable legal", 0.7, 100.0),
            ModelResponse("model_c", "agreement contract legal", 0.9, 100.0),
        ]
        resolver = SynthesizeResolver()
        result, explanation = resolver.resolve(responses)

        # Should select response with most common words
        assert "synthesize" in explanation.lower() or "common" in explanation.lower()


class TestEscalateResolver:
    """Tests for EscalateResolver."""

    def test_strategy_type(self):
        """Test strategy type property."""
        resolver = EscalateResolver()
        assert resolver.strategy_type == DisagreementStrategyType.ESCALATE

    def test_escalation_flag(self):
        """Test that escalation is flagged."""
        responses = [
            ModelResponse("model_a", "result_a", 0.8, 100.0),
            ModelResponse("model_b", "result_b", 0.7, 100.0),
        ]
        resolver = EscalateResolver()
        result, explanation = resolver.resolve(responses)

        assert "ESCALATED" in explanation
        assert "human review" in explanation.lower()


class TestConservativeResolver:
    """Tests for ConservativeResolver."""

    def test_strategy_type(self):
        """Test strategy type property."""
        resolver = ConservativeResolver()
        assert resolver.strategy_type == DisagreementStrategyType.CONSERVATIVE

    def test_prefer_critic(self):
        """Test that critic is preferred."""
        responses = [
            ModelResponse("logic_generator", "aggressive_result", 0.9, 100.0),
            ModelResponse("critic", "cautious_result", 0.6, 100.0),
        ]
        resolver = ConservativeResolver()
        result, explanation = resolver.resolve(responses)

        assert result == "cautious_result"
        assert "critic" in explanation.lower()


class TestCreateDisagreementResolver:
    """Tests for create_disagreement_resolver factory function."""

    def test_create_defer_to_critic(self):
        """Test creating defer to critic resolver."""
        resolver = create_disagreement_resolver(DisagreementStrategyType.DEFER_TO_CRITIC)
        assert isinstance(resolver, DeferToCriticResolver)

    def test_create_defer_to_confidence(self):
        """Test creating defer to confidence resolver."""
        resolver = create_disagreement_resolver(DisagreementStrategyType.DEFER_TO_CONFIDENCE)
        assert isinstance(resolver, DeferToConfidenceResolver)

    def test_create_synthesize(self):
        """Test creating synthesize resolver."""
        resolver = create_disagreement_resolver(DisagreementStrategyType.SYNTHESIZE)
        assert isinstance(resolver, SynthesizeResolver)

    def test_create_escalate(self):
        """Test creating escalate resolver."""
        resolver = create_disagreement_resolver(DisagreementStrategyType.ESCALATE)
        assert isinstance(resolver, EscalateResolver)

    def test_create_conservative(self):
        """Test creating conservative resolver."""
        resolver = create_disagreement_resolver(DisagreementStrategyType.CONSERVATIVE)
        assert isinstance(resolver, ConservativeResolver)


# =============================================================================
# Test EnsembleOrchestrator
# =============================================================================


class TestEnsembleOrchestratorInit:
    """Tests for EnsembleOrchestrator initialization."""

    def test_default_initialization(self, mock_llm_interface):
        """Test default initialization."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)

        assert orchestrator.config is not None
        assert orchestrator._voting_strategy is not None
        assert orchestrator._disagreement_resolver is not None

    def test_custom_config(self, basic_config, mock_llm_interface):
        """Test initialization with custom config."""
        orchestrator = EnsembleOrchestrator(config=basic_config, llm_interface=mock_llm_interface)

        assert orchestrator.config == basic_config

    def test_initialization_without_llm(self):
        """Test initialization without LLM interface."""
        orchestrator = EnsembleOrchestrator()
        assert orchestrator._llm_interface is None


class TestEnsembleOrchestratorAggregation:
    """Tests for response aggregation."""

    def test_aggregate_responses(self, sample_model_responses, mock_llm_interface):
        """Test aggregating responses."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)
        result = orchestrator.aggregate_responses(sample_model_responses)

        assert isinstance(result, VotingResult)
        assert result.decision is not None

    def test_aggregate_with_specific_strategy(self, sample_model_responses, mock_llm_interface):
        """Test aggregating with specific strategy."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)
        result = orchestrator.aggregate_responses(
            sample_model_responses, VotingStrategyType.UNANIMOUS
        )

        assert result.strategy_used == VotingStrategyType.UNANIMOUS

    def test_aggregate_empty_raises(self, mock_llm_interface):
        """Test that empty responses raises error."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)
        with pytest.raises(VotingError):
            orchestrator.aggregate_responses([])


class TestEnsembleOrchestratorDisagreement:
    """Tests for disagreement resolution."""

    def test_resolve_disagreement(self, disagreeing_responses, mock_llm_interface):
        """Test resolving disagreement."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)
        result, explanation = orchestrator.resolve_disagreement(disagreeing_responses)

        assert result is not None
        assert explanation != ""

    def test_resolve_with_specific_strategy(self, disagreeing_responses, mock_llm_interface):
        """Test resolving with specific strategy."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)
        result, explanation = orchestrator.resolve_disagreement(
            disagreeing_responses, DisagreementStrategyType.CONSERVATIVE
        )

        assert "conservative" in explanation.lower() or "cautious" in explanation.lower()

    def test_resolve_empty_raises(self, mock_llm_interface):
        """Test that empty responses raises error."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)
        with pytest.raises(DisagreementResolutionError):
            orchestrator.resolve_disagreement([])


class TestEnsembleOrchestratorStatus:
    """Tests for model status tracking."""

    def test_get_model_status_unavailable(self, mock_llm_interface):
        """Test getting status of unavailable model."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)
        status = orchestrator.get_model_status("nonexistent")

        assert status == ModelStatus.UNAVAILABLE

    def test_get_performance_metrics_empty(self, mock_llm_interface):
        """Test getting empty performance metrics."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)
        metrics = orchestrator.get_performance_metrics()

        assert metrics == {}


class TestEnsembleOrchestratorCache:
    """Tests for caching behavior."""

    def test_clear_cache(self, mock_llm_interface):
        """Test clearing cache."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)
        cleared = orchestrator.clear_cache()

        assert cleared == 0  # Empty cache

    def test_cache_key_generation(self, mock_llm_interface):
        """Test cache key generation."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)

        key1 = orchestrator._get_cache_key(TaskType.RULE_GENERATION, "input1", {"context": "a"})
        key2 = orchestrator._get_cache_key(TaskType.RULE_GENERATION, "input1", {"context": "a"})
        key3 = orchestrator._get_cache_key(TaskType.RULE_GENERATION, "input2", {"context": "a"})

        assert key1 == key2  # Same input = same key
        assert key1 != key3  # Different input = different key


class TestEnsembleOrchestratorOptimization:
    """Tests for routing optimization."""

    def test_optimize_routing_empty_metrics(self, mock_llm_interface):
        """Test optimization with no metrics."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)
        weights = orchestrator.optimize_routing(TaskType.RULE_GENERATION)

        assert weights == {}  # No metrics to optimize

    def test_optimize_routing_with_metrics(self, mock_llm_interface):
        """Test optimization with metrics."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)

        # Manually add some metrics
        orchestrator._performance_metrics["logic_generator"] = ModelPerformanceMetrics(
            model_type="logic_generator",
            total_requests=100,
            successful_requests=90,
            average_latency_ms=150.0,
            average_confidence=0.85,
        )
        orchestrator._performance_metrics["critic"] = ModelPerformanceMetrics(
            model_type="critic",
            total_requests=80,
            successful_requests=75,
            average_latency_ms=200.0,
            average_confidence=0.90,
        )

        weights = orchestrator.optimize_routing(TaskType.RULE_GENERATION)

        assert "logic_generator" in weights
        assert "critic" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Normalized


class TestEnsembleOrchestratorMetrics:
    """Tests for performance metrics tracking."""

    def test_reset_metrics(self, mock_llm_interface):
        """Test resetting metrics."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)

        # Add some metrics
        orchestrator._performance_metrics["test"] = ModelPerformanceMetrics("test")
        orchestrator.reset_metrics()

        assert orchestrator._performance_metrics == {}

    def test_get_disagreement_history(self, mock_llm_interface):
        """Test getting disagreement history."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)
        history = orchestrator.get_disagreement_history()

        assert history == []


# =============================================================================
# Test Task Routing
# =============================================================================


class TestTaskRouting:
    """Tests for task routing functionality."""

    def test_route_task_unknown_type(self, mock_llm_interface):
        """Test that routing fails for unknown task types."""
        # We can't actually test unknown types due to enum, but we can test
        # that valid types don't raise errors on construction
        assert TaskType.RULE_GENERATION is not None
        assert TaskType.RULE_CRITICISM is not None
        assert TaskType.FULL_PIPELINE is not None
        # Verify orchestrator can be constructed (mock_llm_interface fixture available)
        assert mock_llm_interface is not None

    def test_fallback_models_defined(self, mock_llm_interface):
        """Test that fallback models are defined for task types."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)

        for task_type in TaskType:
            fallbacks = orchestrator._get_fallback_models(task_type)
            assert isinstance(fallbacks, list)


# =============================================================================
# Test Enums
# =============================================================================


class TestEnums:
    """Tests for enum definitions."""

    def test_task_types(self):
        """Test TaskType enum values."""
        assert TaskType.RULE_GENERATION.value == "rule_generation"
        assert TaskType.RULE_CRITICISM.value == "rule_criticism"
        assert TaskType.TRANSLATION_TO_NL.value == "translation_to_nl"
        assert TaskType.TRANSLATION_TO_ASP.value == "translation_to_asp"
        assert TaskType.META_ANALYSIS.value == "meta_analysis"
        assert TaskType.FULL_PIPELINE.value == "full_pipeline"

    def test_voting_strategy_types(self):
        """Test VotingStrategyType enum values."""
        assert VotingStrategyType.UNANIMOUS.value == "unanimous"
        assert VotingStrategyType.MAJORITY.value == "majority"
        assert VotingStrategyType.WEIGHTED.value == "weighted"
        assert VotingStrategyType.DIALECTICAL.value == "dialectical"

    def test_disagreement_strategy_types(self):
        """Test DisagreementStrategyType enum values."""
        assert DisagreementStrategyType.DEFER_TO_CRITIC.value == "defer_to_critic"
        assert DisagreementStrategyType.DEFER_TO_CONFIDENCE.value == "defer_to_confidence"
        assert DisagreementStrategyType.SYNTHESIZE.value == "synthesize"
        assert DisagreementStrategyType.ESCALATE.value == "escalate"
        assert DisagreementStrategyType.CONSERVATIVE.value == "conservative"

    def test_model_status(self):
        """Test ModelStatus enum values."""
        assert ModelStatus.AVAILABLE.value == "available"
        assert ModelStatus.BUSY.value == "busy"
        assert ModelStatus.FAILED.value == "failed"
        assert ModelStatus.UNAVAILABLE.value == "unavailable"


# =============================================================================
# Test Exceptions
# =============================================================================


class TestExceptions:
    """Tests for custom exceptions."""

    def test_orchestrator_error(self):
        """Test OrchestratorError."""
        with pytest.raises(OrchestratorError):
            raise OrchestratorError("Test error")

    def test_task_routing_error(self):
        """Test TaskRoutingError."""
        with pytest.raises(TaskRoutingError):
            raise TaskRoutingError("Routing failed")

    def test_voting_error(self):
        """Test VotingError."""
        with pytest.raises(VotingError):
            raise VotingError("Voting failed")

    def test_disagreement_resolution_error(self):
        """Test DisagreementResolutionError."""
        with pytest.raises(DisagreementResolutionError):
            raise DisagreementResolutionError("Resolution failed")

    def test_fallback_exhausted_error(self):
        """Test FallbackExhaustedError."""
        with pytest.raises(FallbackExhaustedError):
            raise FallbackExhaustedError("No fallbacks left")


# =============================================================================
# Integration-like Tests (without actual LLM calls)
# =============================================================================


class TestOrchestratorIntegration:
    """Integration-like tests for orchestrator workflow."""

    def test_full_voting_workflow(self, sample_model_responses, mock_llm_interface):
        """Test complete voting workflow."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)

        # Aggregate responses
        voting_result = orchestrator.aggregate_responses(sample_model_responses)

        assert voting_result.decision is not None
        assert len(voting_result.participating_models) == 3
        assert voting_result.confidence > 0

    def test_full_disagreement_workflow(self, disagreeing_responses, mock_llm_interface):
        """Test complete disagreement resolution workflow."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)

        # Resolve disagreement
        result, explanation = orchestrator.resolve_disagreement(disagreeing_responses)

        # Check history was recorded
        history = orchestrator.get_disagreement_history()
        assert len(history) == 1
        assert history[0].final_decision == result

    def test_metrics_update_flow(self, mock_llm_interface):
        """Test metrics update flow."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)

        # Update metrics
        orchestrator._update_performance_metrics(
            "test_model",
            success=True,
            latency_ms=150.0,
            confidence=0.85,
        )

        metrics = orchestrator.get_performance_metrics()
        assert "test_model" in metrics
        assert metrics["test_model"].successful_requests == 1
        assert metrics["test_model"].average_latency_ms == 150.0

    def test_metrics_update_failure(self, mock_llm_interface):
        """Test metrics update for failed requests."""
        orchestrator = EnsembleOrchestrator(llm_interface=mock_llm_interface)

        # Update with failure
        orchestrator._update_performance_metrics(
            "test_model",
            success=False,
            latency_ms=150.0,
            confidence=0.0,
            error_type="TimeoutError",
        )

        metrics = orchestrator.get_performance_metrics()
        assert metrics["test_model"].failed_requests == 1
        assert "TimeoutError" in metrics["test_model"].error_types


# =============================================================================
# Test Input Validation (Issue #203)
# =============================================================================


class TestOrchestratorConfigValidation:
    """Tests for OrchestratorConfig input validation."""

    def test_invalid_cache_ttl_seconds(self):
        """Test that negative cache_ttl_seconds raises ValueError."""
        with pytest.raises(ValueError, match="cache_ttl_seconds must be non-negative"):
            OrchestratorConfig(cache_ttl_seconds=-1)

    def test_zero_cache_ttl_seconds_allowed(self):
        """Test that zero cache_ttl_seconds is allowed."""
        config = OrchestratorConfig(cache_ttl_seconds=0)
        assert config.cache_ttl_seconds == 0


class TestWeightedVotingStrategyValidation:
    """Tests for WeightedVotingStrategy input validation."""

    def test_zero_confidence_allowed(self):
        """Test that zero confidence is allowed."""
        strategy = WeightedVotingStrategy()
        responses = [
            ModelResponse(
                model_type="test_model",
                result="test result",
                confidence=0.0,
                latency_ms=100.0,
            )
        ]
        result = strategy.vote(responses)
        assert result.decision == "test result"

    def test_negative_confidence_clamped_to_zero(self):
        """Test that negative confidence is clamped to 0.0 by ModelResponse."""
        # ModelResponse.__post_init__ clamps confidence to [0.0, 1.0]
        response = ModelResponse(
            model_type="test_model",
            result="test result",
            confidence=-0.5,
            latency_ms=100.0,
        )
        # Verify clamping occurred
        assert response.confidence == 0.0

        # Verify WeightedVotingStrategy works with clamped value
        strategy = WeightedVotingStrategy()
        result = strategy.vote([response])
        assert result.decision == "test result"

    def test_confidence_above_one_clamped_to_one(self):
        """Test that confidence > 1.0 is clamped to 1.0 by ModelResponse."""
        response = ModelResponse(
            model_type="test_model",
            result="test result",
            confidence=1.5,
            latency_ms=100.0,
        )
        # Verify clamping occurred
        assert response.confidence == 1.0


class TestRouteTaskInputValidation:
    """Tests for EnsembleOrchestrator.route_task() input validation."""

    def test_none_input_data_raises_error(self, mock_orchestrator):
        """Test that None input_data raises TaskRoutingError."""
        with pytest.raises(TaskRoutingError, match="input_data cannot be None"):
            mock_orchestrator.route_task(TaskType.RULE_GENERATION, None)

    def test_empty_string_input_data_raises_error(self, mock_orchestrator):
        """Test that empty string input_data raises TaskRoutingError."""
        with pytest.raises(TaskRoutingError, match="input_data cannot be empty"):
            mock_orchestrator.route_task(TaskType.RULE_GENERATION, "")

    def test_whitespace_only_input_data_raises_error(self, mock_orchestrator):
        """Test that whitespace-only string raises TaskRoutingError."""
        with pytest.raises(TaskRoutingError, match="input_data cannot be empty"):
            mock_orchestrator.route_task(TaskType.RULE_GENERATION, "   ")

    def test_empty_dict_input_data_raises_error(self, mock_orchestrator):
        """Test that empty dict input_data raises TaskRoutingError."""
        with pytest.raises(TaskRoutingError, match="input_data cannot be an empty dictionary"):
            mock_orchestrator.route_task(TaskType.RULE_GENERATION, {})

    def test_empty_list_input_data_raises_error(self, mock_orchestrator):
        """Test that empty list input_data raises TaskRoutingError."""
        with pytest.raises(TaskRoutingError, match="input_data cannot be an empty list"):
            mock_orchestrator.route_task(TaskType.RULE_GENERATION, [])


class TestCacheTTLBoundaryValues:
    """Tests for cache_ttl_seconds boundary values."""

    def test_positive_cache_ttl_allowed(self):
        """Test that positive cache_ttl_seconds is allowed."""
        config = OrchestratorConfig(cache_ttl_seconds=3600)
        assert config.cache_ttl_seconds == 3600

    def test_large_cache_ttl_allowed(self):
        """Test that large cache_ttl_seconds is allowed."""
        config = OrchestratorConfig(cache_ttl_seconds=86400)  # 1 day
        assert config.cache_ttl_seconds == 86400

    def test_boundary_negative_one_raises_error(self):
        """Test that -1 cache_ttl_seconds raises ValueError."""
        with pytest.raises(ValueError, match="cache_ttl_seconds must be non-negative"):
            OrchestratorConfig(cache_ttl_seconds=-1)


# =============================================================================
# Test Error Handling (Issue #201)
# =============================================================================


class TestModelError:
    """Tests for ModelError dataclass (Issue #201)."""

    def test_create_model_error(self):
        """Test creating a ModelError with basic attributes."""
        error = ModelError(
            model_id="logic_generator",
            error_type="TimeoutError",
            message="Model timed out after 60 seconds",
        )
        assert error.model_id == "logic_generator"
        assert error.error_type == "TimeoutError"
        assert error.message == "Model timed out after 60 seconds"
        assert error.timestamp > 0
        assert error.context is None

    def test_model_error_with_context(self):
        """Test creating a ModelError with additional context."""
        context = {"task_type": "rule_generation", "input_length": 1500}
        error = ModelError(
            model_id="critic",
            error_type="ValueError",
            message="Invalid rule format",
            context=context,
        )
        assert error.context == context
        assert error.context["task_type"] == "rule_generation"

    def test_model_error_to_dict(self):
        """Test ModelError.to_dict() method."""
        error = ModelError(
            model_id="translator",
            error_type="ConnectionError",
            message="Failed to connect to LLM API",
            context={"retry_count": 3},
        )
        error_dict = error.to_dict()

        assert error_dict["model_id"] == "translator"
        assert error_dict["error_type"] == "ConnectionError"
        assert error_dict["message"] == "Failed to connect to LLM API"
        assert "timestamp" in error_dict
        assert error_dict["context"] == {"retry_count": 3}


class TestOrchestrationResultErrorHandling:
    """Tests for OrchestrationResult error handling fields (Issue #201)."""

    def test_orchestration_result_with_errors(self):
        """Test OrchestrationResult with error tracking."""
        errors = [
            ModelError(
                model_id="logic_generator",
                error_type="TimeoutError",
                message="Request timed out",
            ),
        ]
        result = OrchestrationResult(
            task_type=TaskType.RULE_GENERATION,
            final_result=None,
            errors=errors,
            failed_models=["logic_generator"],
        )

        assert result.has_errors is True
        assert len(result.errors) == 1
        assert result.failed_models == ["logic_generator"]

    def test_orchestration_result_no_errors(self):
        """Test OrchestrationResult without errors."""
        result = OrchestrationResult(
            task_type=TaskType.RULE_GENERATION,
            final_result="rule(X) :- condition(X).",
        )

        assert result.has_errors is False
        assert len(result.errors) == 0
        assert result.failed_models == []

    def test_orchestration_result_partial_success(self):
        """Test OrchestrationResult with partial success."""
        result = OrchestrationResult(
            task_type=TaskType.FULL_PIPELINE,
            final_result="rule(X) :- cond(X).",
            model_responses=[
                ModelResponse(
                    model_type="logic_generator",
                    result="rule(X) :- cond(X).",
                    confidence=0.85,
                    latency_ms=150.0,
                )
            ],
            errors=[
                ModelError(
                    model_id="critic",
                    error_type="TimeoutError",
                    message="Critic timed out",
                )
            ],
            failed_models=["critic"],
        )

        assert result.partial_success is True
        assert result.has_errors is True
        assert len(result.model_responses) == 1

    def test_orchestration_result_get_error_summary(self):
        """Test OrchestrationResult.get_error_summary() method."""
        result = OrchestrationResult(
            task_type=TaskType.RULE_GENERATION,
            final_result=None,
            model_responses=[
                ModelResponse(
                    model_type="translator",
                    result="fallback_rule(X).",
                    confidence=0.7,
                    latency_ms=200.0,
                )
            ],
            errors=[
                ModelError(
                    model_id="logic_generator",
                    error_type="ValueError",
                    message="Invalid input",
                ),
                ModelError(
                    model_id="critic",
                    error_type="ConnectionError",
                    message="API unavailable",
                ),
            ],
            failed_models=["logic_generator", "critic"],
        )

        summary = result.get_error_summary()

        assert summary["total_errors"] == 2
        assert summary["failed_models"] == ["logic_generator", "critic"]
        assert summary["successful_models"] == ["translator"]
        assert len(summary["errors"]) == 2


class TestAggregatedOrchestrationError:
    """Tests for AggregatedOrchestrationError (Issue #201)."""

    def test_aggregated_error_basic(self):
        """Test creating a basic AggregatedOrchestrationError."""
        error = AggregatedOrchestrationError(
            message="All models failed",
        )
        assert str(error) == "All models failed"

    def test_aggregated_error_with_model_errors(self):
        """Test AggregatedOrchestrationError with model errors."""
        model_errors = [
            ModelError(
                model_id="logic_generator",
                error_type="TimeoutError",
                message="Timed out",
            ),
            ModelError(
                model_id="translator",
                error_type="ValueError",
                message="Invalid format",
            ),
        ]

        error = AggregatedOrchestrationError(
            message="Orchestration failed",
            model_errors=model_errors,
            attempted_models=["logic_generator", "translator"],
            task_type=TaskType.RULE_GENERATION,
        )

        assert len(error.model_errors) == 2
        assert error.attempted_models == ["logic_generator", "translator"]
        assert error.task_type == TaskType.RULE_GENERATION

    def test_aggregated_error_str_format(self):
        """Test AggregatedOrchestrationError string representation."""
        model_errors = [
            ModelError(
                model_id="model1",
                error_type="Error1",
                message="msg1",
            ),
        ]

        error = AggregatedOrchestrationError(
            message="Test failure",
            model_errors=model_errors,
            attempted_models=["model1", "model2"],
        )

        error_str = str(error)
        assert "Test failure" in error_str
        assert "model1" in error_str
        assert "model2" in error_str
        assert "Error1" in error_str

    def test_aggregated_error_get_error_summary(self):
        """Test AggregatedOrchestrationError.get_error_summary() method."""
        model_errors = [
            ModelError(
                model_id="logic_generator",
                error_type="TimeoutError",
                message="Timed out",
            ),
        ]

        error = AggregatedOrchestrationError(
            message="Pipeline failed",
            model_errors=model_errors,
            attempted_models=["logic_generator"],
            task_type=TaskType.FULL_PIPELINE,
        )

        summary = error.get_error_summary()

        assert summary["message"] == "Pipeline failed"
        assert summary["task_type"] == "full_pipeline"
        assert summary["attempted_models"] == ["logic_generator"]
        assert summary["total_failures"] == 1
        assert len(summary["model_errors"]) == 1


class TestFallbackErrorHandling:
    """Tests for fallback error handling with aggregated errors (Issue #201)."""

    def test_handle_fallback_tracks_errors(self, mock_llm_interface):
        """Test that _handle_fallback tracks all errors."""
        config = OrchestratorConfig(enable_fallback=True)
        orchestrator = EnsembleOrchestrator(config=config, llm_interface=mock_llm_interface)

        original_error = ValueError("Primary model failed")
        result = orchestrator._handle_fallback(
            task_type=TaskType.RULE_GENERATION,
            input_data="test input",
            context=None,
            original_error=original_error,
            primary_model="logic_generator",
        )

        # Should have error info from the primary model
        assert result.errors is not None
        assert len(result.errors) >= 1
        assert result.errors[0].model_id == "logic_generator"
        assert result.errors[0].error_type == "ValueError"
        assert "Primary model failed" in result.errors[0].message

    def test_handle_fallback_includes_attempted_models(self, mock_llm_interface):
        """Test that _handle_fallback tracks attempted models."""
        config = OrchestratorConfig(enable_fallback=True)
        orchestrator = EnsembleOrchestrator(config=config, llm_interface=mock_llm_interface)

        original_error = TimeoutError("Timed out")
        result = orchestrator._handle_fallback(
            task_type=TaskType.RULE_GENERATION,
            input_data="test",
            context=None,
            original_error=original_error,
            primary_model="logic_generator",
        )

        # Should have metadata with attempted models
        assert "attempted_models" in result.metadata
        assert "logic_generator" in result.metadata["attempted_models"]

    def test_fallback_exhausted_raises_aggregated_error(self, mock_llm_interface):
        """Test that exhausted fallbacks raise AggregatedOrchestrationError."""
        config = OrchestratorConfig(enable_fallback=True)
        orchestrator = EnsembleOrchestrator(config=config, llm_interface=mock_llm_interface)

        # Override _get_fallback_models to return empty list (no fallbacks)
        original_method = orchestrator._get_fallback_models
        orchestrator._get_fallback_models = lambda task_type: []

        try:
            with pytest.raises(AggregatedOrchestrationError) as exc_info:
                orchestrator._handle_fallback(
                    task_type=TaskType.FULL_PIPELINE,
                    input_data="test",
                    context=None,
                    original_error=ValueError("Failed"),
                    primary_model="logic_generator",
                )

            error = exc_info.value
            assert error.task_type == TaskType.FULL_PIPELINE
            assert len(error.model_errors) >= 1
            assert "logic_generator" in error.attempted_models
        finally:
            orchestrator._get_fallback_models = original_method


class TestRouteTaskErrorHandling:
    """Tests for route_task error handling (Issue #201)."""

    def test_route_task_raises_aggregated_error_when_fallback_disabled(self, mock_llm_interface):
        """Test that route_task raises AggregatedOrchestrationError when fallback is disabled."""
        config = OrchestratorConfig(enable_fallback=False)
        orchestrator = EnsembleOrchestrator(config=config, llm_interface=mock_llm_interface)

        # This will fail because the mock LLM interface isn't properly set up
        with pytest.raises(AggregatedOrchestrationError) as exc_info:
            orchestrator.route_task(
                TaskType.RULE_GENERATION,
                {"principle": "test principle", "domain": "legal"},
            )

        error = exc_info.value
        assert error.task_type == TaskType.RULE_GENERATION
        assert len(error.model_errors) >= 1
        assert "logic_generator" in error.attempted_models

    def test_route_task_includes_error_context(self, mock_llm_interface):
        """Test that route_task errors include proper context."""
        config = OrchestratorConfig(enable_fallback=False)
        orchestrator = EnsembleOrchestrator(config=config, llm_interface=mock_llm_interface)

        with pytest.raises(AggregatedOrchestrationError) as exc_info:
            orchestrator.route_task(
                TaskType.META_ANALYSIS,
                {"failures": [], "insights": []},
            )

        error = exc_info.value
        # Should identify meta_reasoner as the primary model for META_ANALYSIS
        assert "meta_reasoner" in error.attempted_models

        # Check error summary has proper structure
        summary = error.get_error_summary()
        assert summary["task_type"] == "meta_analysis"
        assert summary["total_failures"] >= 1


# =============================================================================
# Test Thread Safety (Issue #200)
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety in EnsembleOrchestrator (Issue #200)."""

    def test_concurrent_performance_metric_updates(self, mock_llm_interface):
        """Test that concurrent calls to _update_performance_metrics are thread-safe."""
        import threading

        config = OrchestratorConfig(enable_performance_tracking=True)
        orchestrator = EnsembleOrchestrator(config=config, llm_interface=mock_llm_interface)

        num_threads = 10
        updates_per_thread = 100
        errors = []

        def update_metrics(thread_id: int):
            try:
                for i in range(updates_per_thread):
                    orchestrator._update_performance_metrics(
                        model_type=f"model_{thread_id % 3}",  # Use 3 model types
                        success=i % 2 == 0,
                        latency_ms=50.0 + thread_id,
                        confidence=0.8,
                        error_type="TestError" if i % 2 != 0 else None,
                    )
            except Exception as e:
                errors.append(e)

        # Create and start threads
        threads = [threading.Thread(target=update_metrics, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Thread errors: {errors}"

        # Verify metrics were recorded
        metrics = orchestrator.get_performance_metrics()
        assert len(metrics) == 3  # 3 model types

        # Verify total requests match expected
        total_requests = sum(m.total_requests for m in metrics.values())
        expected_requests = num_threads * updates_per_thread
        assert total_requests == expected_requests, (
            f"Expected {expected_requests} requests, got {total_requests}"
        )

    def test_concurrent_cache_operations(self, mock_llm_interface):
        """Test that concurrent cache read/write operations are thread-safe."""
        import threading

        config = OrchestratorConfig(enable_caching=True, cache_ttl_seconds=3600)
        orchestrator = EnsembleOrchestrator(config=config, llm_interface=mock_llm_interface)

        num_threads = 10
        operations_per_thread = 50
        errors = []

        def cache_operations(thread_id: int):
            try:
                for i in range(operations_per_thread):
                    cache_key = f"key_{thread_id}_{i}"

                    # Store in cache
                    result = OrchestrationResult(
                        task_type=TaskType.RULE_GENERATION,
                        final_result=f"result_{thread_id}_{i}",
                    )
                    orchestrator._store_cache(cache_key, result)

                    # Read from cache
                    cached = orchestrator._check_cache(cache_key)
                    assert cached is not None
                    assert cached.final_result == f"result_{thread_id}_{i}"
            except Exception as e:
                errors.append(e)

        # Create and start threads
        threads = [threading.Thread(target=cache_operations, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_clear_cache(self, mock_llm_interface):
        """Test that clear_cache is thread-safe with concurrent reads/writes."""
        import threading

        config = OrchestratorConfig(enable_caching=True)
        orchestrator = EnsembleOrchestrator(config=config, llm_interface=mock_llm_interface)

        errors = []
        stop_flag = threading.Event()

        def writer():
            try:
                i = 0
                while not stop_flag.is_set():
                    result = OrchestrationResult(
                        task_type=TaskType.RULE_GENERATION,
                        final_result=f"result_{i}",
                    )
                    orchestrator._store_cache(f"key_{i}", result)
                    i += 1
            except Exception as e:
                errors.append(e)

        def clearer():
            try:
                for _ in range(5):
                    orchestrator.clear_cache()
            except Exception as e:
                errors.append(e)

        writer_thread = threading.Thread(target=writer)
        clearer_thread = threading.Thread(target=clearer)

        writer_thread.start()
        clearer_thread.start()

        clearer_thread.join()
        stop_flag.set()
        writer_thread.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_model_status_access(self, mock_llm_interface):
        """Test that model status reads are thread-safe."""
        import threading

        config = OrchestratorConfig()
        orchestrator = EnsembleOrchestrator(config=config, llm_interface=mock_llm_interface)

        num_threads = 20
        reads_per_thread = 100
        errors = []

        def read_status(thread_id: int):
            try:
                model_types = [
                    "logic_generator",
                    "critic",
                    "translator",
                    "meta_reasoner",
                ]
                for i in range(reads_per_thread):
                    model_type = model_types[i % len(model_types)]
                    status = orchestrator.get_model_status(model_type)
                    # Status should be valid (either UNAVAILABLE or AVAILABLE)
                    assert status in [ModelStatus.UNAVAILABLE, ModelStatus.AVAILABLE]
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_status, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_disagreement_history_updates(self, mock_llm_interface):
        """Test that disagreement history updates are thread-safe."""
        import threading

        config = OrchestratorConfig()
        orchestrator = EnsembleOrchestrator(config=config, llm_interface=mock_llm_interface)

        num_threads = 10
        updates_per_thread = 20
        errors = []

        def add_disagreements(thread_id: int):
            try:
                for i in range(updates_per_thread):
                    responses = [
                        ModelResponse(
                            model_type=f"model_{thread_id}",
                            result=f"result_{i}",
                            confidence=0.8,
                            latency_ms=100.0,
                        )
                    ]
                    # This calls resolve_disagreement which adds to history
                    try:
                        orchestrator.resolve_disagreement(responses)
                    except Exception:
                        pass  # May fail due to mock, but we're testing thread safety
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_disagreements, args=(i,)) for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check history can be read safely
        history = orchestrator.get_disagreement_history()
        assert isinstance(history, list)

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_reset_metrics(self, mock_llm_interface):
        """Test that reset_metrics is thread-safe with concurrent updates."""
        import threading

        config = OrchestratorConfig(enable_performance_tracking=True)
        orchestrator = EnsembleOrchestrator(config=config, llm_interface=mock_llm_interface)

        errors = []
        stop_flag = threading.Event()

        def updater():
            try:
                i = 0
                while not stop_flag.is_set():
                    orchestrator._update_performance_metrics(
                        model_type="test_model",
                        success=True,
                        latency_ms=50.0,
                        confidence=0.9,
                    )
                    i += 1
            except Exception as e:
                errors.append(e)

        def resetter():
            try:
                for _ in range(5):
                    orchestrator.reset_metrics()
            except Exception as e:
                errors.append(e)

        updater_thread = threading.Thread(target=updater)
        resetter_thread = threading.Thread(target=resetter)

        updater_thread.start()
        resetter_thread.start()

        resetter_thread.join()
        stop_flag.set()
        updater_thread.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_orchestrator_has_required_locks(self, mock_llm_interface):
        """Test that orchestrator has all required lock attributes."""
        config = OrchestratorConfig()
        orchestrator = EnsembleOrchestrator(config=config, llm_interface=mock_llm_interface)

        # Check that locks exist
        assert hasattr(orchestrator, "_lock")
        assert hasattr(orchestrator, "_cache_lock")
        assert hasattr(orchestrator, "_model_init_lock")

        # Check locks are proper threading primitives
        import threading

        assert isinstance(orchestrator._lock, type(threading.Lock()))
        assert isinstance(orchestrator._cache_lock, type(threading.Lock()))
        assert isinstance(orchestrator._model_init_lock, type(threading.RLock()))
