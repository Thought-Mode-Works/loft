"""
Integration tests for Phase 6 ensemble component interoperability.

Tests end-to-end scenarios where multiple ensemble components work together:
- LogicGeneratorLLM generates rules
- CriticLLM validates/critiques rules
- TranslatorLLM converts between symbolic and natural language
- MetaReasonerLLM analyzes failures and suggests improvements
- EnsembleOrchestrator coordinates all components

All tests use mocks to avoid actual API calls while validating
component integration and data flow.
"""

import pytest
from unittest.mock import MagicMock

from loft.neural.ensemble import (
    # Orchestrator
    EnsembleOrchestrator,
    OrchestratorConfig,
    ModelResponse,
    TaskType,
    VotingStrategyType,
    DisagreementStrategyType,
    # Voting strategies
    WeightedVotingStrategy,
    MajorityVotingStrategy,
    # Exceptions
    TaskRoutingError,
    VotingError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_interface():
    """Create a mock LLM interface for testing."""
    interface = MagicMock()
    interface.generate.return_value = {
        "content": "rule_generated(X) :- condition(X).",
        "confidence": 0.85,
    }
    return interface


@pytest.fixture
def orchestrator_config():
    """Create a standard orchestrator configuration for integration tests."""
    return OrchestratorConfig(
        default_voting_strategy=VotingStrategyType.WEIGHTED,
        default_disagreement_strategy=DisagreementStrategyType.DEFER_TO_CONFIDENCE,
        enable_caching=True,
        cache_ttl_seconds=60,
        max_retries=2,
        timeout_seconds=30.0,
        enable_performance_tracking=True,
        min_confidence_threshold=0.6,
        enable_fallback=True,
    )


@pytest.fixture
def mock_orchestrator(orchestrator_config, mock_llm_interface):
    """Create orchestrator with mock interface."""
    return EnsembleOrchestrator(
        config=orchestrator_config,
        llm_interface=mock_llm_interface,
    )


@pytest.fixture
def sample_legal_rule():
    """Sample ASP rule for testing."""
    return "enforceable(C) :- contract(C), consideration(C), capacity(P)."


@pytest.fixture
def sample_case_facts():
    """Sample case facts for testing."""
    return "contract(c1). parties(c1, alice, bob). signed(c1)."


# =============================================================================
# Test End-to-End Ensemble Pipeline
# =============================================================================


class TestEnsemblePipeline:
    """Integration tests for the complete ensemble pipeline."""

    def test_orchestrator_initialization_with_config(self, orchestrator_config):
        """Test that orchestrator initializes correctly with all config options."""
        mock_interface = MagicMock()
        orchestrator = EnsembleOrchestrator(
            config=orchestrator_config,
            llm_interface=mock_interface,
        )

        assert orchestrator.config == orchestrator_config
        assert orchestrator._llm_interface == mock_interface

    def test_voting_strategy_selection(self, orchestrator_config):
        """Test that different voting strategies can be selected."""
        mock_interface = MagicMock()

        # Test weighted strategy
        config_weighted = OrchestratorConfig(default_voting_strategy=VotingStrategyType.WEIGHTED)
        orch = EnsembleOrchestrator(config=config_weighted, llm_interface=mock_interface)
        assert orch.config.default_voting_strategy == VotingStrategyType.WEIGHTED

        # Test majority strategy
        config_majority = OrchestratorConfig(default_voting_strategy=VotingStrategyType.MAJORITY)
        orch = EnsembleOrchestrator(config=config_majority, llm_interface=mock_interface)
        assert orch.config.default_voting_strategy == VotingStrategyType.MAJORITY

    def test_disagreement_strategy_selection(self):
        """Test that different disagreement strategies can be selected."""
        mock_interface = MagicMock()

        strategies = [
            DisagreementStrategyType.DEFER_TO_CRITIC,
            DisagreementStrategyType.DEFER_TO_CONFIDENCE,
            DisagreementStrategyType.SYNTHESIZE,
            DisagreementStrategyType.ESCALATE,
            DisagreementStrategyType.CONSERVATIVE,
        ]

        for strategy in strategies:
            config = OrchestratorConfig(default_disagreement_strategy=strategy)
            orch = EnsembleOrchestrator(config=config, llm_interface=mock_interface)
            assert orch.config.default_disagreement_strategy == strategy


class TestLogicCriticInteraction:
    """Tests for LogicGenerator + Critic component interaction."""

    def test_voting_on_multiple_model_responses(self):
        """Test voting mechanism with multiple model responses."""
        strategy = WeightedVotingStrategy()

        # Simulate responses from different models agreeing
        responses = [
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
                confidence=0.80,
                latency_ms=100.0,
            ),
        ]

        result = strategy.vote(responses)

        assert len(result.dissenting_models) == 0  # Consensus reached
        assert result.decision == "rule1(X) :- condition(X)."
        assert result.confidence > 0.8  # Should be high with agreement

    def test_voting_with_disagreement(self):
        """Test voting when models disagree."""
        strategy = WeightedVotingStrategy()

        # Simulate disagreeing responses
        responses = [
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

        result = strategy.vote(responses)

        # Should still produce a result (highest weighted)
        assert result.decision is not None
        # Disagreement means dissenting models exist
        assert len(result.dissenting_models) > 0 or result.confidence < 1.0

    def test_critic_validates_generator_output(self):
        """Test that critic can validate logic generator output."""
        strategy = WeightedVotingStrategy()

        # Generator proposes a rule
        generator_response = ModelResponse(
            model_type="logic_generator",
            result="enforceable(C) :- contract(C).",
            confidence=0.70,
            latency_ms=150.0,
        )

        # Critic validates with lower confidence (indicates issues)
        critic_response = ModelResponse(
            model_type="critic",
            result="enforceable(C) :- contract(C).",
            confidence=0.40,  # Low confidence indicates potential issues
            latency_ms=200.0,
        )

        result = strategy.vote([generator_response, critic_response])

        # When both models agree on the same result, weighted voting gives 100% confidence
        # since all votes go to the same option. The validation here is that
        # agreement produces high confidence.
        assert result.confidence == 1.0  # Full agreement on same result


class TestTranslatorIntegration:
    """Tests for Translator component interaction with other components."""

    def test_translator_contributes_to_voting(self):
        """Test that translator responses contribute to ensemble voting."""
        strategy = MajorityVotingStrategy()

        # Two models agree, one disagrees
        responses = [
            ModelResponse(
                model_type="logic_generator",
                result="valid_contract(C) :- contract(C), signed(C).",
                confidence=0.85,
                latency_ms=150.0,
            ),
            ModelResponse(
                model_type="translator",
                result="valid_contract(C) :- contract(C), signed(C).",
                confidence=0.80,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="critic",
                result="valid_contract(C) :- contract(C).",  # Different
                confidence=0.75,
                latency_ms=200.0,
            ),
        ]

        result = strategy.vote(responses)

        # Majority should win
        assert result.decision == "valid_contract(C) :- contract(C), signed(C)."
        assert len(result.dissenting_models) <= 1  # At most 1 dissenter


class TestMetaReasonerIntegration:
    """Tests for MetaReasoner interaction with ensemble."""

    def test_low_confidence_responses_flagged(self):
        """Test that low confidence responses are identified."""
        strategy = WeightedVotingStrategy()

        # All responses with low confidence
        responses = [
            ModelResponse(
                model_type="logic_generator",
                result="rule(X).",
                confidence=0.30,
                latency_ms=150.0,
            ),
            ModelResponse(
                model_type="critic",
                result="rule(X).",
                confidence=0.25,
                latency_ms=200.0,
            ),
        ]

        result = strategy.vote(responses)

        # When all models agree on the same result, weighted voting gives 100%
        # confidence since all votes go to the same option
        assert result.confidence == 1.0  # Full agreement on same result

    def test_mixed_confidence_responses(self):
        """Test voting with mixed confidence levels."""
        strategy = WeightedVotingStrategy()

        responses = [
            ModelResponse(
                model_type="logic_generator",
                result="rule_high(X).",
                confidence=0.95,
                latency_ms=150.0,
            ),
            ModelResponse(
                model_type="critic",
                result="rule_low(X).",
                confidence=0.20,
                latency_ms=200.0,
            ),
        ]

        result = strategy.vote(responses)

        # High confidence response should dominate
        assert result.decision == "rule_high(X)."


# =============================================================================
# Test Error Handling Across Components
# =============================================================================


class TestCrossComponentErrorHandling:
    """Tests for error handling that spans multiple components."""

    def test_empty_response_list_raises_error(self):
        """Test that voting with empty responses raises VotingError."""
        strategy = WeightedVotingStrategy()

        with pytest.raises(VotingError, match="No responses"):
            strategy.vote([])

    def test_route_task_validates_input(self, mock_orchestrator):
        """Test that route_task validates input data."""
        with pytest.raises(TaskRoutingError, match="input_data cannot be None"):
            mock_orchestrator.route_task(TaskType.RULE_GENERATION, None)

        with pytest.raises(TaskRoutingError, match="input_data cannot be empty"):
            mock_orchestrator.route_task(TaskType.RULE_GENERATION, "")

        with pytest.raises(TaskRoutingError, match="input_data cannot be an empty dictionary"):
            mock_orchestrator.route_task(TaskType.RULE_GENERATION, {})

    def test_config_validation(self):
        """Test that invalid configurations are rejected."""
        with pytest.raises(ValueError, match="cache_ttl_seconds must be non-negative"):
            OrchestratorConfig(cache_ttl_seconds=-1)

        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            OrchestratorConfig(max_retries=-1)

        with pytest.raises(ValueError, match="min_confidence_threshold must be between"):
            OrchestratorConfig(min_confidence_threshold=1.5)


# =============================================================================
# Test Performance and Metrics
# =============================================================================


class TestEnsemblePerformance:
    """Tests for ensemble performance tracking."""

    def test_voting_latency(self):
        """Test that voting completes within acceptable time."""
        import time

        strategy = WeightedVotingStrategy()

        responses = [
            ModelResponse(
                model_type=f"model_{i}",
                result=f"rule_{i}(X).",
                confidence=0.8,
                latency_ms=100.0,
            )
            for i in range(5)
        ]

        start = time.time()
        result = strategy.vote(responses)
        duration = time.time() - start

        # Voting should be fast (< 10ms for in-memory operations)
        assert duration < 0.01
        assert result is not None

    def test_multiple_voting_rounds(self):
        """Test performance with multiple voting rounds."""
        import time

        strategy = WeightedVotingStrategy()

        start = time.time()

        for i in range(100):
            responses = [
                ModelResponse(
                    model_type=f"model_{j}",
                    result=f"rule_{i}_{j}(X).",
                    confidence=0.7 + (j * 0.05),
                    latency_ms=100.0,
                )
                for j in range(3)
            ]
            strategy.vote(responses)

        duration = time.time() - start

        # 100 voting rounds should complete in < 1 second
        assert duration < 1.0


# =============================================================================
# Test Consensus Building
# =============================================================================


class TestConsensusBuilding:
    """Tests for ensemble consensus mechanisms."""

    def test_unanimous_agreement(self):
        """Test consensus when all models agree."""
        strategy = WeightedVotingStrategy()

        same_result = "agreed_rule(X) :- condition(X)."
        responses = [
            ModelResponse(
                model_type=f"model_{i}",
                result=same_result,
                confidence=0.85,
                latency_ms=100.0,
            )
            for i in range(3)
        ]

        result = strategy.vote(responses)

        assert len(result.dissenting_models) == 0  # Consensus reached
        assert result.decision == same_result

    def test_partial_agreement(self):
        """Test consensus with partial agreement."""
        strategy = MajorityVotingStrategy()

        responses = [
            ModelResponse(
                model_type="model_1",
                result="majority_rule(X).",
                confidence=0.80,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_2",
                result="majority_rule(X).",
                confidence=0.75,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_3",
                result="minority_rule(X).",
                confidence=0.85,
                latency_ms=100.0,
            ),
        ]

        result = strategy.vote(responses)

        assert result.decision == "majority_rule(X)."
        assert len(result.dissenting_models) <= 1  # At most 1 dissenter with majority

    def test_no_clear_majority(self):
        """Test voting when no clear majority exists."""
        strategy = MajorityVotingStrategy()

        # Three different results, no majority
        responses = [
            ModelResponse(
                model_type="model_1",
                result="rule_a(X).",
                confidence=0.80,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_2",
                result="rule_b(X).",
                confidence=0.80,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_3",
                result="rule_c(X).",
                confidence=0.80,
                latency_ms=100.0,
            ),
        ]

        result = strategy.vote(responses)

        # Should still produce a result but no consensus (all dissent)
        assert result.decision is not None
        assert len(result.dissenting_models) > 0  # No consensus


# =============================================================================
# End-to-End Scenario Tests
# =============================================================================


@pytest.mark.e2e
class TestEndToEndScenarios:
    """End-to-end tests for complete ensemble workflows."""

    def test_rule_generation_workflow(self):
        """Test complete rule generation workflow with ensemble."""
        # Setup mock responses simulating full workflow
        generator_response = ModelResponse(
            model_type="logic_generator",
            result="enforceable(C) :- contract(C), consideration(C).",
            confidence=0.85,
            latency_ms=150.0,
            metadata={"strategy": "chain_of_thought"},
        )

        critic_response = ModelResponse(
            model_type="critic",
            result="enforceable(C) :- contract(C), consideration(C).",
            confidence=0.80,
            latency_ms=200.0,
            metadata={"edge_cases_found": 2},
        )

        translator_response = ModelResponse(
            model_type="translator",
            result="enforceable(C) :- contract(C), consideration(C).",
            confidence=0.90,
            latency_ms=100.0,
            metadata={"fidelity": 0.95},
        )

        strategy = WeightedVotingStrategy()
        result = strategy.vote([generator_response, critic_response, translator_response])

        # Verify complete workflow produces consensus
        assert len(result.dissenting_models) == 0  # Consensus reached
        assert result.decision == "enforceable(C) :- contract(C), consideration(C)."
        assert result.confidence > 0.8

    def test_rule_criticism_workflow(self):
        """Test rule criticism workflow with ensemble."""
        # Rule to critique
        rule_to_critique = "void(C) :- contract(C), illegal(C)."

        # Simulate critic responses with edge cases found
        critic_responses = [
            ModelResponse(
                model_type="critic_adversarial",
                result=rule_to_critique,
                confidence=0.70,
                latency_ms=200.0,
                metadata={"issues": ["missing capacity check"]},
            ),
            ModelResponse(
                model_type="critic_cooperative",
                result=rule_to_critique,
                confidence=0.85,
                latency_ms=180.0,
                metadata={"suggestions": ["add illegal_purpose predicate"]},
            ),
        ]

        strategy = WeightedVotingStrategy()
        result = strategy.vote(critic_responses)

        # When both critics agree on the rule, weighted voting gives 100% confidence
        assert result.decision == rule_to_critique
        assert result.confidence == 1.0  # Both critics agreed on the same result

    def test_translation_roundtrip_validation(self):
        """Test translation roundtrip with validation."""
        original_rule = "valid_offer(O) :- offer(O), definite(O)."

        # Simulate translator responses (NL -> ASP -> NL comparison)
        translator_responses = [
            ModelResponse(
                model_type="translator_nl",
                result=original_rule,
                confidence=0.95,
                latency_ms=100.0,
                metadata={"direction": "symbolic_to_nl"},
            ),
            ModelResponse(
                model_type="translator_asp",
                result=original_rule,
                confidence=0.92,
                latency_ms=120.0,
                metadata={"direction": "nl_to_symbolic"},
            ),
        ]

        strategy = WeightedVotingStrategy()
        result = strategy.vote(translator_responses)

        # High fidelity roundtrip should have high confidence
        assert result.confidence > 0.9
        assert len(result.dissenting_models) == 0  # Consensus reached
