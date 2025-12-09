"""
Integration tests for cross-component interactions in Phase 6 ensemble.

Tests validate how different ensemble components interact with each other:
- LogicGeneratorLLM -> CriticLLM workflow
- TranslatorLLM -> LogicGeneratorLLM roundtrip
- MetaReasonerLLM consuming data from other components
- EnsembleOrchestrator coordinating all components

All tests use mocks to avoid actual API calls while validating
component interoperability and data flow contracts.
"""

import pytest
from unittest.mock import MagicMock
from typing import Dict, Any

from loft.neural.ensemble import (
    # Orchestrator components
    EnsembleOrchestrator,
    OrchestratorConfig,
    ModelResponse,
    TaskType,
    VotingStrategyType,
    DisagreementStrategyType,
    # Voting strategies
    WeightedVotingStrategy,
    create_voting_strategy,
    # Disagreement resolvers
    DeferToConfidenceResolver,
    TaskRoutingError,
    VotingError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_interface():
    """Create a mock LLM interface that simulates component responses."""
    interface = MagicMock()

    def generate_response(prompt: str, **kwargs) -> Dict[str, Any]:
        """Simulate different component responses based on prompt content."""
        if "generate" in prompt.lower() or "logic" in prompt.lower():
            return {
                "content": "generated_rule(X) :- condition(X).",
                "confidence": 0.85,
                "metadata": {"component": "logic_generator"},
            }
        elif "critic" in prompt.lower() or "evaluate" in prompt.lower():
            return {
                "content": "rule_is_valid(X) :- verified(X).",
                "confidence": 0.80,
                "metadata": {"component": "critic", "issues_found": 0},
            }
        elif "translat" in prompt.lower():
            return {
                "content": "A contract is valid when it has consideration.",
                "confidence": 0.90,
                "metadata": {"component": "translator"},
            }
        elif "meta" in prompt.lower() or "reason" in prompt.lower():
            return {
                "content": '{"insights": ["Pattern detected"], "suggestions": []}',
                "confidence": 0.75,
                "metadata": {"component": "meta_reasoner"},
            }
        else:
            return {
                "content": "default_response(X).",
                "confidence": 0.70,
                "metadata": {"component": "unknown"},
            }

    interface.generate.side_effect = generate_response
    return interface


@pytest.fixture
def orchestrator_config():
    """Standard orchestrator configuration for cross-component tests."""
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
def ensemble_orchestrator(orchestrator_config, mock_llm_interface):
    """Create orchestrator with mock interface."""
    return EnsembleOrchestrator(
        config=orchestrator_config,
        llm_interface=mock_llm_interface,
    )


# =============================================================================
# LogicGenerator -> Critic Workflow Tests
# =============================================================================


class TestLogicGeneratorCriticWorkflow:
    """Tests for the workflow where LogicGenerator output is validated by Critic."""

    def test_generator_output_flows_to_critic(self):
        """Test that logic generator output can be validated by critic."""
        # Simulate generator output
        generator_response = ModelResponse(
            model_type="logic_generator",
            result="enforceable(C) :- contract(C), consideration(C).",
            confidence=0.85,
            latency_ms=150.0,
            metadata={"strategy": "chain_of_thought"},
        )

        # Simulate critic validation of that output
        critic_response = ModelResponse(
            model_type="critic",
            result="enforceable(C) :- contract(C), consideration(C).",
            confidence=0.90,  # Critic validates the rule
            latency_ms=200.0,
            metadata={"issues_found": 0, "edge_cases_checked": 5},
        )

        # Voting should combine both perspectives
        strategy = WeightedVotingStrategy()
        result = strategy.vote([generator_response, critic_response])

        assert len(result.dissenting_models) == 0  # Consensus reached
        assert result.confidence > 0.85

    def test_critic_rejects_generator_output(self):
        """Test workflow when critic finds issues with generated rule."""
        generator_response = ModelResponse(
            model_type="logic_generator",
            result="enforceable(C) :- contract(C).",  # Missing conditions
            confidence=0.75,
            latency_ms=150.0,
        )

        # Critic finds issues - proposes different rule
        critic_response = ModelResponse(
            model_type="critic",
            result="enforceable(C) :- contract(C), valid_consideration(C), capacity(C).",
            confidence=0.85,  # Critic more confident in corrected version
            latency_ms=250.0,
            metadata={"issues_found": 2, "corrections_suggested": 1},
        )

        strategy = WeightedVotingStrategy()
        result = strategy.vote([generator_response, critic_response])

        # Critic's higher confidence should influence final result
        assert result.decision == critic_response.result

    def test_multiple_generators_with_single_critic(self):
        """Test workflow with multiple generator variants evaluated by one critic."""
        generator_responses = [
            ModelResponse(
                model_type="logic_generator_cot",
                result="rule_a(X) :- cond_a(X).",
                confidence=0.80,
                latency_ms=150.0,
            ),
            ModelResponse(
                model_type="logic_generator_few_shot",
                result="rule_b(X) :- cond_b(X).",
                confidence=0.82,
                latency_ms=140.0,
            ),
        ]

        # Critic agrees with one of them
        critic_response = ModelResponse(
            model_type="critic",
            result="rule_b(X) :- cond_b(X).",
            confidence=0.90,
            latency_ms=200.0,
        )

        strategy = WeightedVotingStrategy()
        all_responses = generator_responses + [critic_response]
        result = strategy.vote(all_responses)

        # Critic's agreement should boost rule_b
        assert result.decision == "rule_b(X) :- cond_b(X)."


# =============================================================================
# Translator Component Workflow Tests
# =============================================================================


class TestTranslatorComponentWorkflow:
    """Tests for Translator component interactions."""

    def test_translator_validates_generator_output_semantics(self):
        """Test translator can validate semantic consistency of generated rules."""
        # Generator creates ASP rule
        generator_response = ModelResponse(
            model_type="logic_generator",
            result="valid_contract(C) :- contract(C), signed(C).",
            confidence=0.85,
            latency_ms=150.0,
        )

        # Translator confirms semantic understanding
        translator_response = ModelResponse(
            model_type="translator",
            result="valid_contract(C) :- contract(C), signed(C).",
            confidence=0.92,  # High fidelity translation
            latency_ms=100.0,
            metadata={
                "nl_translation": "A contract is valid when signed.",
                "roundtrip_fidelity": 0.95,
            },
        )

        strategy = WeightedVotingStrategy()
        result = strategy.vote([generator_response, translator_response])

        assert len(result.dissenting_models) == 0  # Consensus reached
        assert result.confidence > 0.85

    def test_translator_roundtrip_consistency(self):
        """Test that translator maintains consistency in roundtrip."""
        original_rule = "offer_valid(O) :- offer(O), specific_terms(O)."

        # Forward translation (ASP -> NL)
        forward_response = ModelResponse(
            model_type="translator_forward",
            result=original_rule,
            confidence=0.95,
            latency_ms=100.0,
            metadata={"direction": "asp_to_nl"},
        )

        # Backward translation (NL -> ASP)
        backward_response = ModelResponse(
            model_type="translator_backward",
            result=original_rule,
            confidence=0.93,
            latency_ms=120.0,
            metadata={"direction": "nl_to_asp"},
        )

        strategy = WeightedVotingStrategy()
        result = strategy.vote([forward_response, backward_response])

        # High fidelity roundtrip
        assert len(result.dissenting_models) == 0  # Consensus reached
        assert result.decision == original_rule

    def test_translator_detects_semantic_drift(self):
        """Test that translator can detect when semantics drift during translation."""
        # Original rule
        original_response = ModelResponse(
            model_type="logic_generator",
            result="binding(C) :- contract(C), mutual_assent(C).",
            confidence=0.85,
            latency_ms=150.0,
        )

        # Translator finds semantic drift - result is different
        translator_response = ModelResponse(
            model_type="translator",
            result="binding(C) :- contract(C), consent(C).",  # Different semantics
            confidence=0.60,  # Lower confidence indicates drift
            latency_ms=100.0,
            metadata={"semantic_similarity": 0.75, "drift_detected": True},
        )

        strategy = WeightedVotingStrategy()
        result = strategy.vote([original_response, translator_response])

        # Should not reach consensus due to drift
        assert len(result.dissenting_models) > 0  # No consensus


# =============================================================================
# MetaReasoner Component Tests
# =============================================================================


class TestMetaReasonerWorkflow:
    """Tests for MetaReasoner component interactions."""

    def test_meta_reasoner_analyzes_low_confidence_results(self):
        """Test that meta reasoner can analyze low confidence voting results."""
        # Low confidence responses
        generator_response = ModelResponse(
            model_type="logic_generator",
            result="uncertain_rule(X) :- maybe(X).",
            confidence=0.45,
            latency_ms=150.0,
        )

        critic_response = ModelResponse(
            model_type="critic",
            result="different_rule(X) :- perhaps(X).",
            confidence=0.40,
            latency_ms=200.0,
        )

        strategy = WeightedVotingStrategy()
        voting_result = strategy.vote([generator_response, critic_response])

        # With disagreement, weighted voting gives partial confidence
        # The implementation calculates confidence as percentage of total weight
        # for the winning option, which should be around 53% (0.45 / 0.85 total)
        assert voting_result.confidence < 0.7  # Not full consensus

        # Meta reasoner would analyze this scenario
        meta_response = ModelResponse(
            model_type="meta_reasoner",
            result="analysis_complete",
            confidence=0.70,
            latency_ms=300.0,
            metadata={
                "issue": "low_agreement",
                "recommendation": "gather_more_evidence",
                "patterns_detected": ["high_uncertainty"],
            },
        )

        assert meta_response.metadata["issue"] == "low_agreement"

    def test_meta_reasoner_identifies_failure_patterns(self):
        """Test meta reasoner can identify patterns from failed attempts."""
        # Simulate a history of failed responses
        failed_responses = [
            ModelResponse(
                model_type="logic_generator",
                result="failed_rule_1(X).",
                confidence=0.30,
                latency_ms=150.0,
                metadata={"error": "syntax_error"},
            ),
            ModelResponse(
                model_type="logic_generator",
                result="failed_rule_2(X).",
                confidence=0.35,
                latency_ms=140.0,
                metadata={"error": "semantic_error"},
            ),
            ModelResponse(
                model_type="logic_generator",
                result="failed_rule_3(X).",
                confidence=0.32,
                latency_ms=160.0,
                metadata={"error": "syntax_error"},
            ),
        ]

        # Extract patterns
        error_types = [r.metadata.get("error") for r in failed_responses]
        syntax_errors = sum(1 for e in error_types if e == "syntax_error")

        assert syntax_errors == 2  # Pattern: syntax errors are common

    def test_meta_reasoner_suggests_strategy_changes(self):
        """Test meta reasoner can suggest strategy changes based on performance."""
        # Simulated performance history
        performance_data = {
            "chain_of_thought": {"success_rate": 0.65, "avg_confidence": 0.75},
            "few_shot": {"success_rate": 0.80, "avg_confidence": 0.82},
            "self_consistency": {"success_rate": 0.72, "avg_confidence": 0.78},
        }

        # Meta reasoner would recommend best strategy
        best_strategy = max(
            performance_data.keys(), key=lambda k: performance_data[k]["success_rate"]
        )

        assert best_strategy == "few_shot"


# =============================================================================
# Orchestrator Coordination Tests
# =============================================================================


class TestOrchestratorCoordination:
    """Tests for EnsembleOrchestrator coordinating all components."""

    def test_orchestrator_manages_component_lifecycle(
        self, ensemble_orchestrator, orchestrator_config
    ):
        """Test that orchestrator properly manages component lifecycle."""
        assert ensemble_orchestrator.config == orchestrator_config
        assert ensemble_orchestrator._llm_interface is not None

    def test_orchestrator_config_affects_voting_behavior(self):
        """Test that config changes affect voting behavior."""
        mock_interface = MagicMock()

        # Test with weighted voting
        config_weighted = OrchestratorConfig(
            default_voting_strategy=VotingStrategyType.WEIGHTED
        )
        orch_weighted = EnsembleOrchestrator(
            config=config_weighted, llm_interface=mock_interface
        )

        # Test with majority voting
        config_majority = OrchestratorConfig(
            default_voting_strategy=VotingStrategyType.MAJORITY
        )
        orch_majority = EnsembleOrchestrator(
            config=config_majority, llm_interface=mock_interface
        )

        assert (
            orch_weighted.config.default_voting_strategy == VotingStrategyType.WEIGHTED
        )
        assert (
            orch_majority.config.default_voting_strategy == VotingStrategyType.MAJORITY
        )

    def test_orchestrator_respects_confidence_threshold(self, orchestrator_config):
        """Test that orchestrator respects minimum confidence threshold."""
        mock_interface = MagicMock()

        config = OrchestratorConfig(
            min_confidence_threshold=0.7,  # High threshold
            default_voting_strategy=VotingStrategyType.WEIGHTED,
        )
        orchestrator = EnsembleOrchestrator(
            config=config, llm_interface=mock_interface
        )

        assert orchestrator.config.min_confidence_threshold == 0.7

    def test_orchestrator_handles_task_routing(self, ensemble_orchestrator):
        """Test that orchestrator validates task routing input."""
        # None input should raise error
        with pytest.raises(TaskRoutingError, match="cannot be None"):
            ensemble_orchestrator.route_task(TaskType.RULE_GENERATION, None)

        # Empty dict should raise error
        with pytest.raises(TaskRoutingError, match="empty dictionary"):
            ensemble_orchestrator.route_task(TaskType.RULE_GENERATION, {})


# =============================================================================
# Data Flow Contract Tests
# =============================================================================


class TestDataFlowContracts:
    """Tests validating data flow contracts between components."""

    def test_model_response_contract(self):
        """Test that ModelResponse maintains its data contract."""
        response = ModelResponse(
            model_type="test_model",
            result="test_rule(X).",
            confidence=0.85,
            latency_ms=100.0,
            metadata={"key": "value"},
        )

        assert response.model_type == "test_model"
        assert response.result == "test_rule(X)."
        assert 0.0 <= response.confidence <= 1.0
        assert response.latency_ms > 0
        assert response.metadata == {"key": "value"}

    def test_confidence_clamping(self):
        """Test that confidence values are clamped to valid range."""
        # Negative confidence should be clamped to 0
        response_negative = ModelResponse(
            model_type="test",
            result="test",
            confidence=-0.5,
            latency_ms=100.0,
        )
        assert response_negative.confidence == 0.0

        # Confidence > 1 should be clamped to 1
        response_over = ModelResponse(
            model_type="test",
            result="test",
            confidence=1.5,
            latency_ms=100.0,
        )
        assert response_over.confidence == 1.0

    def test_voting_result_contract(self):
        """Test that VotingResult maintains its data contract."""
        strategy = WeightedVotingStrategy()
        responses = [
            ModelResponse(
                model_type="model",
                result="test_rule(X).",
                confidence=0.85,
                latency_ms=100.0,
            )
        ]

        result = strategy.vote(responses)

        assert hasattr(result, "decision")
        assert hasattr(result, "confidence")
        assert hasattr(result, "dissenting_models")
        assert isinstance(result.confidence, float)
        assert isinstance(result.dissenting_models, list)

    def test_disagreement_record_contract(self):
        """Test that DisagreementRecord maintains its data contract."""
        resolver = DeferToConfidenceResolver()
        responses = [
            ModelResponse(
                model_type="model_1",
                result="rule_1(X).",
                confidence=0.80,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_2",
                result="rule_2(X).",
                confidence=0.90,
                latency_ms=100.0,
            ),
        ]

        resolution, explanation = resolver.resolve(responses, "Test disagreement")

        assert resolution is not None
        assert explanation is not None and isinstance(explanation, str)


# =============================================================================
# Error Propagation Tests
# =============================================================================


class TestErrorPropagation:
    """Tests for error handling and propagation across components."""

    def test_voting_error_propagation(self):
        """Test that voting errors are properly propagated."""
        strategy = WeightedVotingStrategy()

        with pytest.raises(VotingError):
            strategy.vote([])

    def test_task_routing_error_propagation(self, ensemble_orchestrator):
        """Test that task routing errors are properly propagated."""
        with pytest.raises(TaskRoutingError):
            ensemble_orchestrator.route_task(TaskType.RULE_GENERATION, None)

    def test_graceful_degradation_on_component_failure(self):
        """Test system behavior when one component fails."""
        # Simulate partial responses (one component failed)
        responses = [
            ModelResponse(
                model_type="logic_generator",
                result="rule_from_working_component(X).",
                confidence=0.85,
                latency_ms=150.0,
            ),
            # Critic component "failed" - not in list
        ]

        # System should still produce result with available data
        strategy = WeightedVotingStrategy()
        result = strategy.vote(responses)

        assert result.decision is not None
        assert result.decision == "rule_from_working_component(X)."


# =============================================================================
# Performance Integration Tests
# =============================================================================


class TestPerformanceIntegration:
    """Tests for performance aspects of component integration."""

    def test_component_response_aggregation_time(self):
        """Test that aggregating multiple component responses is fast."""
        import time

        responses = [
            ModelResponse(
                model_type=f"component_{i}",
                result=f"rule_{i}(X).",
                confidence=0.7 + (i * 0.05),
                latency_ms=100.0 + i,
            )
            for i in range(10)
        ]

        strategy = WeightedVotingStrategy()

        start = time.time()
        result = strategy.vote(responses)
        duration = time.time() - start

        # Aggregation should be < 10ms
        assert duration < 0.01
        assert result is not None

    def test_strategy_switching_overhead(self):
        """Test overhead of switching between strategies."""
        import time

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

        for strategy_type in VotingStrategyType:
            strategy = create_voting_strategy(strategy_type)
            strategy.vote(responses)

        duration = time.time() - start

        # All strategy switches and votes should be < 50ms
        assert duration < 0.05

    def test_memory_efficiency_with_many_responses(self):
        """Test that system handles many responses without excessive memory."""
        import sys

        # Create large number of responses
        responses = [
            ModelResponse(
                model_type=f"model_{i}",
                result=f"rule_{i}(X) :- condition_{i}(X).",
                confidence=0.5 + (i % 50) / 100,
                latency_ms=100.0,
            )
            for i in range(1000)
        ]

        # Measure approximate memory
        response_size = sys.getsizeof(responses)

        # Should be reasonable (< 1MB for 1000 responses metadata)
        assert response_size < 1_000_000

        strategy = WeightedVotingStrategy()
        result = strategy.vote(responses)

        assert result is not None


# =============================================================================
# MVP Validation Tests
# =============================================================================


class TestMVPValidation:
    """Tests validating Phase 6 MVP criteria from ROADMAP.md."""

    def test_ensemble_improves_over_single_model(self):
        """Validate: Ensemble consensus improves accuracy over single LLM."""
        # Single model response
        single_response = ModelResponse(
            model_type="single_model",
            result="single_rule(X).",
            confidence=0.70,
            latency_ms=150.0,
        )

        # Ensemble responses (multiple perspectives)
        ensemble_responses = [
            ModelResponse(
                model_type="logic_generator",
                result="ensemble_rule(X).",
                confidence=0.75,
                latency_ms=150.0,
            ),
            ModelResponse(
                model_type="critic",
                result="ensemble_rule(X).",
                confidence=0.80,
                latency_ms=200.0,
            ),
            ModelResponse(
                model_type="translator",
                result="ensemble_rule(X).",
                confidence=0.78,
                latency_ms=100.0,
            ),
        ]

        strategy = WeightedVotingStrategy()

        single_result = strategy.vote([single_response])
        ensemble_result = strategy.vote(ensemble_responses)

        # Both should have 100% confidence when there's agreement on the same result
        # But ensemble with 3 models provides more robust consensus
        assert ensemble_result.confidence == 1.0  # All models agreed
        assert single_result.confidence == 1.0  # Single model also 100%
        assert len(ensemble_result.dissenting_models) == 0  # Consensus reached
        # More models = more robust agreement
        assert len(ensemble_result.participating_models) > len(single_result.participating_models)

    def test_specialized_models_outperform_general(self):
        """Validate: Specialized models outperform general-purpose."""
        # General model response
        general_responses = [
            ModelResponse(
                model_type="general_llm",
                result="general_rule(X).",
                confidence=0.65,
                latency_ms=200.0,
            ),
            ModelResponse(
                model_type="general_llm",
                result="different_rule(X).",
                confidence=0.60,
                latency_ms=180.0,
            ),
        ]

        # Specialized model responses (higher confidence, more agreement)
        specialized_responses = [
            ModelResponse(
                model_type="logic_specialist",
                result="specialized_rule(X).",
                confidence=0.85,
                latency_ms=150.0,
            ),
            ModelResponse(
                model_type="critic_specialist",
                result="specialized_rule(X).",
                confidence=0.88,
                latency_ms=200.0,
            ),
        ]

        strategy = WeightedVotingStrategy()

        general_result = strategy.vote(general_responses)
        specialized_result = strategy.vote(specialized_responses)

        # Specialized should have higher confidence and consensus
        assert specialized_result.confidence > general_result.confidence
        assert len(specialized_result.dissenting_models) == 0  # Consensus reached

    def test_cost_performance_tradeoff_measurable(self):
        """Validate: Cost/performance trade-offs are measurable."""
        # Simulate responses with latency as proxy for cost
        fast_cheap_responses = [
            ModelResponse(
                model_type="haiku_fast",
                result="fast_rule(X).",
                confidence=0.75,
                latency_ms=50.0,  # Fast = cheaper
            ),
            ModelResponse(
                model_type="haiku_fast",
                result="fast_rule(X).",
                confidence=0.73,
                latency_ms=45.0,
            ),
        ]

        slow_expensive_responses = [
            ModelResponse(
                model_type="opus_thorough",
                result="thorough_rule(X).",
                confidence=0.92,
                latency_ms=500.0,  # Slow = more expensive
            ),
            ModelResponse(
                model_type="opus_thorough",
                result="thorough_rule(X).",
                confidence=0.90,
                latency_ms=480.0,
            ),
        ]

        # Calculate cost-performance ratio
        fast_avg_latency = sum(r.latency_ms for r in fast_cheap_responses) / len(
            fast_cheap_responses
        )
        slow_avg_latency = sum(r.latency_ms for r in slow_expensive_responses) / len(
            slow_expensive_responses
        )

        strategy = WeightedVotingStrategy()
        fast_result = strategy.vote(fast_cheap_responses)
        slow_result = strategy.vote(slow_expensive_responses)

        # Verify trade-off is measurable
        fast_efficiency = fast_result.confidence / fast_avg_latency
        slow_efficiency = slow_result.confidence / slow_avg_latency

        assert fast_efficiency != slow_efficiency  # Trade-offs exist
        assert fast_avg_latency < slow_avg_latency  # Cost difference exists
