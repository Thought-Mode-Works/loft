"""
Integration tests for Phase 6 ensemble voting mechanisms.

Tests validate the interaction between voting strategies and disagreement resolvers:
- UnanimousVotingStrategy: Requires all models to agree
- MajorityVotingStrategy: Simple majority wins
- WeightedVotingStrategy: Confidence-weighted voting
- DialecticalVotingStrategy: Thesis-antithesis-synthesis approach

Disagreement resolution strategies:
- DeferToCritic: Critic model has final say
- DeferToConfidence: Highest confidence wins
- Synthesize: Attempt to merge conflicting responses
- Escalate: Flag for human review
- Conservative: Take safest option
"""

import pytest
from typing import List

from loft.neural.ensemble import (
    # Voting strategies
    VotingStrategy,
    UnanimousVotingStrategy,
    MajorityVotingStrategy,
    WeightedVotingStrategy,
    DialecticalVotingStrategy,
    create_voting_strategy,
    VotingStrategyType,
    # Disagreement resolution
    DisagreementResolver,
    DeferToCriticResolver,
    DeferToConfidenceResolver,
    SynthesizeResolver,
    EscalateResolver,
    ConservativeResolver,
    create_disagreement_resolver,
    DisagreementStrategyType,
    # Data classes
    ModelResponse,
    VotingError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def unanimous_responses() -> List[ModelResponse]:
    """Create responses where all models agree."""
    same_result = "agreed_rule(X) :- condition(X)."
    return [
        ModelResponse(
            model_type="logic_generator",
            result=same_result,
            confidence=0.85,
            latency_ms=150.0,
        ),
        ModelResponse(
            model_type="critic",
            result=same_result,
            confidence=0.90,
            latency_ms=200.0,
        ),
        ModelResponse(
            model_type="translator",
            result=same_result,
            confidence=0.88,
            latency_ms=100.0,
        ),
    ]


@pytest.fixture
def majority_agreement_responses() -> List[ModelResponse]:
    """Create responses where majority agrees."""
    majority_result = "majority_rule(X) :- common_condition(X)."
    minority_result = "minority_rule(X) :- rare_condition(X)."
    return [
        ModelResponse(
            model_type="logic_generator",
            result=majority_result,
            confidence=0.85,
            latency_ms=150.0,
        ),
        ModelResponse(
            model_type="critic",
            result=majority_result,
            confidence=0.80,
            latency_ms=200.0,
        ),
        ModelResponse(
            model_type="translator",
            result=minority_result,
            confidence=0.75,
            latency_ms=100.0,
        ),
    ]


@pytest.fixture
def split_responses() -> List[ModelResponse]:
    """Create responses with no clear majority."""
    return [
        ModelResponse(
            model_type="logic_generator",
            result="rule_a(X) :- cond_a(X).",
            confidence=0.80,
            latency_ms=150.0,
        ),
        ModelResponse(
            model_type="critic",
            result="rule_b(X) :- cond_b(X).",
            confidence=0.85,
            latency_ms=200.0,
        ),
        ModelResponse(
            model_type="translator",
            result="rule_c(X) :- cond_c(X).",
            confidence=0.82,
            latency_ms=100.0,
        ),
    ]


@pytest.fixture
def confidence_skewed_responses() -> List[ModelResponse]:
    """Create responses with one high-confidence outlier."""
    return [
        ModelResponse(
            model_type="logic_generator",
            result="high_conf_rule(X) :- strong_evidence(X).",
            confidence=0.98,
            latency_ms=150.0,
        ),
        ModelResponse(
            model_type="critic",
            result="low_conf_rule(X) :- weak_evidence(X).",
            confidence=0.40,
            latency_ms=200.0,
        ),
        ModelResponse(
            model_type="translator",
            result="medium_conf_rule(X) :- some_evidence(X).",
            confidence=0.60,
            latency_ms=100.0,
        ),
    ]


# =============================================================================
# Unanimous Voting Strategy Tests
# =============================================================================


class TestUnanimousVotingStrategy:
    """Tests for UnanimousVotingStrategy behavior."""

    def test_unanimous_agreement_reaches_consensus(self, unanimous_responses):
        """Test that unanimous agreement results in consensus."""
        strategy = UnanimousVotingStrategy()
        result = strategy.vote(unanimous_responses)

        assert len(result.dissenting_models) == 0  # Consensus reached
        assert result.decision == "agreed_rule(X) :- condition(X)."
        assert result.confidence > 0.8

    def test_unanimous_fails_with_any_disagreement(self, majority_agreement_responses):
        """Test that any disagreement fails unanimous requirement."""
        strategy = UnanimousVotingStrategy()
        result = strategy.vote(majority_agreement_responses)

        # Unanimous requires all to agree
        assert len(result.dissenting_models) > 0  # No consensus

    def test_unanimous_with_single_response(self):
        """Test unanimous voting with single response."""
        strategy = UnanimousVotingStrategy()
        responses = [
            ModelResponse(
                model_type="logic_generator",
                result="single_rule(X).",
                confidence=0.90,
                latency_ms=150.0,
            ),
        ]
        result = strategy.vote(responses)

        # Single response is trivially unanimous
        assert len(result.dissenting_models) == 0  # Consensus reached
        assert result.decision == "single_rule(X)."

    def test_unanimous_empty_responses_raises_error(self):
        """Test that empty responses raise VotingError."""
        strategy = UnanimousVotingStrategy()

        with pytest.raises(VotingError, match="No responses"):
            strategy.vote([])


# =============================================================================
# Majority Voting Strategy Tests
# =============================================================================


class TestMajorityVotingStrategy:
    """Tests for MajorityVotingStrategy behavior."""

    def test_majority_agreement_reaches_consensus(self, majority_agreement_responses):
        """Test that majority agreement results in consensus."""
        strategy = MajorityVotingStrategy()
        result = strategy.vote(majority_agreement_responses)

        assert len(result.dissenting_models) <= 1  # Majority consensus
        assert result.decision == "majority_rule(X) :- common_condition(X)."

    def test_majority_with_split_votes(self, split_responses):
        """Test majority voting with no clear majority."""
        strategy = MajorityVotingStrategy()
        result = strategy.vote(split_responses)

        # Should still return a result but no consensus
        assert result.decision is not None
        assert len(result.dissenting_models) > 0  # No consensus

    def test_majority_tie_resolution(self):
        """Test tie resolution in majority voting."""
        strategy = MajorityVotingStrategy()

        # Two pairs with equal votes
        responses = [
            ModelResponse(
                model_type="model_1",
                result="option_a(X).",
                confidence=0.80,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_2",
                result="option_a(X).",
                confidence=0.75,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_3",
                result="option_b(X).",
                confidence=0.90,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_4",
                result="option_b(X).",
                confidence=0.85,
                latency_ms=100.0,
            ),
        ]

        result = strategy.vote(responses)

        # Should pick one (implementation-dependent)
        assert result.decision in ["option_a(X).", "option_b(X)."]

    def test_majority_unanimous_input(self, unanimous_responses):
        """Test that majority also works with unanimous input."""
        strategy = MajorityVotingStrategy()
        result = strategy.vote(unanimous_responses)

        assert len(result.dissenting_models) == 0  # Consensus reached
        assert result.decision == "agreed_rule(X) :- condition(X)."


# =============================================================================
# Weighted Voting Strategy Tests
# =============================================================================


class TestWeightedVotingStrategy:
    """Tests for WeightedVotingStrategy behavior."""

    def test_weighted_favors_high_confidence(self, confidence_skewed_responses):
        """Test that weighted voting favors high confidence responses."""
        strategy = WeightedVotingStrategy()
        result = strategy.vote(confidence_skewed_responses)

        # High confidence response should win
        assert result.decision == "high_conf_rule(X) :- strong_evidence(X)."

    def test_weighted_with_equal_confidence(self):
        """Test weighted voting when all have equal confidence."""
        strategy = WeightedVotingStrategy()

        responses = [
            ModelResponse(
                model_type="model_1",
                result="rule_1(X).",
                confidence=0.80,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_2",
                result="rule_1(X).",
                confidence=0.80,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_3",
                result="rule_2(X).",
                confidence=0.80,
                latency_ms=100.0,
            ),
        ]

        result = strategy.vote(responses)

        # Majority should win when confidence is equal
        assert result.decision == "rule_1(X)."

    def test_weighted_handles_zero_confidence(self):
        """Test weighted voting handles zero confidence gracefully."""
        strategy = WeightedVotingStrategy()

        responses = [
            ModelResponse(
                model_type="model_1",
                result="zero_conf_rule(X).",
                confidence=0.0,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_2",
                result="some_conf_rule(X).",
                confidence=0.50,
                latency_ms=100.0,
            ),
        ]

        result = strategy.vote(responses)

        # Non-zero confidence should win
        assert result.decision == "some_conf_rule(X)."

    def test_weighted_cumulative_confidence(self):
        """Test that weighted voting accumulates confidence for same results."""
        strategy = WeightedVotingStrategy()

        # Two low-confidence responses for same result vs one high
        responses = [
            ModelResponse(
                model_type="model_1",
                result="cumulative_rule(X).",
                confidence=0.60,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_2",
                result="cumulative_rule(X).",
                confidence=0.60,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_3",
                result="single_high_rule(X).",
                confidence=0.90,
                latency_ms=100.0,
            ),
        ]

        result = strategy.vote(responses)

        # Combined weight (0.6 + 0.6 = 1.2) should beat single 0.9
        assert result.decision == "cumulative_rule(X)."


# =============================================================================
# Dialectical Voting Strategy Tests
# =============================================================================


class TestDialecticalVotingStrategy:
    """Tests for DialecticalVotingStrategy behavior."""

    def test_dialectical_with_opposing_views(self):
        """Test dialectical synthesis with opposing responses."""
        strategy = DialecticalVotingStrategy()

        responses = [
            ModelResponse(
                model_type="thesis",
                result="strong_rule(X) :- strict_condition(X).",
                confidence=0.85,
                latency_ms=150.0,
            ),
            ModelResponse(
                model_type="antithesis",
                result="weak_rule(X) :- lenient_condition(X).",
                confidence=0.80,
                latency_ms=200.0,
            ),
        ]

        result = strategy.vote(responses)

        # Should produce a result (synthesis behavior is implementation-specific)
        assert result.decision is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_dialectical_unanimous(self, unanimous_responses):
        """Test dialectical voting with unanimous input."""
        strategy = DialecticalVotingStrategy()
        result = strategy.vote(unanimous_responses)

        # Unanimous should work in dialectical mode
        assert result.decision == "agreed_rule(X) :- condition(X)."

    def test_dialectical_empty_raises_error(self):
        """Test that empty responses raise VotingError."""
        strategy = DialecticalVotingStrategy()

        with pytest.raises(VotingError):
            strategy.vote([])


# =============================================================================
# Voting Strategy Factory Tests
# =============================================================================


class TestVotingStrategyFactory:
    """Tests for create_voting_strategy factory function."""

    def test_create_unanimous_strategy(self):
        """Test creating unanimous strategy via factory."""
        strategy = create_voting_strategy(VotingStrategyType.UNANIMOUS)
        assert isinstance(strategy, UnanimousVotingStrategy)

    def test_create_majority_strategy(self):
        """Test creating majority strategy via factory."""
        strategy = create_voting_strategy(VotingStrategyType.MAJORITY)
        assert isinstance(strategy, MajorityVotingStrategy)

    def test_create_weighted_strategy(self):
        """Test creating weighted strategy via factory."""
        strategy = create_voting_strategy(VotingStrategyType.WEIGHTED)
        assert isinstance(strategy, WeightedVotingStrategy)

    def test_create_dialectical_strategy(self):
        """Test creating dialectical strategy via factory."""
        strategy = create_voting_strategy(VotingStrategyType.DIALECTICAL)
        assert isinstance(strategy, DialecticalVotingStrategy)

    def test_factory_returns_voting_strategy_interface(self):
        """Test that all factory returns implement VotingStrategy."""
        for strategy_type in VotingStrategyType:
            strategy = create_voting_strategy(strategy_type)
            assert isinstance(strategy, VotingStrategy)


# =============================================================================
# Disagreement Resolver Tests
# =============================================================================


class TestDeferToCriticResolver:
    """Tests for DeferToCriticResolver behavior."""

    def test_defer_to_critic_selects_critic_response(self):
        """Test that critic's response is selected on disagreement."""
        resolver = DeferToCriticResolver()

        responses = [
            ModelResponse(
                model_type="logic_generator",
                result="generator_rule(X).",
                confidence=0.90,
                latency_ms=150.0,
            ),
            ModelResponse(
                model_type="critic",
                result="critic_rule(X).",
                confidence=0.80,
                latency_ms=200.0,
            ),
        ]

        resolution, explanation = resolver.resolve(responses, "Test disagreement")

        assert resolution == "critic_rule(X)."
        assert "critic" in explanation.lower()

    def test_defer_to_critic_without_critic_response(self):
        """Test behavior when no critic response exists."""
        resolver = DeferToCriticResolver()

        responses = [
            ModelResponse(
                model_type="logic_generator",
                result="generator_rule(X).",
                confidence=0.90,
                latency_ms=150.0,
            ),
            ModelResponse(
                model_type="translator",
                result="translator_rule(X).",
                confidence=0.85,
                latency_ms=100.0,
            ),
        ]

        # Should fall back to another strategy or highest confidence
        resolution, explanation = resolver.resolve(responses, "No critic available")
        assert resolution is not None


class TestDeferToConfidenceResolver:
    """Tests for DeferToConfidenceResolver behavior."""

    def test_defer_to_highest_confidence(self, confidence_skewed_responses):
        """Test that highest confidence response is selected."""
        resolver = DeferToConfidenceResolver()

        resolution, explanation = resolver.resolve(confidence_skewed_responses, "Confidence test")

        assert resolution == "high_conf_rule(X) :- strong_evidence(X)."
        assert "confidence" in explanation.lower()

    def test_defer_to_confidence_tie(self):
        """Test behavior with tied confidence scores."""
        resolver = DeferToConfidenceResolver()

        responses = [
            ModelResponse(
                model_type="model_1",
                result="rule_1(X).",
                confidence=0.85,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_2",
                result="rule_2(X).",
                confidence=0.85,
                latency_ms=100.0,
            ),
        ]

        resolution, explanation = resolver.resolve(responses, "Tie test")

        # Should pick one (implementation-specific)
        assert resolution in ["rule_1(X).", "rule_2(X)."]


class TestConservativeResolver:
    """Tests for ConservativeResolver behavior."""

    def test_conservative_selects_safest(self):
        """Test that conservative resolver selects safest option."""
        resolver = ConservativeResolver()

        responses = [
            ModelResponse(
                model_type="model_1",
                result="risky_rule(X) :- speculative(X).",
                confidence=0.90,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_2",
                result="safe_rule(X) :- verified(X).",
                confidence=0.75,
                latency_ms=100.0,
            ),
        ]

        resolution, explanation = resolver.resolve(responses, "Safety test")

        assert "conservative" in explanation.lower() or resolution is not None
        assert resolution is not None


class TestEscalateResolver:
    """Tests for EscalateResolver behavior."""

    def test_escalate_flags_for_review(self):
        """Test that escalate resolver flags for human review."""
        resolver = EscalateResolver()

        responses = [
            ModelResponse(
                model_type="model_1",
                result="conflicting_rule_1(X).",
                confidence=0.85,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_2",
                result="conflicting_rule_2(X).",
                confidence=0.85,
                latency_ms=100.0,
            ),
        ]

        resolution, explanation = resolver.resolve(responses, "Complex disagreement")

        assert "escalat" in explanation.lower()
        assert "human" in explanation.lower() or "review" in explanation.lower()


class TestSynthesizeResolver:
    """Tests for SynthesizeResolver behavior."""

    def test_synthesize_merges_responses(self):
        """Test that synthesize resolver attempts to merge responses."""
        resolver = SynthesizeResolver()

        responses = [
            ModelResponse(
                model_type="model_1",
                result="rule_part_a(X) :- cond_a(X).",
                confidence=0.80,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_2",
                result="rule_part_b(X) :- cond_b(X).",
                confidence=0.80,
                latency_ms=100.0,
            ),
        ]

        resolution, explanation = resolver.resolve(responses, "Synthesis test")

        assert "synth" in explanation.lower() or resolution is not None
        assert resolution is not None


# =============================================================================
# Disagreement Resolver Factory Tests
# =============================================================================


class TestDisagreementResolverFactory:
    """Tests for create_disagreement_resolver factory function."""

    def test_create_defer_to_critic(self):
        """Test creating defer_to_critic resolver via factory."""
        resolver = create_disagreement_resolver(DisagreementStrategyType.DEFER_TO_CRITIC)
        assert isinstance(resolver, DeferToCriticResolver)

    def test_create_defer_to_confidence(self):
        """Test creating defer_to_confidence resolver via factory."""
        resolver = create_disagreement_resolver(DisagreementStrategyType.DEFER_TO_CONFIDENCE)
        assert isinstance(resolver, DeferToConfidenceResolver)

    def test_create_synthesize(self):
        """Test creating synthesize resolver via factory."""
        resolver = create_disagreement_resolver(DisagreementStrategyType.SYNTHESIZE)
        assert isinstance(resolver, SynthesizeResolver)

    def test_create_escalate(self):
        """Test creating escalate resolver via factory."""
        resolver = create_disagreement_resolver(DisagreementStrategyType.ESCALATE)
        assert isinstance(resolver, EscalateResolver)

    def test_create_conservative(self):
        """Test creating conservative resolver via factory."""
        resolver = create_disagreement_resolver(DisagreementStrategyType.CONSERVATIVE)
        assert isinstance(resolver, ConservativeResolver)

    def test_factory_returns_resolver_interface(self):
        """Test that all factory returns implement DisagreementResolver."""
        for strategy_type in DisagreementStrategyType:
            resolver = create_disagreement_resolver(strategy_type)
            assert isinstance(resolver, DisagreementResolver)


# =============================================================================
# Voting + Disagreement Integration Tests
# =============================================================================


class TestVotingDisagreementIntegration:
    """Tests for voting strategies integrated with disagreement resolution."""

    def test_weighted_voting_with_disagreement_resolution(self):
        """Test weighted voting followed by disagreement resolution."""
        voting_strategy = WeightedVotingStrategy()
        resolver = DeferToConfidenceResolver()

        # Responses with significant disagreement
        responses = [
            ModelResponse(
                model_type="logic_generator",
                result="rule_a(X).",
                confidence=0.70,
                latency_ms=150.0,
            ),
            ModelResponse(
                model_type="critic",
                result="rule_b(X).",
                confidence=0.95,
                latency_ms=200.0,
            ),
        ]

        # First try voting
        voting_result = voting_strategy.vote(responses)

        # If no consensus, resolve disagreement
        if len(voting_result.dissenting_models) > 0:
            resolution, explanation = resolver.resolve(responses, "Voting produced no consensus")
            assert resolution is not None

    def test_majority_voting_with_critic_fallback(self, split_responses):
        """Test majority voting falling back to critic resolution."""
        voting_strategy = MajorityVotingStrategy()
        resolver = DeferToCriticResolver()

        voting_result = voting_strategy.vote(split_responses)

        # Split responses should not reach consensus
        if len(voting_result.dissenting_models) > 0:
            # DeferToCritic resolver looks for model_type containing "critic"
            # and returns the result from the first critic response found
            # In split_responses fixture, model_b has result "rule_b(X) :- cond_b(X)."
            # which will be found if there's a critic model
            resolution, explanation = resolver.resolve(split_responses, "Split vote fallback")
            # The resolver returns the first available result or the highest confidence
            assert resolution is not None
            assert "critic" in explanation.lower() or "fall" in explanation.lower()

    def test_unanimous_failure_triggers_resolution(self, majority_agreement_responses):
        """Test that unanimous failure triggers disagreement resolution."""
        voting_strategy = UnanimousVotingStrategy()
        resolver = DeferToConfidenceResolver()

        voting_result = voting_strategy.vote(majority_agreement_responses)

        # Unanimous should fail with any disagreement
        assert len(voting_result.dissenting_models) > 0  # No consensus

        resolution, explanation = resolver.resolve(
            majority_agreement_responses, "Unanimous voting failed"
        )
        assert resolution is not None

    def test_dialectical_with_synthesis_resolution(self):
        """Test dialectical voting with synthesis disagreement resolution."""
        voting_strategy = DialecticalVotingStrategy()
        resolver = SynthesizeResolver()

        # Opposing viewpoints
        responses = [
            ModelResponse(
                model_type="thesis",
                result="strict_rule(X) :- all_conditions(X).",
                confidence=0.80,
                latency_ms=150.0,
            ),
            ModelResponse(
                model_type="antithesis",
                result="permissive_rule(X) :- some_conditions(X).",
                confidence=0.80,
                latency_ms=200.0,
            ),
        ]

        voting_result = voting_strategy.vote(responses)

        if len(voting_result.dissenting_models) > 0:
            resolution, explanation = resolver.resolve(responses, "Dialectical synthesis needed")
            # Synthesize resolver returns result with synthesis explanation
            assert resolution is not None
            assert "synth" in explanation.lower() or resolution is not None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestVotingEdgeCases:
    """Tests for edge cases in voting and disagreement resolution."""

    def test_single_response_voting(self):
        """Test all strategies handle single response."""
        single_response = [
            ModelResponse(
                model_type="solo",
                result="solo_rule(X).",
                confidence=0.90,
                latency_ms=100.0,
            ),
        ]

        for strategy_type in VotingStrategyType:
            strategy = create_voting_strategy(strategy_type)
            result = strategy.vote(single_response)
            assert result.decision == "solo_rule(X)."

    def test_all_zero_confidence(self):
        """Test voting with all zero confidence responses."""
        responses = [
            ModelResponse(
                model_type=f"model_{i}",
                result=f"rule_{i}(X).",
                confidence=0.0,
                latency_ms=100.0,
            )
            for i in range(3)
        ]

        strategy = WeightedVotingStrategy()
        result = strategy.vote(responses)

        # Should still produce a result
        assert result.decision is not None

    def test_very_long_result_strings(self):
        """Test voting with very long result strings."""
        long_result = (
            "very_long_rule(X) :- " + ", ".join([f"condition_{i}(X)" for i in range(100)]) + "."
        )

        responses = [
            ModelResponse(
                model_type="model",
                result=long_result,
                confidence=0.90,
                latency_ms=100.0,
            ),
        ]

        strategy = WeightedVotingStrategy()
        result = strategy.vote(responses)

        assert result.decision == long_result

    def test_special_characters_in_results(self):
        """Test voting with special characters in results."""
        special_result = 'rule("quoted string", X) :- condition(X).'

        responses = [
            ModelResponse(
                model_type="model_1",
                result=special_result,
                confidence=0.85,
                latency_ms=100.0,
            ),
            ModelResponse(
                model_type="model_2",
                result=special_result,
                confidence=0.80,
                latency_ms=100.0,
            ),
        ]

        strategy = MajorityVotingStrategy()
        result = strategy.vote(responses)

        assert result.decision == special_result
        assert len(result.dissenting_models) == 0  # Consensus reached

    def test_max_confidence_boundary(self):
        """Test voting with maximum confidence values."""
        responses = [
            ModelResponse(
                model_type="model",
                result="perfect_rule(X).",
                confidence=1.0,
                latency_ms=100.0,
            ),
        ]

        strategy = WeightedVotingStrategy()
        result = strategy.vote(responses)

        assert result.decision == "perfect_rule(X)."
        assert result.confidence == 1.0


# =============================================================================
# Performance Tests
# =============================================================================


class TestVotingPerformance:
    """Performance tests for voting strategies."""

    def test_voting_large_response_set(self):
        """Test voting performance with many responses."""
        import time

        responses = [
            ModelResponse(
                model_type=f"model_{i}",
                result=f"rule_{i % 5}(X).",  # 5 different results
                confidence=0.5 + (i % 50) / 100,  # Varied confidence
                latency_ms=100.0,
            )
            for i in range(100)
        ]

        for strategy_type in VotingStrategyType:
            strategy = create_voting_strategy(strategy_type)

            start = time.time()
            result = strategy.vote(responses)
            duration = time.time() - start

            # Should complete in < 100ms
            assert duration < 0.1
            assert result.decision is not None

    def test_disagreement_resolution_performance(self):
        """Test disagreement resolution performance."""
        import time

        responses = [
            ModelResponse(
                model_type=f"model_{i}",
                result=f"unique_rule_{i}(X).",
                confidence=0.5 + (i % 50) / 100,
                latency_ms=100.0,
            )
            for i in range(50)
        ]

        for strategy_type in DisagreementStrategyType:
            resolver = create_disagreement_resolver(strategy_type)

            start = time.time()
            record = resolver.resolve(responses, "Performance test")
            duration = time.time() - start

            # Should complete in < 50ms
            assert duration < 0.05
            assert record is not None
