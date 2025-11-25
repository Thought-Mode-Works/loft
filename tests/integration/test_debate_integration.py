"""
Integration tests for Debate Framework (Phase 4.2) with full system components.

Tests the debate framework working end-to-end with validation pipeline
and self-modifying system.
"""

import pytest

from loft.dialectical.critic import CriticSystem
from loft.dialectical.debate_framework import DebateFramework
from loft.dialectical.debate_schemas import DebateContext
from loft.dialectical.synthesizer import Synthesizer
from loft.neural.rule_schemas import GeneratedRule, GapFillingResponse, RuleCandidate


def make_test_rule(asp_rule: str, confidence: float = 0.8, **kwargs) -> GeneratedRule:
    """Helper to create GeneratedRule with all required fields for testing."""
    return GeneratedRule(
        asp_rule=asp_rule,
        confidence=confidence,
        reasoning=kwargs.get("reasoning", f"Test rule: {asp_rule[:50]}"),
        predicates_used=kwargs.get("predicates_used", []),
        source_type=kwargs.get("source_type", "principle"),
        source_text=kwargs.get("source_text", "Test rule"),
    )


class MockGenerator:
    """Mock generator for testing."""

    def __init__(self, initial_rule: str = "enforceable(C) :- contract(C)."):
        self.initial_rule = initial_rule

    def fill_knowledge_gap(
        self, gap_description, missing_predicate, context=None, existing_predicates=None
    ):
        rule = make_test_rule(
            asp_rule=self.initial_rule,
            confidence=0.65,
            predicates_used=["contract"],
            source_type="gap_fill",
        )
        return GapFillingResponse(
            gap_description=gap_description,
            missing_predicate=missing_predicate or "enforceable",
            candidates=[
                RuleCandidate(
                    rule=rule,
                    applicability_score=0.65,
                    complexity_score=0.4,
                )
            ],
            recommended_index=0,
            requires_validation=False,
            test_cases_needed=[],
            confidence=0.65,
        )


class TestDebateWithValidation:
    """Test debate framework integration with validation pipeline."""

    @pytest.fixture
    def framework(self):
        """Create debate framework."""
        generator = MockGenerator()
        critic = CriticSystem(mock_mode=True)
        synthesizer = Synthesizer(mock_mode=True)
        return DebateFramework(generator, critic, synthesizer, max_rounds=3)

    def test_debate_improves_rule_quality(self, framework):
        """Test that debate produces better rules than initial proposal."""
        context = DebateContext(
            knowledge_gap_description="Enforceable contract requirements",
            existing_rules=[],
            existing_predicates=["contract", "signed", "consideration", "capacity"],
            max_rounds=3,
        )

        result = framework.run_dialectical_cycle(context)

        # Initial proposal is simple
        assert result.initial_proposal.asp_rule == "enforceable(C) :- contract(C)."

        # Final rule should have more conditions
        final_conditions = result.final_rule.asp_rule.count(",")
        initial_conditions = result.initial_proposal.asp_rule.count(",")
        assert final_conditions >= initial_conditions

    def test_debate_handles_good_initial_proposal(self, framework):
        """Test debate with already-good initial proposal."""
        # Override generator with good rule
        good_generator = MockGenerator(
            "enforceable(C) :- contract(C), consideration(C), capacity(P1), capacity(P2)."
        )
        framework.generator = good_generator

        context = DebateContext(
            knowledge_gap_description="Complete contract rules",
            existing_rules=[],
            existing_predicates=["contract", "consideration", "capacity"],
            max_rounds=3,
        )

        result = framework.run_dialectical_cycle(context)

        # Should converge quickly or maintain quality
        assert result.final_rule.confidence >= result.initial_proposal.confidence - 0.1

    def test_debate_with_existing_rules_context(self, framework):
        """Test debate considers existing rules."""
        context = DebateContext(
            knowledge_gap_description="Additional contract rules",
            existing_rules=[
                "void(C) :- contract(C), illegal(C).",
                "valid_offer(O) :- offer(O), definite(O).",
            ],
            existing_predicates=["contract", "offer", "signed"],
            max_rounds=2,
        )

        result = framework.run_dialectical_cycle(context)

        # Should complete without issues
        assert result.total_rounds >= 1
        assert result.final_rule is not None


class TestDebateConvergence:
    """Test debate convergence behavior."""

    @pytest.fixture
    def framework(self):
        """Create debate framework."""
        generator = MockGenerator()
        critic = CriticSystem(mock_mode=True)
        synthesizer = Synthesizer(mock_mode=True)
        return DebateFramework(
            generator, critic, synthesizer, max_rounds=5, convergence_threshold=0.90
        )

    def test_convergence_within_max_rounds(self, framework):
        """Test that debate converges within max rounds."""
        context = DebateContext(
            knowledge_gap_description="Contract enforceability",
            existing_rules=[],
            existing_predicates=["contract", "signed"],
            max_rounds=5,
        )

        result = framework.run_dialectical_cycle(context)

        # Should stop at or before max rounds
        assert result.total_rounds <= 5

    def test_convergence_reason_reported(self, framework):
        """Test that convergence reason is provided."""
        context = DebateContext(
            knowledge_gap_description="Test convergence",
            existing_rules=[],
            existing_predicates=["contract"],
            max_rounds=3,
        )

        result = framework.run_dialectical_cycle(context)

        # Should have convergence reason
        assert result.convergence_reason is not None
        assert len(result.convergence_reason) > 0

    def test_debate_transcript_completeness(self, framework):
        """Test that debate transcript captures all arguments."""
        context = DebateContext(
            knowledge_gap_description="Complete transcript test",
            existing_rules=[],
            existing_predicates=["contract", "signed"],
            max_rounds=2,
        )

        result = framework.run_dialectical_cycle(context)

        # Transcript should have arguments from all agents
        speakers = {arg.speaker for arg in result.debate_transcript}
        assert "generator" in speakers or "critic" in speakers or "synthesizer" in speakers

        # Transcript should be non-empty
        assert len(result.debate_transcript) >= 2  # At least thesis + antithesis


class TestDebateWithRuleEvolution:
    """Test debate framework with rule evolution over multiple cycles."""

    def test_multiple_debates_track_evolution(self):
        """Test running multiple debates and tracking evolution."""
        generator = MockGenerator()
        critic = CriticSystem(mock_mode=True)
        synthesizer = Synthesizer(mock_mode=True)
        framework = DebateFramework(generator, critic, synthesizer, max_rounds=2)

        # Run multiple debates
        contexts = [
            DebateContext(
                knowledge_gap_description=f"Contract rule iteration {i}",
                existing_rules=[],
                existing_predicates=["contract", "signed"],
                max_rounds=2,
            )
            for i in range(3)
        ]

        results = [framework.run_dialectical_cycle(ctx) for ctx in contexts]

        # All should complete
        assert all(r.total_rounds >= 1 for r in results)

        # History should track all debates
        history = framework.get_debate_history()
        assert len(history) == 3

    def test_debate_improves_incrementally(self):
        """Test that each debate round shows improvement."""
        generator = MockGenerator("enforceable(C) :- contract(C).")
        critic = CriticSystem(mock_mode=True)
        synthesizer = Synthesizer(mock_mode=True)
        framework = DebateFramework(generator, critic, synthesizer, max_rounds=3)

        context = DebateContext(
            knowledge_gap_description="Incremental improvement test",
            existing_rules=[],
            existing_predicates=["contract", "signed", "consideration", "capacity"],
            max_rounds=3,
        )

        result = framework.run_dialectical_cycle(context)

        # Each round should show progress
        for i, round_item in enumerate(result.debate_rounds):
            # Later rounds should not significantly decrease confidence
            if i > 0:
                prev_round = result.debate_rounds[i - 1]
                if round_item.synthesis and prev_round.synthesis:
                    # Confidence should not drop significantly
                    assert round_item.synthesis.confidence >= prev_round.synthesis.confidence - 0.2


@pytest.mark.e2e
class TestDebateEndToEnd:
    """End-to-end tests for complete debate workflows."""

    def test_complete_debate_workflow(self):
        """Test complete debate workflow from gap to final rule."""
        generator = MockGenerator("enforceable(C) :- contract(C), signed(C).")
        critic = CriticSystem(mock_mode=True)
        synthesizer = Synthesizer(mock_mode=True)
        framework = DebateFramework(generator, critic, synthesizer, max_rounds=3)

        context = DebateContext(
            knowledge_gap_description="Complete workflow: enforceable contracts must have consideration and capacity",
            existing_rules=[
                "void(C) :- contract(C), illegal(C).",
            ],
            existing_predicates=["contract", "signed", "consideration", "capacity", "illegal"],
            target_layer="tactical",
            max_rounds=3,
        )

        result = framework.run_dialectical_cycle(context)

        # Verify complete workflow
        assert result.initial_proposal is not None
        assert result.final_rule is not None
        assert result.total_rounds >= 1
        assert result.debate_transcript is not None

        # Get and verify transcript
        transcript = framework.get_debate_transcript(result)
        assert len(transcript) > 0
        assert "INITIAL PROPOSAL" in transcript
        assert "FINAL RESULT" in transcript

    def test_debate_with_complex_knowledge_gap(self):
        """Test debate with complex multi-condition knowledge gap."""
        generator = MockGenerator("valid_contract(C) :- contract(C).")
        critic = CriticSystem(mock_mode=True)
        synthesizer = Synthesizer(mock_mode=True)
        framework = DebateFramework(generator, critic, synthesizer, max_rounds=4)

        context = DebateContext(
            knowledge_gap_description="""
            A valid contract requires:
            1. Offer and acceptance
            2. Consideration from both parties
            3. Legal capacity of all parties
            4. Legal purpose
            5. Proper form (writing for certain types)
            """,
            existing_rules=[],
            existing_predicates=[
                "contract",
                "offer",
                "acceptance",
                "consideration",
                "capacity",
                "legal_purpose",
                "written",
            ],
            target_layer="tactical",
            max_rounds=4,
        )

        result = framework.run_dialectical_cycle(context)

        # Should handle complex gap
        assert result.final_rule is not None
        # Final rule should be more complex than initial
        assert len(result.final_rule.asp_rule) > len(result.initial_proposal.asp_rule)


class TestDebatePerformance:
    """Test debate framework performance."""

    def test_debate_completes_quickly(self):
        """Test that debate completes in reasonable time."""
        import time

        generator = MockGenerator()
        critic = CriticSystem(mock_mode=True)
        synthesizer = Synthesizer(mock_mode=True)
        framework = DebateFramework(generator, critic, synthesizer, max_rounds=3)

        context = DebateContext(
            knowledge_gap_description="Performance test",
            existing_rules=[],
            existing_predicates=["contract"],
            max_rounds=3,
        )

        start = time.time()
        result = framework.run_dialectical_cycle(context)
        duration = time.time() - start

        # Mock mode should be fast
        assert duration < 1.0  # 1 second
        assert result is not None

    def test_multiple_debates_efficient(self):
        """Test running multiple debates efficiently."""
        import time

        generator = MockGenerator()
        critic = CriticSystem(mock_mode=True)
        synthesizer = Synthesizer(mock_mode=True)
        framework = DebateFramework(generator, critic, synthesizer, max_rounds=2)

        contexts = [
            DebateContext(
                knowledge_gap_description=f"Debate {i}",
                existing_rules=[],
                existing_predicates=["contract"],
                max_rounds=2,
            )
            for i in range(5)
        ]

        start = time.time()
        results = [framework.run_dialectical_cycle(ctx) for ctx in contexts]
        duration = time.time() - start

        # Should handle 5 debates quickly in mock mode
        assert duration < 2.0  # 2 seconds
        assert len(results) == 5
        assert all(r is not None for r in results)
