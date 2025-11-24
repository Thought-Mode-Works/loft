"""
Unit tests for the Debate Framework (Phase 4.2).

Tests multi-agent debate orchestration, synthesis, and convergence detection.
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


class TestDebateSchemas:
    """Test debate data structures."""

    def test_debate_context_creation(self):
        """Test DebateContext creation."""
        context = DebateContext(
            knowledge_gap_description="Rules for contract enforceability",
            existing_rules=["void(C) :- contract(C), illegal(C)."],
            existing_predicates=["contract", "signed", "consideration"],
            target_layer="tactical",
            max_rounds=3,
        )

        assert context.knowledge_gap_description == "Rules for contract enforceability"
        assert len(context.existing_rules) == 1
        assert len(context.existing_predicates) == 3
        assert context.max_rounds == 3
        assert context.convergence_threshold == 0.85

    def test_dialectical_cycle_result_summary(self):
        """Test DialecticalCycleResult summary generation."""
        from loft.dialectical.debate_schemas import DialecticalCycleResult

        initial = make_test_rule("enforceable(C) :- contract(C).")
        final = make_test_rule("enforceable(C) :- contract(C), consideration(C).")

        result = DialecticalCycleResult(
            initial_proposal=initial,
            final_rule=final,
            debate_rounds=[],
            total_rounds=2,
            converged=True,
            convergence_reason="Threshold exceeded",
            improvement_score=0.15,
        )

        summary = result.get_summary()
        assert "Initial Proposal" in summary
        assert "Final Rule" in summary
        assert "Total Rounds: 2" in summary
        assert "Converged: True" in summary


class TestSynthesizer:
    """Test Synthesizer agent."""

    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer in mock mode."""
        return Synthesizer(mock_mode=True)

    def test_synthesizer_initialization(self, synthesizer):
        """Test synthesizer initialization."""
        assert synthesizer.mock_mode is True
        assert synthesizer.llm_client is None

    def test_synthesizer_combines_thesis_and_antithesis(self, synthesizer):
        """Test basic synthesis functionality."""
        from loft.dialectical.critique_schemas import (
            CritiqueIssue,
            CritiqueReport,
            CritiqueSeverity,
        )

        thesis = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["contract"],
        )

        critique = CritiqueReport(
            rule=thesis.asp_rule,
            issues=[
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing consideration requirement",
                    severity=CritiqueSeverity.MEDIUM,
                )
            ],
            suggested_revision="enforceable(C) :- contract(C), consideration(C).",
        )

        synthesis, argument = synthesizer.synthesize(thesis, critique, [])

        assert isinstance(synthesis, GeneratedRule)
        assert "consideration" in synthesis.asp_rule
        assert argument.speaker == "synthesizer"
        assert synthesis.confidence >= thesis.confidence

    def test_synthesizer_no_changes_needed(self, synthesizer):
        """Test synthesis when thesis is already good."""
        from loft.dialectical.critique_schemas import CritiqueReport

        thesis = make_test_rule(
            asp_rule="enforceable(C) :- contract(C), consideration(C), capacity(P1).",
            predicates_used=["contract", "consideration", "capacity"],
        )

        critique = CritiqueReport(
            rule=thesis.asp_rule,
            recommendation="accept",
        )

        synthesis, argument = synthesizer.synthesize(thesis, critique, [])

        # Should return thesis unchanged
        assert synthesis.asp_rule == thesis.asp_rule
        assert "satisfactory" in argument.content.lower()


class TestDebateFramework:
    """Test DebateFramework orchestration."""

    @pytest.fixture
    def mock_generator(self):
        """Create mock generator."""

        class MockGenerator:
            def fill_knowledge_gap(
                self, gap_description, existing_rules, existing_predicates, target_layer
            ):
                rule = make_test_rule(
                    asp_rule="enforceable(C) :- contract(C), signed(C).",
                    confidence=0.7,
                    predicates_used=["contract", "signed"],
                    source_type="gap_fill",
                )
                return GapFillingResponse(
                    gap_description=gap_description,
                    missing_predicate="enforceable",
                    candidates=[
                        RuleCandidate(
                            rule=rule,
                            applicability_score=0.7,
                            complexity_score=0.5,
                        )
                    ],
                    recommended_index=0,
                    requires_validation=False,
                    test_cases_needed=[],
                    confidence=0.7,
                )

        return MockGenerator()

    @pytest.fixture
    def framework(self, mock_generator):
        """Create debate framework with mock agents."""
        critic = CriticSystem(mock_mode=True)
        synthesizer = Synthesizer(mock_mode=True)
        return DebateFramework(
            generator=mock_generator,
            critic=critic,
            synthesizer=synthesizer,
            max_rounds=3,
            convergence_threshold=0.85,
        )

    def test_framework_initialization(self, framework):
        """Test framework initialization."""
        assert framework.max_rounds == 3
        assert framework.convergence_threshold == 0.85
        assert len(framework.debate_history) == 0

    def test_run_dialectical_cycle(self, framework):
        """Test running complete dialectical cycle."""
        context = DebateContext(
            knowledge_gap_description="Rules for contract enforceability",
            existing_rules=[],
            existing_predicates=["contract", "signed", "consideration"],
            max_rounds=2,
        )

        result = framework.run_dialectical_cycle(context)

        assert result is not None
        assert result.initial_proposal is not None
        assert result.final_rule is not None
        assert result.total_rounds >= 1
        assert len(result.debate_rounds) >= 1
        assert len(result.debate_transcript) >= 2  # At least thesis + antithesis

    def test_early_convergence_on_acceptance(self, framework):
        """Test early convergence when critic accepts thesis."""
        # Create a very good initial rule that critic will accept
        context = DebateContext(
            knowledge_gap_description="Complete enforceable contract rule",
            existing_rules=[],
            existing_predicates=["contract", "signed", "consideration", "capacity"],
            max_rounds=5,
        )

        result = framework.run_dialectical_cycle(context)

        # Mock critic should eventually accept or we hit max rounds
        assert result.total_rounds <= 5

    def test_convergence_detection(self, framework):
        """Test convergence score calculation."""
        thesis = make_test_rule("enforceable(C) :- contract(C).")
        synthesis_same = make_test_rule("enforceable(C) :- contract(C).")
        synthesis_different = make_test_rule("valid(C) :- offer(C), acceptance(C).")

        score_same = framework._calculate_convergence(thesis, synthesis_same)
        score_different = framework._calculate_convergence(thesis, synthesis_different)

        assert score_same == 1.0  # Identical rules
        assert 0.0 <= score_different < 1.0  # Different rules

    def test_debate_transcript_generation(self, framework):
        """Test debate transcript formatting."""
        context = DebateContext(
            knowledge_gap_description="Contract rules",
            existing_rules=[],
            existing_predicates=["contract"],
            max_rounds=2,
        )

        result = framework.run_dialectical_cycle(context)
        transcript = framework.get_debate_transcript(result)

        assert "DIALECTICAL DEBATE TRANSCRIPT" in transcript
        assert "INITIAL PROPOSAL" in transcript
        assert "FINAL RESULT" in transcript
        assert "ROUND" in transcript

    def test_multiple_debate_rounds(self, framework):
        """Test framework handles multiple rounds correctly."""
        context = DebateContext(
            knowledge_gap_description="Multi-round test",
            existing_rules=[],
            existing_predicates=["contract", "signed"],
            max_rounds=3,
        )

        result = framework.run_dialectical_cycle(context)

        # Should run multiple rounds or converge early
        assert 1 <= result.total_rounds <= 3

        # Each round should have complete structure
        for round_item in result.debate_rounds:
            assert round_item.thesis is not None
            assert round_item.antithesis is not None
            assert round_item.thesis_argument is not None
            assert round_item.antithesis_argument is not None

    def test_debate_history_tracking(self, framework):
        """Test that debates are stored in history."""
        context1 = DebateContext(
            knowledge_gap_description="First debate",
            existing_rules=[],
            existing_predicates=["contract"],
            max_rounds=2,
        )

        context2 = DebateContext(
            knowledge_gap_description="Second debate",
            existing_rules=[],
            existing_predicates=["offer"],
            max_rounds=2,
        )

        framework.run_dialectical_cycle(context1)
        framework.run_dialectical_cycle(context2)

        history = framework.get_debate_history()
        assert len(history) == 2
        assert history[0].metadata["knowledge_gap"] == "First debate"
        assert history[1].metadata["knowledge_gap"] == "Second debate"

    def test_improvement_score_calculation(self, framework):
        """Test that improvement scores are calculated."""
        context = DebateContext(
            knowledge_gap_description="Test improvement",
            existing_rules=[],
            existing_predicates=["contract", "signed"],
            max_rounds=2,
        )

        result = framework.run_dialectical_cycle(context)

        # Improvement score should be difference in confidence
        assert isinstance(result.improvement_score, float)
        # Mock synthesis should improve or maintain confidence
        assert result.improvement_score >= -0.1  # Allow small decrease


class TestDebateIntegration:
    """Integration tests for debate framework with all components."""

    def test_full_dialectical_cycle(self):
        """Test complete dialectical cycle with all agents."""
        from loft.neural.rule_schemas import GapFillingResponse

        class MockGenerator:
            def fill_knowledge_gap(
                self, gap_description, existing_rules, existing_predicates, target_layer
            ):
                rule = make_test_rule(
                    asp_rule="enforceable(C) :- contract(C).",
                    confidence=0.6,
                    predicates_used=["contract"],
                    source_type="gap_fill",
                )
                return GapFillingResponse(
                    gap_description=gap_description,
                    missing_predicate="enforceable",
                    candidates=[
                        RuleCandidate(
                            rule=rule,
                            applicability_score=0.6,
                            complexity_score=0.3,
                        )
                    ],
                    recommended_index=0,
                    requires_validation=False,
                    test_cases_needed=[],
                    confidence=0.6,
                )

        generator = MockGenerator()
        critic = CriticSystem(mock_mode=True)
        synthesizer = Synthesizer(mock_mode=True)

        framework = DebateFramework(generator, critic, synthesizer, max_rounds=3)

        context = DebateContext(
            knowledge_gap_description="Rules for enforceable contracts",
            existing_rules=[],
            existing_predicates=["contract", "signed", "consideration", "capacity"],
            max_rounds=3,
        )

        result = framework.run_dialectical_cycle(context)

        # Verify complete cycle
        assert result.initial_proposal.asp_rule == "enforceable(C) :- contract(C)."
        assert result.final_rule is not None
        assert result.total_rounds >= 1

        # Final rule should be improved (have more conditions)
        assert len(result.final_rule.asp_rule) >= len(result.initial_proposal.asp_rule)
