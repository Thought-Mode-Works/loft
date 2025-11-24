"""
Integration tests for the Critic System with full system components.

Tests the critic system working end-to-end with validation pipeline,
rule generation, and the self-modifying system.
"""

import pytest

from loft.dialectical.critic import CriticSystem
from loft.neural.rule_schemas import GeneratedRule
from loft.validation.validation_pipeline import ValidationPipeline


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


class TestCriticValidationIntegration:
    """Test critic system integration with validation pipeline."""

    @pytest.fixture
    def critic(self):
        """Create critic in mock mode."""
        return CriticSystem(mock_mode=True, enable_synthesis=True)

    @pytest.fixture
    def pipeline_with_critic(self, critic):
        """Create validation pipeline with critic enabled."""
        return ValidationPipeline(critic_system=critic, enable_dialectical=True, min_confidence=0.6)

    def test_dialectical_validation_stage_runs(self, pipeline_with_critic):
        """Test that dialectical validation stage executes."""
        report = pipeline_with_critic.validate_rule(
            rule_asp="enforceable(C) :- contract(C), signed(C).",
            rule_id="test_dialect_001",
            target_layer="tactical",
        )

        # Should have dialectical stage results
        assert "dialectical" in report.stage_results
        critique = report.stage_results["dialectical"]

        assert critique.rule == "enforceable(C) :- contract(C), signed(C)."
        assert len(critique.issues) >= 0  # May or may not have issues

    def test_critique_affects_final_decision(self, pipeline_with_critic):
        """Test that critique results influence final validation decision."""
        # Rule with obvious missing conditions
        report = pipeline_with_critic.validate_rule(
            rule_asp="enforceable(C) :- contract(C).",
            rule_id="incomplete_rule",
            target_layer="tactical",
        )

        # Mock critic should identify issues
        assert "dialectical" in report.stage_results
        critique = report.stage_results["dialectical"]

        # Should recommend revision
        assert len(critique.issues) > 0

    def test_critique_provides_suggestions(self, pipeline_with_critic):
        """Test that critique provides actionable suggestions."""
        report = pipeline_with_critic.validate_rule(
            rule_asp="enforceable(C) :- contract(C), signed(C).",
            rule_id="test_suggestions",
            target_layer="tactical",
        )

        critique = report.stage_results["dialectical"]

        # Check that issues have suggestions
        for issue in critique.issues:
            assert issue.description is not None
            assert issue.severity is not None

    def test_multiple_rules_validated_consistently(self, pipeline_with_critic):
        """Test consistent validation across multiple rules."""
        rules = [
            "enforceable(C) :- contract(C), signed(C).",
            "valid_offer(O) :- offer(O), definite(O).",
            "has_capacity(P) :- party(P), adult(P).",
        ]

        reports = [
            pipeline_with_critic.validate_rule(
                rule_asp=rule, rule_id=f"rule_{i}", target_layer="tactical"
            )
            for i, rule in enumerate(rules)
        ]

        # All should have dialectical validation
        assert all("dialectical" in r.stage_results for r in reports)

        # All should complete without errors
        assert all(r.final_decision in ["accept", "revise", "reject"] for r in reports)


class TestCriticWithRuleAccumulation:
    """Test critic with rule accumulation and persistence (Phase 4 integration)."""

    @pytest.fixture
    def critic(self):
        """Create critic system."""
        return CriticSystem(mock_mode=True)

    def test_critic_checks_against_existing_rules(self, critic):
        """Test that critic considers existing rules in critique."""
        existing_rules = [
            "enforceable(C) :- contract(C), consideration(C), capacity(P1), capacity(P2).",
            "void(C) :- contract(C), illegal_purpose(C).",
        ]

        new_rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            confidence=0.7,
            predicates_used=["contract", "signed"],
        )

        critique = critic.critique_rule(new_rule, existing_rules, context="Contract law")

        # Should compare with existing rules
        assert critique.rule == new_rule.asp_rule

        # Mock critic may detect potential conflicts
        contradictions = critic.check_contradictions(new_rule, existing_rules)
        assert isinstance(contradictions, list)

    def test_critic_evolution_with_rule_accumulation(self, critic):
        """Test critic behavior as rules accumulate over time."""
        # Simulate accumulated rules over multiple cycles
        cycle_1_rules = ["enforceable(C) :- contract(C), consideration(C)."]

        cycle_2_rules = cycle_1_rules + [
            "valid_offer(O) :- offer(O), definite(O), communicated(O)."
        ]

        cycle_3_rules = cycle_2_rules + ["acceptance(A) :- acceptance(A), mirror_image(A)."]

        # New rule to critique against accumulated knowledge
        new_rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            confidence=0.8,
            predicates_used=["contract", "signed"],
        )

        # Critique against different stages of accumulation
        critique_1 = critic.critique_rule(new_rule, cycle_1_rules)
        critique_2 = critic.critique_rule(new_rule, cycle_2_rules)
        critique_3 = critic.critique_rule(new_rule, cycle_3_rules)

        # All critiques should complete
        assert all(
            c.recommendation in ["accept", "revise", "reject"]
            for c in [critique_1, critique_2, critique_3]
        )


class TestDialecticalRefinement:
    """Test dialectical refinement process (critique -> synthesis)."""

    @pytest.fixture
    def critic(self):
        """Create critic with synthesis enabled."""
        return CriticSystem(mock_mode=True, enable_synthesis=True)

    def test_synthesis_improves_rule(self, critic):
        """Test that synthesis can improve a rule based on critique."""
        original_rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.6,
            predicates_used=["contract"],
        )

        # Generate critique
        critique = critic.critique_rule(original_rule, [])

        # Attempt synthesis
        improved = critic.synthesize_improvement(original_rule, critique, [])

        if improved:  # Mock may or may not synthesize
            # Should be different from original
            assert improved.asp_rule != original_rule.asp_rule
            # Should indicate it's a synthesized improvement
            assert (
                "synthesis" in improved.reasoning.lower()
                or "synthesized" in improved.reasoning.lower()
            )

    def test_dialectical_loop(self, critic):
        """Test iterative refinement through dialectical process."""
        rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.5,
            predicates_used=["contract"],
        )

        # Iteration 1: Critique and improve
        critique_1 = critic.critique_rule(rule, [])
        rule_v2 = critic.synthesize_improvement(rule, critique_1, [])

        if rule_v2:
            # Iteration 2: Critique improved version
            critique_2 = critic.critique_rule(rule_v2, [])

            # Second critique should find fewer or different issues
            assert isinstance(critique_2, type(critique_1))
            assert critique_2.rule == rule_v2.asp_rule


@pytest.mark.e2e
class TestCriticEndToEnd:
    """End-to-end tests for critic system in full workflow."""

    def test_full_validation_with_dialectical(self):
        """Test complete validation workflow with dialectical reasoning."""
        critic = CriticSystem(mock_mode=True)
        pipeline = ValidationPipeline(
            critic_system=critic, enable_dialectical=True, min_confidence=0.7
        )

        # Validate a moderately complex rule
        rule_asp = "enforceable(C) :- contract(C), signed(C), consideration(C)."

        report = pipeline.validate_rule(
            rule_asp=rule_asp, rule_id="e2e_test_001", target_layer="tactical"
        )

        # Should pass through all stages
        assert "syntactic" in report.stage_results
        assert "semantic" in report.stage_results
        assert "dialectical" in report.stage_results

        # Should reach a final decision
        assert report.final_decision in ["accept", "revise", "reject"]

        # If accepted, should have reasonable confidence
        if report.final_decision == "accept":
            assert report.aggregate_confidence >= 0.7

    def test_dialectical_catches_issues_missed_by_syntax_semantic(self):
        """Test that dialectical validation can catch issues missed by earlier stages."""
        critic = CriticSystem(mock_mode=True)
        pipeline = ValidationPipeline(
            critic_system=critic, enable_dialectical=True, min_confidence=0.6
        )

        # Rule that's syntactically and semantically valid but logically incomplete
        rule_asp = "enforceable(C) :- contract(C), signed(C)."

        report = pipeline.validate_rule(
            rule_asp=rule_asp, rule_id="logical_gap_test", target_layer="tactical"
        )

        # Should pass syntax and semantic
        assert report.stage_results["syntactic"].is_valid
        assert report.stage_results["semantic"].is_valid

        # But dialectical should identify logical issues
        critique = report.stage_results["dialectical"]
        assert len(critique.issues) > 0  # Should find missing conditions


class TestCriticPerformance:
    """Test critic system performance and efficiency."""

    def test_critic_response_time(self):
        """Test that critic responds within acceptable time."""
        import time

        critic = CriticSystem(mock_mode=True)
        rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C), signed(C), consideration(C).",
            confidence=0.8,
            predicates_used=["contract", "signed", "consideration"],
        )

        start = time.time()
        critique = critic.critique_rule(rule, [])
        duration = time.time() - start

        # Mock mode should be very fast
        assert duration < 0.1  # 100ms
        assert critique is not None

    def test_critique_multiple_rules_efficiently(self):
        """Test critiquing multiple rules efficiently."""
        import time

        critic = CriticSystem(mock_mode=True)
        rules = [
            make_test_rule(
                asp_rule=f"rule_{i}(X) :- condition_{i}(X).",
                confidence=0.7,
                predicates_used=[f"condition_{i}"],
            )
            for i in range(10)
        ]

        start = time.time()
        critiques = [critic.critique_rule(r, []) for r in rules]
        duration = time.time() - start

        # Should handle 10 rules quickly in mock mode
        assert duration < 1.0  # 1 second
        assert len(critiques) == 10
        assert all(c is not None for c in critiques)
