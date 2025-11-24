"""
Unit tests for the Critic System (Phase 4.1).

Tests dialectical reasoning capabilities including critique generation,
edge case identification, and contradiction detection.
"""

import pytest

from loft.dialectical.critic import CriticSystem
from loft.dialectical.critique_schemas import (
    Contradiction,
    CritiqueIssue,
    CritiqueReport,
    CritiqueSeverity,
    EdgeCase,
)
from loft.neural.rule_schemas import GeneratedRule


class TestCritiqueSchemas:
    """Test critique data structures."""

    def test_edge_case_creation(self):
        """Test EdgeCase creation."""
        edge_case = EdgeCase(
            description="Contract with minor party",
            scenario={"party_age": 17},
            expected_outcome="Contract voidable",
            severity=CritiqueSeverity.HIGH,
        )

        assert edge_case.description == "Contract with minor party"
        assert edge_case.scenario["party_age"] == 17
        assert edge_case.severity == CritiqueSeverity.HIGH

    def test_contradiction_creation(self):
        """Test Contradiction creation."""
        contradiction = Contradiction(
            description="Conflicting enforceability rules",
            new_rule="enforceable(C) :- contract(C).",
            conflicting_rule="enforceable(C) :- contract(C), consideration(C).",
            example_scenario={"contract": "c1"},
            severity=CritiqueSeverity.HIGH,
        )

        assert "Conflicting" in contradiction.description
        assert contradiction.severity == CritiqueSeverity.HIGH

    def test_critique_report_should_reject(self):
        """Test CritiqueReport rejection logic."""
        report = CritiqueReport(
            rule="test_rule",
            issues=[
                CritiqueIssue(
                    category="logical_flaw",
                    description="Critical flaw",
                    severity=CritiqueSeverity.CRITICAL,
                )
            ],
            overall_severity=CritiqueSeverity.CRITICAL,
        )

        assert report.has_critical_issues()
        assert report.should_reject()

    def test_critique_report_should_revise(self):
        """Test CritiqueReport revision logic."""
        report = CritiqueReport(
            rule="test_rule",
            issues=[
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing consideration",
                    severity=CritiqueSeverity.MEDIUM,
                )
            ],
            suggested_revision="enforceable(C) :- contract(C), consideration(C).",
            overall_severity=CritiqueSeverity.MEDIUM,
        )

        assert not report.should_reject()
        assert report.should_revise()


class TestCriticSystemMockMode:
    """Test CriticSystem in mock mode (no LLM required)."""

    @pytest.fixture
    def critic(self):
        """Create critic system in mock mode."""
        return CriticSystem(mock_mode=True)

    @pytest.fixture
    def sample_rule(self):
        """Create sample rule for testing."""
        return GeneratedRule(
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            confidence=0.8,
            reasoning="A contract is enforceable if it is both created and signed by parties",
            predicates_used=["contract", "signed"],
            source_type="principle",
            source_text="General contract law principle",
        )

    def test_critic_initialization(self, critic):
        """Test critic system initialization."""
        assert critic.mock_mode is True
        assert critic.enable_synthesis is True

    def test_critique_rule_basic(self, critic, sample_rule):
        """Test basic rule critique in mock mode."""
        critique = critic.critique_rule(
            sample_rule, existing_rules=[], context="Contract law"
        )

        assert isinstance(critique, CritiqueReport)
        assert critique.rule == sample_rule.asp_rule
        assert critique.recommendation in ["accept", "revise", "reject"]

    def test_critique_identifies_missing_consideration(self, critic):
        """Test that critic identifies missing consideration."""
        rule = GeneratedRule(
            rule_id="test",
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            confidence=0.8,
            strategy="conservative",
        )

        critique = critic.critique_rule(rule, [])

        # Mock critic should identify missing consideration
        assert len(critique.issues) > 0
        assert any("consideration" in i.description.lower() for i in critique.issues)

    def test_critique_identifies_missing_capacity(self, critic):
        """Test that critic identifies missing capacity checks."""
        rule = GeneratedRule(
            rule_id="test",
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            confidence=0.8,
            strategy="conservative",
        )

        critique = critic.critique_rule(rule, [])

        # Mock critic should identify missing capacity
        assert any("capacity" in i.description.lower() for i in critique.issues)

    def test_find_edge_cases(self, critic, sample_rule):
        """Test edge case identification."""
        edge_cases = critic.find_edge_cases(sample_rule)

        assert isinstance(edge_cases, list)
        assert len(edge_cases) > 0
        assert all(isinstance(ec, EdgeCase) for ec in edge_cases)

    def test_edge_cases_include_minor_scenario(self, critic, sample_rule):
        """Test that edge cases include minor party scenario."""
        edge_cases = critic.find_edge_cases(sample_rule)

        assert any("minor" in ec.description.lower() for ec in edge_cases)

    def test_check_contradictions_no_existing_rules(self, critic, sample_rule):
        """Test contradiction check with no existing rules."""
        contradictions = critic.check_contradictions(sample_rule, [])

        assert isinstance(contradictions, list)
        assert len(contradictions) == 0  # No contradictions possible

    def test_check_contradictions_with_similar_rule(self, critic):
        """Test contradiction detection with similar existing rule."""
        new_rule = GeneratedRule(
            rule_id="new",
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.8,
            strategy="permissive",
        )

        existing = ["enforceable(C) :- contract(C), consideration(C), capacity(P)."]

        contradictions = critic.check_contradictions(new_rule, existing)

        # Should detect potential conflict on enforceability
        assert len(contradictions) > 0

    def test_synthesize_improvement_disabled(self):
        """Test synthesis when disabled."""
        critic = CriticSystem(mock_mode=True, enable_synthesis=False)
        rule = GeneratedRule(
            rule_id="test",
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.8,
            strategy="conservative",
        )
        critique = CritiqueReport(
            rule=rule.asp_rule,
            issues=[
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing consideration",
                    severity=CritiqueSeverity.MEDIUM,
                )
            ],
        )

        improved = critic.synthesize_improvement(rule, critique, [])

        assert improved is None

    def test_synthesize_improvement_mock(self, critic):
        """Test synthesis in mock mode."""
        rule = GeneratedRule(
            rule_id="test",
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.8,
            strategy="conservative",
        )

        critique = CritiqueReport(
            rule=rule.asp_rule,
            issues=[
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing consideration check",
                    severity=CritiqueSeverity.MEDIUM,
                    suggestion="Add consideration(C)",
                )
            ],
            suggested_revision="enforceable(C) :- contract(C), consideration(C).",
        )

        improved = critic.synthesize_improvement(rule, critique, [])

        if improved:  # Mock may or may not synthesize
            assert isinstance(improved, GeneratedRule)
            assert improved.rule_id == "test_improved"
            assert "consideration" in improved.asp_rule


class TestCriticSystemValidation:
    """Test critic system validation capabilities."""

    @pytest.fixture
    def critic(self):
        """Create critic system."""
        return CriticSystem(mock_mode=True)

    def test_critique_recommends_accept_for_good_rule(self, critic):
        """Test that good rules are recommended for acceptance."""
        good_rule = GeneratedRule(
            rule_id="good",
            asp_rule="enforceable(C) :- contract(C), consideration(C), capacity(P1), capacity(P2), signed(C).",
            confidence=0.9,
            strategy="conservative",
        )

        critique = critic.critique_rule(good_rule, [])

        # Should have fewer issues
        assert len(critique.issues) < 2
        assert critique.recommendation in ["accept", "revise"]

    def test_critique_recommends_revise_for_incomplete_rule(self, critic):
        """Test that incomplete rules are recommended for revision."""
        incomplete_rule = GeneratedRule(
            rule_id="incomplete",
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.6,
            strategy="permissive",
        )

        critique = critic.critique_rule(incomplete_rule, [])

        # Should identify missing conditions
        assert len(critique.issues) >= 1
        assert critique.recommendation == "revise"


@pytest.mark.integration
class TestCriticIntegration:
    """Integration tests requiring full system setup."""

    def test_critic_with_validation_pipeline(self):
        """Test critic integration with validation pipeline."""
        from loft.validation.validation_pipeline import ValidationPipeline

        critic = CriticSystem(mock_mode=True)
        pipeline = ValidationPipeline(
            critic_system=critic, enable_dialectical=True, min_confidence=0.7
        )

        # Validate a rule with dialectical validation
        report = pipeline.validate_rule(
            rule_asp="enforceable(C) :- contract(C), signed(C).",
            rule_id="test_001",
            target_layer="tactical",
        )

        # Should have dialectical stage
        assert "dialectical" in report.stage_results
        assert isinstance(report.stage_results["dialectical"], CritiqueReport)

    def test_critic_rejects_flawed_rule(self):
        """Test that critic can reject critically flawed rules."""
        from loft.validation.validation_pipeline import ValidationPipeline

        critic = CriticSystem(mock_mode=True)
        pipeline = ValidationPipeline(
            critic_system=critic, enable_dialectical=True, min_confidence=0.7
        )

        # For this test, we'll manually create a rule that should be rejected
        # In mock mode, the critic is lenient, so this is more of a structure test
        report = pipeline.validate_rule(
            rule_asp="invalid_rule :- .",  # Malformed rule
            rule_id="bad_rule",
            target_layer="tactical",
        )

        # Should be rejected at syntax stage
        assert report.final_decision == "reject"
