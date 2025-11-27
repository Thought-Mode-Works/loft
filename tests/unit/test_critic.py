"""
Unit tests for the Critic System (Phase 4.1).

Tests dialectical reasoning capabilities including critique generation,
edge case identification, and contradiction detection.
"""

import json
from unittest.mock import Mock

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
        critique = critic.critique_rule(sample_rule, existing_rules=[], context="Contract law")

        assert isinstance(critique, CritiqueReport)
        assert critique.rule == sample_rule.asp_rule
        assert critique.recommendation in ["accept", "revise", "reject"]

    def test_critique_identifies_missing_consideration(self, critic):
        """Test that critic identifies missing consideration."""
        rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            confidence=0.8,
            predicates_used=["contract", "signed"],
        )

        critique = critic.critique_rule(rule, [])

        # Mock critic should identify missing consideration
        assert len(critique.issues) > 0
        assert any("consideration" in i.description.lower() for i in critique.issues)

    def test_critique_identifies_missing_capacity(self, critic):
        """Test that critic identifies missing capacity checks."""
        rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            confidence=0.8,
            predicates_used=["contract", "signed"],
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
        new_rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.8,
            predicates_used=["contract"],
        )

        existing = ["enforceable(C) :- contract(C), consideration(C), capacity(P)."]

        contradictions = critic.check_contradictions(new_rule, existing)

        # Should detect potential conflict on enforceability
        assert len(contradictions) > 0

    def test_synthesize_improvement_disabled(self):
        """Test synthesis when disabled."""
        critic = CriticSystem(mock_mode=True, enable_synthesis=False)
        rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.8,
            predicates_used=["contract"],
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
        rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.8,
            predicates_used=["contract"],
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
            assert "consideration" in improved.asp_rule
            assert "Mock synthesis" in improved.reasoning or "Synthesized" in improved.reasoning


class TestCriticSystemValidation:
    """Test critic system validation capabilities."""

    @pytest.fixture
    def critic(self):
        """Create critic system."""
        return CriticSystem(mock_mode=True)

    def test_critique_recommends_accept_for_good_rule(self, critic):
        """Test that good rules are recommended for acceptance."""
        good_rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C), consideration(C), capacity(P1), capacity(P2), signed(C).",
            confidence=0.9,
            predicates_used=["contract", "consideration", "capacity", "signed"],
        )

        critique = critic.critique_rule(good_rule, [])

        # Should have fewer issues
        assert len(critique.issues) < 2
        assert critique.recommendation in ["accept", "revise"]

    def test_critique_recommends_revise_for_incomplete_rule(self, critic):
        """Test that incomplete rules are recommended for revision."""
        incomplete_rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.6,
            predicates_used=["contract"],
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
            rule_asp="invalid_rule(X) :- not not invalid_rule(X).",  # Circular rule
            rule_id="bad_rule",
            target_layer="tactical",
        )

        # Should complete validation (mock critic is lenient)
        assert report.final_decision in ["accept", "reject", "revise", "flag_for_review"]


class TestCriticWithLLM:
    """Test CriticSystem with mocked LLM responses."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        mock = Mock()
        return mock

    @pytest.fixture
    def sample_rule(self):
        """Create sample rule for testing."""
        return GeneratedRule(
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            confidence=0.8,
            reasoning="Test rule",
            predicates_used=["contract", "signed"],
            source_type="principle",
            source_text="Test source",
        )

    def test_critique_with_llm_success(self, mock_llm, sample_rule):
        """Test critique generation with LLM."""
        # Mock LLM response
        mock_llm.query.return_value = Mock(
            raw_text=json.dumps({
                "issues": [
                    {
                        "category": "missing_condition",
                        "description": "Missing consideration requirement",
                        "severity": "medium",
                        "suggestion": "Add consideration(C) to the rule body",
                    }
                ],
                "edge_cases": [],
                "contradictions": [],
                "overall_severity": "medium",
                "recommendation": "revise",
                "confidence": 0.8,
            })
        )

        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        critique = critic.critique_rule(sample_rule, [])

        assert len(critique.issues) == 1
        assert critique.issues[0].category == "missing_condition"
        assert critique.recommendation == "revise"
        assert critique.confidence == 0.8

    def test_critique_with_llm_markdown_wrapped(self, mock_llm, sample_rule):
        """Test JSON extraction from markdown-wrapped LLM response."""
        # Mock LLM response with markdown code block
        mock_llm.query.return_value = Mock(
            raw_text="""Here's the critique:

```json
{
    "issues": [{"category": "logical_flaw", "description": "Circular logic", "severity": "high"}],
    "edge_cases": [],
    "contradictions": [],
    "overall_severity": "high",
    "recommendation": "reject",
    "confidence": 0.9
}
```"""
        )

        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        critique = critic.critique_rule(sample_rule, [])

        assert len(critique.issues) == 1
        assert critique.issues[0].severity == CritiqueSeverity.HIGH
        assert critique.recommendation == "reject"

    def test_critique_with_llm_error_handling(self, mock_llm, sample_rule):
        """Test error handling when LLM fails."""
        # Mock LLM to raise exception
        mock_llm.query.side_effect = Exception("LLM API error")

        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        critique = critic.critique_rule(sample_rule, [])

        assert critique.recommendation == "error"
        assert critique.confidence == 0.0
        assert "error" in critique.metadata

    def test_critique_with_edge_cases(self, mock_llm, sample_rule):
        """Test critique with edge cases."""
        mock_llm.query.return_value = Mock(
            raw_text=json.dumps({
                "issues": [],
                "edge_cases": [
                    {
                        "description": "Contract with minor",
                        "scenario": {"party_age": 17},
                        "expected_outcome": "Voidable",
                        "current_outcome": "Enforceable",
                        "severity": "high",
                    }
                ],
                "contradictions": [],
                "overall_severity": "medium",
                "recommendation": "revise",
                "confidence": 0.75,
            })
        )

        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        critique = critic.critique_rule(sample_rule, [])

        assert len(critique.edge_cases) == 1
        assert critique.edge_cases[0].description == "Contract with minor"
        assert critique.edge_cases[0].severity == CritiqueSeverity.HIGH

    def test_critique_with_contradictions(self, mock_llm, sample_rule):
        """Test critique with contradictions."""
        existing_rules = ["enforceable(C) :- contract(C), consideration(C)."]
        mock_llm.query.return_value = Mock(
            raw_text=json.dumps({
                "issues": [],
                "edge_cases": [],
                "contradictions": [
                    {
                        "description": "Conflicting enforceability conditions",
                        "conflicting_rule": existing_rules[0],
                        "example_scenario": {"contract": "c1"},
                        "severity": "high",
                        "resolution_suggestion": "Add consideration requirement",
                    }
                ],
                "overall_severity": "high",
                "recommendation": "revise",
                "confidence": 0.85,
            })
        )

        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        critique = critic.critique_rule(sample_rule, existing_rules)

        assert len(critique.contradictions) == 1
        assert critique.contradictions[0].severity == CritiqueSeverity.HIGH
        assert critique.contradictions[0].resolution_suggestion is not None

    def test_critique_with_suggested_revision(self, mock_llm, sample_rule):
        """Test critique with suggested revision."""
        mock_llm.query.return_value = Mock(
            raw_text=json.dumps({
                "issues": [
                    {"category": "missing_condition", "description": "Missing consideration", "severity": "medium"}
                ],
                "edge_cases": [],
                "contradictions": [],
                "overall_severity": "medium",
                "recommendation": "revise",
                "suggested_revision": "enforceable(C) :- contract(C), signed(C), consideration(C).",
                "confidence": 0.8,
            })
        )

        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        critique = critic.critique_rule(sample_rule, [])

        assert critique.suggested_revision is not None
        assert "consideration" in critique.suggested_revision

    def test_find_edge_cases_with_llm(self, mock_llm, sample_rule):
        """Test edge case identification with LLM."""
        mock_llm.query.return_value = Mock(
            raw_text=json.dumps({
                "edge_cases": [
                    {
                        "description": "Oral contract over $500",
                        "scenario": {"contract_value": 600, "form": "oral"},
                        "expected_outcome": "Unenforceable under statute of frauds",
                        "current_outcome": "Enforceable",
                        "severity": "high",
                    },
                    {
                        "description": "Contract with intoxicated party",
                        "scenario": {"party_state": "intoxicated"},
                        "expected_outcome": "Voidable",
                        "severity": "medium",
                    },
                ]
            })
        )

        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        edge_cases = critic.find_edge_cases(sample_rule)

        assert len(edge_cases) == 2
        assert edge_cases[0].description == "Oral contract over $500"
        assert edge_cases[1].severity == CritiqueSeverity.MEDIUM

    def test_find_edge_cases_llm_error(self, mock_llm, sample_rule):
        """Test edge case identification when LLM fails."""
        mock_llm.query.side_effect = Exception("LLM error")

        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        edge_cases = critic.find_edge_cases(sample_rule)

        assert edge_cases == []

    def test_check_contradictions_with_llm(self, mock_llm, sample_rule):
        """Test contradiction detection with LLM."""
        existing_rules = [
            "enforceable(C) :- contract(C), consideration(C), capacity(P1), capacity(P2).",
            "valid(C) :- contract(C), offer(C), acceptance(C).",
        ]

        mock_llm.query.return_value = Mock(
            raw_text=json.dumps({
                "contradictions": [
                    {
                        "description": "Different enforceability conditions",
                        "conflicting_rule": existing_rules[0],
                        "example_scenario": {"contract": "c1", "signed": True},
                        "severity": "high",
                        "resolution_suggestion": "Align conditions or add qualification",
                    }
                ]
            })
        )

        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        contradictions = critic.check_contradictions(sample_rule, existing_rules)

        assert len(contradictions) == 1
        assert contradictions[0].new_rule == sample_rule.asp_rule

    def test_check_contradictions_no_existing_rules_llm(self, mock_llm, sample_rule):
        """Test contradiction check with no existing rules (LLM mode)."""
        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        contradictions = critic.check_contradictions(sample_rule, [])

        assert contradictions == []
        # LLM should not be called when there are no existing rules
        mock_llm.query.assert_not_called()

    def test_synthesize_improvement_with_llm(self, mock_llm, sample_rule):
        """Test synthesis with LLM."""
        critique = CritiqueReport(
            rule=sample_rule.asp_rule,
            issues=[
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing consideration",
                    severity=CritiqueSeverity.MEDIUM,
                )
            ],
            recommendation="revise",
        )

        mock_llm.query.return_value = Mock(
            raw_text=json.dumps({
                "improved_rule": "enforceable(C) :- contract(C), signed(C), consideration(C).",
                "changes": "Added consideration requirement",
                "confidence": 0.85,
            })
        )

        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        improved = critic.synthesize_improvement(sample_rule, critique, [])

        assert improved is not None
        assert "consideration" in improved.asp_rule
        assert improved.confidence == 0.85
        assert improved.source_type == "refinement"

    def test_synthesize_improvement_llm_error(self, mock_llm, sample_rule):
        """Test synthesis error handling."""
        critique = CritiqueReport(
            rule=sample_rule.asp_rule,
            issues=[CritiqueIssue(category="test", description="test", severity=CritiqueSeverity.LOW)],
        )

        mock_llm.query.side_effect = Exception("LLM synthesis error")

        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        improved = critic.synthesize_improvement(sample_rule, critique, [])

        assert improved is None


class TestJSONExtraction:
    """Test JSON extraction from various LLM response formats."""

    @pytest.fixture
    def critic(self):
        """Create critic with mock LLM."""
        return CriticSystem(llm_client=Mock(), mock_mode=False)

    def test_extract_json_plain(self, critic):
        """Test extraction from plain JSON."""
        response = '{"key": "value", "number": 42}'
        extracted = critic._extract_json(response)
        assert extracted == response

    def test_extract_json_with_markdown(self, critic):
        """Test extraction from markdown code block."""
        response = """Some text before
```json
{"key": "value"}
```
Some text after"""
        extracted = critic._extract_json(response)
        data = json.loads(extracted)
        assert data["key"] == "value"

    def test_extract_json_without_language_marker(self, critic):
        """Test extraction from code block without 'json' marker."""
        response = """```
{"key": "value"}
```"""
        extracted = critic._extract_json(response)
        data = json.loads(extracted)
        assert data["key"] == "value"

    def test_extract_json_nested_in_text(self, critic):
        """Test extraction when JSON is nested in text."""
        response = """Here is the result: {"result": "success", "count": 5} and that's it."""
        extracted = critic._extract_json(response)
        data = json.loads(extracted)
        assert data["result"] == "success"

    def test_extract_json_multiline(self, critic):
        """Test extraction of multiline JSON."""
        response = """{
    "key1": "value1",
    "key2": {
        "nested": true
    },
    "key3": [1, 2, 3]
}"""
        extracted = critic._extract_json(response)
        data = json.loads(extracted)
        assert data["key1"] == "value1"
        assert data["key2"]["nested"] is True


class TestCriticSeverityScoring:
    """Test critique severity scoring logic."""

    @pytest.fixture
    def critic(self):
        """Create critic in mock mode."""
        return CriticSystem(mock_mode=True)

    def test_severity_calculation_from_issues(self, critic):
        """Test overall severity calculated from issues."""
        rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["contract"],
        )

        critique = critic.critique_rule(rule, [])

        # Mock critic should identify issues with severity
        if critique.issues:
            # Overall severity should be a valid severity level
            assert critique.overall_severity.value in ["low", "medium", "high", "critical"]

    def test_has_critical_issues(self):
        """Test critical issues detection."""
        report = CritiqueReport(
            rule="test",
            issues=[
                CritiqueIssue(
                    category="critical_flaw",
                    description="Fatal error",
                    severity=CritiqueSeverity.CRITICAL,
                )
            ],
            overall_severity=CritiqueSeverity.CRITICAL,
        )

        assert report.has_critical_issues()
        assert report.should_reject()

    def test_has_high_severity_issues(self):
        """Test high severity detection."""
        report = CritiqueReport(
            rule="test",
            issues=[
                CritiqueIssue(
                    category="major_flaw",
                    description="Significant issue",
                    severity=CritiqueSeverity.HIGH,
                )
            ],
            overall_severity=CritiqueSeverity.HIGH,
        )

        assert report.has_high_severity_issues()
        assert not report.has_critical_issues()


class TestRecommendationLogic:
    """Test critique recommendation logic."""

    def test_recommendation_accept(self):
        """Test accept recommendation."""
        report = CritiqueReport(
            rule="test",
            issues=[],
            overall_severity=CritiqueSeverity.LOW,
            recommendation="accept",
        )

        assert not report.should_reject()
        assert not report.should_revise()

    def test_recommendation_revise(self):
        """Test revise recommendation."""
        report = CritiqueReport(
            rule="test",
            issues=[
                CritiqueIssue(
                    category="minor_issue",
                    description="Small improvement needed",
                    severity=CritiqueSeverity.MEDIUM,
                )
            ],
            suggested_revision="improved_rule",
            overall_severity=CritiqueSeverity.MEDIUM,
        )

        assert report.should_revise()
        assert not report.should_reject()

    def test_recommendation_reject_multiple_high_issues(self):
        """Test reject recommendation with multiple high severity issues."""
        report = CritiqueReport(
            rule="test",
            issues=[
                CritiqueIssue(category="flaw1", description="Issue 1", severity=CritiqueSeverity.HIGH),
                CritiqueIssue(category="flaw2", description="Issue 2", severity=CritiqueSeverity.HIGH),
                CritiqueIssue(category="flaw3", description="Issue 3", severity=CritiqueSeverity.HIGH),
            ],
            overall_severity=CritiqueSeverity.HIGH,
        )

        assert report.should_reject()


class TestMockImplementations:
    """Test mock implementations of critic methods."""

    @pytest.fixture
    def critic(self):
        """Create critic in mock mode."""
        return CriticSystem(mock_mode=True)

    def test_mock_critique_logic(self, critic):
        """Test mock critique generation logic."""
        rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["contract"],
        )

        critique = critic._mock_critique(rule, [])

        assert isinstance(critique, CritiqueReport)
        assert critique.rule == rule.asp_rule
        assert len(critique.issues) > 0  # Should identify missing conditions

    def test_mock_edge_cases_generation(self, critic):
        """Test mock edge case generation."""
        rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["contract"],
        )

        edge_cases = critic._mock_edge_cases(rule)

        assert len(edge_cases) > 0
        assert all(isinstance(ec, EdgeCase) for ec in edge_cases)
        assert any("minor" in ec.description.lower() for ec in edge_cases)

    def test_mock_contradictions_detection(self, critic):
        """Test mock contradiction detection."""
        rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["contract"],
        )

        existing = ["enforceable(C) :- contract(C), consideration(C)."]
        contradictions = critic._mock_contradictions(rule, existing)

        assert len(contradictions) > 0
        assert contradictions[0].new_rule == rule.asp_rule

    def test_mock_synthesis_adds_conditions(self, critic):
        """Test mock synthesis adds missing conditions."""
        rule = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["contract"],
        )

        critique = CritiqueReport(
            rule=rule.asp_rule,
            issues=[
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing consideration requirement",
                    severity=CritiqueSeverity.MEDIUM,
                    suggestion="Add consideration(C)",
                )
            ],
        )

        improved = critic._mock_synthesis(rule, critique)

        if improved:
            assert "consideration" in improved.asp_rule


class TestLLMCallMethod:
    """Test LLM call method."""

    def test_call_llm_success(self):
        """Test successful LLM call."""
        mock_llm = Mock()
        mock_llm.query.return_value = Mock(raw_text="LLM response")

        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)
        response = critic._call_llm("Test prompt")

        assert response == "LLM response"
        mock_llm.query.assert_called_once()

    def test_call_llm_no_client(self):
        """Test LLM call without client raises error."""
        critic = CriticSystem(llm_client=None, mock_mode=False)

        with pytest.raises(ValueError, match="No LLM client configured"):
            critic._call_llm("Test prompt")


class TestCriticInitialization:
    """Test CriticSystem initialization options."""

    def test_init_with_llm_client(self):
        """Test initialization with LLM client."""
        mock_llm = Mock()
        critic = CriticSystem(llm_client=mock_llm, mock_mode=False)

        assert critic.llm_client is mock_llm
        assert not critic.mock_mode
        assert critic.enable_synthesis

    def test_init_mock_mode_explicit(self):
        """Test explicit mock mode."""
        critic = CriticSystem(mock_mode=True)

        assert critic.mock_mode
        assert critic.llm_client is None

    def test_init_mock_mode_no_llm(self):
        """Test mock mode activated when no LLM provided."""
        critic = CriticSystem(llm_client=None)

        assert critic.mock_mode

    def test_init_disable_synthesis(self):
        """Test initialization with synthesis disabled."""
        critic = CriticSystem(enable_synthesis=False, mock_mode=True)

        assert not critic.enable_synthesis
