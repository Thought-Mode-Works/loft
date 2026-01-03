"""
Unit tests for rule refiner.

Issue #278: Rule Refinement and Feedback Loop
"""

from unittest.mock import Mock

import pytest

from loft.feedback.refiner import RuleRefiner
from loft.feedback.schemas import (
    PerformanceIssue,
    RuleFeedbackEntry,
    RuleOutcome,
    RulePerformanceMetrics,
)


@pytest.fixture
def mock_llm():
    """Create mock LLM interface."""
    llm = Mock()
    llm.generate = Mock()
    return llm


@pytest.fixture
def refiner(mock_llm):
    """Create rule refiner with mock LLM."""
    return RuleRefiner(mock_llm)


@pytest.fixture
def sample_metrics():
    """Create sample underperforming metrics."""
    metrics = RulePerformanceMetrics(rule_id="underperforming_rule")
    for i in range(10):
        outcome = RuleOutcome.CORRECT if i < 4 else RuleOutcome.INCORRECT
        actual = "yes" if i < 4 else "no"
        metrics.update_from_entry(
            RuleFeedbackEntry(
                rule_id="underperforming_rule",
                question=f"Contract question {i}",
                expected_answer="yes",
                actual_answer=actual,
                outcome=outcome,
                rule_used=True,
                confidence=0.6,
                domain="contracts",
            )
        )
    return metrics


@pytest.fixture
def sample_issues():
    """Create sample performance issues."""
    return [
        PerformanceIssue(
            issue_type="low_accuracy",
            severity=0.6,
            description="Rule has 40% accuracy",
            example_failures=["Failed on question 1", "Failed on question 2"],
            suggested_action="Strengthen rule conditions",
        )
    ]


class TestRuleRefiner:
    """Test RuleRefiner."""

    def test_refiner_initialization(self, mock_llm):
        """Test creating rule refiner."""
        refiner = RuleRefiner(mock_llm)
        assert refiner.llm == mock_llm

    def test_propose_refinement_success(
        self, refiner, mock_llm, sample_metrics, sample_issues
    ):
        """Test successfully generating a refinement proposal."""
        # Mock LLM response
        mock_llm.generate.return_value = """
REFINEMENT_TYPE: strengthen

REFINED_RULE:
```asp
valid_contract(C) :- offer(C), acceptance(C), consideration(C), capacity(C).
```

RATIONALE:
Added capacity check to prevent contracts with incapacitated parties.

EXPECTED_IMPACT:
Should reduce false positives where parties lack capacity.

TEST_CASES:
1. Minor entering contract should fail
2. Intoxicated party contract should fail
3. Valid adult contract should pass
"""

        rule_text = "valid_contract(C) :- offer(C), acceptance(C), consideration(C)."
        rule_id = "underperforming_rule"

        proposal = refiner.propose_refinement(
            rule_text, rule_id, sample_metrics, sample_issues
        )

        assert proposal is not None
        assert proposal.original_rule_id == rule_id
        assert proposal.refinement_type == "strengthen"
        assert "capacity" in proposal.proposed_asp_rule
        assert "capacity" in proposal.rationale.lower()
        assert len(proposal.test_cases) == 3
        assert proposal.confidence > 0.5

    def test_build_refinement_prompt(self, refiner, sample_metrics, sample_issues):
        """Test building refinement prompt."""
        rule_text = "valid_contract(C) :- offer(C), acceptance(C)."

        prompt = refiner._build_refinement_prompt(
            rule_text, sample_metrics, sample_issues
        )

        assert "Rule Refinement Task" in prompt
        assert rule_text in prompt
        assert "Performance Metrics" in prompt
        assert f"{sample_metrics.accuracy_when_used:.1%}" in prompt
        assert "Identified Issues" in prompt
        assert sample_issues[0].description in prompt

    def test_parse_refinement_response(self, refiner, sample_issues):
        """Test parsing LLM response."""
        response = """
REFINEMENT_TYPE: generalize

REFINED_RULE:
```asp
enforceable(C) :- contract(C), not void(C).
```

RATIONALE:
Simplified the rule to be more general and cover edge cases.

EXPECTED_IMPACT:
Will improve accuracy on ambiguous contract validity questions.

TEST_CASES:
1. Valid contract should be enforceable
2. Void contract should not be enforceable
"""

        rule_id = "test_rule"
        original_rule = "enforceable(C) :- written(C), signed(C)."

        proposal = refiner._parse_refinement_response(
            response, rule_id, original_rule, sample_issues
        )

        assert proposal.refinement_type == "generalize"
        assert "enforceable(C)" in proposal.proposed_asp_rule
        assert "void" in proposal.proposed_asp_rule
        assert "Simplified" in proposal.rationale
        assert "improve accuracy" in proposal.expected_impact
        assert len(proposal.test_cases) == 2

    def test_extract_field(self, refiner):
        """Test extracting fields from response."""
        text = """
REFINEMENT_TYPE: strengthen

RATIONALE:
This is the rationale section.
It has multiple lines.

EXPECTED_IMPACT:
This is the impact section.
"""

        refinement_type = refiner._extract_field(text, "REFINEMENT_TYPE:")
        assert refinement_type == "strengthen"

        rationale = refiner._extract_field(text, "RATIONALE:")
        assert "This is the rationale section" in rationale
        assert "multiple lines" in rationale

        impact = refiner._extract_field(text, "EXPECTED_IMPACT:")
        assert "This is the impact section" in impact

    def test_extract_code_block(self, refiner):
        """Test extracting code from markdown code block."""
        text = """
REFINED_RULE:
```asp
valid(C) :- offer(C), acceptance(C).
```

Some other text.
"""

        code = refiner._extract_code_block(text, "REFINED_RULE:")
        assert "valid(C)" in code
        assert ":-" in code
        assert "```" not in code  # Should be stripped

    def test_extract_test_cases(self, refiner):
        """Test extracting test cases."""
        text = """
TEST_CASES:
1. First test case
2. Second test case
3. Third test case

Some other text.
"""

        test_cases = refiner._extract_test_cases(text)
        assert len(test_cases) == 3
        assert "First test case" in test_cases
        assert "Second test case" in test_cases

    def test_extract_test_cases_with_bullets(self, refiner):
        """Test extracting test cases with bullet points."""
        text = """
TEST_CASES:
- Test with bullet
- Another bullet test
"""

        test_cases = refiner._extract_test_cases(text)
        assert len(test_cases) == 2
        assert "Test with bullet" in test_cases

    def test_estimate_confidence(self, refiner):
        """Test confidence estimation."""
        # Well-structured response
        good_response = """
REFINEMENT_TYPE: strengthen
REFINED_RULE:
```asp
new_rule(X) :- different(X).
```
RATIONALE:
This is a detailed rationale explaining why the refinement will help improve accuracy.
TEST_CASES:
1. Test case 1
"""

        refined_rule = "new_rule(X) :- different(X)."
        original_rule = "old_rule(X) :- same(X)."

        confidence = refiner._estimate_confidence(
            good_response, refined_rule, original_rule
        )

        # Should have high confidence (all markers present, rule changed, etc.)
        assert confidence >= 0.8

        # Minimal response
        bad_response = "Some text without proper structure"
        confidence_low = refiner._estimate_confidence(
            bad_response, original_rule, original_rule
        )

        # Should have low confidence (no markers, rule unchanged)
        assert confidence_low <= 0.6

    def test_handle_llm_error(self, refiner, mock_llm, sample_metrics, sample_issues):
        """Test handling LLM errors gracefully."""
        # Make LLM raise an error
        mock_llm.generate.side_effect = Exception("LLM API error")

        rule_text = "test_rule(X) :- condition(X)."
        rule_id = "test_rule"

        proposal = refiner.propose_refinement(
            rule_text, rule_id, sample_metrics, sample_issues
        )

        # Should return None on error
        assert proposal is None

    def test_fallback_values_on_parse_failure(self, refiner, sample_issues):
        """Test fallback values when parsing fails."""
        # Response missing most fields
        incomplete_response = "Just some random text"

        proposal = refiner._parse_refinement_response(
            incomplete_response,
            "test_rule",
            "original_rule(X) :- cond(X).",
            sample_issues,
        )

        # Should have fallback values
        assert proposal.refinement_type == "general_refinement"
        assert proposal.proposed_asp_rule == "original_rule(X) :- cond(X)."
        assert "LLM-proposed refinement" in proposal.rationale
        assert "Improved accuracy" in proposal.expected_impact

    def test_proposal_includes_issues(
        self, refiner, mock_llm, sample_metrics, sample_issues
    ):
        """Test that proposal includes the issues it addresses."""
        mock_llm.generate.return_value = """
REFINEMENT_TYPE: strengthen
REFINED_RULE:
```asp
new_rule(X) :- cond1(X), cond2(X).
```
RATIONALE:
Added conditions to improve accuracy.
EXPECTED_IMPACT:
Better performance on edge cases.
TEST_CASES:
1. Test edge case
"""

        proposal = refiner.propose_refinement(
            "old_rule(X) :- cond1(X).",
            "test_rule",
            sample_metrics,
            sample_issues,
        )

        assert proposal.issues_addressed == sample_issues

    def test_domain_metrics_in_prompt(self, refiner):
        """Test that domain metrics are included in prompt."""
        metrics = RulePerformanceMetrics(rule_id="test_rule")
        # Add domain-specific feedback
        for i in range(3):
            metrics.update_from_entry(
                RuleFeedbackEntry(
                    rule_id="test_rule",
                    question=f"Q{i}",
                    expected_answer="yes",
                    actual_answer="yes" if i < 2 else "no",
                    outcome=RuleOutcome.CORRECT if i < 2 else RuleOutcome.INCORRECT,
                    rule_used=True,
                    confidence=0.8,
                    domain="contracts",
                )
            )

        prompt = refiner._build_refinement_prompt(
            "test_rule(X) :- cond(X).", metrics, []
        )

        assert "Performance by Domain" in prompt
        assert "contracts" in prompt

    def test_example_failures_in_prompt(self, refiner, sample_metrics):
        """Test that example failures are included in prompt."""
        issues = [
            PerformanceIssue(
                issue_type="low_accuracy",
                severity=0.8,
                description="Rule fails frequently",
                example_failures=["Failure case 1", "Failure case 2"],
                suggested_action="Review conditions",
            )
        ]

        prompt = refiner._build_refinement_prompt(
            "test_rule(X) :- cond(X).", sample_metrics, issues
        )

        assert "Example failures:" in prompt
        assert "Failure case 1" in prompt
        assert "Failure case 2" in prompt
