"""
Unit tests for critique prompt generation functions.

Tests prompt generation for various critique scenarios including
rule critique, edge case identification, contradiction detection,
and synthesis prompting.
"""

from loft.dialectical.critique_prompts import (
    get_contradiction_check_prompt,
    get_critique_prompt,
    get_edge_case_prompt,
    get_synthesis_prompt,
)


class TestGetCritiquePrompt:
    """Test get_critique_prompt function."""

    def test_critique_prompt_basic(self):
        """Test basic critique prompt generation."""
        rule = "enforceable(C) :- contract(C), signed(C)."
        existing_rules = ["valid(C) :- contract(C)."]
        context = "Contract law domain"

        prompt = get_critique_prompt(rule, existing_rules, context)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert rule in prompt
        assert context in prompt

    def test_critique_prompt_includes_rule(self):
        """Test prompt includes the rule to critique."""
        rule = "enforceable(C) :- contract(C), consideration(C)."
        prompt = get_critique_prompt(rule, [], "")

        assert rule in prompt
        assert "PROPOSED RULE" in prompt

    def test_critique_prompt_includes_existing_rules(self):
        """Test prompt includes existing rules."""
        rule = "test_rule(X) :- condition(X)."
        existing_rules = [
            "existing_rule_1(X) :- foo(X).",
            "existing_rule_2(Y) :- bar(Y).",
        ]

        prompt = get_critique_prompt(rule, existing_rules, "")

        assert "existing_rule_1" in prompt
        assert "existing_rule_2" in prompt
        assert "EXISTING RULES" in prompt

    def test_critique_prompt_empty_existing_rules(self):
        """Test prompt with empty existing rules."""
        rule = "test_rule(X) :- condition(X)."
        prompt = get_critique_prompt(rule, [], "")

        assert "no existing rules" in prompt.lower()
        assert rule in prompt

    def test_critique_prompt_includes_context(self):
        """Test prompt includes domain context."""
        rule = "enforceable(C) :- contract(C)."
        context = "Specific contract law context with UCC rules"

        prompt = get_critique_prompt(rule, [], context)

        assert context in prompt
        assert "CONTEXT" in prompt

    def test_critique_prompt_default_context(self):
        """Test prompt uses default context when not provided."""
        rule = "enforceable(C) :- contract(C)."
        prompt = get_critique_prompt(rule, [], "")

        assert "Contract law" in prompt or "contract law" in prompt.lower()

    def test_critique_prompt_requests_json_format(self):
        """Test prompt requests JSON format response."""
        rule = "test_rule(X) :- condition(X)."
        prompt = get_critique_prompt(rule, [], "")

        assert "JSON" in prompt or "json" in prompt.lower()
        assert "issues" in prompt
        assert "edge_cases" in prompt
        assert "contradictions" in prompt

    def test_critique_prompt_specifies_categories(self):
        """Test prompt specifies issue categories."""
        rule = "test_rule(X) :- condition(X)."
        prompt = get_critique_prompt(rule, [], "")

        assert "logical_flaw" in prompt or "LOGICAL FLAW" in prompt
        assert "missing_condition" in prompt or "MISSING CONDITION" in prompt
        assert "contradiction" in prompt or "CONTRADICTION" in prompt

    def test_critique_prompt_specifies_severity_levels(self):
        """Test prompt specifies severity levels."""
        rule = "test_rule(X) :- condition(X)."
        prompt = get_critique_prompt(rule, [], "")

        assert "low" in prompt.lower()
        assert "medium" in prompt.lower()
        assert "high" in prompt.lower()
        assert "critical" in prompt.lower()

    def test_critique_prompt_requests_recommendation(self):
        """Test prompt requests recommendation."""
        rule = "test_rule(X) :- condition(X)."
        prompt = get_critique_prompt(rule, [], "")

        assert "recommendation" in prompt.lower()
        assert "accept" in prompt.lower()
        assert "revise" in prompt.lower()
        assert "reject" in prompt.lower()

    def test_critique_prompt_requests_confidence(self):
        """Test prompt requests confidence score."""
        rule = "test_rule(X) :- condition(X)."
        prompt = get_critique_prompt(rule, [], "")

        assert "confidence" in prompt.lower()


class TestGetEdgeCasePrompt:
    """Test get_edge_case_prompt function."""

    def test_edge_case_prompt_basic(self):
        """Test basic edge case prompt generation."""
        rule = "enforceable(C) :- contract(C), signed(C)."
        context = "Contract law"

        prompt = get_edge_case_prompt(rule, context)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert rule in prompt
        assert context in prompt

    def test_edge_case_prompt_includes_rule(self):
        """Test prompt includes the rule to analyze."""
        rule = "enforceable(C) :- contract(C), consideration(C)."
        prompt = get_edge_case_prompt(rule, "")

        assert rule in prompt
        assert "RULE TO ANALYZE" in prompt

    def test_edge_case_prompt_includes_context(self):
        """Test prompt includes context."""
        rule = "test_rule(X) :- condition(X)."
        context = "Specific legal domain context"

        prompt = get_edge_case_prompt(rule, context)

        assert context in prompt
        assert "CONTEXT" in prompt

    def test_edge_case_prompt_default_context(self):
        """Test prompt uses default context."""
        rule = "test_rule(X) :- condition(X)."
        prompt = get_edge_case_prompt(rule, "")

        assert "Contract law" in prompt or "contract law" in prompt.lower()

    def test_edge_case_prompt_requests_json_format(self):
        """Test prompt requests JSON format."""
        rule = "test_rule(X) :- condition(X)."
        prompt = get_edge_case_prompt(rule, "")

        assert "JSON" in prompt or "json" in prompt.lower()
        assert "edge_cases" in prompt

    def test_edge_case_prompt_specifies_considerations(self):
        """Test prompt specifies what to consider."""
        rule = "test_rule(X) :- condition(X)."
        prompt = get_edge_case_prompt(rule, "")

        # Should mention various types of edge cases to consider
        assert "BOUNDARY" in prompt or "boundary" in prompt.lower()
        assert "NULL" in prompt or "MISSING" in prompt or "null" in prompt.lower()

    def test_edge_case_prompt_requests_scenario_details(self):
        """Test prompt requests scenario details."""
        rule = "test_rule(X) :- condition(X)."
        prompt = get_edge_case_prompt(rule, "")

        assert "scenario" in prompt.lower()
        assert "description" in prompt.lower()
        assert "expected_outcome" in prompt.lower()

    def test_edge_case_prompt_requests_severity(self):
        """Test prompt requests severity for edge cases."""
        rule = "test_rule(X) :- condition(X)."
        prompt = get_edge_case_prompt(rule, "")

        assert "severity" in prompt.lower()


class TestGetContradictionCheckPrompt:
    """Test get_contradiction_check_prompt function."""

    def test_contradiction_prompt_basic(self):
        """Test basic contradiction check prompt."""
        new_rule = "enforceable(C) :- contract(C), signed(C)."
        existing_rules = ["enforceable(C) :- contract(C), consideration(C)."]

        prompt = get_contradiction_check_prompt(new_rule, existing_rules)

        assert prompt is not None
        assert isinstance(prompt, str)
        assert new_rule in prompt

    def test_contradiction_prompt_no_existing_rules(self):
        """Test contradiction check with no existing rules returns None."""
        new_rule = "enforceable(C) :- contract(C)."
        prompt = get_contradiction_check_prompt(new_rule, [])

        assert prompt is None

    def test_contradiction_prompt_includes_new_rule(self):
        """Test prompt includes new rule."""
        new_rule = "new_rule(X) :- condition(X)."
        existing_rules = ["old_rule(Y) :- other(Y)."]

        prompt = get_contradiction_check_prompt(new_rule, existing_rules)

        assert new_rule in prompt
        assert "NEW RULE" in prompt

    def test_contradiction_prompt_includes_existing_rules(self):
        """Test prompt includes all existing rules."""
        new_rule = "new_rule(X) :- condition(X)."
        existing_rules = [
            "existing_1(X) :- foo(X).",
            "existing_2(Y) :- bar(Y).",
            "existing_3(Z) :- baz(Z).",
        ]

        prompt = get_contradiction_check_prompt(new_rule, existing_rules)

        for rule in existing_rules:
            assert rule in prompt
        assert "EXISTING RULES" in prompt

    def test_contradiction_prompt_requests_json_format(self):
        """Test prompt requests JSON format."""
        new_rule = "test(X) :- condition(X)."
        existing_rules = ["other(Y) :- stuff(Y)."]

        prompt = get_contradiction_check_prompt(new_rule, existing_rules)

        assert "JSON" in prompt or "json" in prompt.lower()
        assert "contradictions" in prompt

    def test_contradiction_prompt_defines_contradiction(self):
        """Test prompt defines what constitutes a contradiction."""
        new_rule = "test(X) :- condition(X)."
        existing_rules = ["other(Y) :- stuff(Y)."]

        prompt = get_contradiction_check_prompt(new_rule, existing_rules)

        # Should explain what contradictions are
        assert "contradiction" in prompt.lower()
        assert "conflict" in prompt.lower()

    def test_contradiction_prompt_requests_example_scenario(self):
        """Test prompt requests example scenarios."""
        new_rule = "test(X) :- condition(X)."
        existing_rules = ["other(Y) :- stuff(Y)."]

        prompt = get_contradiction_check_prompt(new_rule, existing_rules)

        assert "example_scenario" in prompt or "scenario" in prompt.lower()

    def test_contradiction_prompt_requests_resolution(self):
        """Test prompt requests resolution suggestions."""
        new_rule = "test(X) :- condition(X)."
        existing_rules = ["other(Y) :- stuff(Y)."]

        prompt = get_contradiction_check_prompt(new_rule, existing_rules)

        assert "resolution" in prompt.lower() or "resolve" in prompt.lower()

    def test_contradiction_prompt_requests_severity(self):
        """Test prompt requests severity assessment."""
        new_rule = "test(X) :- condition(X)."
        existing_rules = ["other(Y) :- stuff(Y)."]

        prompt = get_contradiction_check_prompt(new_rule, existing_rules)

        assert "severity" in prompt.lower()


class TestGetSynthesisPrompt:
    """Test get_synthesis_prompt function."""

    def test_synthesis_prompt_basic(self):
        """Test basic synthesis prompt generation."""
        rule = "enforceable(C) :- contract(C), signed(C)."
        critique = '{"issues": [{"category": "missing_condition", "description": "Missing consideration"}]}'
        existing_rules = ["valid(C) :- contract(C)."]

        prompt = get_synthesis_prompt(rule, critique, existing_rules)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert rule in prompt

    def test_synthesis_prompt_includes_original_rule(self):
        """Test prompt includes original rule."""
        rule = "original_rule(X) :- condition(X)."
        critique = '{"issues": []}'

        prompt = get_synthesis_prompt(rule, critique, [])

        assert rule in prompt
        assert "ORIGINAL RULE" in prompt

    def test_synthesis_prompt_includes_critique(self):
        """Test prompt includes critique."""
        rule = "test_rule(X) :- condition(X)."
        critique = '{"issues": [{"description": "Missing important condition"}]}'

        prompt = get_synthesis_prompt(rule, critique, [])

        assert "Missing important condition" in prompt or critique in prompt
        assert "CRITIQUE" in prompt

    def test_synthesis_prompt_includes_existing_rules(self):
        """Test prompt includes existing rules for context."""
        rule = "test_rule(X) :- condition(X)."
        critique = '{"issues": []}'
        existing_rules = [
            "existing_1(X) :- foo(X).",
            "existing_2(Y) :- bar(Y).",
        ]

        prompt = get_synthesis_prompt(rule, critique, existing_rules)

        assert "existing_1" in prompt
        assert "existing_2" in prompt

    def test_synthesis_prompt_empty_existing_rules(self):
        """Test prompt with empty existing rules."""
        rule = "test_rule(X) :- condition(X)."
        critique = '{"issues": []}'

        prompt = get_synthesis_prompt(rule, critique, [])

        # Should still generate valid prompt
        assert rule in prompt
        assert len(prompt) > 0

    def test_synthesis_prompt_requests_json_format(self):
        """Test prompt requests JSON format response."""
        rule = "test_rule(X) :- condition(X)."
        critique = '{"issues": []}'

        prompt = get_synthesis_prompt(rule, critique, [])

        assert "JSON" in prompt or "json" in prompt.lower()
        assert "improved_rule" in prompt

    def test_synthesis_prompt_specifies_requirements(self):
        """Test prompt specifies synthesis requirements."""
        rule = "test_rule(X) :- condition(X)."
        critique = '{"issues": []}'

        prompt = get_synthesis_prompt(rule, critique, [])

        # Should specify what the synthesis should do
        assert "intent" in prompt.lower() or "maintain" in prompt.lower()
        assert "flaw" in prompt.lower() or "fix" in prompt.lower() or "address" in prompt.lower()

    def test_synthesis_prompt_requests_explanation(self):
        """Test prompt requests explanation of changes."""
        rule = "test_rule(X) :- condition(X)."
        critique = '{"issues": []}'

        prompt = get_synthesis_prompt(rule, critique, [])

        assert "changes" in prompt.lower() or "explanation" in prompt.lower()

    def test_synthesis_prompt_requests_confidence(self):
        """Test prompt requests confidence score."""
        rule = "test_rule(X) :- condition(X)."
        critique = '{"issues": []}'

        prompt = get_synthesis_prompt(rule, critique, [])

        assert "confidence" in prompt.lower()

    def test_synthesis_prompt_requires_asp_syntax(self):
        """Test prompt requires ASP syntax in output."""
        rule = "test_rule(X) :- condition(X)."
        critique = '{"issues": []}'

        prompt = get_synthesis_prompt(rule, critique, [])

        assert "ASP" in prompt or "asp" in prompt.lower()


class TestPromptTemplateValidation:
    """Test that prompt templates contain all required sections."""

    def test_critique_prompt_has_required_sections(self):
        """Test critique prompt has all required sections."""
        rule = "test(X) :- condition(X)."
        prompt = get_critique_prompt(rule, [], "")

        # Required sections
        assert "PROPOSED RULE" in prompt or "proposed rule" in prompt.lower()
        assert "TASK" in prompt or "task" in prompt.lower()
        # Should request structured output
        assert "{" in prompt  # JSON structure example

    def test_edge_case_prompt_has_required_sections(self):
        """Test edge case prompt has required sections."""
        rule = "test(X) :- condition(X)."
        prompt = get_edge_case_prompt(rule, "")

        assert "RULE" in prompt
        assert "TASK" in prompt or "task" in prompt.lower()
        assert "{" in prompt  # JSON structure example

    def test_contradiction_prompt_has_required_sections(self):
        """Test contradiction prompt has required sections."""
        new_rule = "test(X) :- condition(X)."
        existing = ["other(Y) :- stuff(Y)."]
        prompt = get_contradiction_check_prompt(new_rule, existing)

        assert "NEW RULE" in prompt
        assert "EXISTING RULES" in prompt or "existing rules" in prompt.lower()
        assert "TASK" in prompt or "task" in prompt.lower()

    def test_synthesis_prompt_has_required_sections(self):
        """Test synthesis prompt has required sections."""
        rule = "test(X) :- condition(X)."
        critique = '{"issues": []}'
        prompt = get_synthesis_prompt(rule, critique, [])

        assert "ORIGINAL RULE" in prompt or "original rule" in prompt.lower()
        assert "CRITIQUE" in prompt or "critique" in prompt.lower()
        assert "TASK" in prompt or "task" in prompt.lower()


class TestVariableSubstitution:
    """Test that variables are properly substituted in prompts."""

    def test_critique_prompt_substitutes_rule(self):
        """Test rule is properly substituted in critique prompt."""
        rule = "unique_test_rule_12345(X) :- special_condition(X)."
        prompt = get_critique_prompt(rule, [], "")

        assert "unique_test_rule_12345" in prompt
        assert "special_condition" in prompt

    def test_critique_prompt_substitutes_multiple_existing_rules(self):
        """Test multiple existing rules are substituted."""
        rule = "test(X) :- condition(X)."
        existing = [
            "first_rule(X) :- a(X).",
            "second_rule(Y) :- b(Y).",
            "third_rule(Z) :- c(Z).",
        ]
        prompt = get_critique_prompt(rule, existing, "")

        assert "first_rule" in prompt
        assert "second_rule" in prompt
        assert "third_rule" in prompt

    def test_critique_prompt_substitutes_context(self):
        """Test context is properly substituted."""
        rule = "test(X) :- condition(X)."
        context = "Very specific and unique context 98765"
        prompt = get_critique_prompt(rule, [], context)

        assert "Very specific and unique context 98765" in prompt

    def test_edge_case_prompt_substitutes_rule(self):
        """Test rule substitution in edge case prompt."""
        rule = "unique_edge_case_rule(X) :- special(X)."
        prompt = get_edge_case_prompt(rule, "")

        assert "unique_edge_case_rule" in prompt
        assert "special" in prompt

    def test_contradiction_prompt_substitutes_rules(self):
        """Test rule substitution in contradiction prompt."""
        new_rule = "new_unique_rule(X) :- new_condition(X)."
        existing = ["existing_unique_rule(Y) :- existing_condition(Y)."]
        prompt = get_contradiction_check_prompt(new_rule, existing)

        assert "new_unique_rule" in prompt
        assert "existing_unique_rule" in prompt

    def test_synthesis_prompt_substitutes_rule_and_critique(self):
        """Test rule and critique substitution in synthesis prompt."""
        rule = "synthesis_test_rule(X) :- condition(X)."
        critique = '{"issues": [{"description": "Unique issue 54321"}]}'
        prompt = get_synthesis_prompt(rule, critique, [])

        assert "synthesis_test_rule" in prompt
        # Critique should be included somehow
        assert "Unique issue 54321" in prompt or critique in prompt


class TestPromptReturnTypes:
    """Test that prompt functions return correct types."""

    def test_critique_prompt_returns_string(self):
        """Test get_critique_prompt returns string."""
        result = get_critique_prompt("test(X).", [], "")
        assert isinstance(result, str)

    def test_edge_case_prompt_returns_string(self):
        """Test get_edge_case_prompt returns string."""
        result = get_edge_case_prompt("test(X).", "")
        assert isinstance(result, str)

    def test_contradiction_prompt_returns_string_or_none(self):
        """Test get_contradiction_check_prompt returns string or None."""
        result = get_contradiction_check_prompt("test(X).", ["other(Y)."])
        assert isinstance(result, str)

        result_empty = get_contradiction_check_prompt("test(X).", [])
        assert result_empty is None

    def test_synthesis_prompt_returns_string(self):
        """Test get_synthesis_prompt returns string."""
        result = get_synthesis_prompt("test(X).", '{"issues": []}', [])
        assert isinstance(result, str)


class TestPromptNonEmpty:
    """Test that prompts are non-empty and substantial."""

    def test_critique_prompt_not_empty(self):
        """Test critique prompt is not empty."""
        prompt = get_critique_prompt("test(X).", [], "")
        assert len(prompt) > 100  # Should be substantial

    def test_edge_case_prompt_not_empty(self):
        """Test edge case prompt is not empty."""
        prompt = get_edge_case_prompt("test(X).", "")
        assert len(prompt) > 100

    def test_contradiction_prompt_not_empty(self):
        """Test contradiction prompt is not empty."""
        prompt = get_contradiction_check_prompt("test(X).", ["other(Y)."])
        assert len(prompt) > 100

    def test_synthesis_prompt_not_empty(self):
        """Test synthesis prompt is not empty."""
        prompt = get_synthesis_prompt("test(X).", '{"issues": []}', [])
        assert len(prompt) > 100
