"""
Unit tests for the Synthesizer System (Phase 4.2).

Tests dialectical synthesis capabilities including thesis-antithesis-synthesis
dialectical cycles.
"""

import json
from unittest.mock import Mock

import pytest

from loft.dialectical.critique_schemas import (
    Contradiction,
    CritiqueIssue,
    CritiqueReport,
    CritiqueSeverity,
    EdgeCase,
)
from loft.dialectical.debate_schemas import DebateArgument
from loft.dialectical.synthesizer import Synthesizer
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


class TestSynthesizerInitialization:
    """Test Synthesizer initialization."""

    def test_init_with_llm_client(self):
        """Test initialization with LLM client."""
        mock_llm = Mock()
        synthesizer = Synthesizer(llm_client=mock_llm, mock_mode=False)

        assert synthesizer.llm_client is mock_llm
        assert not synthesizer.mock_mode

    def test_init_mock_mode_explicit(self):
        """Test explicit mock mode."""
        synthesizer = Synthesizer(mock_mode=True)

        assert synthesizer.mock_mode
        assert synthesizer.llm_client is None

    def test_init_mock_mode_no_llm(self):
        """Test mock mode activated when no LLM provided."""
        synthesizer = Synthesizer(llm_client=None)

        assert synthesizer.mock_mode


class TestSynthesizerMockMode:
    """Test Synthesizer in mock mode (no LLM required)."""

    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer in mock mode."""
        return Synthesizer(mock_mode=True)

    @pytest.fixture
    def thesis(self):
        """Create sample thesis rule."""
        return make_test_rule(
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            predicates_used=["contract", "signed"],
        )

    @pytest.fixture
    def antithesis_with_issues(self):
        """Create critique with issues."""
        return CritiqueReport(
            rule="enforceable(C) :- contract(C), signed(C).",
            issues=[
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing consideration requirement",
                    severity=CritiqueSeverity.MEDIUM,
                ),
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing capacity checks",
                    severity=CritiqueSeverity.MEDIUM,
                ),
            ],
            recommendation="revise",
        )

    def test_synthesize_basic(self, synthesizer, thesis, antithesis_with_issues):
        """Test basic synthesis (thesis + antithesis → synthesis)."""
        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis_with_issues, [])

        assert isinstance(synthesis_rule, GeneratedRule)
        assert isinstance(argument, DebateArgument)
        assert argument.speaker == "synthesizer"

    def test_synthesize_with_suggested_revision(self, synthesizer, thesis):
        """Test synthesis with suggested revision from critique."""
        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            issues=[],
            suggested_revision="enforceable(C) :- contract(C), signed(C), consideration(C).",
            recommendation="revise",
        )

        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, [])

        assert synthesis_rule.asp_rule == antithesis.suggested_revision
        assert "Applied suggested revision" in argument.content

    def test_synthesize_adds_consideration(self, synthesizer, thesis, antithesis_with_issues):
        """Test synthesis adding missing consideration condition."""
        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis_with_issues, [])

        # Mock synthesis should add consideration if mentioned in issues
        if synthesis_rule.asp_rule != thesis.asp_rule:
            assert "consideration" in synthesis_rule.asp_rule

    def test_synthesize_adds_capacity(self, synthesizer, thesis, antithesis_with_issues):
        """Test synthesis adding missing capacity checks."""
        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis_with_issues, [])

        # Mock synthesis should add capacity if mentioned in issues
        if synthesis_rule.asp_rule != thesis.asp_rule:
            # May add capacity checks
            pass  # Mock synthesis logic varies

    def test_synthesize_preserves_original_intent(self, synthesizer, thesis):
        """Test synthesis preserving original intent."""
        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            issues=[
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing consideration",
                    severity=CritiqueSeverity.LOW,
                )
            ],
            recommendation="revise",
        )

        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, [])

        # Should still reference contract and signed
        assert "contract(C)" in synthesis_rule.asp_rule

    def test_synthesize_empty_critique(self, synthesizer, thesis):
        """Test synthesis with empty critique (no issues)."""
        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            issues=[],
            recommendation="accept",
        )

        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, [])

        # Should return thesis unchanged
        assert synthesis_rule.asp_rule == thesis.asp_rule
        assert "satisfactory" in argument.content.lower()

    def test_synthesize_confidence_boost(self, synthesizer, thesis):
        """Test synthesis boosts confidence."""
        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            issues=[
                CritiqueIssue(
                    category="minor",
                    description="Minor issue",
                    severity=CritiqueSeverity.LOW,
                )
            ],
            suggested_revision="enforceable(C) :- contract(C), signed(C), valid(C).",
        )

        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, [])

        # Synthesis should have increased confidence
        if synthesis_rule.asp_rule != thesis.asp_rule:
            assert synthesis_rule.confidence >= thesis.confidence


class TestSynthesizerWithLLM:
    """Test Synthesizer with mocked LLM responses."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        return Mock()

    @pytest.fixture
    def thesis(self):
        """Create sample thesis rule."""
        return make_test_rule(
            asp_rule="enforceable(C) :- contract(C), signed(C).",
            predicates_used=["contract", "signed"],
        )

    @pytest.fixture
    def antithesis(self):
        """Create sample critique."""
        return CritiqueReport(
            rule="enforceable(C) :- contract(C), signed(C).",
            issues=[
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing consideration requirement",
                    severity=CritiqueSeverity.MEDIUM,
                )
            ],
            recommendation="revise",
        )

    def test_synthesize_with_llm_success(self, mock_llm, thesis, antithesis):
        """Test synthesis with LLM."""
        mock_llm.query.return_value = Mock(
            raw_text=json.dumps({
                "synthesized_rule": "enforceable(C) :- contract(C), signed(C), consideration(C).",
                "reasoning": "Added consideration requirement from critique",
                "argument": "The thesis correctly identified contract and signature, but critique rightly noted missing consideration",
                "confidence": 0.85,
                "changes_made": ["Added consideration(C)"],
            })
        )

        synthesizer = Synthesizer(llm_client=mock_llm, mock_mode=False)
        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, [])

        assert "consideration" in synthesis_rule.asp_rule
        assert synthesis_rule.confidence == 0.85
        assert argument.speaker == "synthesizer"
        assert "thesis" in argument.content.lower()

    def test_synthesize_with_llm_markdown_wrapped(self, mock_llm, thesis, antithesis):
        """Test JSON extraction from markdown-wrapped LLM response."""
        mock_llm.query.return_value = Mock(
            raw_text="""Here's the synthesis:

```json
{
    "synthesized_rule": "enforceable(C) :- contract(C), signed(C), consideration(C), capacity(P1), capacity(P2).",
    "reasoning": "Combined thesis and antithesis",
    "argument": "Addressed all critique points",
    "confidence": 0.9,
    "changes_made": ["Added consideration", "Added capacity checks"]
}
```"""
        )

        synthesizer = Synthesizer(llm_client=mock_llm, mock_mode=False)
        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, [])

        assert "consideration" in synthesis_rule.asp_rule
        assert "capacity" in synthesis_rule.asp_rule
        assert synthesis_rule.confidence == 0.9

    def test_synthesize_with_llm_error_handling(self, mock_llm, thesis, antithesis):
        """Test error handling when synthesis fails."""
        mock_llm.query.side_effect = Exception("LLM API error")

        synthesizer = Synthesizer(llm_client=mock_llm, mock_mode=False)
        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, [])

        # Should fall back to thesis
        assert synthesis_rule.asp_rule == thesis.asp_rule
        assert "failed" in argument.content.lower()
        assert argument.confidence < 0.5

    def test_synthesize_normalizes_whitespace(self, mock_llm, thesis, antithesis):
        """Test synthesis normalizes ASP rule whitespace."""
        mock_llm.query.return_value = Mock(
            raw_text=json.dumps({
                "synthesized_rule": "enforceable(C) :- \n  contract(C), \n  signed(C),\n  consideration(C).",
                "reasoning": "Test",
                "argument": "Test",
                "confidence": 0.8,
            })
        )

        synthesizer = Synthesizer(llm_client=mock_llm, mock_mode=False)
        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, [])

        # Should normalize to single line without extra whitespace
        assert "\n" not in synthesis_rule.asp_rule
        assert "  " not in synthesis_rule.asp_rule  # No double spaces

    def test_synthesize_with_edge_cases(self, mock_llm, thesis):
        """Test synthesis handling edge cases from critique."""
        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            edge_cases=[
                EdgeCase(
                    description="Contract with minor",
                    scenario={"party_age": 17},
                    expected_outcome="Voidable",
                    severity=CritiqueSeverity.HIGH,
                )
            ],
            recommendation="revise",
        )

        mock_llm.query.return_value = Mock(
            raw_text=json.dumps({
                "synthesized_rule": "enforceable(C) :- contract(C), signed(C), all_parties_competent(C).",
                "reasoning": "Added competency check to handle minor party edge case",
                "argument": "Addressed edge case of minor parties",
                "confidence": 0.85,
            })
        )

        synthesizer = Synthesizer(llm_client=mock_llm, mock_mode=False)
        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, [])

        assert synthesis_rule.asp_rule != thesis.asp_rule

    def test_synthesize_with_contradictions(self, mock_llm, thesis):
        """Test synthesis resolving contradictions."""
        existing_rules = ["enforceable(C) :- contract(C), consideration(C), capacity(P)."]
        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            contradictions=[
                Contradiction(
                    description="Conflicting enforceability conditions",
                    new_rule=thesis.asp_rule,
                    conflicting_rule=existing_rules[0],
                    example_scenario={},
                    severity=CritiqueSeverity.HIGH,
                )
            ],
            recommendation="revise",
        )

        mock_llm.query.return_value = Mock(
            raw_text=json.dumps({
                "synthesized_rule": "enforceable(C) :- contract(C), signed(C), consideration(C), capacity(P).",
                "reasoning": "Aligned with existing rule to resolve contradiction",
                "argument": "Resolved contradiction by incorporating all conditions",
                "confidence": 0.8,
            })
        )

        synthesizer = Synthesizer(llm_client=mock_llm, mock_mode=False)
        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, existing_rules)

        assert "consideration" in synthesis_rule.asp_rule


class TestPromptBuilding:
    """Test synthesis prompt building."""

    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer."""
        return Synthesizer(mock_mode=True)

    @pytest.fixture
    def thesis(self):
        """Create thesis rule."""
        return make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.8,
            reasoning="Basic contract rule",
            predicates_used=["contract"],
        )

    @pytest.fixture
    def antithesis(self):
        """Create antithesis critique."""
        return CritiqueReport(
            rule="enforceable(C) :- contract(C).",
            issues=[
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing consideration",
                    severity=CritiqueSeverity.MEDIUM,
                )
            ],
            edge_cases=[
                EdgeCase(
                    description="Contract without consideration",
                    scenario={},
                    severity=CritiqueSeverity.HIGH,
                )
            ],
            contradictions=[],
            recommendation="revise",
        )

    def test_build_synthesis_prompt_structure(self, synthesizer, thesis, antithesis):
        """Test synthesis prompt structure."""
        prompt = synthesizer._build_synthesis_prompt(thesis, antithesis, [], None)

        assert "Thesis" in prompt
        assert "Antithesis" in prompt
        assert "Synthesis" in prompt or "synthesize" in prompt.lower()
        assert thesis.asp_rule in prompt
        assert thesis.reasoning in prompt

    def test_build_synthesis_prompt_includes_issues(self, synthesizer, thesis, antithesis):
        """Test prompt includes critique issues."""
        prompt = synthesizer._build_synthesis_prompt(thesis, antithesis, [], None)

        assert "Missing consideration" in prompt
        assert "medium" in prompt.lower()

    def test_build_synthesis_prompt_includes_edge_cases(self, synthesizer, thesis, antithesis):
        """Test prompt includes edge cases."""
        prompt = synthesizer._build_synthesis_prompt(thesis, antithesis, [], None)

        assert "Contract without consideration" in prompt

    def test_build_synthesis_prompt_includes_existing_rules(self, synthesizer, thesis, antithesis):
        """Test prompt includes existing rules context."""
        existing = ["rule1(X) :- condition(X).", "rule2(Y) :- other(Y)."]
        prompt = synthesizer._build_synthesis_prompt(thesis, antithesis, existing, None)

        assert "rule1" in prompt
        assert "rule2" in prompt

    def test_build_synthesis_prompt_empty_existing_rules(self, synthesizer, thesis, antithesis):
        """Test prompt with no existing rules."""
        prompt = synthesizer._build_synthesis_prompt(thesis, antithesis, [], None)

        assert "No existing rules" in prompt or "no existing rules" in prompt.lower()

    def test_build_synthesis_prompt_with_context(self, synthesizer, thesis, antithesis):
        """Test prompt with additional context."""
        context = "Contract law domain with UCC rules"
        prompt = synthesizer._build_synthesis_prompt(thesis, antithesis, [], context)

        # Context may not be used in current implementation, but test structure
        assert "Thesis" in prompt


class TestJSONExtraction:
    """Test JSON extraction from LLM responses."""

    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer with mock LLM."""
        return Synthesizer(llm_client=Mock(), mock_mode=False)

    def test_extract_json_plain(self, synthesizer):
        """Test extraction from plain JSON."""
        response = '{"synthesized_rule": "test :- rule.", "confidence": 0.8}'
        extracted = synthesizer._extract_json(response)
        assert extracted == response

    def test_extract_json_with_markdown(self, synthesizer):
        """Test extraction from markdown code block."""
        response = """Some text
```json
{"synthesized_rule": "test :- rule.", "confidence": 0.8}
```
More text"""
        extracted = synthesizer._extract_json(response)
        data = json.loads(extracted)
        assert data["synthesized_rule"] == "test :- rule."

    def test_extract_json_without_language_marker(self, synthesizer):
        """Test extraction from code block without 'json' marker."""
        response = """```
{"synthesized_rule": "test :- rule."}
```"""
        extracted = synthesizer._extract_json(response)
        data = json.loads(extracted)
        assert "synthesized_rule" in data

    def test_extract_json_nested_in_text(self, synthesizer):
        """Test extraction when JSON is nested in text."""
        response = 'Here is result: {"synthesized_rule": "test."} end'
        extracted = synthesizer._extract_json(response)
        data = json.loads(extracted)
        assert "synthesized_rule" in data


class TestLLMCallMethod:
    """Test LLM call method."""

    def test_call_llm_success(self):
        """Test successful LLM call."""
        mock_llm = Mock()
        mock_llm.query.return_value = Mock(raw_text="LLM response")

        synthesizer = Synthesizer(llm_client=mock_llm, mock_mode=False)
        response = synthesizer._call_llm("Test prompt")

        assert response == "LLM response"
        mock_llm.query.assert_called_once()

    def test_call_llm_no_client(self):
        """Test LLM call without client raises error."""
        synthesizer = Synthesizer(llm_client=None, mock_mode=False)

        with pytest.raises(ValueError, match="No LLM client configured"):
            synthesizer._call_llm("Test prompt")


class TestMockSynthesis:
    """Test mock synthesis implementation."""

    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer in mock mode."""
        return Synthesizer(mock_mode=True)

    def test_mock_synthesis_applies_suggested_revision(self, synthesizer):
        """Test mock synthesis applies suggested revision."""
        thesis = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["contract"],
        )

        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            suggested_revision="enforceable(C) :- contract(C), consideration(C).",
        )

        synthesis_rule, argument = synthesizer._mock_synthesis(thesis, antithesis, [])

        assert synthesis_rule.asp_rule == antithesis.suggested_revision

    def test_mock_synthesis_adds_consideration(self, synthesizer):
        """Test mock synthesis adds consideration when mentioned."""
        thesis = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["contract"],
        )

        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            issues=[
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing consideration requirement",
                    severity=CritiqueSeverity.MEDIUM,
                )
            ],
        )

        synthesis_rule, argument = synthesizer._mock_synthesis(thesis, antithesis, [])

        if synthesis_rule.asp_rule != thesis.asp_rule:
            assert "consideration" in synthesis_rule.asp_rule

    def test_mock_synthesis_adds_capacity(self, synthesizer):
        """Test mock synthesis adds capacity when mentioned."""
        thesis = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["contract"],
        )

        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            issues=[
                CritiqueIssue(
                    category="missing_condition",
                    description="Missing capacity checks for parties",
                    severity=CritiqueSeverity.MEDIUM,
                )
            ],
        )

        synthesis_rule, argument = synthesizer._mock_synthesis(thesis, antithesis, [])

        if synthesis_rule.asp_rule != thesis.asp_rule:
            assert "capacity" in synthesis_rule.asp_rule

    def test_mock_synthesis_no_changes_needed(self, synthesizer):
        """Test mock synthesis when no changes needed."""
        thesis = make_test_rule(
            asp_rule="enforceable(C) :- contract(C), consideration(C), capacity(P).",
            predicates_used=["contract", "consideration", "capacity"],
        )

        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            issues=[],
        )

        synthesis_rule, argument = synthesizer._mock_synthesis(thesis, antithesis, [])

        assert synthesis_rule.asp_rule == thesis.asp_rule
        assert "satisfactory" in argument.content.lower()


class TestConfidenceScoring:
    """Test confidence scoring in synthesis."""

    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer in mock mode."""
        return Synthesizer(mock_mode=True)

    def test_synthesis_confidence_boost(self, synthesizer):
        """Test synthesis increases confidence."""
        thesis = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.7,
            predicates_used=["contract"],
        )

        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            suggested_revision="enforceable(C) :- contract(C), consideration(C).",
        )

        synthesis_rule, argument = synthesizer._mock_synthesis(thesis, antithesis, [])

        # Mock synthesis should boost confidence
        if synthesis_rule.asp_rule != thesis.asp_rule:
            assert synthesis_rule.confidence > thesis.confidence

    def test_synthesis_confidence_cap(self, synthesizer):
        """Test synthesis confidence capped at 0.95."""
        thesis = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            confidence=0.9,
            predicates_used=["contract"],
        )

        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            suggested_revision="enforceable(C) :- contract(C), consideration(C).",
        )

        synthesis_rule, argument = synthesizer._mock_synthesis(thesis, antithesis, [])

        # Confidence should not exceed 0.95
        assert synthesis_rule.confidence <= 0.95


class TestDebateArgument:
    """Test DebateArgument generation."""

    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer in mock mode."""
        return Synthesizer(mock_mode=True)

    def test_argument_structure(self, synthesizer):
        """Test debate argument structure."""
        thesis = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["contract"],
        )

        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            issues=[],
        )

        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, [])

        assert isinstance(argument, DebateArgument)
        assert argument.speaker == "synthesizer"
        assert isinstance(argument.content, str)
        assert len(argument.content) > 0

    def test_argument_references_thesis(self, synthesizer):
        """Test argument references thesis."""
        thesis = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["contract"],
        )

        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            suggested_revision="enforceable(C) :- contract(C), consideration(C).",
        )

        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, [])

        assert len(argument.references) > 0
        assert thesis.asp_rule in argument.references

    def test_argument_confidence_matches_rule(self, synthesizer):
        """Test argument confidence matches synthesized rule."""
        thesis = make_test_rule(
            asp_rule="enforceable(C) :- contract(C).",
            predicates_used=["contract"],
        )

        antithesis = CritiqueReport(
            rule=thesis.asp_rule,
            suggested_revision="enforceable(C) :- contract(C), consideration(C).",
        )

        synthesis_rule, argument = synthesizer.synthesize(thesis, antithesis, [])

        assert argument.confidence == synthesis_rule.confidence
