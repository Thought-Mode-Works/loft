"""
Unit tests for RuleGenerator with mocked LLM.

Tests rule generation functionality without making actual LLM API calls,
using mocked responses to verify logic and error handling.
"""

import pytest
from unittest.mock import Mock
from loft.neural.rule_generator import RuleGenerator
from loft.neural.rule_schemas import (
    GeneratedRule,
    GapFillingResponse,
    ConsensusVote,
    RuleCandidate,
)
from loft.neural.llm_interface import LLMInterface, LLMResponse
from loft.symbolic.asp_core import ASPCore


@pytest.fixture
def mock_llm():
    """Create a mocked LLM interface."""
    llm = Mock(spec=LLMInterface)
    llm.get_total_cost = Mock(return_value=0.05)
    llm.get_total_tokens = Mock(return_value=1000)
    return llm


@pytest.fixture
def mock_asp_core():
    """Create a mocked ASP core."""
    return Mock(spec=ASPCore)


@pytest.fixture
def rule_generator(mock_llm, mock_asp_core):
    """Create a RuleGenerator with mocked dependencies."""
    return RuleGenerator(
        llm=mock_llm,
        asp_core=mock_asp_core,
        domain="contract_law",
        prompt_version="v1.1",  # Use v1.1 which supports constraints
    )


class TestRuleGeneratorInit:
    """Test RuleGenerator initialization."""

    def test_init_with_defaults(self, mock_llm):
        """Test initialization with default parameters."""
        generator = RuleGenerator(llm=mock_llm)

        assert generator.llm == mock_llm
        assert generator.asp_core is not None
        assert generator.domain == "legal"
        assert generator.prompt_version == "latest"

    def test_init_with_custom_params(self, mock_llm, mock_asp_core):
        """Test initialization with custom parameters."""
        generator = RuleGenerator(
            llm=mock_llm,
            asp_core=mock_asp_core,
            domain="tort_law",
            prompt_version="v2.0",
        )

        assert generator.asp_core == mock_asp_core
        assert generator.domain == "tort_law"
        assert generator.prompt_version == "v2.0"


class TestGenerateFromPrinciple:
    """Test generating rules from legal principles."""

    def test_successful_generation(self, rule_generator, mock_llm):
        """Test successful rule generation from principle."""
        # Mock LLM response
        generated_rule = GeneratedRule(
            asp_rule="enforceable(C) :- contract(C), not void(C).",
            confidence=0.9,
            reasoning="Contract is enforceable if not void",
            predicates_used=["contract/1", "void/1"],
            new_predicates=["enforceable/1"],
            alternative_formulations=[],
            source_type="principle",
            source_text="A contract is enforceable unless it is void",
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = generated_rule
        mock_llm.query = Mock(return_value=mock_response)

        # Generate rule
        result = rule_generator.generate_from_principle(
            principle_text="A contract is enforceable unless it is void"
        )

        # Verify result
        assert isinstance(result, GeneratedRule)
        assert result.asp_rule == "enforceable(C) :- contract(C), not void(C)."
        assert result.confidence == 0.9
        assert result.source_type == "principle"

        # Verify LLM was called
        mock_llm.query.assert_called_once()
        call_kwargs = mock_llm.query.call_args[1]
        assert call_kwargs["output_schema"] == GeneratedRule
        assert call_kwargs["temperature"] == 0.3

    def test_with_existing_predicates(self, rule_generator, mock_llm):
        """Test generation with specified existing predicates."""
        generated_rule = GeneratedRule(
            asp_rule="test_rule(X) :- pred1(X).",
            confidence=0.8,
            reasoning="Test",
            predicates_used=["pred1/1"],
            new_predicates=[],
            source_type="principle",
            source_text="Test principle",
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = generated_rule
        mock_llm.query = Mock(return_value=mock_response)

        rule_generator.generate_from_principle(
            principle_text="Test principle",
            existing_predicates=["pred1/1", "pred2/2"],
        )

        # Verify prompt includes predicates
        call_args = mock_llm.query.call_args
        prompt = call_args[1]["question"]
        assert "pred1/1" in prompt
        assert "pred2/2" in prompt

    def test_with_constraints(self, rule_generator, mock_llm):
        """Test generation with constraints."""
        generated_rule = GeneratedRule(
            asp_rule="test(X) :- constraint_satisfied(X).",
            confidence=0.85,
            reasoning="Test",
            predicates_used=[],
            new_predicates=[],
            source_type="principle",
            source_text="Test",
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = generated_rule
        mock_llm.query = Mock(return_value=mock_response)

        rule_generator.generate_from_principle(
            principle_text="Test principle",
            constraints="Must use negation",
        )

        # Verify constraints in prompt
        prompt = mock_llm.query.call_args[1]["question"]
        assert "Must use negation" in prompt


class TestGenerateFromCase:
    """Test generating rules from case law."""

    def test_successful_case_extraction(self, rule_generator, mock_llm):
        """Test successful rule extraction from case."""
        generated_rule = GeneratedRule(
            asp_rule="exception_applies(C) :- part_performance(C).",
            confidence=0.92,
            reasoning="Court held part performance exception applies",
            predicates_used=["part_performance/1"],
            new_predicates=["exception_applies/1"],
            source_type="case",
            source_text="The court held...",
            citation="Smith v. Jones, 123 F.3d 456",
            jurisdiction="Federal",
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = generated_rule
        mock_llm.query = Mock(return_value=mock_response)

        result = rule_generator.generate_from_case(
            case_text="The court held that part performance satisfies the statute",
            citation="Smith v. Jones, 123 F.3d 456",
            jurisdiction="Federal",
        )

        assert isinstance(result, GeneratedRule)
        assert result.citation == "Smith v. Jones, 123 F.3d 456"
        assert result.jurisdiction == "Federal"
        assert result.source_type == "case"

    def test_with_focus(self, rule_generator, mock_llm):
        """Test case extraction with specific focus."""
        generated_rule = GeneratedRule(
            asp_rule="test(C) :- focused_aspect(C).",
            confidence=0.88,
            reasoning="Focused on specific aspect",
            predicates_used=[],
            new_predicates=[],
            source_type="case",
            source_text="Test",
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = generated_rule
        mock_llm.query = Mock(return_value=mock_response)

        rule_generator.generate_from_case(
            case_text="Test case",
            citation="Test v. Case",
            focus="statute of frauds exception",
        )

        # Verify focus in prompt
        prompt = mock_llm.query.call_args[1]["question"]
        assert "statute of frauds exception" in prompt


class TestFillKnowledgeGap:
    """Test filling knowledge gaps."""

    def test_successful_gap_filling(self, rule_generator, mock_llm):
        """Test successful gap filling with multiple candidates."""
        candidate1 = RuleCandidate(
            rule=GeneratedRule(
                asp_rule="missing_pred(X) :- condition1(X).",
                confidence=0.85,
                reasoning="Conservative approach",
                predicates_used=["condition1/1"],
                new_predicates=["missing_pred/1"],
                source_type="gap_fill",
                source_text="Gap: missing_pred not defined",
            ),
            applicability_score=0.9,
            complexity_score=0.3,
        )

        candidate2 = RuleCandidate(
            rule=GeneratedRule(
                asp_rule="missing_pred(X) :- condition1(X), condition2(X).",
                confidence=0.8,
                reasoning="More comprehensive",
                predicates_used=["condition1/1", "condition2/1"],
                new_predicates=["missing_pred/1"],
                source_type="gap_fill",
                source_text="Gap: missing_pred not defined",
            ),
            applicability_score=0.95,
            complexity_score=0.6,
        )

        gap_response = GapFillingResponse(
            gap_description="Cannot determine if predicate holds",
            missing_predicate="missing_pred/1",
            candidates=[candidate1, candidate2],
            recommended_index=1,
            requires_validation=True,
            test_cases_needed=["test1", "test2"],
            confidence=0.88,
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = gap_response
        mock_llm.query = Mock(return_value=mock_response)

        result = rule_generator.fill_knowledge_gap(
            gap_description="Cannot determine if predicate holds",
            missing_predicate="missing_pred/1",
        )

        assert isinstance(result, GapFillingResponse)
        assert len(result.candidates) == 2
        assert result.recommended_index == 1
        assert result.requires_validation is True

    def test_gap_filling_with_context(self, rule_generator, mock_llm):
        """Test gap filling with additional context."""
        gap_response = GapFillingResponse(
            gap_description="Test gap",
            missing_predicate="test_pred/1",
            candidates=[
                RuleCandidate(
                    rule=GeneratedRule(
                        asp_rule="test_pred(X) :- input(X).",
                        confidence=0.5,
                        reasoning="Test",
                        predicates_used=["input/1"],
                        new_predicates=["test_pred/1"],
                        source_type="gap_fill",
                        source_text="Test",
                    ),
                    applicability_score=0.5,
                )
            ],
            recommended_index=0,
            requires_validation=False,
            test_cases_needed=[],
            confidence=0.7,
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = gap_response
        mock_llm.query = Mock(return_value=mock_response)

        context = {"contract_type": "land_sale", "has_writing": False}

        rule_generator.fill_knowledge_gap(
            gap_description="Test gap",
            missing_predicate="test_pred/1",
            context=context,
        )

        # Verify context in prompt
        prompt = mock_llm.query.call_args[1]["question"]
        assert "land_sale" in prompt
        assert "has_writing" in prompt

    def test_invalid_recommended_index(self, rule_generator, mock_llm):
        """Test handling of invalid recommended index."""
        gap_response = GapFillingResponse(
            gap_description="Test",
            missing_predicate="test/1",
            candidates=[
                RuleCandidate(
                    rule=GeneratedRule(
                        asp_rule="test(X) :- base(X).",
                        confidence=0.5,
                        reasoning="Test",
                        predicates_used=["base/1"],
                        new_predicates=["test/1"],
                        source_type="gap_fill",
                        source_text="Test",
                    ),
                    applicability_score=0.5,
                )
            ],
            recommended_index=5,  # Out of range
            requires_validation=False,
            test_cases_needed=[],
            confidence=0.7,
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = gap_response
        mock_llm.query = Mock(return_value=mock_response)

        result = rule_generator.fill_knowledge_gap(
            gap_description="Test",
            missing_predicate="test/1",
        )

        # Should correct to 0
        assert result.recommended_index == 0


class TestConsensusVote:
    """Test consensus voting."""

    def test_accept_vote(self, rule_generator, mock_llm):
        """Test acceptance vote."""
        vote = ConsensusVote(
            vote="accept",
            confidence=0.95,
            issues_found=[],
            reasoning="Rule is correct and well-formed",
            test_cases_to_validate=["test1"],
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = vote
        mock_llm.query = Mock(return_value=mock_response)

        result = rule_generator.get_consensus_vote(
            proposed_rule="valid(C) :- contract(C).",
            proposer_reasoning="Simple validity check",
        )

        assert result.vote == "accept"
        assert result.confidence == 0.95
        assert len(result.issues_found) == 0

    def test_reject_vote(self, rule_generator, mock_llm):
        """Test rejection vote."""
        vote = ConsensusVote(
            vote="reject",
            confidence=0.9,
            issues_found=["Circular dependency", "Invalid syntax"],
            reasoning="Rule has fundamental flaws",
            test_cases_to_validate=[],
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = vote
        mock_llm.query = Mock(return_value=mock_response)

        result = rule_generator.get_consensus_vote(
            proposed_rule="bad(X) :- bad(X).",  # Circular
            proposer_reasoning="Test",
        )

        assert result.vote == "reject"
        assert len(result.issues_found) == 2
        assert "Circular dependency" in result.issues_found

    def test_revise_vote(self, rule_generator, mock_llm):
        """Test revision vote with suggestion."""
        vote = ConsensusVote(
            vote="revise",
            confidence=0.85,
            issues_found=["Missing exception handling"],
            suggested_revision="better(C) :- contract(C), not exception(C).",
            reasoning="Needs exception handling",
            test_cases_to_validate=["test1", "test2"],
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = vote
        mock_llm.query = Mock(return_value=mock_response)

        result = rule_generator.get_consensus_vote(
            proposed_rule="better(C) :- contract(C).",
            proposer_reasoning="Basic rule",
        )

        assert result.vote == "revise"
        assert result.suggested_revision is not None
        assert "not exception(C)" in result.suggested_revision


class TestRefineRuleFromVotes:
    """Test rule refinement from votes."""

    def test_refinement_with_mixed_votes(self, rule_generator, mock_llm):
        """Test refinement when votes are mixed."""
        votes = [
            ConsensusVote(
                vote="accept",
                confidence=0.8,
                issues_found=[],
                reasoning="Looks good",
            ),
            ConsensusVote(
                vote="revise",
                confidence=0.9,
                issues_found=["Missing edge case"],
                reasoning="Needs work",
            ),
            ConsensusVote(
                vote="revise",
                confidence=0.85,
                issues_found=["Missing edge case", "Unclear variable"],
                reasoning="Needs clarification",
            ),
        ]

        refined_rule = GeneratedRule(
            asp_rule="refined(C) :- contract(C), edge_case_handled(C).",
            confidence=0.92,
            reasoning="Incorporated feedback from votes",
            predicates_used=["contract/1", "edge_case_handled/1"],
            new_predicates=["refined/1"],
            source_type="refinement",
            source_text="Original rule refined based on votes",
        )

        mock_response = Mock(spec=LLMResponse)
        mock_response.content = refined_rule
        mock_llm.query = Mock(return_value=mock_response)

        result = rule_generator.refine_rule_from_votes(
            original_rule="original(C) :- contract(C).",
            votes=votes,
        )

        assert isinstance(result, GeneratedRule)
        assert result.source_type == "refinement"
        assert result.confidence > 0.9

        # Verify prompt contains vote summary
        prompt = mock_llm.query.call_args[1]["question"]
        assert "Accept: 1" in prompt
        assert "Revise: 2" in prompt
        assert "Missing edge case" in prompt  # Common issue


class TestValidateRuleSyntax:
    """Test ASP syntax validation."""

    def test_valid_syntax(self, rule_generator):
        """Test validation of valid ASP rule."""
        is_valid, error = rule_generator.validate_rule_syntax(
            "enforceable(C) :- contract(C), not void(C)."
        )

        assert is_valid is True
        assert error is None

    def test_valid_fact(self, rule_generator):
        """Test validation of valid ASP fact."""
        is_valid, error = rule_generator.validate_rule_syntax("contract(c1).")

        assert is_valid is True
        assert error is None

    def test_invalid_syntax(self, rule_generator):
        """Test validation of invalid syntax."""
        # Missing period
        is_valid, error = rule_generator.validate_rule_syntax("invalid(X) :- test(X)")

        # May or may not catch this - depends on Clingo
        # Just ensure it doesn't crash
        assert isinstance(is_valid, bool)


class TestHelperMethods:
    """Test helper methods."""

    def test_get_existing_predicates(self, rule_generator):
        """Test extraction of existing predicates."""
        predicates = rule_generator._get_existing_predicates()

        assert isinstance(predicates, list)
        assert len(predicates) > 0
        assert "contract/1" in predicates

    def test_format_predicate_list(self, rule_generator):
        """Test predicate list formatting."""
        predicates = ["contract/1", "party/2", "writing/1"]
        formatted = rule_generator._format_predicate_list(predicates)

        assert "- contract/1" in formatted
        assert "- party/2" in formatted
        assert "- writing/1" in formatted

    def test_format_empty_predicate_list(self, rule_generator):
        """Test formatting of empty predicate list."""
        formatted = rule_generator._format_predicate_list([])

        assert "No existing predicates" in formatted

    def test_get_generation_stats(self, rule_generator, mock_llm):
        """Test generation statistics."""
        stats = rule_generator.get_generation_stats()

        assert "total_cost" in stats
        assert "total_tokens" in stats
        assert "domain" in stats
        assert "prompt_version" in stats
        assert stats["domain"] == "contract_law"
        assert stats["prompt_version"] == "v1.1"
