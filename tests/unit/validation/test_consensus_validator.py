"""
Unit tests for consensus validator.

Tests multi-LLM consensus validation of LLM-generated ASP rules.
"""

import pytest
from unittest.mock import Mock, MagicMock
from loft.validation.consensus_validator import ConsensusValidator
from loft.validation.validation_schemas import ConsensusValidationResult
from loft.neural.rule_schemas import ConsensusVote


class TestConsensusValidator:
    """Tests for ConsensusValidator class."""

    @pytest.fixture
    def mock_rule_generators(self):
        """Create mock rule generators for testing."""
        gen1 = Mock()
        gen2 = Mock()
        gen3 = Mock()
        return [gen1, gen2, gen3]

    @pytest.fixture
    def validator(self, mock_rule_generators):
        """Create a consensus validator with mock generators."""
        return ConsensusValidator(
            rule_generators=mock_rule_generators,
            consensus_threshold=0.6,
        )

    def test_initialization(self, mock_rule_generators):
        """Test validator initialization."""
        validator = ConsensusValidator(
            rule_generators=mock_rule_generators,
            weights=[0.5, 0.3, 0.2],
            consensus_threshold=0.7,
        )

        assert len(validator.rule_generators) == 3
        assert validator.weights == [0.5, 0.3, 0.2]
        assert validator.consensus_threshold == 0.7

    def test_initialization_weight_mismatch(self, mock_rule_generators):
        """Test that mismatched weights raise error."""
        with pytest.raises(ValueError, match="Number of weights must match"):
            ConsensusValidator(
                rule_generators=mock_rule_generators,
                weights=[0.5, 0.5],  # Only 2 weights for 3 generators
            )

    def test_validate_rule_unanimous_accept(self, validator, mock_rule_generators):
        """Test validation with unanimous accept votes."""
        # All generators vote accept with high confidence
        for gen in mock_rule_generators:
            gen.get_consensus_vote.return_value = ConsensusVote(
                vote="accept",
                confidence=0.9,
                reasoning="Rule looks good",
                issues_found=[],
                suggested_revision=None,
                test_cases_to_validate=[],
                voter_id="test_voter",
            )

        result = validator.validate_rule(
            rule_text="enforceable(C) :- contract(C), not void(C).",
            proposer_reasoning="Contract is enforceable unless void",
        )

        assert isinstance(result, ConsensusValidationResult)
        assert result.decision == "accept"
        assert result.is_valid
        assert result.consensus_strength > 0.6
        assert len(result.votes) == 3

    def test_validate_rule_unanimous_reject(self, validator, mock_rule_generators):
        """Test validation with unanimous reject votes."""
        for gen in mock_rule_generators:
            gen.get_consensus_vote.return_value = ConsensusVote(
                vote="reject",
                confidence=0.8,
                reasoning="Rule has logical errors",
                issues_found=["circular_dependency", "undefined_predicate"],
                suggested_revision=None,
                test_cases_to_validate=[],
                voter_id="test_voter",
            )

        result = validator.validate_rule(
            rule_text="bad(X) :- bad(X).",  # Circular rule
            proposer_reasoning="Self-referential rule",
        )

        assert result.decision == "reject"
        assert not result.is_valid
        assert len(result.votes) == 3

    def test_validate_rule_needs_revision(self, validator, mock_rule_generators):
        """Test validation with revise votes."""
        for gen in mock_rule_generators:
            gen.get_consensus_vote.return_value = ConsensusVote(
                vote="revise",
                confidence=0.7,
                reasoning="Rule needs minor adjustments",
                issues_found=["missing_negation"],
                suggested_revision="Add 'not void(C)' condition", test_cases_to_validate=[],
                voter_id="test_voter",
            )

        result = validator.validate_rule(
            rule_text="enforceable(C) :- contract(C).",
            proposer_reasoning="Simple enforceability rule",
        )

        assert result.decision == "revise"
        assert not result.is_valid
        assert len(result.suggested_revisions) == 3  # One from each voter

    def test_validate_rule_mixed_votes(self, validator, mock_rule_generators):
        """Test validation with mixed votes."""
        # Gen 1: accept, Gen 2: accept, Gen 3: revise
        mock_rule_generators[0].get_consensus_vote.return_value = ConsensusVote(
            vote="accept",
            confidence=0.9,
            reasoning="Good rule",
            issues_found=[],
            suggested_revision=None, test_cases_to_validate=[],
            voter_id="voter_0",
        )

        mock_rule_generators[1].get_consensus_vote.return_value = ConsensusVote(
            vote="accept",
            confidence=0.85,
            reasoning="Acceptable",
            issues_found=[],
            suggested_revision=None, test_cases_to_validate=[],
            voter_id="voter_1",
        )

        mock_rule_generators[2].get_consensus_vote.return_value = ConsensusVote(
            vote="revise",
            confidence=0.6,
            reasoning="Could be better",
            issues_found=["clarity"],
            suggested_revision="Rephrase", test_cases_to_validate=[],
            voter_id="voter_2",
        )

        result = validator.validate_rule(
            rule_text="test(X) :- condition(X).",
            proposer_reasoning="Test rule",
        )

        # Accept should win (2 accept vs 1 revise)
        assert result.decision == "accept"
        assert result.accept_weight > result.revise_weight

    def test_validate_rule_weighted_voting(self):
        """Test that vote weights are properly applied."""
        gen1, gen2, gen3 = Mock(), Mock(), Mock()

        # Give gen1 high weight
        validator = ConsensusValidator(
            rule_generators=[gen1, gen2, gen3],
            weights=[0.7, 0.15, 0.15],
            consensus_threshold=0.5,
        )

        # Gen1 votes reject (high weight), others vote accept
        gen1.get_consensus_vote.return_value = ConsensusVote(
            vote="reject",
            confidence=1.0,
            reasoning="Major issue",
            issues_found=["critical_error"],
            suggested_revision=None, test_cases_to_validate=[],
            voter_id="voter_0",
        )

        gen2.get_consensus_vote.return_value = ConsensusVote(
            vote="accept",
            confidence=1.0,
            reasoning="Looks good",
            issues_found=[],
            suggested_revision=None, test_cases_to_validate=[],
            voter_id="voter_1",
        )

        gen3.get_consensus_vote.return_value = ConsensusVote(
            vote="accept",
            confidence=1.0,
            reasoning="Looks good",
            issues_found=[],
            suggested_revision=None, test_cases_to_validate=[],
            voter_id="voter_2",
        )

        result = validator.validate_rule(
            rule_text="test.",
            proposer_reasoning="Test",
        )

        # Reject should win due to higher weight
        assert result.decision == "reject"
        assert result.reject_weight == 0.7
        assert result.accept_weight == 0.3

    def test_validate_rule_consensus_threshold(self):
        """Test that consensus threshold is enforced."""
        gen1, gen2, gen3 = Mock(), Mock(), Mock()

        validator = ConsensusValidator(
            rule_generators=[gen1, gen2, gen3],
            consensus_threshold=0.8,  # High threshold
        )

        # 2 accept, 1 reject (weak consensus)
        gen1.get_consensus_vote.return_value = ConsensusVote(
            vote="accept", confidence=0.6, reasoning="OK", issues_found=[], suggested_revision=None, test_cases_to_validate=[], voter_id="v0"
        )
        gen2.get_consensus_vote.return_value = ConsensusVote(
            vote="accept", confidence=0.6, reasoning="OK", issues_found=[], suggested_revision=None, test_cases_to_validate=[], voter_id="v1"
        )
        gen3.get_consensus_vote.return_value = ConsensusVote(
            vote="reject", confidence=0.6, reasoning="No", issues_found=[], suggested_revision=None, test_cases_to_validate=[], voter_id="v2"
        )

        result = validator.validate_rule(
            rule_text="test.",
            proposer_reasoning="Test",
        )

        # Accept wins but consensus_strength might be below threshold
        # is_valid requires both accept AND sufficient consensus
        if result.decision == "accept":
            assert result.is_valid == (result.consensus_strength >= 0.8)

    def test_validate_rule_with_vote_failure(self, validator, mock_rule_generators):
        """Test handling of vote failures."""
        # Gen1 and Gen2 succeed, Gen3 fails
        mock_rule_generators[0].get_consensus_vote.return_value = ConsensusVote(
            vote="accept", confidence=0.9, reasoning="Good", issues_found=[], suggested_revision=None, test_cases_to_validate=[], voter_id="v0"
        )

        mock_rule_generators[1].get_consensus_vote.return_value = ConsensusVote(
            vote="accept", confidence=0.85, reasoning="Good", issues_found=[], suggested_revision=None, test_cases_to_validate=[], voter_id="v1"
        )

        # Gen3 raises an exception
        mock_rule_generators[2].get_consensus_vote.side_effect = Exception("API error")

        result = validator.validate_rule(
            rule_text="test.",
            proposer_reasoning="Test",
        )

        # Should still get result with 3 votes (failed vote becomes abstain/revise)
        assert len(result.votes) == 3
        assert result.votes[2].vote == "revise"
        assert result.votes[2].confidence == 0.0
        assert "voting_error" in result.votes[2].issues_found

    def test_validate_rule_with_existing_predicates(self, validator, mock_rule_generators):
        """Test that existing predicates are passed to voters."""
        for gen in mock_rule_generators:
            gen.get_consensus_vote.return_value = ConsensusVote(
                vote="accept", confidence=0.9, reasoning="Good", issues_found=[], suggested_revision=None, test_cases_to_validate=[], voter_id="v"
            )

        existing_predicates = ["contract", "void", "enforceable"]

        validator.validate_rule(
            rule_text="test.",
            proposer_reasoning="Test",
            existing_predicates=existing_predicates,
        )

        # Verify existing_predicates were passed to each generator
        for gen in mock_rule_generators:
            gen.get_consensus_vote.assert_called_once()
            call_kwargs = gen.get_consensus_vote.call_args[1]
            assert call_kwargs["existing_predicates"] == existing_predicates

    def test_validate_rule_source_type(self, validator, mock_rule_generators):
        """Test that source_type is passed to voters."""
        for gen in mock_rule_generators:
            gen.get_consensus_vote.return_value = ConsensusVote(
                vote="accept", confidence=0.9, reasoning="Good", issues_found=[], suggested_revision=None, test_cases_to_validate=[], voter_id="v"
            )

        validator.validate_rule(
            rule_text="test.",
            proposer_reasoning="Test",
            source_type="principle",
        )

        # Verify source_type was passed
        for gen in mock_rule_generators:
            call_kwargs = gen.get_consensus_vote.call_args[1]
            assert call_kwargs["source_type"] == "principle"

    def test_validate_batch(self, validator, mock_rule_generators):
        """Test batch validation of multiple rules."""
        for gen in mock_rule_generators:
            gen.get_consensus_vote.return_value = ConsensusVote(
                vote="accept", confidence=0.9, reasoning="Good", issues_found=[], suggested_revision=None, test_cases_to_validate=[], voter_id="v"
            )

        rules = [
            "rule1(X) :- condition(X).",
            "rule2(Y) :- other(Y).",
            "rule3.",
        ]
        reasonings = [
            "First rule reasoning",
            "Second rule reasoning",
            "Third rule reasoning",
        ]

        results = validator.validate_batch(rules, reasonings)

        assert len(results) == 3
        assert all(isinstance(r, ConsensusValidationResult) for r in results)
        assert all(r.decision == "accept" for r in results)

    def test_validate_batch_length_mismatch(self, validator):
        """Test that batch validation rejects mismatched lengths."""
        rules = ["rule1.", "rule2."]
        reasonings = ["reason1."]  # Mismatch

        with pytest.raises(ValueError, match="Number of rules must match number of reasonings"):
            validator.validate_batch(rules, reasonings)

    def test_consensus_strength_calculation(self, validator, mock_rule_generators):
        """Test consensus strength calculation."""
        # All vote accept with confidence 1.0
        for gen in mock_rule_generators:
            gen.get_consensus_vote.return_value = ConsensusVote(
                vote="accept", confidence=1.0, reasoning="Perfect", issues_found=[], suggested_revision=None, test_cases_to_validate=[], voter_id="v"
            )

        result = validator.validate_rule(
            rule_text="test.",
            proposer_reasoning="Test",
        )

        # With equal weights and all voting accept, consensus_strength should be 1.0
        assert result.consensus_strength == 1.0

    def test_zero_total_weight_handling(self, validator, mock_rule_generators):
        """Test handling when total weight is zero (all abstain)."""
        # All votes with zero confidence
        for gen in mock_rule_generators:
            gen.get_consensus_vote.return_value = ConsensusVote(
                vote="revise", confidence=0.0, reasoning="Unsure", issues_found=[], suggested_revision="Needs review", test_cases_to_validate=[], voter_id="v"
            )

        result = validator.validate_rule(
            rule_text="test.",
            proposer_reasoning="Test",
        )

        # When all confidence is 0, should default to revise
        assert result.decision == "revise"
        assert result.consensus_strength == 0.0
        assert not result.is_valid

    def test_suggested_revisions_collection(self, validator, mock_rule_generators):
        """Test that suggested revisions are collected from all votes."""
        mock_rule_generators[0].get_consensus_vote.return_value = ConsensusVote(
            vote="revise",
            confidence=0.7,
            reasoning="Needs work",
            issues_found=[],
            suggested_revision="Add negation; Check consistency", test_cases_to_validate=[],
            voter_id="v0",
        )

        mock_rule_generators[1].get_consensus_vote.return_value = ConsensusVote(
            vote="revise",
            confidence=0.8,
            reasoning="Minor issues",
            issues_found=[],
            suggested_revision="Simplify",
            test_cases_to_validate=[],
            voter_id="v1",
        )

        mock_rule_generators[2].get_consensus_vote.return_value = ConsensusVote(
            vote="accept",
            confidence=0.9,
            reasoning="OK",
            issues_found=[],
            suggested_revision=None, test_cases_to_validate=[],
            voter_id="v2",
        )

        result = validator.validate_rule(
            rule_text="test.",
            proposer_reasoning="Test",
        )

        # Should collect suggested revisions from revise votes
        assert len(result.suggested_revisions) == 2  # From v0 and v1 (both have suggested_revision)
        # v0 has a single string "Add negation; Check consistency"
        # v1 has "Simplify"
        # v2 (accept) has no suggestion
        assert any("Add negation" in rev for rev in result.suggested_revisions)
        assert "Simplify" in result.suggested_revisions

    def test_validation_result_structure(self, validator, mock_rule_generators):
        """Test that validation result has all expected fields."""
        for gen in mock_rule_generators:
            gen.get_consensus_vote.return_value = ConsensusVote(
                vote="accept", confidence=0.9, reasoning="Good", issues_found=[], suggested_revision=None, test_cases_to_validate=[], voter_id="v"
            )

        result = validator.validate_rule(
            rule_text="test.",
            proposer_reasoning="Test",
        )

        # Check all required fields
        assert hasattr(result, "decision")
        assert hasattr(result, "votes")
        assert hasattr(result, "accept_weight")
        assert hasattr(result, "reject_weight")
        assert hasattr(result, "revise_weight")
        assert hasattr(result, "consensus_strength")
        assert hasattr(result, "suggested_revisions")
        assert hasattr(result, "is_valid")

    def test_empty_generator_list(self):
        """Test that empty generator list is handled."""
        # While not a normal use case, ensure it doesn't crash
        validator = ConsensusValidator(
            rule_generators=[],
            consensus_threshold=0.6,
        )

        result = validator.validate_rule(
            rule_text="test.",
            proposer_reasoning="Test",
        )

        # With no generators, total_weight will be 0
        assert result.decision == "revise"
        assert result.consensus_strength == 0.0
        assert len(result.votes) == 0
