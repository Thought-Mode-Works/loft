"""
Consensus validator for LLM-generated ASP rules.

This module validates rules using multi-LLM consensus voting,
aggregating opinions from multiple models for robust validation.
"""

from typing import List, Optional
from loguru import logger

from loft.validation.validation_schemas import ConsensusValidationResult
from loft.neural.rule_generator import RuleGenerator
from loft.neural.rule_schemas import ConsensusVote


class ConsensusValidator:
    """
    Validates rules using multi-LLM consensus.

    Uses multiple LLM instances to vote on rule quality,
    aggregating opinions for robust validation.
    """

    def __init__(
        self,
        rule_generators: List[RuleGenerator],
        weights: Optional[List[float]] = None,
        consensus_threshold: float = 0.6,
    ):
        """
        Initialize consensus validator.

        Args:
            rule_generators: List of RuleGenerator instances (different models/prompts)
            weights: Optional weights for each generator's vote (defaults to equal)
            consensus_threshold: Minimum consensus strength required (0.0-1.0)
        """
        self.rule_generators = rule_generators
        self.weights = weights or [1.0] * len(rule_generators)
        self.consensus_threshold = consensus_threshold

        if len(self.weights) != len(self.rule_generators):
            raise ValueError("Number of weights must match number of generators")

        logger.info(
            f"Initialized ConsensusValidator with {len(rule_generators)} generators, "
            f"threshold={consensus_threshold}"
        )

    def validate_rule(
        self,
        rule_text: str,
        proposer_reasoning: str,
        source_type: str = "generated",
        existing_predicates: Optional[List[str]] = None,
    ) -> ConsensusValidationResult:
        """
        Validate rule using multi-LLM consensus.

        Args:
            rule_text: The ASP rule to validate
            proposer_reasoning: Original reasoning for the rule
            source_type: Type of source (principle/case/gap_fill/refinement)
            existing_predicates: List of existing predicates

        Returns:
            ConsensusValidationResult with aggregated votes

        Example:
            >>> validator = ConsensusValidator([generator1, generator2])
            >>> result = validator.validate_rule(
            ...     "enforceable(C) :- contract(C), not void(C).",
            ...     "Contract is enforceable unless void"
            ... )
            >>> assert result.decision in ["accept", "reject", "revise"]
        """
        votes = []

        # Collect votes from each generator
        for i, generator in enumerate(self.rule_generators):
            try:
                vote = generator.get_consensus_vote(
                    proposed_rule=rule_text,
                    proposer_reasoning=proposer_reasoning,
                    source_type=source_type,
                    existing_predicates=existing_predicates,
                )
                votes.append(vote)
                logger.debug(
                    f"Generator {i} voted: {vote.vote} (confidence: {vote.confidence:.2f})"
                )

            except Exception as e:
                logger.warning(f"Generator {i} failed to vote: {e}")
                # Create abstain vote
                votes.append(
                    ConsensusVote(
                        vote="revise",
                        confidence=0.0,
                        reasoning=f"Vote failed: {str(e)}",
                        issues_found=["voting_error"],
                        suggested_revisions=[],
                        voter_id=f"generator_{i}",
                    )
                )

        # Aggregate votes
        accept_weight = sum(
            self.weights[i] * vote.confidence
            for i, vote in enumerate(votes)
            if vote.vote == "accept"
        )

        reject_weight = sum(
            self.weights[i] * vote.confidence
            for i, vote in enumerate(votes)
            if vote.vote == "reject"
        )

        revise_weight = sum(
            self.weights[i] * vote.confidence
            for i, vote in enumerate(votes)
            if vote.vote == "revise"
        )

        total_weight = accept_weight + reject_weight + revise_weight

        # Determine consensus decision
        if total_weight == 0:
            decision = "revise"
            consensus_strength = 0.0
        else:
            max_weight = max(accept_weight, reject_weight, revise_weight)
            consensus_strength = max_weight / total_weight

            if accept_weight == max_weight:
                decision = "accept"
            elif reject_weight == max_weight:
                decision = "reject"
            else:
                decision = "revise"

        # Collect suggested revisions
        suggested_revisions = []
        for vote in votes:
            suggested_revisions.extend(vote.suggested_revisions)

        # Check if valid (accept with sufficient consensus)
        is_valid = (
            decision == "accept" and consensus_strength >= self.consensus_threshold
        )

        return ConsensusValidationResult(
            decision=decision,
            votes=votes,
            accept_weight=accept_weight,
            reject_weight=reject_weight,
            revise_weight=revise_weight,
            consensus_strength=consensus_strength,
            suggested_revisions=suggested_revisions,
            is_valid=is_valid,
        )

    def validate_batch(
        self,
        rules: List[str],
        reasonings: List[str],
        source_type: str = "generated",
        existing_predicates: Optional[List[str]] = None,
    ) -> List[ConsensusValidationResult]:
        """
        Validate multiple rules using consensus.

        Args:
            rules: List of rules to validate
            reasonings: List of reasoning for each rule
            source_type: Source type for all rules
            existing_predicates: Existing predicates

        Returns:
            List of ConsensusValidationResult objects
        """
        if len(rules) != len(reasonings):
            raise ValueError("Number of rules must match number of reasonings")

        results = []
        for rule, reasoning in zip(rules, reasonings):
            result = self.validate_rule(
                rule, reasoning, source_type, existing_predicates
            )
            results.append(result)

        return results
