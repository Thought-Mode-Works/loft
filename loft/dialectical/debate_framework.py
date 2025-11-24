"""
Multi-agent debate framework for dialectical rule refinement (Phase 4.2).

Orchestrates thesis-antithesis-synthesis cycles between generator, critic,
and synthesizer agents to iteratively improve rules.
"""

from typing import List

from loguru import logger

from loft.dialectical.critic import CriticSystem
from loft.dialectical.debate_schemas import (
    DebateArgument,
    DebateContext,
    DebatePhase,
    DebateRound,
    DialecticalCycleResult,
)
from loft.dialectical.synthesizer import Synthesizer
from loft.neural.rule_generator import RuleGenerator
from loft.neural.rule_schemas import GeneratedRule


class DebateFramework:
    """
    Orchestrates multi-round dialectical debates between LLM agents.

    Architecture:
    - Generator (thesis): Proposes rules
    - Critic (antithesis): Identifies flaws, edge cases, contradictions
    - Synthesizer (synthesis): Combines insights into improved rules

    The framework runs iterative cycles until convergence or max rounds.
    """

    def __init__(
        self,
        generator: RuleGenerator,
        critic: CriticSystem,
        synthesizer: Synthesizer,
        max_rounds: int = 3,
        convergence_threshold: float = 0.85,
    ):
        """
        Initialize debate framework.

        Args:
            generator: Rule generator agent (thesis)
            critic: Critic system agent (antithesis)
            synthesizer: Synthesizer agent (synthesis)
            max_rounds: Maximum debate rounds before stopping
            convergence_threshold: Similarity threshold for convergence (0-1)
        """
        self.generator = generator
        self.critic = critic
        self.synthesizer = synthesizer
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.debate_history: List[DialecticalCycleResult] = []

        logger.info(
            f"Initialized DebateFramework (max_rounds={max_rounds}, "
            f"convergence_threshold={convergence_threshold})"
        )

    def run_dialectical_cycle(
        self,
        context: DebateContext,
    ) -> DialecticalCycleResult:
        """
        Run complete dialectical cycle with multiple rounds.

        Args:
            context: Debate context with gap description, existing rules, etc.

        Returns:
            DialecticalCycleResult with final rule and debate transcript

        Example:
            >>> framework = DebateFramework(generator, critic, synthesizer)
            >>> context = DebateContext(
            ...     knowledge_gap_description="Rules for contract enforceability",
            ...     existing_rules=[],
            ...     existing_predicates=["contract", "signed"],
            ... )
            >>> result = framework.run_dialectical_cycle(context)
            >>> print(result.final_rule.asp_rule)
        """
        logger.info(f"Starting dialectical cycle: {context.knowledge_gap_description[:80]}...")

        # Round 0: Initial thesis from generator
        logger.info("Round 0: Generating initial thesis")
        initial_proposal = self._generate_initial_thesis(context)

        debate_rounds: List[DebateRound] = []
        current_thesis = initial_proposal
        converged = False
        convergence_reason = None

        # Run debate rounds
        for round_num in range(1, context.max_rounds + 1):
            logger.info(f"Round {round_num}: Running dialectical cycle")

            # Antithesis: Critic analyzes thesis
            logger.info(f"Round {round_num}: Critic analyzing thesis")
            critique = self.critic.critique_rule(
                current_thesis,
                context.existing_rules,
                context=context.domain,
            )

            antithesis_arg = DebateArgument(
                speaker="critic",
                content=f"Identified {len(critique.issues)} issues, "
                f"{len(critique.edge_cases)} edge cases, "
                f"{len(critique.contradictions)} contradictions. "
                f"Recommendation: {critique.recommendation}",
                references=[current_thesis.asp_rule],
                confidence=critique.confidence,
            )

            # Check if critique recommends acceptance (early convergence)
            if critique.recommendation == "accept":
                logger.info(f"Round {round_num}: Critic recommends acceptance. Converging early.")
                debate_round = DebateRound(
                    round_number=round_num,
                    thesis=current_thesis,
                    thesis_argument=DebateArgument(
                        speaker="generator",
                        content=f"Proposed: {current_thesis.asp_rule}",
                        confidence=current_thesis.confidence,
                    ),
                    antithesis=critique,
                    antithesis_argument=antithesis_arg,
                    synthesis=current_thesis,
                    synthesis_argument=DebateArgument(
                        speaker="synthesizer",
                        content="No synthesis needed. Thesis accepted by critic.",
                        confidence=current_thesis.confidence,
                    ),
                    convergence_score=1.0,
                    phase=DebatePhase.CONVERGED,
                )
                debate_rounds.append(debate_round)
                converged = True
                convergence_reason = "Critic accepted thesis"
                break

            # Synthesis: Synthesizer combines thesis and antithesis
            logger.info(f"Round {round_num}: Synthesizer creating synthesis")
            synthesis, synthesis_arg = self.synthesizer.synthesize(
                current_thesis,
                critique,
                context.existing_rules,
                context=context.domain,
            )

            # Calculate convergence score
            convergence_score = self._calculate_convergence(current_thesis, synthesis)

            debate_round = DebateRound(
                round_number=round_num,
                thesis=current_thesis,
                thesis_argument=DebateArgument(
                    speaker="generator",
                    content=f"Proposed: {current_thesis.asp_rule}. Reasoning: {current_thesis.reasoning}",
                    confidence=current_thesis.confidence,
                ),
                antithesis=critique,
                antithesis_argument=antithesis_arg,
                synthesis=synthesis,
                synthesis_argument=synthesis_arg,
                convergence_score=convergence_score,
                improvement_score=synthesis.confidence - current_thesis.confidence,
                phase=DebatePhase.SYNTHESIS,
            )
            debate_rounds.append(debate_round)

            # Check convergence
            if convergence_score >= self.convergence_threshold:
                logger.info(
                    f"Round {round_num}: Converged (score: {convergence_score:.2f} >= "
                    f"{self.convergence_threshold})"
                )
                converged = True
                convergence_reason = f"Convergence score {convergence_score:.2f} exceeded threshold"
                break

            # Update thesis for next round
            current_thesis = synthesis

        # If didn't converge, use last synthesis
        if not converged:
            logger.info(
                f"Did not converge after {context.max_rounds} rounds. "
                f"Using last synthesis as final rule."
            )
            convergence_reason = f"Max rounds ({context.max_rounds}) reached"

        final_rule = debate_rounds[-1].synthesis if debate_rounds else initial_proposal

        # Calculate overall improvement
        improvement_score = final_rule.confidence - initial_proposal.confidence

        # Build transcript
        transcript = []
        for round_item in debate_rounds:
            transcript.append(round_item.thesis_argument)
            transcript.append(round_item.antithesis_argument)
            if round_item.synthesis_argument:
                transcript.append(round_item.synthesis_argument)

        result = DialecticalCycleResult(
            initial_proposal=initial_proposal,
            final_rule=final_rule,
            debate_rounds=debate_rounds,
            total_rounds=len(debate_rounds),
            converged=converged,
            convergence_reason=convergence_reason,
            improvement_score=improvement_score,
            debate_transcript=transcript,
            metadata={
                "max_rounds": context.max_rounds,
                "convergence_threshold": self.convergence_threshold,
                "knowledge_gap": context.knowledge_gap_description,
            },
        )

        # Store in history
        self.debate_history.append(result)

        logger.info(
            f"Dialectical cycle complete. Rounds: {result.total_rounds}, "
            f"Converged: {result.converged}, Improvement: {result.improvement_score:+.2f}"
        )

        return result

    def _generate_initial_thesis(self, context: DebateContext) -> GeneratedRule:
        """Generate initial thesis from knowledge gap."""
        logger.debug("Generating initial thesis from knowledge gap")

        # Use generator to create initial proposal
        response = self.generator.fill_knowledge_gap(
            gap_description=context.knowledge_gap_description,
            existing_rules="\n".join(context.existing_rules),
            existing_predicates=context.existing_predicates,
            target_layer=context.target_layer,
        )

        # Extract recommended rule from response
        if response.candidates and len(response.candidates) > response.recommended_index:
            thesis = response.candidates[response.recommended_index].rule
        else:
            # Fallback: create simple rule
            thesis = GeneratedRule(
                asp_rule=f"rule :- {context.existing_predicates[0]}."
                if context.existing_predicates
                else "rule.",
                confidence=0.3,
                reasoning="Fallback rule due to empty generation response",
                predicates_used=context.existing_predicates[:1]
                if context.existing_predicates
                else [],
                source_type="gap_fill",
                source_text=context.knowledge_gap_description,
            )

        logger.info(f"Initial thesis: {thesis.asp_rule}")
        return thesis

    def _calculate_convergence(self, thesis: GeneratedRule, synthesis: GeneratedRule) -> float:
        """
        Calculate convergence score between thesis and synthesis.

        Returns:
            Float between 0.0 (completely different) and 1.0 (identical)
        """
        # Simple heuristic: compare ASP rules as strings
        if thesis.asp_rule == synthesis.asp_rule:
            return 1.0

        # Calculate similarity based on shared tokens
        thesis_tokens = set(thesis.asp_rule.split())
        synthesis_tokens = set(synthesis.asp_rule.split())

        if not thesis_tokens and not synthesis_tokens:
            return 1.0
        if not thesis_tokens or not synthesis_tokens:
            return 0.0

        intersection = thesis_tokens & synthesis_tokens
        union = thesis_tokens | synthesis_tokens

        jaccard = len(intersection) / len(union) if union else 0.0

        logger.debug(f"Convergence score: {jaccard:.2f} (Jaccard similarity)")
        return jaccard

    def get_debate_history(self) -> List[DialecticalCycleResult]:
        """Get history of all debates run by this framework."""
        return self.debate_history

    def get_debate_transcript(self, result: DialecticalCycleResult) -> str:
        """
        Get human-readable transcript of debate.

        Args:
            result: The debate result to format

        Returns:
            Formatted transcript string
        """
        lines = [
            "=" * 80,
            "DIALECTICAL DEBATE TRANSCRIPT",
            "=" * 80,
            f"Knowledge Gap: {result.metadata.get('knowledge_gap', 'N/A')}",
            f"Total Rounds: {result.total_rounds}",
            f"Converged: {result.converged}",
            f"Improvement: {result.improvement_score:+.2f}",
            "",
            "INITIAL PROPOSAL",
            "-" * 80,
            f"Rule: {result.initial_proposal.asp_rule}",
            f"Reasoning: {result.initial_proposal.reasoning}",
            f"Confidence: {result.initial_proposal.confidence:.2f}",
            "",
        ]

        for round_item in result.debate_rounds:
            lines.extend(
                [
                    f"ROUND {round_item.round_number}",
                    "-" * 80,
                    f"Phase: {round_item.phase.value}",
                    "",
                    "Thesis:",
                    f"  {round_item.thesis.asp_rule}",
                    f"  Argument: {round_item.thesis_argument.content}",
                    f"  Confidence: {round_item.thesis_argument.confidence:.2f}",
                    "",
                    "Antithesis:",
                    f"  Issues: {len(round_item.antithesis.issues)}",
                    f"  Recommendation: {round_item.antithesis.recommendation}",
                    f"  Argument: {round_item.antithesis_argument.content}",
                    "",
                ]
            )

            if round_item.synthesis:
                lines.extend(
                    [
                        "Synthesis:",
                        f"  {round_item.synthesis.asp_rule}",
                        f"  Argument: {round_item.synthesis_argument.content if round_item.synthesis_argument else 'N/A'}",
                        f"  Convergence Score: {round_item.convergence_score:.2f}",
                        "",
                    ]
                )

        lines.extend(
            [
                "FINAL RESULT",
                "-" * 80,
                f"Rule: {result.final_rule.asp_rule}",
                f"Confidence: {result.final_rule.confidence:.2f}",
                f"Reasoning: {result.final_rule.reasoning}",
                f"Convergence Reason: {result.convergence_reason}",
                "=" * 80,
            ]
        )

        return "\n".join(lines)
