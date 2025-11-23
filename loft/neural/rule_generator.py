"""
LLM-based rule generation for symbolic reasoning systems.

This module implements the RuleGenerator class that uses LLMs to generate
candidate ASP rules from natural language descriptions, legal cases, and
identified knowledge gaps.
"""

from typing import List, Optional, Dict, Any
from loguru import logger

from .llm_interface import LLMInterface
from .rule_schemas import (
    GeneratedRule,
    GapFillingResponse,
    ConsensusVote,
)
from .rule_prompts import get_prompt
from loft.symbolic.asp_core import ASPCore


class RuleGenerator:
    """
    Generate ASP rules from natural language using LLMs.

    This class implements Phase 2.1 functionality, converting legal principles,
    case law, and identified gaps into candidate ASP rules for validation.
    """

    def __init__(
        self,
        llm: LLMInterface,
        asp_core: Optional[ASPCore] = None,
        domain: str = "legal",
        prompt_version: str = "latest",
    ):
        """
        Initialize rule generator.

        Args:
            llm: LLM interface for generation
            asp_core: ASP core for validation and context
            domain: Legal domain (e.g., "contract_law", "torts")
            prompt_version: Version of prompts to use ("latest" or specific version)
        """
        self.llm = llm
        self.asp_core = asp_core or ASPCore()
        self.domain = domain
        self.prompt_version = prompt_version
        logger.info(
            f"Initialized RuleGenerator for domain={domain}, prompt_version={prompt_version}"
        )

    def generate_from_principle(
        self,
        principle_text: str,
        existing_predicates: Optional[List[str]] = None,
        constraints: Optional[str] = None,
    ) -> GeneratedRule:
        """
        Generate ASP rule from a legal principle.

        Args:
            principle_text: Natural language statement of legal principle
            existing_predicates: List of predicates that can be referenced
            constraints: Optional constraints on generation

        Returns:
            GeneratedRule with ASP rule and metadata

        Example:
            >>> generator = RuleGenerator(llm, asp_core)
            >>> rule = generator.generate_from_principle(
            ...     "A contract requires offer, acceptance, and consideration."
            ... )
            >>> print(rule.asp_rule)
            valid_contract(C) :- has_offer(C), has_acceptance(C), has_consideration(C).
        """
        logger.info(f"Generating rule from principle: {principle_text[:100]}...")

        # Get existing predicates from ASP core if not provided
        if existing_predicates is None:
            existing_predicates = self._get_existing_predicates()

        # Format prompt
        prompt_template = get_prompt("principle_to_rule", self.prompt_version)
        prompt = prompt_template.format(
            principle_text=principle_text,
            domain=self.domain,
            existing_predicates=self._format_predicate_list(existing_predicates),
            constraints=f"\n**Constraints:** {constraints}" if constraints else "",
        )

        # Query LLM with structured output
        response = self.llm.query(
            question=prompt,
            output_schema=GeneratedRule,
            temperature=0.3,  # Lower temperature for more consistent rule generation
        )

        rule = response.content
        logger.info(f"Generated rule with confidence {rule.confidence:.2f}: {rule.asp_rule[:100]}")

        return rule

    def generate_from_case(
        self,
        case_text: str,
        citation: str,
        jurisdiction: str = "Unknown",
        existing_predicates: Optional[List[str]] = None,
        focus: Optional[str] = None,
    ) -> GeneratedRule:
        """
        Extract ASP rule from case law.

        Args:
            case_text: Excerpt from judicial opinion
            citation: Case citation (e.g., "Smith v. Jones, 123 F.3d 456")
            jurisdiction: Jurisdiction (e.g., "CA", "Federal")
            existing_predicates: List of predicates that can be referenced
            focus: Specific aspect to focus on

        Returns:
            GeneratedRule extracted from case holding

        Example:
            >>> rule = generator.generate_from_case(
            ...     case_text="The court held that part performance...",
            ...     citation="Shaughnessy v. Eidsmo, 23 P.2d 362",
            ...     jurisdiction="Montana"
            ... )
        """
        logger.info(f"Generating rule from case: {citation}")

        if existing_predicates is None:
            existing_predicates = self._get_existing_predicates()

        # Format prompt - fallback to latest if version doesn't exist
        prompt_template = self._get_prompt_safe("case_to_rule")
        prompt = prompt_template.format(
            case_text=case_text,
            citation=citation,
            jurisdiction=jurisdiction,
            domain=self.domain,
            existing_predicates=self._format_predicate_list(existing_predicates),
            focus=f"\n**Focus:** {focus}" if focus else "",
        )

        # Query LLM
        response = self.llm.query(
            question=prompt,
            output_schema=GeneratedRule,
            temperature=0.3,
        )

        rule = response.content
        logger.info(f"Extracted rule from {citation} with confidence {rule.confidence:.2f}")

        return rule

    def fill_knowledge_gap(
        self,
        gap_description: str,
        missing_predicate: str,
        context: Optional[Dict[str, Any]] = None,
        existing_predicates: Optional[List[str]] = None,
    ) -> GapFillingResponse:
        """
        Generate rules to fill an identified knowledge gap.

        This is called when the symbolic core identifies a missing predicate
        or incomplete rule that prevents reasoning.

        Args:
            gap_description: Description of the knowledge gap
            missing_predicate: The specific predicate that needs definition
            context: Additional context about the gap
            existing_predicates: List of available predicates

        Returns:
            GapFillingResponse with multiple candidate rules and recommendation

        Example:
            >>> response = generator.fill_knowledge_gap(
            ...     gap_description="Cannot determine if writing is sufficient",
            ...     missing_predicate="contains_essential_terms(W)",
            ... )
            >>> best_candidate = response.candidates[response.recommended_index]
            >>> print(best_candidate.rule.asp_rule)
        """
        logger.info(f"Filling knowledge gap: {gap_description}")

        if existing_predicates is None:
            existing_predicates = self._get_existing_predicates()

        # Format context
        context_str = ""
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())

        # Format prompt
        prompt_template = get_prompt("gap_filling", self.prompt_version)
        prompt = prompt_template.format(
            gap_description=gap_description,
            missing_predicate=missing_predicate,
            existing_predicates=self._format_predicate_list(existing_predicates),
            context=context_str if context_str else "No additional context provided.",
        )

        # Query LLM
        response = self.llm.query(
            question=prompt,
            output_schema=GapFillingResponse,
            temperature=0.5,  # Higher temperature for diverse candidates
            max_tokens=8192,  # May need more tokens for multiple candidates
        )

        gap_response = response.content

        # Validate recommended index
        if gap_response.recommended_index >= len(gap_response.candidates):
            logger.warning(
                f"Recommended index {gap_response.recommended_index} out of range, using 0"
            )
            gap_response.recommended_index = 0

        logger.info(
            f"Generated {len(gap_response.candidates)} candidates for gap, "
            f"recommended: {gap_response.recommended_index}"
        )

        return gap_response

    def get_consensus_vote(
        self,
        proposed_rule: str,
        proposer_reasoning: str,
        source_type: str = "unknown",
        existing_predicates: Optional[List[str]] = None,
    ) -> ConsensusVote:
        """
        Get consensus vote from LLM on a proposed rule.

        Uses a different LLM instance or prompt to review a rule generated
        by another LLM, enabling multi-LLM consensus validation.

        Args:
            proposed_rule: The ASP rule to evaluate
            proposer_reasoning: Original reasoning for the rule
            source_type: Type of source ("principle", "case", "gap_fill", "refinement")
            existing_predicates: List of available predicates

        Returns:
            ConsensusVote with accept/reject/revise decision

        Example:
            >>> vote = generator.get_consensus_vote(
            ...     proposed_rule="enforceable(C) :- ...",
            ...     proposer_reasoning="This rule captures the statute of frauds..."
            ... )
            >>> if vote.vote == "accept":
            ...     # Add rule to knowledge base
            ...     pass
        """
        logger.info(f"Getting consensus vote on rule: {proposed_rule[:100]}...")

        if existing_predicates is None:
            existing_predicates = self._get_existing_predicates()

        # Format prompt
        prompt_template = get_prompt("consensus_vote", self.prompt_version)
        prompt = prompt_template.format(
            proposed_rule=proposed_rule,
            proposer_reasoning=proposer_reasoning,
            domain=self.domain,
            source_type=source_type,
            existing_predicates=self._format_predicate_list(existing_predicates),
        )

        # Query LLM
        response = self.llm.query(
            question=prompt,
            output_schema=ConsensusVote,
            temperature=0.2,  # Low temperature for consistent evaluation
        )

        vote = response.content
        logger.info(
            f"Consensus vote: {vote.vote} (confidence: {vote.confidence:.2f}), "
            f"issues: {len(vote.issues_found)}"
        )

        return vote

    def refine_rule_from_votes(
        self,
        original_rule: str,
        votes: List[ConsensusVote],
    ) -> GeneratedRule:
        """
        Refine a rule based on consensus votes.

        When a rule receives mixed votes or multiple "revise" suggestions,
        this method synthesizes the feedback into an improved rule.

        Args:
            original_rule: The original proposed rule
            votes: List of consensus votes from multiple LLMs

        Returns:
            GeneratedRule with refined ASP rule

        Example:
            >>> votes = [vote1, vote2, vote3]
            >>> refined = generator.refine_rule_from_votes(original_rule, votes)
        """
        logger.info(f"Refining rule based on {len(votes)} votes")

        # Summarize votes
        accept_count = sum(1 for v in votes if v.vote == "accept")
        reject_count = sum(1 for v in votes if v.vote == "reject")
        revise_count = sum(1 for v in votes if v.vote == "revise")

        votes_summary = f"""
- Accept: {accept_count}
- Reject: {reject_count}
- Revise: {revise_count}
        """.strip()

        # Collect common issues
        all_issues = []
        for vote in votes:
            all_issues.extend(vote.issues_found)

        # Count issue frequency
        from collections import Counter

        issue_counts = Counter(all_issues)
        common_issues = [
            f"- {issue} (mentioned {count}x)" for issue, count in issue_counts.most_common(5)
        ]
        common_issues_str = (
            "\n".join(common_issues) if common_issues else "No common issues identified"
        )

        # Format prompt - fallback to latest if version doesn't exist
        prompt_template = self._get_prompt_safe("refinement")
        prompt = prompt_template.format(
            original_rule=original_rule,
            votes_summary=votes_summary,
            common_issues=common_issues_str,
        )

        # Query LLM
        response = self.llm.query(
            question=prompt,
            output_schema=GeneratedRule,
            temperature=0.3,
        )

        refined_rule = response.content
        logger.info(f"Refined rule with confidence {refined_rule.confidence:.2f}")

        return refined_rule

    def _get_existing_predicates(self) -> List[str]:
        """
        Extract list of existing predicates from ASP core.

        Returns:
            List of predicate signatures (e.g., ["contract/1", "party/1"])
        """
        # TODO: Implement predicate extraction from ASP core
        # For now, return common legal predicates
        return [
            "contract/1",
            "party/1",
            "party_to_contract/2",
            "writing/1",
            "signed_by/2",
            "enforceable/1",
            "unenforceable/1",
            "within_statute/1",
            "has_sufficient_writing/1",
            "exception_applies/1",
            "land_sale_contract/1",
            "goods_sale_contract/1",
            "sale_amount/2",
        ]

    def _format_predicate_list(self, predicates: List[str]) -> str:
        """
        Format predicate list for prompt inclusion.

        Args:
            predicates: List of predicate signatures

        Returns:
            Formatted string for prompt
        """
        if not predicates:
            return "No existing predicates defined."

        return "\n".join(f"- {pred}" for pred in sorted(predicates))

    def validate_rule_syntax(self, asp_rule: str) -> tuple[bool, Optional[str]]:
        """
        Validate ASP syntax of a generated rule.

        Args:
            asp_rule: The ASP rule to validate

        Returns:
            Tuple of (is_valid, error_message)

        Example:
            >>> is_valid, error = generator.validate_rule_syntax(
            ...     "enforceable(C) :- contract(C), not void(C)."
            ... )
            >>> if not is_valid:
            ...     print(f"Syntax error: {error}")
        """
        try:
            import clingo

            # Try to parse the rule with Clingo
            ctl = clingo.Control(["0"])
            ctl.add("base", [], asp_rule)
            ctl.ground([("base", [])])

            return True, None

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"ASP syntax validation failed: {error_msg}")
            return False, error_msg

    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about rule generation.

        Returns:
            Dictionary with generation metrics
        """
        return {
            "total_cost": self.llm.get_total_cost(),
            "total_tokens": self.llm.get_total_tokens(),
            "domain": self.domain,
            "prompt_version": self.prompt_version,
        }

    def _get_prompt_safe(self, template_name: str) -> str:
        """
        Get prompt template with safe fallback.

        If the requested version doesn't exist for this template,
        falls back to 'latest'.

        Args:
            template_name: Name of the prompt template

        Returns:
            Prompt template string
        """
        try:
            return get_prompt(template_name, self.prompt_version)
        except KeyError:
            logger.warning(
                f"Version {self.prompt_version} not found for {template_name}, using 'latest'"
            )
            return get_prompt(template_name, "latest")
