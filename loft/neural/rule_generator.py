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
    validate_asp_rule_completeness,
)
from .rule_prompts import get_prompt
from .errors import LLMParsingError
from .rule_generalizer import RuleGeneralizer, GeneralizationResult
from loft.symbolic.asp_core import ASPCore


# Default and escalation token limits for rule generation
DEFAULT_MAX_TOKENS = 4096
RETRY_MAX_TOKENS = 8192  # Increased limit for retry attempts
MAX_GENERATION_RETRIES = 3  # Maximum retry attempts for truncation


class RuleGenerationError(Exception):
    """Raised when rule generation fails after all retries."""

    def __init__(self, message: str, attempts: int, last_error: Optional[str] = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


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
        max_retries: int = MAX_GENERATION_RETRIES,
    ) -> GeneratedRule:
        """
        Generate ASP rule from a legal principle.

        Implements retry logic with increased token limits to handle truncation.
        Validates that generated rules are complete before returning.

        Args:
            principle_text: Natural language statement of legal principle
            existing_predicates: List of predicates that can be referenced
            constraints: Optional constraints on generation
            max_retries: Maximum number of retry attempts for truncated rules

        Returns:
            GeneratedRule with ASP rule and metadata

        Raises:
            RuleGenerationError: If rule generation fails after all retries

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

        # Format prompt with explicit instruction for complete rules
        prompt_template = get_prompt("principle_to_rule", self.prompt_version)
        prompt = prompt_template.format(
            principle_text=principle_text,
            domain=self.domain,
            existing_predicates=self._format_predicate_list(existing_predicates),
            constraints=f"\n**Constraints:** {constraints}" if constraints else "",
        )

        # Add explicit completeness instruction to prompt
        completeness_instruction = (
            "\n\n**CRITICAL:** The ASP rule MUST be complete and well-formed. "
            "Ensure the rule:\n"
            "1. Ends with a period '.'\n"
            "2. Has balanced parentheses\n"
            "3. Has a complete rule body (no trailing underscores or incomplete predicates)\n"
            "4. Is valid Clingo ASP syntax\n"
        )
        prompt = prompt + completeness_instruction

        last_error: Optional[str] = None

        for attempt in range(max_retries):
            # Escalate max_tokens on retries
            current_max_tokens = DEFAULT_MAX_TOKENS if attempt == 0 else RETRY_MAX_TOKENS

            try:
                logger.debug(
                    f"Generation attempt {attempt + 1}/{max_retries}, "
                    f"max_tokens={current_max_tokens}"
                )

                # Query LLM with structured output
                response = self.llm.query(
                    question=prompt,
                    output_schema=GeneratedRule,
                    temperature=0.3,  # Lower temperature for more consistent rule generation
                    max_tokens=current_max_tokens,
                )

                rule = response.content

                # Double-check completeness (schema validation should catch this,
                # but we verify explicitly for safety)
                is_valid, error_msg = validate_asp_rule_completeness(rule.asp_rule)
                if not is_valid:
                    logger.warning(f"Generated rule failed completeness check: {error_msg}")
                    last_error = error_msg
                    continue  # Retry

                logger.info(
                    f"Generated rule with confidence {rule.confidence:.2f}: {rule.asp_rule[:100]}"
                )
                return rule

            except LLMParsingError as e:
                logger.warning(f"LLM parsing error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                continue  # Retry

            except ValueError as e:
                # Pydantic validation error (including our ASP validation)
                logger.warning(f"Validation error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                continue  # Retry

        # All retries exhausted
        error_message = (
            f"Failed to generate complete ASP rule after {max_retries} attempts. "
            f"Last error: {last_error}"
        )
        logger.error(error_message)
        raise RuleGenerationError(
            message=error_message,
            attempts=max_retries,
            last_error=last_error,
        )

    def generate_from_case(
        self,
        case_text: str,
        citation: str,
        jurisdiction: str = "Unknown",
        existing_predicates: Optional[List[str]] = None,
        focus: Optional[str] = None,
        max_retries: int = MAX_GENERATION_RETRIES,
    ) -> GeneratedRule:
        """
        Extract ASP rule from case law.

        Implements retry logic with increased token limits to handle truncation.
        Validates that generated rules are complete before returning.

        Args:
            case_text: Excerpt from judicial opinion
            citation: Case citation (e.g., "Smith v. Jones, 123 F.3d 456")
            jurisdiction: Jurisdiction (e.g., "CA", "Federal")
            existing_predicates: List of predicates that can be referenced
            focus: Specific aspect to focus on
            max_retries: Maximum number of retry attempts for truncated rules

        Returns:
            GeneratedRule extracted from case holding

        Raises:
            RuleGenerationError: If rule generation fails after all retries

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

        # Add explicit completeness instruction to prompt
        completeness_instruction = (
            "\n\n**CRITICAL:** The ASP rule MUST be complete and well-formed. "
            "Ensure the rule:\n"
            "1. Ends with a period '.'\n"
            "2. Has balanced parentheses\n"
            "3. Has a complete rule body (no trailing underscores or incomplete predicates)\n"
            "4. Is valid Clingo ASP syntax\n"
        )
        prompt = prompt + completeness_instruction

        last_error: Optional[str] = None

        for attempt in range(max_retries):
            # Escalate max_tokens on retries
            current_max_tokens = DEFAULT_MAX_TOKENS if attempt == 0 else RETRY_MAX_TOKENS

            try:
                logger.debug(
                    f"Case rule generation attempt {attempt + 1}/{max_retries}, "
                    f"max_tokens={current_max_tokens}"
                )

                # Query LLM
                response = self.llm.query(
                    question=prompt,
                    output_schema=GeneratedRule,
                    temperature=0.3,
                    max_tokens=current_max_tokens,
                )

                rule = response.content

                # Double-check completeness
                is_valid, error_msg = validate_asp_rule_completeness(rule.asp_rule)
                if not is_valid:
                    logger.warning(f"Generated rule failed completeness check: {error_msg}")
                    last_error = error_msg
                    continue  # Retry

                logger.info(f"Extracted rule from {citation} with confidence {rule.confidence:.2f}")
                return rule

            except LLMParsingError as e:
                logger.warning(f"LLM parsing error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                continue  # Retry

            except ValueError as e:
                # Pydantic validation error (including our ASP validation)
                logger.warning(f"Validation error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                continue  # Retry

        # All retries exhausted
        error_message = (
            f"Failed to generate complete ASP rule from case {citation} "
            f"after {max_retries} attempts. Last error: {last_error}"
        )
        logger.error(error_message)
        raise RuleGenerationError(
            message=error_message,
            attempts=max_retries,
            last_error=last_error,
        )

    def fill_knowledge_gap(
        self,
        gap_description: str,
        missing_predicate: str,
        context: Optional[Dict[str, Any]] = None,
        existing_predicates: Optional[List[str]] = None,
        dataset_predicates: Optional[List[str]] = None,
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
            dataset_predicates: List of predicates extracted from case facts for
                alignment (issue #166). When provided, generated rules will be
                constrained to use these exact predicates.

        Returns:
            GapFillingResponse with multiple candidate rules and recommendation

        Example:
            >>> response = generator.fill_knowledge_gap(
            ...     gap_description="Cannot determine if writing is sufficient",
            ...     missing_predicate="contains_essential_terms(W)",
            ...     dataset_predicates=["contract(X)", "amount(X, A)", "party(X, P)"],
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

        # Format dataset predicates section (issue #166)
        dataset_predicates_section = ""
        if dataset_predicates:
            formatted_predicates = "\n".join(f"  - {p}" for p in dataset_predicates)
            dataset_predicates_section = f"""
**Dataset Predicates (from case facts - USE THESE EXACTLY):**
{formatted_predicates}

"""

        # Format prompt
        prompt_template = get_prompt("gap_filling", self.prompt_version)
        prompt = prompt_template.format(
            gap_description=gap_description,
            missing_predicate=missing_predicate,
            existing_predicates=self._format_predicate_list(existing_predicates),
            context=context_str if context_str else "No additional context provided.",
            dataset_predicates_section=dataset_predicates_section,
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

    def repair_malformed_rule(
        self,
        malformed_rule: str,
        error_message: str,
        original_context: str,
        available_predicates: Optional[List[str]] = None,
    ) -> GeneratedRule:
        """
        Attempt to repair a malformed or truncated ASP rule.

        Uses a specialized repair prompt to ask the LLM to fix syntax errors,
        complete truncated rules, and align predicates with the dataset.

        Args:
            malformed_rule: The broken/truncated ASP rule
            error_message: The validation error message
            original_context: The original principle or context for the rule
            available_predicates: List of predicates from the dataset

        Returns:
            GeneratedRule with the repaired ASP rule

        Raises:
            RuleGenerationError: If repair fails
        """
        logger.info(f"Attempting to repair malformed rule: {malformed_rule[:50]}...")

        # Format available predicates
        predicates_str = (
            self._format_predicate_list(available_predicates)
            if available_predicates
            else "No predicate list available."
        )

        # Get repair prompt
        prompt_template = get_prompt("rule_repair", "latest")
        prompt = prompt_template.format(
            malformed_rule=malformed_rule,
            error_message=error_message,
            original_context=original_context,
            available_predicates=predicates_str,
        )

        try:
            response = self.llm.query(
                question=prompt,
                output_schema=GeneratedRule,
                temperature=0.2,  # Low temperature for consistent repair
                max_tokens=RETRY_MAX_TOKENS,
            )

            repaired_rule = response.content

            # Validate the repaired rule
            is_valid, validation_error = validate_asp_rule_completeness(repaired_rule.asp_rule)
            if not is_valid:
                logger.warning(f"Repaired rule still invalid: {validation_error}")
                raise RuleGenerationError(
                    message=f"Rule repair failed: {validation_error}",
                    attempts=1,
                    last_error=validation_error,
                )

            logger.info(
                f"Successfully repaired rule with confidence {repaired_rule.confidence:.2f}"
            )
            return repaired_rule

        except Exception as e:
            logger.error(f"Rule repair failed: {e}")
            raise RuleGenerationError(
                message=f"Rule repair failed: {e}",
                attempts=1,
                last_error=str(e),
            )

    def generate_from_principle_aligned(
        self,
        principle_text: str,
        dataset_predicates: List[str],
        max_retries: int = MAX_GENERATION_RETRIES,
    ) -> GeneratedRule:
        """
        Generate ASP rule aligned with dataset predicates.

        Uses a specialized prompt that includes example predicates from the
        dataset to ensure the generated rule uses matching predicate formats.

        Args:
            principle_text: Natural language statement of legal principle
            dataset_predicates: List of predicates extracted from dataset facts
            max_retries: Maximum number of retry attempts

        Returns:
            GeneratedRule with aligned predicates

        Raises:
            RuleGenerationError: If rule generation fails after all retries
        """
        logger.info(f"Generating aligned rule from principle: {principle_text[:100]}...")

        # Format dataset predicates as examples
        predicates_str = "\n".join(f"- {pred}" for pred in dataset_predicates[:20])

        # Get aligned prompt
        prompt_template = get_prompt("aligned_principle_to_rule", "latest")
        prompt = prompt_template.format(
            principle_text=principle_text,
            domain=self.domain,
            dataset_predicates=predicates_str,
        )

        last_error: Optional[str] = None
        last_malformed_rule: Optional[str] = None

        for attempt in range(max_retries):
            current_max_tokens = DEFAULT_MAX_TOKENS if attempt == 0 else RETRY_MAX_TOKENS

            try:
                logger.debug(
                    f"Aligned generation attempt {attempt + 1}/{max_retries}, "
                    f"max_tokens={current_max_tokens}"
                )

                response = self.llm.query(
                    question=prompt,
                    output_schema=GeneratedRule,
                    temperature=0.3,
                    max_tokens=current_max_tokens,
                )

                rule = response.content

                # Validate completeness
                is_valid, error_msg = validate_asp_rule_completeness(rule.asp_rule)
                if not is_valid:
                    logger.warning(f"Generated rule failed completeness check: {error_msg}")
                    last_error = error_msg
                    last_malformed_rule = rule.asp_rule
                    continue

                logger.info(
                    f"Generated aligned rule with confidence {rule.confidence:.2f}: "
                    f"{rule.asp_rule[:100]}"
                )
                return rule

            except LLMParsingError as e:
                logger.warning(f"LLM parsing error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                continue

            except ValueError as e:
                logger.warning(f"Validation error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                # Try to extract the malformed rule from the error
                if hasattr(e, "__context__") and e.__context__:
                    last_malformed_rule = str(e.__context__)
                continue

        # All retries exhausted - try repair as last resort
        if last_malformed_rule:
            logger.info("Attempting rule repair as last resort...")
            try:
                return self.repair_malformed_rule(
                    malformed_rule=last_malformed_rule,
                    error_message=last_error or "Unknown error",
                    original_context=principle_text,
                    available_predicates=dataset_predicates,
                )
            except RuleGenerationError:
                pass  # Fall through to raise error

        error_message = (
            f"Failed to generate aligned ASP rule after {max_retries} attempts. "
            f"Last error: {last_error}"
        )
        logger.error(error_message)
        raise RuleGenerationError(
            message=error_message,
            attempts=max_retries,
            last_error=last_error,
        )

    def generalize_rule(
        self,
        generated_rule: GeneratedRule,
        facts: Optional[str] = None,
        known_entities: Optional[set] = None,
    ) -> tuple[GeneratedRule, GeneralizationResult]:
        """
        Generalize a generated rule to improve coverage.

        Replaces specific entity constants (like 'alice', 'c1') with anonymous
        variables while preserving intentional constants (like 'yes', 'no', 'land').

        Args:
            generated_rule: The rule to generalize
            facts: ASP facts to extract entity names from (auto-extracts if provided)
            known_entities: Explicit set of entity names to generalize

        Returns:
            Tuple of (modified GeneratedRule, GeneralizationResult)

        Example:
            >>> rule = generator.generate_from_principle("...")
            >>> generalized, result = generator.generalize_rule(
            ...     rule, facts="claim(c1). claimant(c1, alice)."
            ... )
            >>> # Original: "enforceable(X) :- claimant(X, alice)."
            >>> # Generalized: "enforceable(X) :- claimant(X, _)."
        """
        from .rule_generalizer import extract_entities_from_facts

        # Extract entities from facts if not provided
        if known_entities is None and facts:
            known_entities = extract_entities_from_facts(facts)

        generalizer = RuleGeneralizer(known_entities=known_entities)
        result = generalizer.generalize(generated_rule.asp_rule)

        if result.was_modified:
            logger.info(
                f"Generalized rule: {len(result.changes)} changes made - "
                f"{', '.join(result.changes[:3])}"
            )
            # Create new GeneratedRule with generalized ASP
            generalized_rule = GeneratedRule(
                asp_rule=result.generalized_rule,
                confidence=generated_rule.confidence,
                reasoning=generated_rule.reasoning
                + f"\n[Generalized: {', '.join(result.changes)}]",
                predicates_used=generated_rule.predicates_used,
                predicates_defined=generated_rule.predicates_defined,
            )
            return generalized_rule, result
        else:
            logger.debug("Rule unchanged after generalization attempt")
            return generated_rule, result

    def generate_and_generalize(
        self,
        principle_text: str,
        facts: str,
        existing_predicates: Optional[List[str]] = None,
        constraints: Optional[str] = None,
    ) -> tuple[GeneratedRule, GeneralizationResult]:
        """
        Generate a rule from a principle and automatically generalize it.

        Combines generate_from_principle and generalize_rule into a single
        operation for convenience.

        Args:
            principle_text: Natural language statement of legal principle
            facts: ASP facts from the dataset (used to identify entities)
            existing_predicates: List of predicates that can be referenced
            constraints: Optional constraints on generation

        Returns:
            Tuple of (GeneratedRule with generalized ASP, GeneralizationResult)

        Example:
            >>> rule, gen_result = generator.generate_and_generalize(
            ...     "Adverse possession requires continuous occupation",
            ...     facts="claim(c1). claimant(c1, alice). occupation_years(c1, 25)."
            ... )
        """
        # Generate initial rule
        rule = self.generate_from_principle(
            principle_text=principle_text,
            existing_predicates=existing_predicates,
            constraints=constraints,
        )

        # Generalize it
        return self.generalize_rule(rule, facts=facts)


def extract_predicates_from_asp_facts(asp_facts: str) -> List[str]:
    """
    Extract predicate patterns from ASP facts.

    Parses ASP facts and extracts predicate names with their arities and
    example arguments for use in aligned rule generation.

    Args:
        asp_facts: String containing ASP facts (e.g., "contract(c1). party(c1, alice).")

    Returns:
        List of predicate patterns (e.g., ["contract(X)", "party(X, Y)"])

    Example:
        >>> facts = "claim(c1). occupation_years(c1, 22). occupation_continuous(c1, yes)."
        >>> predicates = extract_predicates_from_asp_facts(facts)
        >>> print(predicates)
        ['claim(X)', 'occupation_years(X, N)', 'occupation_continuous(X, V)']
    """
    import re

    predicates = set()

    # Pattern to match predicates: name(args).
    predicate_pattern = r"([a-z_][a-z0-9_]*)\(([^)]*)\)"

    for match in re.finditer(predicate_pattern, asp_facts):
        pred_name = match.group(1)
        args = match.group(2)

        # Count arguments
        if args.strip():
            arg_list = [a.strip() for a in args.split(",")]
            arity = len(arg_list)

            # Create variable pattern based on arity
            if arity == 1:
                predicates.add(f"{pred_name}(X)")
            elif arity == 2:
                # Try to infer argument types
                second_arg = arg_list[1]
                if second_arg in ("yes", "no"):
                    predicates.add(f"{pred_name}(X, yes/no)")
                elif second_arg.isdigit():
                    predicates.add(f"{pred_name}(X, N)")
                else:
                    predicates.add(f"{pred_name}(X, Y)")
            else:
                vars_str = ", ".join([chr(ord("X") + i) for i in range(min(arity, 3))])
                if arity > 3:
                    vars_str += ", ..."
                predicates.add(f"{pred_name}({vars_str})")
        else:
            predicates.add(f"{pred_name}")

    return sorted(predicates)
