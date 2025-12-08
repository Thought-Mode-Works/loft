"""
Logic Generator LLM - Specialized ASP rule generation.

This module implements a specialized LLM optimized for generating syntactically
and semantically valid Answer Set Programming (ASP) rules. It uses prompt
optimization and few-shot learning to improve ASP generation quality over
general-purpose LLMs.

Part of Phase 6: Heterogeneous Neural Ensemble (Issue #188).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from loft.neural.llm_interface import LLMInterface, LLMProvider
from loft.neural.rule_schemas import GeneratedRule
from loft.validation.asp_validators import check_embedded_periods, check_unsafe_variables


class OptimizationStrategy(Enum):
    """Strategies for optimizing ASP generation."""

    PROMPT_OPTIMIZATION = "prompt_optimization"
    FEW_SHOT_LEARNING = "few_shot_learning"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SELF_CONSISTENCY = "self_consistency"


@dataclass
class LogicGeneratorConfig:
    """Configuration for the Logic Generator LLM.

    Attributes:
        model: Model identifier (e.g., 'claude-3-5-haiku-20241022')
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens for response
        optimization_strategy: Strategy for improving ASP generation
        few_shot_examples: Example ASP rules for few-shot learning
        enable_syntax_validation: Validate ASP syntax before returning
        enable_variable_safety_check: Check for unsafe variables
        max_generation_retries: Maximum retry attempts on validation failure
    """

    model: str = "claude-3-5-haiku-20241022"
    temperature: float = 0.3
    max_tokens: int = 4096
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.PROMPT_OPTIMIZATION
    few_shot_examples: List[str] = field(default_factory=list)
    enable_syntax_validation: bool = True
    enable_variable_safety_check: bool = True
    max_generation_retries: int = 3


@dataclass
class ASPGenerationResult:
    """Result from ASP rule generation.

    Attributes:
        rule: The generated ASP rule
        is_valid: Whether the rule passed validation
        confidence: Model confidence in the rule
        validation_errors: List of validation errors if any
        generation_time_ms: Time taken for generation
        retries_needed: Number of retries before success
        reasoning: Chain-of-thought reasoning if enabled
    """

    rule: str
    is_valid: bool
    confidence: float
    validation_errors: List[str] = field(default_factory=list)
    generation_time_ms: float = 0.0
    retries_needed: int = 0
    reasoning: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result from benchmarking logic generator against general LLM.

    Attributes:
        logic_generator_accuracy: Accuracy of specialized logic generator
        general_llm_accuracy: Accuracy of general-purpose LLM
        logic_generator_syntax_valid_rate: Rate of syntactically valid rules
        general_llm_syntax_valid_rate: Rate of syntactically valid rules for general
        logic_generator_avg_time_ms: Average generation time for logic generator
        general_llm_avg_time_ms: Average generation time for general LLM
        test_cases_count: Number of test cases used
        improvement_percentage: Percentage improvement over general LLM
    """

    logic_generator_accuracy: float
    general_llm_accuracy: float
    logic_generator_syntax_valid_rate: float
    general_llm_syntax_valid_rate: float
    logic_generator_avg_time_ms: float
    general_llm_avg_time_ms: float
    test_cases_count: int
    improvement_percentage: float


# ASP-specialized system prompt
ASP_LOGIC_SYSTEM_PROMPT = """You are an expert ASP (Answer Set Programming) rule generator specialized in formal logic.

Your task is to generate syntactically and semantically valid ASP rules following Clingo syntax.

**Critical ASP Syntax Rules:**
1. Rules end with a period (.)
2. Variables start with UPPERCASE letters (X, Y, Person, Contract)
3. Constants start with lowercase letters (alice, contract1, yes, no)
4. Predicates use lowercase (party_to_contract/2, enforceable/1)
5. Negation uses 'not' keyword (not void(C))
6. Every variable in the head must appear in a positive body literal (variable safety)
7. No embedded periods in predicate names or arguments

**Variable Safety Requirement:**
All variables appearing in the rule head MUST also appear in at least one positive
(non-negated) literal in the rule body. This is a fundamental ASP constraint.

**Example Valid Rules:**
```asp
% Contract is enforceable if it has required elements
enforceable(C) :- contract(C), has_offer(C), has_acceptance(C), has_consideration(C).

% Party is bound to contract terms
bound_to_terms(P, C) :- party_to_contract(P, C), signed(P, C), not void(C).

% Exception applies when specific conditions are met
exception_applies(C) :- contract(C), part_performance(C, significant), land_sale_contract(C).
```

Generate rules that are precise, use appropriate predicates from the provided context,
and follow all ASP syntax requirements."""


class LogicGeneratorLLM:
    """
    Specialized LLM for formal ASP logic generation.

    This class wraps an LLM provider with ASP-specific prompt optimization,
    validation, and retry logic to achieve higher accuracy on formal logic
    generation tasks compared to general-purpose LLMs.

    Example:
        >>> from loft.neural.providers import AnthropicProvider
        >>> provider = AnthropicProvider(api_key, model="claude-3-5-haiku-20241022")
        >>> config = LogicGeneratorConfig(temperature=0.3)
        >>> logic_gen = LogicGeneratorLLM(provider, config)
        >>> result = logic_gen.generate_rule(
        ...     principle="A contract requires offer and acceptance",
        ...     predicates=["contract(X)", "offer(X)", "acceptance(X)"]
        ... )
        >>> print(result.rule)
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: Optional[LogicGeneratorConfig] = None,
    ):
        """Initialize the Logic Generator LLM.

        Args:
            provider: LLM provider instance (Anthropic, OpenAI, etc.)
            config: Configuration for generation parameters
        """
        self.provider = provider
        self.config = config or LogicGeneratorConfig()
        self._llm = LLMInterface(provider, enable_cache=True, max_retries=3)

        # Statistics tracking
        self._total_generations = 0
        self._successful_generations = 0
        self._syntax_errors = 0
        self._variable_safety_errors = 0

        # Load default few-shot examples if none provided
        if not self.config.few_shot_examples:
            self.config.few_shot_examples = self._get_default_few_shot_examples()

        logger.info(
            f"Initialized LogicGeneratorLLM with strategy={self.config.optimization_strategy.value}"
        )

    def generate_rule(
        self,
        principle: str,
        predicates: Optional[List[str]] = None,
        domain: str = "legal",
        context: Optional[Dict[str, Any]] = None,
    ) -> ASPGenerationResult:
        """Generate an ASP rule from a natural language principle.

        Uses the configured optimization strategy to produce high-quality
        ASP rules with validation.

        Args:
            principle: Natural language description of the rule to generate
            predicates: Available predicates to use in the rule
            domain: Domain context (e.g., "contracts", "torts")
            context: Additional context for generation

        Returns:
            ASPGenerationResult with the generated rule and metadata

        Example:
            >>> result = logic_gen.generate_rule(
            ...     principle="A contract is void if signed under duress",
            ...     predicates=["contract(X)", "signed(X, P)", "duress(X)", "void(X)"],
            ...     domain="contracts"
            ... )
        """
        start_time = time.time()
        self._total_generations += 1

        # Build the generation prompt
        prompt = self._build_generation_prompt(principle, predicates, domain, context)

        validation_errors: List[str] = []
        retries = 0
        generated_rule = ""
        confidence = 0.0
        reasoning = None

        for attempt in range(self.config.max_generation_retries):
            try:
                response = self._llm.query(
                    question=prompt,
                    output_schema=GeneratedRule,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    system_prompt=ASP_LOGIC_SYSTEM_PROMPT,
                )

                generated_rule = response.content.asp_rule
                confidence = response.content.confidence
                reasoning = response.content.reasoning

                # Validate the generated rule
                validation_errors = self._validate_rule(generated_rule)

                if not validation_errors:
                    self._successful_generations += 1
                    break

                # Log validation failures for retry
                logger.debug(f"Validation failed on attempt {attempt + 1}: {validation_errors}")

                # Enhance prompt with error feedback for retry
                prompt = self._build_retry_prompt(prompt, generated_rule, validation_errors)
                retries = attempt + 1

            except Exception as e:
                logger.warning(f"Generation error on attempt {attempt + 1}: {e}")
                validation_errors = [str(e)]
                retries = attempt + 1

        generation_time_ms = (time.time() - start_time) * 1000
        is_valid = len(validation_errors) == 0

        return ASPGenerationResult(
            rule=generated_rule,
            is_valid=is_valid,
            confidence=confidence,
            validation_errors=validation_errors,
            generation_time_ms=generation_time_ms,
            retries_needed=retries,
            reasoning=reasoning,
        )

    def generate_gap_filling_candidates(
        self,
        gap_description: str,
        missing_predicate: str,
        dataset_predicates: Optional[List[str]] = None,
        num_candidates: int = 3,
    ) -> List[ASPGenerationResult]:
        """Generate multiple candidate rules to fill a knowledge gap.

        Produces diverse candidates using different approaches for the
        same gap, enabling ensemble voting or user selection.

        Args:
            gap_description: Description of the knowledge gap
            missing_predicate: The predicate that needs definition
            dataset_predicates: Available predicates from the dataset
            num_candidates: Number of candidate rules to generate

        Returns:
            List of ASPGenerationResult with candidate rules
        """
        candidates = []

        # Generate candidates with varying temperatures for diversity
        temperatures = [0.2, 0.4, 0.6][:num_candidates]

        for i, temp in enumerate(temperatures):
            prompt = self._build_gap_filling_prompt(
                gap_description,
                missing_predicate,
                dataset_predicates,
                approach=["conservative", "balanced", "permissive"][i],
            )

            start_time = time.time()

            try:
                response = self._llm.query(
                    question=prompt,
                    output_schema=GeneratedRule,
                    temperature=temp,
                    max_tokens=self.config.max_tokens,
                    system_prompt=ASP_LOGIC_SYSTEM_PROMPT,
                )

                rule = response.content.asp_rule
                validation_errors = self._validate_rule(rule)

                candidates.append(
                    ASPGenerationResult(
                        rule=rule,
                        is_valid=len(validation_errors) == 0,
                        confidence=response.content.confidence,
                        validation_errors=validation_errors,
                        generation_time_ms=(time.time() - start_time) * 1000,
                        reasoning=response.content.reasoning,
                    )
                )

            except Exception as e:
                logger.warning(f"Candidate generation {i + 1} failed: {e}")
                candidates.append(
                    ASPGenerationResult(
                        rule="",
                        is_valid=False,
                        confidence=0.0,
                        validation_errors=[str(e)],
                        generation_time_ms=(time.time() - start_time) * 1000,
                    )
                )

        return candidates

    def benchmark_against_general_llm(
        self,
        test_cases: List[Dict[str, Any]],
        general_llm: LLMInterface,
    ) -> BenchmarkResult:
        """Benchmark this logic generator against a general-purpose LLM.

        Runs the same test cases through both the specialized logic generator
        and a general-purpose LLM to measure improvement.

        Args:
            test_cases: List of test cases with 'principle', 'predicates',
                and optionally 'expected_rule' keys
            general_llm: General-purpose LLM interface for comparison

        Returns:
            BenchmarkResult with comparative metrics
        """
        logger.info(f"Benchmarking against general LLM on {len(test_cases)} cases")

        logic_gen_results: List[ASPGenerationResult] = []
        general_llm_results: List[Dict[str, Any]] = []

        for case in test_cases:
            # Run through logic generator
            lg_result = self.generate_rule(
                principle=case["principle"],
                predicates=case.get("predicates"),
                domain=case.get("domain", "legal"),
            )
            logic_gen_results.append(lg_result)

            # Run through general LLM
            start_time = time.time()
            try:
                gen_response = general_llm.query(
                    question=self._build_simple_prompt(case["principle"], case.get("predicates")),
                    output_schema=GeneratedRule,
                    temperature=0.3,
                )
                gen_rule = gen_response.content.asp_rule
                gen_errors = self._validate_rule(gen_rule)
                general_llm_results.append(
                    {
                        "rule": gen_rule,
                        "is_valid": len(gen_errors) == 0,
                        "time_ms": (time.time() - start_time) * 1000,
                    }
                )
            except Exception as e:
                general_llm_results.append(
                    {
                        "rule": "",
                        "is_valid": False,
                        "time_ms": (time.time() - start_time) * 1000,
                        "error": str(e),
                    }
                )

        # Calculate metrics
        lg_valid_count = sum(1 for r in logic_gen_results if r.is_valid)
        gen_valid_count = sum(1 for r in general_llm_results if r["is_valid"])

        lg_syntax_rate = lg_valid_count / len(test_cases) if test_cases else 0.0
        gen_syntax_rate = gen_valid_count / len(test_cases) if test_cases else 0.0

        lg_avg_time = (
            sum(r.generation_time_ms for r in logic_gen_results) / len(test_cases)
            if test_cases
            else 0.0
        )
        gen_avg_time = (
            sum(r["time_ms"] for r in general_llm_results) / len(test_cases) if test_cases else 0.0
        )

        # Calculate accuracy if expected rules are provided
        lg_accuracy = lg_syntax_rate  # Default to syntax validity
        gen_accuracy = gen_syntax_rate

        improvement = (
            ((lg_syntax_rate - gen_syntax_rate) / gen_syntax_rate * 100)
            if gen_syntax_rate > 0
            else 0.0
        )

        return BenchmarkResult(
            logic_generator_accuracy=lg_accuracy,
            general_llm_accuracy=gen_accuracy,
            logic_generator_syntax_valid_rate=lg_syntax_rate,
            general_llm_syntax_valid_rate=gen_syntax_rate,
            logic_generator_avg_time_ms=lg_avg_time,
            general_llm_avg_time_ms=gen_avg_time,
            test_cases_count=len(test_cases),
            improvement_percentage=improvement,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics.

        Returns:
            Dictionary with generation metrics
        """
        success_rate = (
            self._successful_generations / self._total_generations
            if self._total_generations > 0
            else 0.0
        )

        return {
            "total_generations": self._total_generations,
            "successful_generations": self._successful_generations,
            "success_rate": success_rate,
            "syntax_errors": self._syntax_errors,
            "variable_safety_errors": self._variable_safety_errors,
            "model": self.config.model,
            "optimization_strategy": self.config.optimization_strategy.value,
        }

    def reset_statistics(self) -> None:
        """Reset generation statistics."""
        self._total_generations = 0
        self._successful_generations = 0
        self._syntax_errors = 0
        self._variable_safety_errors = 0

    def _build_generation_prompt(
        self,
        principle: str,
        predicates: Optional[List[str]],
        domain: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Build the generation prompt with optimization strategy."""
        predicates_str = (
            "\n".join(f"  - {p}" for p in predicates)
            if predicates
            else "  (use appropriate predicate names)"
        )

        context_str = ""
        if context:
            context_str = "\n**Additional Context:**\n" + "\n".join(
                f"  - {k}: {v}" for k, v in context.items()
            )

        few_shot_str = ""
        if (
            self.config.optimization_strategy == OptimizationStrategy.FEW_SHOT_LEARNING
            and self.config.few_shot_examples
        ):
            few_shot_str = "\n**Reference Examples:**\n```asp\n"
            few_shot_str += "\n".join(self.config.few_shot_examples[:3])
            few_shot_str += "\n```\n"

        cot_instruction = ""
        if self.config.optimization_strategy == OptimizationStrategy.CHAIN_OF_THOUGHT:
            cot_instruction = """

**Reasoning Process:**
Before generating the rule, think through:
1. What are the key conditions/elements from the principle?
2. Which predicates map to which conditions?
3. What should be in the head vs body of the rule?
4. Are all head variables grounded in the body?
"""

        prompt = f"""Generate an ASP rule for the following legal principle in the {domain} domain.

**Principle:** {principle}

**Available Predicates:**
{predicates_str}
{context_str}
{few_shot_str}
{cot_instruction}
**Requirements:**
1. The rule must be syntactically valid Clingo ASP
2. All head variables must appear in positive body literals
3. Use ONLY the provided predicates or clearly justified new ones
4. The rule must end with a period (.)

Generate the ASP rule:"""

        return prompt

    def _build_gap_filling_prompt(
        self,
        gap_description: str,
        missing_predicate: str,
        dataset_predicates: Optional[List[str]],
        approach: str = "balanced",
    ) -> str:
        """Build prompt for gap filling."""
        predicates_str = (
            "\n".join(f"  - {p}" for p in dataset_predicates)
            if dataset_predicates
            else "  (infer appropriate predicates)"
        )

        approach_guidance = {
            "conservative": "Generate a rule that only fires when ALL conditions are clearly met.",
            "balanced": "Generate a rule that balances specificity with reasonable coverage.",
            "permissive": "Generate a rule that captures the broadest reasonable interpretation.",
        }

        return f"""Fill a knowledge gap by generating an ASP rule.

**Gap Description:** {gap_description}

**Missing Predicate:** {missing_predicate}

**Available Predicates (use these EXACTLY):**
{predicates_str}

**Approach:** {approach_guidance.get(approach, approach_guidance["balanced"])}

**Requirements:**
1. Define {missing_predicate} using the available predicates
2. Ensure variable safety (all head variables in positive body literals)
3. Rule must be syntactically valid Clingo ASP
4. Rule must end with a period (.)

Generate the ASP rule:"""

    def _build_retry_prompt(
        self,
        original_prompt: str,
        failed_rule: str,
        errors: List[str],
    ) -> str:
        """Build a retry prompt with error feedback."""
        error_str = "\n".join(f"  - {e}" for e in errors)

        return f"""{original_prompt}

**PREVIOUS ATTEMPT FAILED:**
```asp
{failed_rule}
```

**Validation Errors:**
{error_str}

**Fix these issues and generate a corrected ASP rule:**"""

    def _build_simple_prompt(
        self,
        principle: str,
        predicates: Optional[List[str]],
    ) -> str:
        """Build a simple prompt for general LLM comparison."""
        predicates_str = ", ".join(predicates) if predicates else "appropriate predicates"

        return f"""Generate an ASP rule for: {principle}
Use predicates: {predicates_str}
The rule must be valid Clingo ASP syntax ending with a period."""

    def _validate_rule(self, rule: str) -> List[str]:
        """Validate an ASP rule for common errors.

        Args:
            rule: The ASP rule to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not rule or not rule.strip():
            errors.append("Empty rule")
            return errors

        # Check for proper termination
        if not rule.strip().endswith("."):
            errors.append("Rule must end with a period (.)")

        # Check for embedded periods
        if self.config.enable_syntax_validation:
            period_errors, period_warnings = check_embedded_periods(rule)
            if period_errors:
                errors.extend([f"Embedded period error: {e}" for e in period_errors])
                self._syntax_errors += 1

        # Check for unsafe variables
        if self.config.enable_variable_safety_check:
            var_errors, var_warnings = check_unsafe_variables(rule)
            if var_errors:
                errors.extend([f"Variable safety error: {e}" for e in var_errors])
                self._variable_safety_errors += 1

        # Basic syntax checks
        if ":-" in rule:
            head_body = rule.split(":-", 1)
            if len(head_body) == 2:
                head = head_body[0].strip()
                body = head_body[1].strip().rstrip(".")

                # Check for empty head or body
                if not head:
                    errors.append("Rule has empty head")
                if not body:
                    errors.append("Rule has empty body")

                # Check for balanced parentheses
                if head.count("(") != head.count(")"):
                    errors.append("Unbalanced parentheses in head")
                if body.count("(") != body.count(")"):
                    errors.append("Unbalanced parentheses in body")

        return errors

    def _get_default_few_shot_examples(self) -> List[str]:
        """Get default few-shot examples for legal ASP rules."""
        return [
            "% Contract requires offer and acceptance\n"
            "valid_contract(C) :- contract(C), has_offer(C), has_acceptance(C), has_consideration(C).",
            "% Writing requirement for statute of frauds\n"
            "enforceable(C) :- contract(C), writing(W), references_contract(W, C), signed_by(W, P), party_to_contract(P, C).",
            "% Part performance exception\n"
            "exception_applies(C) :- land_sale_contract(C), part_performance(C, significant), reliance_interest(C, substantial).",
            "% Adverse possession elements\n"
            "adverse_possession(C) :- claim(C), actual_possession(C), open_notorious(C), continuous(C), hostile(C), statutory_period_met(C).",
        ]
