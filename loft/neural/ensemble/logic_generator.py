"""
Logic Generator LLM - Specialized ASP rule generation.

This module implements a specialized LLM optimized for generating syntactically
and semantically valid Answer Set Programming (ASP) rules. It uses prompt
optimization and few-shot learning to improve ASP generation quality over
general-purpose LLMs.

Part of Phase 6: Heterogeneous Neural Ensemble (Issue #188).

Enhancements based on multi-agent code review (PR #194):
- Strategy Pattern for optimization strategies
- Retry logic with exponential backoff
- Abstract base class for extensibility
- Comprehensive logging
- Caching mechanism for repeated queries
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from loguru import logger

from loft.neural.llm_interface import LLMInterface, LLMProvider
from loft.neural.rule_schemas import GeneratedRule
from loft.validation.asp_validators import check_embedded_periods, check_unsafe_variables


# Custom exception for ASP generation errors
class ASPGenerationError(Exception):
    """Custom exception for ASP generation failures after all retries."""

    def __init__(self, message: str, attempts: int = 0, last_error: Optional[str] = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class OptimizationStrategyType(Enum):
    """Enum identifying optimization strategy types."""

    PROMPT_OPTIMIZATION = "prompt_optimization"
    FEW_SHOT_LEARNING = "few_shot_learning"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SELF_CONSISTENCY = "self_consistency"


# ============================================================================
# Strategy Pattern: Abstract base and concrete strategies
# ============================================================================


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies.

    Follows the Strategy pattern to allow different prompt optimization
    approaches to be swapped without changing the LogicGeneratorLLM class.
    """

    @property
    @abstractmethod
    def strategy_type(self) -> OptimizationStrategyType:
        """Return the strategy type enum."""
        pass

    @abstractmethod
    def prepare_prompt(
        self,
        base_prompt: str,
        principle: str,
        predicates: Optional[List[str]],
        domain: str,
        context: Optional[Dict[str, Any]],
        few_shot_examples: Optional[List[str]] = None,
    ) -> str:
        """Prepare the prompt using this strategy.

        Args:
            base_prompt: The base prompt template
            principle: Natural language principle to convert
            predicates: Available predicates
            domain: Domain context
            context: Additional context
            few_shot_examples: Example rules for few-shot learning

        Returns:
            The prepared prompt string
        """
        pass

    @abstractmethod
    def process_response(self, response: str, reasoning: Optional[str] = None) -> str:
        """Post-process the response from the LLM.

        Args:
            response: The raw ASP rule from the LLM
            reasoning: Optional reasoning from chain-of-thought

        Returns:
            The processed ASP rule
        """
        pass


class PromptOptimizationStrategy(OptimizationStrategy):
    """Default strategy using optimized prompts for ASP generation."""

    @property
    def strategy_type(self) -> OptimizationStrategyType:
        return OptimizationStrategyType.PROMPT_OPTIMIZATION

    def prepare_prompt(
        self,
        base_prompt: str,
        principle: str,
        predicates: Optional[List[str]],
        domain: str,
        context: Optional[Dict[str, Any]],
        few_shot_examples: Optional[List[str]] = None,
    ) -> str:
        logger.debug("Preparing prompt with PROMPT_OPTIMIZATION strategy")
        return base_prompt

    def process_response(self, response: str, reasoning: Optional[str] = None) -> str:
        return response.strip()


class FewShotLearningStrategy(OptimizationStrategy):
    """Strategy using few-shot examples to guide generation."""

    @property
    def strategy_type(self) -> OptimizationStrategyType:
        return OptimizationStrategyType.FEW_SHOT_LEARNING

    def prepare_prompt(
        self,
        base_prompt: str,
        principle: str,
        predicates: Optional[List[str]],
        domain: str,
        context: Optional[Dict[str, Any]],
        few_shot_examples: Optional[List[str]] = None,
    ) -> str:
        logger.debug("Preparing prompt with FEW_SHOT_LEARNING strategy")
        if few_shot_examples:
            few_shot_str = "\n**Reference Examples:**\n```asp\n"
            few_shot_str += "\n".join(few_shot_examples[:3])
            few_shot_str += "\n```\n"
            return base_prompt + few_shot_str
        return base_prompt

    def process_response(self, response: str, reasoning: Optional[str] = None) -> str:
        return response.strip()


class ChainOfThoughtStrategy(OptimizationStrategy):
    """Strategy using chain-of-thought reasoning before generation."""

    @property
    def strategy_type(self) -> OptimizationStrategyType:
        return OptimizationStrategyType.CHAIN_OF_THOUGHT

    def prepare_prompt(
        self,
        base_prompt: str,
        principle: str,
        predicates: Optional[List[str]],
        domain: str,
        context: Optional[Dict[str, Any]],
        few_shot_examples: Optional[List[str]] = None,
    ) -> str:
        logger.debug("Preparing prompt with CHAIN_OF_THOUGHT strategy")
        cot_instruction = """

**Reasoning Process:**
Before generating the rule, think through:
1. What are the key conditions/elements from the principle?
2. Which predicates map to which conditions?
3. What should be in the head vs body of the rule?
4. Are all head variables grounded in the body?
"""
        return base_prompt + cot_instruction

    def process_response(self, response: str, reasoning: Optional[str] = None) -> str:
        if reasoning:
            logger.debug(f"Chain-of-thought reasoning: {reasoning[:200]}...")
        return response.strip()


class SelfConsistencyStrategy(OptimizationStrategy):
    """Strategy that generates multiple candidates for consistency checking."""

    @property
    def strategy_type(self) -> OptimizationStrategyType:
        return OptimizationStrategyType.SELF_CONSISTENCY

    def prepare_prompt(
        self,
        base_prompt: str,
        principle: str,
        predicates: Optional[List[str]],
        domain: str,
        context: Optional[Dict[str, Any]],
        few_shot_examples: Optional[List[str]] = None,
    ) -> str:
        logger.debug("Preparing prompt with SELF_CONSISTENCY strategy")
        consistency_instruction = """

**Self-Consistency Check:**
Generate the most reliable interpretation of the rule. Focus on:
1. Clear, unambiguous conditions
2. Minimal assumptions beyond stated requirements
3. Conservative variable usage
"""
        return base_prompt + consistency_instruction

    def process_response(self, response: str, reasoning: Optional[str] = None) -> str:
        return response.strip()


# Strategy factory for creating strategies from enum
def create_strategy(strategy_type: OptimizationStrategyType) -> OptimizationStrategy:
    """Factory function to create strategy instances from type enum.

    Args:
        strategy_type: The type of optimization strategy

    Returns:
        An instance of the corresponding strategy class
    """
    strategy_map: Dict[OptimizationStrategyType, Type[OptimizationStrategy]] = {
        OptimizationStrategyType.PROMPT_OPTIMIZATION: PromptOptimizationStrategy,
        OptimizationStrategyType.FEW_SHOT_LEARNING: FewShotLearningStrategy,
        OptimizationStrategyType.CHAIN_OF_THOUGHT: ChainOfThoughtStrategy,
        OptimizationStrategyType.SELF_CONSISTENCY: SelfConsistencyStrategy,
    }
    return strategy_map[strategy_type]()


# Backwards compatibility alias
OptimizationStrategy_Enum = OptimizationStrategyType


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
        retry_base_delay_seconds: Base delay for exponential backoff
        enable_cache: Enable caching for repeated queries
        cache_max_size: Maximum number of cached results
    """

    model: str = "claude-3-5-haiku-20241022"
    temperature: float = 0.3
    max_tokens: int = 4096
    optimization_strategy: OptimizationStrategyType = OptimizationStrategyType.PROMPT_OPTIMIZATION
    few_shot_examples: List[str] = field(default_factory=list)
    enable_syntax_validation: bool = True
    enable_variable_safety_check: bool = True
    max_generation_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    enable_cache: bool = True
    cache_max_size: int = 100


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
        from_cache: Whether result was retrieved from cache
    """

    rule: str
    is_valid: bool
    confidence: float
    validation_errors: List[str] = field(default_factory=list)
    generation_time_ms: float = 0.0
    retries_needed: int = 0
    reasoning: Optional[str] = None
    from_cache: bool = False


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


# ============================================================================
# Abstract Base Class for Logic Generators
# ============================================================================


class LogicGenerator(ABC):
    """Abstract base class for logic generators.

    Defines the interface that all logic generators must implement,
    enabling future extensibility for CriticLLM, TranslatorLLM, etc.
    """

    @abstractmethod
    def generate_rule(
        self,
        principle: str,
        predicates: Optional[List[str]] = None,
        domain: str = "legal",
        context: Optional[Dict[str, Any]] = None,
    ) -> ASPGenerationResult:
        """Generate an ASP rule from a natural language principle.

        Args:
            principle: Natural language description of the rule to generate
            predicates: Available predicates to use in the rule
            domain: Domain context (e.g., "contracts", "torts")
            context: Additional context for generation

        Returns:
            ASPGenerationResult with the generated rule and metadata
        """
        pass

    @abstractmethod
    def validate_rule(self, rule: str) -> Tuple[bool, List[str]]:
        """Validate an ASP rule.

        Args:
            rule: The ASP rule to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics.

        Returns:
            Dictionary with generation metrics
        """
        pass


# ============================================================================
# Main Implementation
# ============================================================================


class LogicGeneratorLLM(LogicGenerator):
    """
    Specialized LLM for formal ASP logic generation.

    This class wraps an LLM provider with ASP-specific prompt optimization,
    validation, and retry logic to achieve higher accuracy on formal logic
    generation tasks compared to general-purpose LLMs.

    Features (based on multi-agent review recommendations):
    - Strategy Pattern for flexible optimization approaches
    - Exponential backoff retry logic for resilience
    - LRU caching for repeated queries
    - Comprehensive logging throughout

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
        strategy: Optional[OptimizationStrategy] = None,
    ):
        """Initialize the Logic Generator LLM.

        Args:
            provider: LLM provider instance (Anthropic, OpenAI, etc.)
            config: Configuration for generation parameters
            strategy: Optional custom optimization strategy (overrides config)
        """
        self.provider = provider
        self.config = config or LogicGeneratorConfig()
        self._llm = LLMInterface(provider, enable_cache=True, max_retries=3)

        # Initialize strategy (custom or from config)
        if strategy:
            self._strategy = strategy
            logger.info(f"Using custom strategy: {strategy.strategy_type.value}")
        else:
            self._strategy = create_strategy(self.config.optimization_strategy)
            logger.info(f"Using config strategy: {self.config.optimization_strategy.value}")

        # Statistics tracking
        self._total_generations = 0
        self._successful_generations = 0
        self._syntax_errors = 0
        self._variable_safety_errors = 0
        self._cache_hits = 0
        self._total_retries = 0

        # Cache for repeated queries
        self._cache: Dict[str, ASPGenerationResult] = {}

        # Load default few-shot examples if none provided
        if not self.config.few_shot_examples:
            self.config.few_shot_examples = self._get_default_few_shot_examples()

        logger.info(
            f"Initialized LogicGeneratorLLM: "
            f"strategy={self._strategy.strategy_type.value}, "
            f"model={self.config.model}, "
            f"cache={'enabled' if self.config.enable_cache else 'disabled'}"
        )

    def set_strategy(self, strategy: OptimizationStrategy) -> None:
        """Change the optimization strategy at runtime.

        Args:
            strategy: The new optimization strategy to use
        """
        old_strategy = self._strategy.strategy_type.value
        self._strategy = strategy
        logger.info(f"Strategy changed: {old_strategy} -> {strategy.strategy_type.value}")

    def generate_rule(
        self,
        principle: str,
        predicates: Optional[List[str]] = None,
        domain: str = "legal",
        context: Optional[Dict[str, Any]] = None,
    ) -> ASPGenerationResult:
        """Generate an ASP rule from a natural language principle.

        Uses the configured optimization strategy with retry logic and
        exponential backoff for resilience.

        Args:
            principle: Natural language description of the rule to generate
            predicates: Available predicates to use in the rule
            domain: Domain context (e.g., "contracts", "torts")
            context: Additional context for generation

        Returns:
            ASPGenerationResult with the generated rule and metadata

        Raises:
            ASPGenerationError: If generation fails after all retries
        """
        start_time = time.time()
        self._total_generations += 1

        logger.debug(
            f"Starting rule generation #{self._total_generations}: "
            f"principle='{principle[:50]}...', domain={domain}"
        )

        # Check cache first
        cache_key = self._get_cache_key(principle, predicates, domain, context)
        if self.config.enable_cache and cache_key in self._cache:
            cached_result = self._cache[cache_key]
            self._cache_hits += 1
            logger.debug(f"Cache hit for key {cache_key[:16]}... (hits: {self._cache_hits})")
            return ASPGenerationResult(
                rule=cached_result.rule,
                is_valid=cached_result.is_valid,
                confidence=cached_result.confidence,
                validation_errors=cached_result.validation_errors,
                generation_time_ms=0.0,
                retries_needed=0,
                reasoning=cached_result.reasoning,
                from_cache=True,
            )

        # Build the base generation prompt
        base_prompt = self._build_generation_prompt(principle, predicates, domain, context)

        # Apply strategy to prepare final prompt
        prompt = self._strategy.prepare_prompt(
            base_prompt=base_prompt,
            principle=principle,
            predicates=predicates,
            domain=domain,
            context=context,
            few_shot_examples=self.config.few_shot_examples,
        )

        validation_errors: List[str] = []
        retries = 0
        generated_rule = ""
        confidence = 0.0
        reasoning = None

        for attempt in range(self.config.max_generation_retries):
            try:
                logger.debug(
                    f"Generation attempt {attempt + 1}/{self.config.max_generation_retries}"
                )

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

                # Process response through strategy
                generated_rule = self._strategy.process_response(generated_rule, reasoning)

                # Validate the generated rule
                is_valid, validation_errors = self.validate_rule(generated_rule)

                if is_valid:
                    self._successful_generations += 1
                    logger.info(
                        f"Rule generated successfully on attempt {attempt + 1}: "
                        f"confidence={confidence:.2f}"
                    )
                    break

                # Log validation failures for retry
                logger.warning(f"Validation failed on attempt {attempt + 1}: {validation_errors}")

                # Enhance prompt with error feedback for retry
                prompt = self._build_retry_prompt(prompt, generated_rule, validation_errors)
                retries = attempt + 1
                self._total_retries += 1

                # Exponential backoff before retry
                if attempt < self.config.max_generation_retries - 1:
                    delay = self.config.retry_base_delay_seconds * (2**attempt)
                    logger.debug(f"Waiting {delay:.1f}s before retry (exponential backoff)")
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"Generation error on attempt {attempt + 1}: {e}")
                validation_errors = [str(e)]
                retries = attempt + 1
                self._total_retries += 1

                # Exponential backoff before retry
                if attempt < self.config.max_generation_retries - 1:
                    delay = self.config.retry_base_delay_seconds * (2**attempt)
                    logger.debug(f"Waiting {delay:.1f}s before retry after error")
                    time.sleep(delay)

        generation_time_ms = (time.time() - start_time) * 1000
        is_valid = len(validation_errors) == 0

        result = ASPGenerationResult(
            rule=generated_rule,
            is_valid=is_valid,
            confidence=confidence,
            validation_errors=validation_errors,
            generation_time_ms=generation_time_ms,
            retries_needed=retries,
            reasoning=reasoning,
            from_cache=False,
        )

        # Cache successful results
        if is_valid and self.config.enable_cache:
            if len(self._cache) >= self.config.cache_max_size:
                # Remove oldest entry (simple LRU approximation)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Cache eviction: removed {oldest_key[:16]}...")
            self._cache[cache_key] = result
            logger.debug(f"Cached result for key {cache_key[:16]}...")

        logger.info(
            f"Generation complete: valid={is_valid}, "
            f"retries={retries}, time={generation_time_ms:.1f}ms"
        )

        return result

    def validate_rule(self, rule: str) -> Tuple[bool, List[str]]:
        """Validate an ASP rule for common errors.

        Args:
            rule: The ASP rule to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = self._validate_rule_internal(rule)
        return len(errors) == 0, errors

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
        logger.info(f"Generating {num_candidates} gap-filling candidates for: {missing_predicate}")
        candidates = []

        # Generate candidates with varying temperatures for diversity
        temperatures = [0.2, 0.4, 0.6][:num_candidates]

        for i, temp in enumerate(temperatures):
            approach = ["conservative", "balanced", "permissive"][i]
            logger.debug(f"Generating candidate {i + 1} with approach={approach}, temp={temp}")

            prompt = self._build_gap_filling_prompt(
                gap_description,
                missing_predicate,
                dataset_predicates,
                approach=approach,
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
                is_valid, validation_errors = self.validate_rule(rule)

                candidates.append(
                    ASPGenerationResult(
                        rule=rule,
                        is_valid=is_valid,
                        confidence=response.content.confidence,
                        validation_errors=validation_errors,
                        generation_time_ms=(time.time() - start_time) * 1000,
                        reasoning=response.content.reasoning,
                    )
                )
                logger.debug(f"Candidate {i + 1} generated: valid={is_valid}")

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

        valid_count = sum(1 for c in candidates if c.is_valid)
        logger.info(f"Gap-filling complete: {valid_count}/{num_candidates} valid candidates")

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
        logger.info(f"Starting benchmark against general LLM: {len(test_cases)} test cases")

        logic_gen_results: List[ASPGenerationResult] = []
        general_llm_results: List[Dict[str, Any]] = []

        for i, case in enumerate(test_cases):
            logger.debug(f"Benchmark case {i + 1}/{len(test_cases)}")

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
                is_valid, gen_errors = self.validate_rule(gen_rule)
                general_llm_results.append(
                    {
                        "rule": gen_rule,
                        "is_valid": is_valid,
                        "time_ms": (time.time() - start_time) * 1000,
                    }
                )
            except Exception as e:
                logger.warning(f"General LLM failed on case {i + 1}: {e}")
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

        improvement = (
            ((lg_syntax_rate - gen_syntax_rate) / gen_syntax_rate * 100)
            if gen_syntax_rate > 0
            else 0.0
        )

        result = BenchmarkResult(
            logic_generator_accuracy=lg_syntax_rate,
            general_llm_accuracy=gen_syntax_rate,
            logic_generator_syntax_valid_rate=lg_syntax_rate,
            general_llm_syntax_valid_rate=gen_syntax_rate,
            logic_generator_avg_time_ms=lg_avg_time,
            general_llm_avg_time_ms=gen_avg_time,
            test_cases_count=len(test_cases),
            improvement_percentage=improvement,
        )

        logger.info(
            f"Benchmark complete: "
            f"LogicGen={lg_syntax_rate:.1%} vs General={gen_syntax_rate:.1%} "
            f"(improvement={improvement:+.1f}%)"
        )

        return result

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

        cache_hit_rate = (
            self._cache_hits / self._total_generations if self._total_generations > 0 else 0.0
        )

        return {
            "total_generations": self._total_generations,
            "successful_generations": self._successful_generations,
            "success_rate": success_rate,
            "syntax_errors": self._syntax_errors,
            "variable_safety_errors": self._variable_safety_errors,
            "total_retries": self._total_retries,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
            "model": self.config.model,
            "optimization_strategy": self._strategy.strategy_type.value,
        }

    def reset_statistics(self) -> None:
        """Reset generation statistics."""
        self._total_generations = 0
        self._successful_generations = 0
        self._syntax_errors = 0
        self._variable_safety_errors = 0
        self._total_retries = 0
        self._cache_hits = 0
        logger.debug("Statistics reset")

    def clear_cache(self) -> None:
        """Clear the generation cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared ({cache_size} entries removed)")

    def _get_cache_key(
        self,
        principle: str,
        predicates: Optional[List[str]],
        domain: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Generate a cache key for the given inputs."""
        key_parts = [
            principle,
            str(sorted(predicates) if predicates else []),
            domain,
            str(sorted(context.items()) if context else {}),
            self._strategy.strategy_type.value,
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _build_generation_prompt(
        self,
        principle: str,
        predicates: Optional[List[str]],
        domain: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Build the base generation prompt."""
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

        prompt = f"""Generate an ASP rule for the following legal principle in the {domain} domain.

**Principle:** {principle}

**Available Predicates:**
{predicates_str}
{context_str}

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

    def _validate_rule_internal(self, rule: str) -> List[str]:
        """Validate an ASP rule for common errors (internal method).

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
