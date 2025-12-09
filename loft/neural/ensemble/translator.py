"""
Translator LLM - Specialized symbolic-NL bidirectional translation.

This module implements a specialized LLM optimized for high-fidelity translation
between ASP (Answer Set Programming) rules and natural language. It provides
bidirectional translation with roundtrip validation to ensure semantic preservation.

Part of Phase 6: Heterogeneous Neural Ensemble (Issue #190).

Complements the Logic Generator (Issue #188) and Critic LLM (Issue #189) by
providing the ontological bridge between symbolic reasoning and neural pattern
recognition.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from loft.neural.llm_interface import LLMInterface, LLMProvider


# =============================================================================
# Custom Exceptions
# =============================================================================


class TranslationError(Exception):
    """Custom exception for translation failures."""

    def __init__(
        self,
        message: str,
        direction: str = "unknown",
        attempts: int = 0,
        last_error: Optional[str] = None,
    ):
        super().__init__(message)
        self.direction = direction
        self.attempts = attempts
        self.last_error = last_error


class LLMResponseParsingError(Exception):
    """Raised when LLM response cannot be parsed into expected schema."""

    pass


# =============================================================================
# Data Classes for Translation Results
# =============================================================================


@dataclass
class TranslationResult:
    """Result from a translation operation.

    Attributes:
        source: Original input text
        target: Translated output text
        direction: Translation direction (asp_to_nl, nl_to_asp)
        confidence: Confidence in the translation (0.0-1.0)
        predicates_used: Predicates referenced in the translation
        ambiguities: Identified ambiguities in the translation
        reasoning: Explanation of translation choices
        translation_time_ms: Time taken for translation
        from_cache: Whether result was from cache
    """

    source: str
    target: str
    direction: str
    confidence: float = 0.7
    predicates_used: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)
    reasoning: Optional[str] = None
    translation_time_ms: float = 0.0
    from_cache: bool = False


@dataclass
class RoundtripResult:
    """Result from roundtrip validation.

    Attributes:
        original: Original input
        intermediate: Intermediate translation
        final: Final translation back to original format
        fidelity_score: Semantic preservation score (0.0-1.0)
        is_asp_original: Whether original was ASP (True) or NL (False)
        preserved_predicates: Predicates preserved through roundtrip
        lost_information: Information lost during translation
        added_information: Information added/hallucinated during translation
        confidence: Confidence in the fidelity assessment
        total_time_ms: Total time for roundtrip
    """

    original: str
    intermediate: str
    final: str
    fidelity_score: float
    is_asp_original: bool
    preserved_predicates: List[str] = field(default_factory=list)
    lost_information: List[str] = field(default_factory=list)
    added_information: List[str] = field(default_factory=list)
    confidence: float = 0.7
    total_time_ms: float = 0.0


@dataclass
class TranslatorBenchmarkResult:
    """Result from benchmarking translator against general LLM.

    Attributes:
        translator_fidelity: Average fidelity score for specialized translator
        general_llm_fidelity: Average fidelity score for general-purpose LLM
        translator_avg_time_ms: Average translation time for translator
        general_llm_avg_time_ms: Average translation time for general LLM
        test_cases_count: Number of test cases used
        improvement_percentage: Percentage improvement over general LLM
        translator_roundtrip_success_rate: Rate of successful roundtrips
        general_llm_roundtrip_success_rate: Rate of successful roundtrips
    """

    translator_fidelity: float
    general_llm_fidelity: float
    translator_avg_time_ms: float
    general_llm_avg_time_ms: float
    test_cases_count: int
    improvement_percentage: float
    translator_roundtrip_success_rate: float
    general_llm_roundtrip_success_rate: float


# =============================================================================
# Pydantic Schemas for LLM Structured Output
# =============================================================================


class ASPToNLSchema(BaseModel):
    """Schema for ASP to natural language translation."""

    natural_language: str = Field(
        description="Natural language explanation of the ASP rule"
    )
    predicates_identified: List[str] = Field(
        default_factory=list, description="Predicates identified in the ASP rule"
    )
    reasoning: str = Field(description="Explanation of how the translation was derived")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the translation"
    )
    ambiguities: List[str] = Field(
        default_factory=list,
        description="Ambiguous aspects that could be interpreted differently",
    )


class NLToASPSchema(BaseModel):
    """Schema for natural language to ASP translation."""

    asp_rule: str = Field(description="Generated ASP rule from the description")
    predicates_used: List[str] = Field(
        default_factory=list, description="Predicates used in the generated rule"
    )
    reasoning: str = Field(
        description="Explanation of how predicates were chosen and structured"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the translation"
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions made during translation",
    )


class FidelityAssessmentSchema(BaseModel):
    """Schema for semantic fidelity assessment."""

    fidelity_score: float = Field(
        ge=0.0, le=1.0, description="Semantic preservation score"
    )
    preserved_elements: List[str] = Field(
        default_factory=list,
        description="Semantic elements preserved through roundtrip",
    )
    lost_elements: List[str] = Field(
        default_factory=list, description="Semantic elements lost during translation"
    )
    added_elements: List[str] = Field(
        default_factory=list,
        description="Elements added/hallucinated during translation",
    )
    reasoning: str = Field(description="Explanation of the fidelity assessment")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the assessment"
    )


# =============================================================================
# Translation Strategy Pattern
# =============================================================================


class TranslationStrategyType(Enum):
    """Enum identifying translation strategy types."""

    LITERAL = "literal"
    CONTEXTUAL = "contextual"
    LEGAL_DOMAIN = "legal_domain"
    PEDAGOGICAL = "pedagogical"


class TranslationStrategy(ABC):
    """Abstract base class for translation strategies.

    Different strategies for translating between ASP and natural language,
    enabling different translation styles and accuracy trade-offs.
    """

    @property
    @abstractmethod
    def strategy_type(self) -> TranslationStrategyType:
        """Return the strategy type enum."""
        pass

    @abstractmethod
    def prepare_asp_to_nl_prompt(
        self,
        base_prompt: str,
        asp_rule: str,
        context: Dict[str, Any],
    ) -> str:
        """Prepare the prompt for ASP to NL translation."""
        pass

    @abstractmethod
    def prepare_nl_to_asp_prompt(
        self,
        base_prompt: str,
        description: str,
        predicates: List[str],
        context: Dict[str, Any],
    ) -> str:
        """Prepare the prompt for NL to ASP translation."""
        pass


class LiteralStrategy(TranslationStrategy):
    """Direct, close-to-source translation preserving structure."""

    @property
    def strategy_type(self) -> TranslationStrategyType:
        return TranslationStrategyType.LITERAL

    def prepare_asp_to_nl_prompt(
        self,
        base_prompt: str,
        asp_rule: str,
        context: Dict[str, Any],
    ) -> str:
        logger.debug("Preparing prompt with LITERAL strategy")
        literal_addition = """

**Literal Translation Instructions:**
Translate as closely as possible to the original ASP structure:
1. Preserve the logical relationship (head :- body means "head is true if body is true")
2. Translate predicates directly to their semantic meaning
3. Maintain variable bindings and their relationships
4. Do NOT paraphrase or simplify beyond direct translation"""
        return base_prompt + literal_addition

    def prepare_nl_to_asp_prompt(
        self,
        base_prompt: str,
        description: str,
        predicates: List[str],
        context: Dict[str, Any],
    ) -> str:
        literal_addition = """

**Literal Translation Instructions:**
Generate ASP that directly mirrors the natural language structure:
1. Map each condition to a single predicate when possible
2. Preserve the logical order from the description
3. Use predicates that closely match the terminology used
4. Avoid adding implicit conditions not in the description"""
        return base_prompt + literal_addition


class ContextualStrategy(TranslationStrategy):
    """Context-aware translation with domain understanding."""

    @property
    def strategy_type(self) -> TranslationStrategyType:
        return TranslationStrategyType.CONTEXTUAL

    def prepare_asp_to_nl_prompt(
        self,
        base_prompt: str,
        asp_rule: str,
        context: Dict[str, Any],
    ) -> str:
        logger.debug("Preparing prompt with CONTEXTUAL strategy")
        contextual_addition = """

**Contextual Translation Instructions:**
Translate with awareness of the domain context:
1. Use domain-appropriate terminology
2. Explain implicit domain knowledge where helpful
3. Connect to related concepts in the domain
4. Produce natural, fluent language appropriate for the context"""
        return base_prompt + contextual_addition

    def prepare_nl_to_asp_prompt(
        self,
        base_prompt: str,
        description: str,
        predicates: List[str],
        context: Dict[str, Any],
    ) -> str:
        contextual_addition = """

**Contextual Translation Instructions:**
Generate ASP with domain context awareness:
1. Use predicates that match domain conventions
2. Include implicit domain constraints when appropriate
3. Consider how this rule interacts with other domain rules
4. Balance completeness with specificity"""
        return base_prompt + contextual_addition


class LegalDomainStrategy(TranslationStrategy):
    """Specialized strategy for legal domain translations."""

    @property
    def strategy_type(self) -> TranslationStrategyType:
        return TranslationStrategyType.LEGAL_DOMAIN

    def prepare_asp_to_nl_prompt(
        self,
        base_prompt: str,
        asp_rule: str,
        context: Dict[str, Any],
    ) -> str:
        logger.debug("Preparing prompt with LEGAL_DOMAIN strategy")
        legal_addition = """

**Legal Domain Translation Instructions:**
Translate for legal reasoning clarity:
1. Use precise legal terminology
2. Distinguish between necessary and sufficient conditions
3. Clarify elements, requirements, and exceptions
4. Reference relevant legal concepts (statute of frauds, consideration, etc.)
5. Explain how the rule would apply to legal fact patterns"""
        return base_prompt + legal_addition

    def prepare_nl_to_asp_prompt(
        self,
        base_prompt: str,
        description: str,
        predicates: List[str],
        context: Dict[str, Any],
    ) -> str:
        legal_addition = """

**Legal Domain Translation Instructions:**
Generate ASP appropriate for legal reasoning:
1. Model legal elements as separate predicates
2. Distinguish positive requirements from negative defenses
3. Use predicates that match legal doctrine
4. Handle exceptions using negation-as-failure where appropriate
5. Ensure variable safety for all legal entities"""
        return base_prompt + legal_addition


class PedagogicalStrategy(TranslationStrategy):
    """Translation optimized for explanations and learning."""

    @property
    def strategy_type(self) -> TranslationStrategyType:
        return TranslationStrategyType.PEDAGOGICAL

    def prepare_asp_to_nl_prompt(
        self,
        base_prompt: str,
        asp_rule: str,
        context: Dict[str, Any],
    ) -> str:
        logger.debug("Preparing prompt with PEDAGOGICAL strategy")
        pedagogical_addition = """

**Pedagogical Translation Instructions:**
Translate for educational clarity:
1. Explain what the rule means in plain language
2. Break down complex conditions into understandable parts
3. Provide examples of when the rule applies
4. Highlight key concepts and relationships
5. Make implicit logic explicit for learners"""
        return base_prompt + pedagogical_addition

    def prepare_nl_to_asp_prompt(
        self,
        base_prompt: str,
        description: str,
        predicates: List[str],
        context: Dict[str, Any],
    ) -> str:
        pedagogical_addition = """

**Pedagogical Translation Instructions:**
Generate ASP with educational clarity:
1. Use clear, descriptive predicate names
2. Add comments explaining each condition
3. Structure the rule for readability
4. Include reasoning for predicate choices"""
        return base_prompt + pedagogical_addition


def create_translation_strategy(
    strategy_type: TranslationStrategyType,
) -> TranslationStrategy:
    """Factory function to create translation strategy instances."""
    strategy_map: Dict[TranslationStrategyType, Type[TranslationStrategy]] = {
        TranslationStrategyType.LITERAL: LiteralStrategy,
        TranslationStrategyType.CONTEXTUAL: ContextualStrategy,
        TranslationStrategyType.LEGAL_DOMAIN: LegalDomainStrategy,
        TranslationStrategyType.PEDAGOGICAL: PedagogicalStrategy,
    }
    return strategy_map[strategy_type]()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TranslatorConfig:
    """Configuration for the Translator LLM.

    Attributes:
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        strategy: Translation strategy type
        max_retries: Maximum retry attempts
        retry_base_delay_seconds: Base delay for exponential backoff
        enable_cache: Enable caching
        cache_max_size: Maximum cache entries
        fidelity_threshold: Minimum fidelity score for success
    """

    model: str = "claude-3-5-haiku-20241022"
    temperature: float = 0.2  # Lower for more deterministic translations
    max_tokens: int = 4096
    strategy: TranslationStrategyType = TranslationStrategyType.LEGAL_DOMAIN
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    enable_cache: bool = True
    cache_max_size: int = 100
    fidelity_threshold: float = 0.95  # Target >95% fidelity per issue #190


# =============================================================================
# System Prompts
# =============================================================================


TRANSLATOR_SYSTEM_PROMPT = """You are an expert translator specializing in bidirectional translation between Answer Set Programming (ASP) rules and natural language.

Your task is to provide high-fidelity translations that preserve semantic meaning while being appropriate for the target format.

**ASP Syntax Understanding:**
- Rules have the form: head :- body. (head is true if body is true)
- Facts are rules without a body: fact.
- Variables are UPPERCASE (X, Y, Contract, Person)
- Constants are lowercase (alice, contract1, yes)
- Predicates are lowercase with arity (party_to_contract/2, enforceable/1)
- Negation uses 'not' keyword (negation-as-failure)
- Conjunction in body uses comma (,)
- Disjunction in head uses semicolon (;)

**Translation Principles:**
1. SEMANTIC PRESERVATION: The meaning must be preserved across translation
2. PREDICATE CLARITY: Predicate names should map clearly to concepts
3. VARIABLE COHERENCE: Variables should be consistently tracked
4. CONDITION COMPLETENESS: All conditions should be translated
5. AMBIGUITY ACKNOWLEDGMENT: Flag ambiguous translations

**Legal Domain Context:**
- Contract elements: offer, acceptance, consideration, capacity, legality
- Writing requirements: statute of frauds, signed writings
- Exceptions: part performance, promissory estoppel
- Property: adverse possession, title, ownership
- Torts: negligence, duty, breach, causation, damages"""


ASP_TO_NL_PROMPT_TEMPLATE = """Translate this ASP rule to natural language.

**ASP Rule:**
```asp
{rule}
```

**Domain Context:**
{context}

**Available Predicates for Reference:**
{predicates}

**Translation Task:**
1. Explain what this rule means in clear natural language
2. Identify all predicates used and their roles
3. Describe the logical relationship (conditions → conclusion)
4. Note any ambiguities or multiple interpretations

Generate your translation:"""


NL_TO_ASP_PROMPT_TEMPLATE = """Translate this natural language description to an ASP rule.

**Description:**
{description}

**Available Predicates (use these):**
{predicates}

**Domain Context:**
{context}

**Translation Task:**
1. Generate a syntactically valid ASP rule
2. Map the description's conditions to appropriate predicates
3. Ensure all head variables appear in positive body literals
4. List the predicates used and explain your choices
5. Note any assumptions made during translation

**Requirements:**
- Rule must end with a period (.)
- Variables must be UPPERCASE
- Constants must be lowercase
- All head variables must be grounded in body

Generate your translation:"""


FIDELITY_ASSESSMENT_PROMPT = """Assess the semantic fidelity of this roundtrip translation.

**Original ({original_type}):**
```
{original}
```

**Intermediate Translation ({intermediate_type}):**
```
{intermediate}
```

**Final Translation Back ({final_type}):**
```
{final}
```

**Assessment Task:**
Compare the original and final translations to assess semantic preservation:
1. What semantic elements were preserved?
2. What information was lost?
3. What information was added or hallucinated?
4. Calculate an overall fidelity score (0.0 = completely different, 1.0 = semantically identical)

Generate your assessment:"""


# =============================================================================
# Abstract Base Class
# =============================================================================


class Translator(ABC):
    """Abstract base class for translator implementations.

    Defines the interface that all translator implementations must follow,
    enabling future extensibility for different translation approaches.
    """

    @abstractmethod
    def asp_to_nl(
        self,
        asp_rule: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TranslationResult:
        """Convert ASP rule to natural language."""
        pass

    @abstractmethod
    def nl_to_asp(
        self,
        description: str,
        predicates: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> TranslationResult:
        """Convert natural language to ASP rule."""
        pass

    @abstractmethod
    def roundtrip_validate(
        self,
        original: str,
        is_asp: bool,
        predicates: Optional[List[str]] = None,
    ) -> RoundtripResult:
        """Validate semantic preservation through roundtrip translation."""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get translation statistics."""
        pass


# =============================================================================
# Main Implementation
# =============================================================================


class TranslatorLLM(Translator):
    """Specialized LLM for symbolic-NL bidirectional translation.

    This class provides high-fidelity translation between ASP rules and
    natural language, with roundtrip validation to ensure semantic preservation.

    Features:
    - Strategy Pattern for different translation styles
    - Exponential backoff retry logic
    - LRU caching for repeated translations
    - Roundtrip validation with fidelity scoring
    - Comprehensive logging
    - Integration with existing translation infrastructure

    Example:
        >>> from loft.neural.providers import AnthropicProvider
        >>> provider = AnthropicProvider(api_key, model="claude-3-5-haiku-20241022")
        >>> config = TranslatorConfig(strategy=TranslationStrategyType.LEGAL_DOMAIN)
        >>> translator = TranslatorLLM(provider, config)
        >>> result = translator.asp_to_nl(
        ...     asp_rule="enforceable(C) :- contract(C), signed(C).",
        ...     context={"domain": "contracts"}
        ... )
        >>> print(result.target)
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: Optional[TranslatorConfig] = None,
        strategy: Optional[TranslationStrategy] = None,
    ):
        """Initialize the Translator LLM.

        Args:
            provider: LLM provider instance
            config: Configuration for translation parameters
            strategy: Optional custom translation strategy (overrides config)

        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("provider cannot be None")

        self.provider = provider
        self.config = config or TranslatorConfig()
        self._llm = LLMInterface(provider, enable_cache=True, max_retries=3)

        # Thread-safe strategy access
        self._strategy_lock = threading.Lock()

        # Initialize strategy
        if strategy:
            self._strategy = strategy
            logger.info(f"Using custom strategy: {strategy.strategy_type.value}")
        else:
            self._strategy = create_translation_strategy(self.config.strategy)
            logger.info(f"Using config strategy: {self.config.strategy.value}")

        # Statistics tracking
        self._total_translations = 0
        self._asp_to_nl_count = 0
        self._nl_to_asp_count = 0
        self._roundtrip_count = 0
        self._successful_roundtrips = 0
        self._high_fidelity_roundtrips = 0
        self._cache_hits = 0
        self._total_retries = 0
        self._total_fidelity_score = 0.0

        # Cache for repeated translations
        self._cache: Dict[str, TranslationResult] = {}
        self._cache_lock = threading.Lock()

        logger.info(
            f"Initialized TranslatorLLM: "
            f"strategy={self._strategy.strategy_type.value}, "
            f"model={self.config.model}, "
            f"cache={'enabled' if self.config.enable_cache else 'disabled'}, "
            f"fidelity_threshold={self.config.fidelity_threshold}"
        )

    def set_strategy(self, strategy: TranslationStrategy) -> None:
        """Change the translation strategy at runtime.

        Thread-safe method to update the translation strategy.

        Args:
            strategy: New strategy to use for translation

        Raises:
            ValueError: If strategy is None
        """
        if strategy is None:
            raise ValueError("strategy cannot be None")

        with self._strategy_lock:
            old_strategy = self._strategy.strategy_type.value
            self._strategy = strategy
            logger.info(
                f"Strategy changed: {old_strategy} -> {strategy.strategy_type.value}"
            )

    def asp_to_nl(
        self,
        asp_rule: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TranslationResult:
        """Convert ASP rule to natural language explanation.

        Args:
            asp_rule: The ASP rule to translate
            context: Optional context including domain, predicates

        Returns:
            TranslationResult with natural language explanation

        Raises:
            ValueError: If asp_rule is empty
            TranslationError: If translation fails after all retries
        """
        if not asp_rule or not asp_rule.strip():
            raise ValueError("asp_rule cannot be empty")

        start_time = time.time()
        self._total_translations += 1
        self._asp_to_nl_count += 1
        context = context or {}

        logger.debug(f"Starting ASP→NL translation: {asp_rule[:50]}...")

        # Check cache with thread safety
        cache_key = self._get_cache_key("asp_to_nl", asp_rule, context)
        with self._cache_lock:
            if self.config.enable_cache and cache_key in self._cache:
                self._cache_hits += 1
                cached = self._cache[cache_key]
                logger.debug("Cache hit for ASP→NL translation")
                return TranslationResult(
                    source=asp_rule,
                    target=cached.target,
                    direction="asp_to_nl",
                    confidence=cached.confidence,
                    predicates_used=cached.predicates_used,
                    ambiguities=cached.ambiguities,
                    reasoning=cached.reasoning,
                    translation_time_ms=0.0,
                    from_cache=True,
                )

        # Get strategy with thread safety
        with self._strategy_lock:
            current_strategy = self._strategy

        # Prepare context strings
        predicates_str = (
            "\n".join(f"  - {p}" for p in context.get("predicates", []))
            or "  (infer from rule)"
        )
        context_str = (
            "\n".join(f"  - {k}: {v}" for k, v in context.items() if k != "predicates")
            or "  (general legal context)"
        )

        # Build prompt
        base_prompt = ASP_TO_NL_PROMPT_TEMPLATE.format(
            rule=asp_rule,
            context=context_str,
            predicates=predicates_str,
        )

        prompt = current_strategy.prepare_asp_to_nl_prompt(
            base_prompt, asp_rule, context
        )

        result = TranslationResult(
            source=asp_rule,
            target="",
            direction="asp_to_nl",
            confidence=0.0,
        )

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"ASP→NL translation attempt {attempt + 1}")

                response = self._llm.query(
                    question=prompt,
                    output_schema=ASPToNLSchema,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    system_prompt=TRANSLATOR_SYSTEM_PROMPT,
                )

                # Validate response content
                if not hasattr(response, "content") or response.content is None:
                    raise LLMResponseParsingError("LLM response has no content")

                response_data = response.content
                if not isinstance(response_data, ASPToNLSchema):
                    raise LLMResponseParsingError(
                        f"Expected ASPToNLSchema, got {type(response_data).__name__}"
                    )

                result = TranslationResult(
                    source=asp_rule,
                    target=response_data.natural_language,
                    direction="asp_to_nl",
                    confidence=response_data.confidence,
                    predicates_used=list(response_data.predicates_identified),
                    ambiguities=list(response_data.ambiguities),
                    reasoning=response_data.reasoning,
                    translation_time_ms=(time.time() - start_time) * 1000,
                    from_cache=False,
                )

                logger.info(
                    f"ASP→NL translation complete: confidence={result.confidence:.2f}"
                )
                break

            except (ValidationError, LLMResponseParsingError) as e:
                logger.warning(f"ASP→NL translation attempt {attempt + 1} failed: {e}")
                self._total_retries += 1

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay_seconds * (2**attempt)
                    logger.debug(f"Waiting {delay:.1f}s before retry")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"ASP→NL translation failed after {self.config.max_retries} attempts"
                    )

            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(
                    f"ASP→NL translation attempt {attempt + 1} failed (network): {e}"
                )
                self._total_retries += 1

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay_seconds * (2**attempt)
                    time.sleep(delay)

            except KeyboardInterrupt:
                raise

        # Cache successful result
        if result.target and self.config.enable_cache:
            with self._cache_lock:
                self._cache_result(cache_key, result)

        return result

    def nl_to_asp(
        self,
        description: str,
        predicates: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> TranslationResult:
        """Convert natural language description to ASP rule.

        Args:
            description: Natural language description to translate
            predicates: Available predicates to use in the rule
            context: Optional context including domain

        Returns:
            TranslationResult with ASP rule

        Raises:
            ValueError: If description is empty
            TranslationError: If translation fails after all retries
        """
        if not description or not description.strip():
            raise ValueError("description cannot be empty")

        start_time = time.time()
        self._total_translations += 1
        self._nl_to_asp_count += 1
        context = context or {}

        logger.debug(f"Starting NL→ASP translation: {description[:50]}...")

        # Check cache with thread safety
        cache_context = {**context, "predicates": tuple(predicates)}
        cache_key = self._get_cache_key("nl_to_asp", description, cache_context)
        with self._cache_lock:
            if self.config.enable_cache and cache_key in self._cache:
                self._cache_hits += 1
                cached = self._cache[cache_key]
                logger.debug("Cache hit for NL→ASP translation")
                return TranslationResult(
                    source=description,
                    target=cached.target,
                    direction="nl_to_asp",
                    confidence=cached.confidence,
                    predicates_used=cached.predicates_used,
                    ambiguities=cached.ambiguities,
                    reasoning=cached.reasoning,
                    translation_time_ms=0.0,
                    from_cache=True,
                )

        # Get strategy with thread safety
        with self._strategy_lock:
            current_strategy = self._strategy

        # Prepare context strings
        predicates_str = (
            "\n".join(f"  - {p}" for p in predicates)
            if predicates
            else "  (infer appropriate predicates)"
        )
        context_str = (
            "\n".join(f"  - {k}: {v}" for k, v in context.items() if k != "predicates")
            or "  (general legal context)"
        )

        # Build prompt
        base_prompt = NL_TO_ASP_PROMPT_TEMPLATE.format(
            description=description,
            predicates=predicates_str,
            context=context_str,
        )

        prompt = current_strategy.prepare_nl_to_asp_prompt(
            base_prompt, description, predicates, context
        )

        result = TranslationResult(
            source=description,
            target="",
            direction="nl_to_asp",
            confidence=0.0,
        )

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"NL→ASP translation attempt {attempt + 1}")

                response = self._llm.query(
                    question=prompt,
                    output_schema=NLToASPSchema,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    system_prompt=TRANSLATOR_SYSTEM_PROMPT,
                )

                # Validate response content
                if not hasattr(response, "content") or response.content is None:
                    raise LLMResponseParsingError("LLM response has no content")

                response_data = response.content
                if not isinstance(response_data, NLToASPSchema):
                    raise LLMResponseParsingError(
                        f"Expected NLToASPSchema, got {type(response_data).__name__}"
                    )

                result = TranslationResult(
                    source=description,
                    target=response_data.asp_rule,
                    direction="nl_to_asp",
                    confidence=response_data.confidence,
                    predicates_used=list(response_data.predicates_used),
                    ambiguities=list(response_data.assumptions),
                    reasoning=response_data.reasoning,
                    translation_time_ms=(time.time() - start_time) * 1000,
                    from_cache=False,
                )

                logger.info(
                    f"NL→ASP translation complete: confidence={result.confidence:.2f}"
                )
                break

            except (ValidationError, LLMResponseParsingError) as e:
                logger.warning(f"NL→ASP translation attempt {attempt + 1} failed: {e}")
                self._total_retries += 1

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay_seconds * (2**attempt)
                    logger.debug(f"Waiting {delay:.1f}s before retry")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"NL→ASP translation failed after {self.config.max_retries} attempts"
                    )

            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(
                    f"NL→ASP translation attempt {attempt + 1} failed (network): {e}"
                )
                self._total_retries += 1

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay_seconds * (2**attempt)
                    time.sleep(delay)

            except KeyboardInterrupt:
                raise

        # Cache successful result
        if result.target and self.config.enable_cache:
            with self._cache_lock:
                self._cache_result(cache_key, result)

        return result

    def roundtrip_validate(
        self,
        original: str,
        is_asp: bool,
        predicates: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoundtripResult:
        """Validate semantic preservation through roundtrip translation.

        Translates the original to the other format and back, then assesses
        how much semantic meaning was preserved.

        Args:
            original: Original text to validate
            is_asp: True if original is ASP, False if natural language
            predicates: Available predicates (required if is_asp=False)
            context: Optional context for translation

        Returns:
            RoundtripResult with fidelity assessment

        Raises:
            ValueError: If original is empty or predicates missing for NL input
        """
        if not original or not original.strip():
            raise ValueError("original cannot be empty")

        if not is_asp and not predicates:
            predicates = []  # Allow empty predicates, translator will infer

        start_time = time.time()
        self._roundtrip_count += 1
        context = context or {}

        logger.info(
            f"Starting roundtrip validation: "
            f"{'ASP→NL→ASP' if is_asp else 'NL→ASP→NL'}"
        )

        # Step 1: First translation
        if is_asp:
            first_result = self.asp_to_nl(original, context)
            intermediate = first_result.target
            intermediate_predicates = first_result.predicates_used
        else:
            first_result = self.nl_to_asp(original, predicates or [], context)
            intermediate = first_result.target
            intermediate_predicates = first_result.predicates_used

        # Step 2: Second translation (back to original format)
        if is_asp:
            second_result = self.nl_to_asp(
                intermediate,
                intermediate_predicates or predicates or [],
                context,
            )
            final = second_result.target
        else:
            second_result = self.asp_to_nl(intermediate, context)
            final = second_result.target

        # Step 3: Assess fidelity
        fidelity_result = self._assess_fidelity(
            original, intermediate, final, is_asp, context
        )

        total_time = (time.time() - start_time) * 1000

        result = RoundtripResult(
            original=original,
            intermediate=intermediate,
            final=final,
            fidelity_score=fidelity_result["fidelity_score"],
            is_asp_original=is_asp,
            preserved_predicates=fidelity_result["preserved"],
            lost_information=fidelity_result["lost"],
            added_information=fidelity_result["added"],
            confidence=fidelity_result["confidence"],
            total_time_ms=total_time,
        )

        # Update statistics
        self._total_fidelity_score += result.fidelity_score
        if result.fidelity_score >= self.config.fidelity_threshold:
            self._successful_roundtrips += 1
            self._high_fidelity_roundtrips += 1

        logger.info(
            f"Roundtrip validation complete: "
            f"fidelity={result.fidelity_score:.2f} "
            f"(threshold={self.config.fidelity_threshold})"
        )

        return result

    def _assess_fidelity(
        self,
        original: str,
        intermediate: str,
        final: str,
        is_asp_original: bool,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess semantic fidelity using LLM comparison."""
        original_type = "ASP" if is_asp_original else "Natural Language"
        intermediate_type = "Natural Language" if is_asp_original else "ASP"
        final_type = "ASP" if is_asp_original else "Natural Language"

        prompt = FIDELITY_ASSESSMENT_PROMPT.format(
            original=original,
            original_type=original_type,
            intermediate=intermediate,
            intermediate_type=intermediate_type,
            final=final,
            final_type=final_type,
        )

        try:
            response = self._llm.query(
                question=prompt,
                output_schema=FidelityAssessmentSchema,
                temperature=0.1,  # Low temperature for consistent assessment
                max_tokens=self.config.max_tokens,
                system_prompt=TRANSLATOR_SYSTEM_PROMPT,
            )

            if hasattr(response, "content") and response.content is not None:
                result = response.content
                return {
                    "fidelity_score": result.fidelity_score,
                    "preserved": list(result.preserved_elements),
                    "lost": list(result.lost_elements),
                    "added": list(result.added_elements),
                    "confidence": result.confidence,
                }

        except Exception as e:
            logger.warning(f"Fidelity assessment failed: {e}, using heuristic")

        # Fallback: Simple heuristic-based fidelity assessment
        return self._heuristic_fidelity_assessment(original, final, is_asp_original)

    def _heuristic_fidelity_assessment(
        self,
        original: str,
        final: str,
        is_asp_original: bool,
    ) -> Dict[str, Any]:
        """Fallback heuristic fidelity assessment."""
        if is_asp_original:
            # For ASP: compare predicates and structure
            original_predicates = self._extract_predicates(original)
            final_predicates = self._extract_predicates(final)

            preserved = set(original_predicates) & set(final_predicates)
            lost = set(original_predicates) - set(final_predicates)
            added = set(final_predicates) - set(original_predicates)

            if original_predicates:
                fidelity = len(preserved) / len(original_predicates)
            else:
                fidelity = 1.0 if final.strip() == original.strip() else 0.5
        else:
            # For NL: simple word overlap
            original_words = set(original.lower().split())
            final_words = set(final.lower().split())

            common_words = original_words & final_words
            if original_words:
                fidelity = len(common_words) / len(original_words)
            else:
                fidelity = 0.5

            preserved = list(common_words)[:5]
            lost = list(original_words - final_words)[:5]
            added = list(final_words - original_words)[:5]

        return {
            "fidelity_score": fidelity,
            "preserved": list(preserved) if isinstance(preserved, set) else preserved,
            "lost": list(lost) if isinstance(lost, set) else lost,
            "added": list(added) if isinstance(added, set) else added,
            "confidence": 0.5,  # Lower confidence for heuristic
        }

    def _extract_predicates(self, asp_code: str) -> List[str]:
        """Extract predicate names from ASP code."""
        import re

        predicates = []
        # Match predicate patterns: predicate_name(...)
        pattern = r"([a-z_][a-z0-9_]*)\s*\("
        matches = re.findall(pattern, asp_code.lower())
        for match in matches:
            if match not in ["not"] and match not in predicates:
                predicates.append(match)
        return predicates

    def benchmark_against_general_llm(
        self,
        test_cases: List[Dict[str, Any]],
        general_llm: LLMInterface,
    ) -> TranslatorBenchmarkResult:
        """Benchmark this translator against a general-purpose LLM.

        Args:
            test_cases: List of test cases with 'input', 'is_asp', 'predicates' keys
            general_llm: General-purpose LLM interface for comparison

        Returns:
            TranslatorBenchmarkResult with comparative metrics
        """
        logger.info(
            f"Starting benchmark against general LLM: {len(test_cases)} test cases"
        )

        translator_fidelities: List[float] = []
        general_fidelities: List[float] = []
        translator_times: List[float] = []
        general_times: List[float] = []
        translator_successes = 0
        general_successes = 0

        for i, case in enumerate(test_cases):
            logger.debug(f"Benchmark case {i + 1}/{len(test_cases)}")

            input_text = case["input"]
            is_asp = case.get("is_asp", True)
            predicates = case.get("predicates", [])
            context = case.get("context", {})

            # Test specialized translator
            start_time = time.time()
            try:
                translator_result = self.roundtrip_validate(
                    input_text, is_asp, predicates, context
                )
                translator_fidelities.append(translator_result.fidelity_score)
                translator_times.append(translator_result.total_time_ms)
                if translator_result.fidelity_score >= self.config.fidelity_threshold:
                    translator_successes += 1
            except Exception as e:
                logger.warning(f"Translator failed on case {i + 1}: {e}")
                translator_fidelities.append(0.0)
                translator_times.append((time.time() - start_time) * 1000)

            # Test general LLM (simplified roundtrip)
            start_time = time.time()
            try:
                # Simple prompt for general LLM
                if is_asp:
                    prompt1 = f"Translate this ASP rule to plain English: {input_text}"
                    resp1 = general_llm.query(question=prompt1, temperature=0.3)
                    intermediate = str(resp1.content)

                    prompt2 = f"Translate this back to ASP: {intermediate}"
                    resp2 = general_llm.query(question=prompt2, temperature=0.3)
                    final = str(resp2.content)
                else:
                    prompt1 = f"Convert this to an ASP rule: {input_text}"
                    resp1 = general_llm.query(question=prompt1, temperature=0.3)
                    intermediate = str(resp1.content)

                    prompt2 = (
                        f"Translate this ASP back to plain English: {intermediate}"
                    )
                    resp2 = general_llm.query(question=prompt2, temperature=0.3)
                    final = str(resp2.content)

                # Heuristic fidelity for general LLM
                fidelity_info = self._heuristic_fidelity_assessment(
                    input_text, final, is_asp
                )
                general_fidelities.append(fidelity_info["fidelity_score"])
                if fidelity_info["fidelity_score"] >= self.config.fidelity_threshold:
                    general_successes += 1

            except Exception as e:
                logger.warning(f"General LLM failed on case {i + 1}: {e}")
                general_fidelities.append(0.0)

            general_times.append((time.time() - start_time) * 1000)

        # Calculate metrics
        avg_translator_fidelity = (
            sum(translator_fidelities) / len(translator_fidelities)
            if translator_fidelities
            else 0.0
        )
        avg_general_fidelity = (
            sum(general_fidelities) / len(general_fidelities)
            if general_fidelities
            else 0.0
        )

        improvement = (
            (
                (avg_translator_fidelity - avg_general_fidelity)
                / avg_general_fidelity
                * 100
            )
            if avg_general_fidelity > 0
            else 0.0
        )

        result = TranslatorBenchmarkResult(
            translator_fidelity=avg_translator_fidelity,
            general_llm_fidelity=avg_general_fidelity,
            translator_avg_time_ms=(
                sum(translator_times) / len(translator_times)
                if translator_times
                else 0.0
            ),
            general_llm_avg_time_ms=(
                sum(general_times) / len(general_times) if general_times else 0.0
            ),
            test_cases_count=len(test_cases),
            improvement_percentage=improvement,
            translator_roundtrip_success_rate=(
                translator_successes / len(test_cases) if test_cases else 0.0
            ),
            general_llm_roundtrip_success_rate=(
                general_successes / len(test_cases) if test_cases else 0.0
            ),
        )

        logger.info(
            f"Benchmark complete: "
            f"Translator={avg_translator_fidelity:.1%} vs General={avg_general_fidelity:.1%} "
            f"(improvement={improvement:+.1f}%)"
        )

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get translation statistics."""
        avg_fidelity = (
            self._total_fidelity_score / self._roundtrip_count
            if self._roundtrip_count > 0
            else 0.0
        )

        return {
            "total_translations": self._total_translations,
            "asp_to_nl_count": self._asp_to_nl_count,
            "nl_to_asp_count": self._nl_to_asp_count,
            "roundtrip_count": self._roundtrip_count,
            "successful_roundtrips": self._successful_roundtrips,
            "high_fidelity_roundtrips": self._high_fidelity_roundtrips,
            "average_fidelity_score": avg_fidelity,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                self._cache_hits / self._total_translations
                if self._total_translations > 0
                else 0.0
            ),
            "total_retries": self._total_retries,
            "cache_size": len(self._cache),
            "model": self.config.model,
            "strategy": self._strategy.strategy_type.value,
            "fidelity_threshold": self.config.fidelity_threshold,
        }

    def reset_statistics(self) -> None:
        """Reset translation statistics."""
        self._total_translations = 0
        self._asp_to_nl_count = 0
        self._nl_to_asp_count = 0
        self._roundtrip_count = 0
        self._successful_roundtrips = 0
        self._high_fidelity_roundtrips = 0
        self._cache_hits = 0
        self._total_retries = 0
        self._total_fidelity_score = 0.0
        logger.debug("Statistics reset")

    def clear_cache(self) -> None:
        """Clear the translation cache."""
        with self._cache_lock:
            cache_size = len(self._cache)
            self._cache.clear()
        logger.info(f"Cache cleared ({cache_size} entries removed)")

    def _get_cache_key(
        self,
        translation_type: str,
        source: str,
        context: Any,
    ) -> str:
        """Generate a cache key for the given translation."""
        try:
            context_str = json.dumps(context, sort_keys=True, default=str)
        except (TypeError, ValueError):
            context_str = str(context)

        with self._strategy_lock:
            strategy_type = self._strategy.strategy_type.value

        key_parts = [
            translation_type,
            source,
            context_str,
            strategy_type,
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _cache_result(self, cache_key: str, result: TranslationResult) -> None:
        """Cache a translation result."""
        if len(self._cache) >= self.config.cache_max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Cache eviction: removed {oldest_key[:16]}...")

        self._cache[cache_key] = result
        logger.debug(f"Cached result for key {cache_key[:16]}...")
