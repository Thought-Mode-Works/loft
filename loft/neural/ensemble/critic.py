"""
Critic LLM - Specialized edge case and contradiction detection.

This module implements a specialized LLM optimized for finding flaws in generated
ASP rules. It uses adversarial analysis to identify edge cases, detect contradictions
between rules, and assess generalization capabilities.

Part of Phase 6: Heterogeneous Neural Ensemble (Issue #189).

Complements the Logic Generator (Issue #188) by providing adversarial review of
generated rules before they are incorporated into the knowledge base.
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
# Specific Exceptions for Retry Logic
# =============================================================================


class LLMConnectionError(Exception):
    """Raised when LLM connection fails (network issues, timeouts)."""

    pass


class LLMResponseParsingError(Exception):
    """Raised when LLM response cannot be parsed into expected schema."""

    pass


class LLMRateLimitError(Exception):
    """Raised when LLM rate limit is exceeded."""

    pass


# =============================================================================
# Data Classes for Critic Results
# =============================================================================


@dataclass
class EdgeCase:
    """Represents an identified edge case where a rule may fail.

    Attributes:
        description: Natural language description of the edge case
        scenario: Concrete example scenario that triggers the failure
        failure_mode: How the rule fails (false positive, false negative, etc.)
        severity: Severity of the edge case (low, medium, high, critical)
        confidence: Confidence that this is a real edge case (0.0-1.0)
        suggested_fix: Optional suggestion for addressing the edge case
        related_predicates: Predicates involved in the edge case
    """

    description: str
    scenario: str
    failure_mode: str
    severity: str
    confidence: float
    suggested_fix: Optional[str] = None
    related_predicates: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate severity values."""
        valid_severities = {"low", "medium", "high", "critical"}
        if self.severity.lower() not in valid_severities:
            logger.warning(f"Unknown severity '{self.severity}', defaulting to 'medium'")
            self.severity = "medium"
        self.severity = self.severity.lower()


@dataclass
class Contradiction:
    """Represents a logical contradiction between rules.

    Attributes:
        description: Natural language description of the contradiction
        rule1: First conflicting rule
        rule2: Second conflicting rule (or existing knowledge base reference)
        conflict_type: Type of conflict (direct, implicit, contextual)
        example_trigger: Example input that triggers both rules inconsistently
        resolution_suggestion: Suggested way to resolve the contradiction
        confidence: Confidence in the contradiction assessment (0.0-1.0)
        affected_predicates: Predicates affected by the contradiction
    """

    description: str
    rule1: str
    rule2: str
    conflict_type: str
    example_trigger: Optional[str] = None
    resolution_suggestion: Optional[str] = None
    confidence: float = 0.0
    affected_predicates: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate conflict type."""
        valid_types = {"direct", "implicit", "contextual", "semantic"}
        if self.conflict_type.lower() not in valid_types:
            logger.warning(
                f"Unknown conflict type '{self.conflict_type}', defaulting to 'implicit'"
            )
            self.conflict_type = "implicit"
        self.conflict_type = self.conflict_type.lower()


@dataclass
class GeneralizationAssessment:
    """Assessment of a rule's ability to generalize.

    Attributes:
        generalization_score: Overall generalization score (0.0-1.0)
        coverage_estimate: Estimated coverage of relevant cases
        overfitting_risk: Risk of overfitting to specific cases (0.0-1.0)
        underfitting_risk: Risk of being too general (0.0-1.0)
        test_cases_needed: Suggested test cases for validation
        edge_cases_found: Edge cases identified during assessment
        confidence: Confidence in the assessment (0.0-1.0)
    """

    generalization_score: float
    coverage_estimate: float
    overfitting_risk: float
    underfitting_risk: float
    test_cases_needed: List[str] = field(default_factory=list)
    edge_cases_found: List[EdgeCase] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class CriticResult:
    """Complete result from critic analysis.

    Attributes:
        rule: The rule that was analyzed
        edge_cases: List of identified edge cases
        contradictions: List of identified contradictions
        generalization: Generalization assessment
        overall_quality_score: Overall quality score (0.0-1.0)
        recommendation: Recommendation (accept, revise, reject)
        analysis_time_ms: Time taken for analysis
        from_cache: Whether result was from cache
    """

    rule: str
    edge_cases: List[EdgeCase] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    generalization: Optional[GeneralizationAssessment] = None
    overall_quality_score: float = 0.0
    recommendation: str = "revise"
    analysis_time_ms: float = 0.0
    from_cache: bool = False


# =============================================================================
# Pydantic Schemas for LLM Structured Output
# =============================================================================


class EdgeCaseSchema(BaseModel):
    """Schema for LLM-generated edge case analysis."""

    description: str = Field(description="Natural language description of the edge case")
    scenario: str = Field(description="Concrete example scenario that triggers the failure")
    failure_mode: str = Field(
        description="How the rule fails: false_positive, false_negative, undefined_behavior"
    )
    severity: str = Field(description="Severity: low, medium, high, critical")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence this is a real edge case")
    suggested_fix: Optional[str] = Field(
        default=None, description="Suggestion for addressing the edge case"
    )
    related_predicates: List[str] = Field(default_factory=list, description="Predicates involved")


class EdgeCaseAnalysisResult(BaseModel):
    """Schema for complete edge case analysis response."""

    edge_cases: List[EdgeCaseSchema] = Field(description="List of identified edge cases")
    analysis_summary: str = Field(description="Summary of the edge case analysis")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in the analysis")


class ContradictionSchema(BaseModel):
    """Schema for LLM-generated contradiction detection."""

    description: str = Field(description="Description of the contradiction")
    rule1: str = Field(description="First conflicting rule")
    rule2: str = Field(description="Second conflicting rule or knowledge base reference")
    conflict_type: str = Field(description="Type: direct, implicit, contextual, semantic")
    example_trigger: Optional[str] = Field(
        default=None, description="Example that triggers inconsistency"
    )
    resolution_suggestion: Optional[str] = Field(
        default=None, description="How to resolve the conflict"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this contradiction")
    affected_predicates: List[str] = Field(default_factory=list, description="Affected predicates")


class ContradictionAnalysisResult(BaseModel):
    """Schema for complete contradiction analysis response."""

    contradictions: List[ContradictionSchema] = Field(
        description="List of identified contradictions"
    )
    is_consistent: bool = Field(description="Whether the rule is consistent with existing rules")
    analysis_summary: str = Field(description="Summary of the contradiction analysis")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in the analysis")


class GeneralizationSchema(BaseModel):
    """Schema for LLM-generated generalization assessment."""

    generalization_score: float = Field(ge=0.0, le=1.0, description="Overall generalization score")
    coverage_estimate: float = Field(
        ge=0.0, le=1.0, description="Estimated coverage of relevant cases"
    )
    overfitting_risk: float = Field(ge=0.0, le=1.0, description="Risk of overfitting")
    underfitting_risk: float = Field(ge=0.0, le=1.0, description="Risk of being too general")
    test_cases_needed: List[str] = Field(description="Suggested test cases for validation")
    reasoning: str = Field(description="Explanation of the assessment")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the assessment")


# =============================================================================
# Custom Exception
# =============================================================================


class CriticAnalysisError(Exception):
    """Custom exception for critic analysis failures."""

    def __init__(self, message: str, analysis_type: str = "unknown", attempts: int = 0):
        super().__init__(message)
        self.analysis_type = analysis_type
        self.attempts = attempts


# =============================================================================
# Critic Strategy Pattern
# =============================================================================


class CriticStrategyType(Enum):
    """Enum identifying critic strategy types."""

    ADVERSARIAL = "adversarial"
    COOPERATIVE = "cooperative"
    SYSTEMATIC = "systematic"
    DIALECTICAL = "dialectical"


class CriticStrategy(ABC):
    """Abstract base class for critic strategies.

    Different strategies for analyzing rules, enabling different
    approaches to edge case detection and contradiction finding.
    """

    @property
    @abstractmethod
    def strategy_type(self) -> CriticStrategyType:
        """Return the strategy type enum."""
        pass

    @abstractmethod
    def prepare_edge_case_prompt(
        self,
        base_prompt: str,
        rule: str,
        context: Dict[str, Any],
    ) -> str:
        """Prepare the prompt for edge case detection."""
        pass

    @abstractmethod
    def prepare_contradiction_prompt(
        self,
        base_prompt: str,
        rule: str,
        existing_rules: List[str],
    ) -> str:
        """Prepare the prompt for contradiction detection."""
        pass


class AdversarialStrategy(CriticStrategy):
    """Aggressively seeks out flaws and edge cases."""

    @property
    def strategy_type(self) -> CriticStrategyType:
        return CriticStrategyType.ADVERSARIAL

    def prepare_edge_case_prompt(
        self,
        base_prompt: str,
        rule: str,
        context: Dict[str, Any],
    ) -> str:
        logger.debug("Preparing prompt with ADVERSARIAL strategy")
        adversarial_addition = """

**Adversarial Analysis Instructions:**
Your goal is to find EVERY possible way this rule could fail. Consider:
1. Extreme boundary conditions
2. Unusual but valid inputs
3. Real-world scenarios that violate assumptions
4. Edge cases the rule author likely didn't consider
5. Combinations of predicates that create unexpected behavior

Be thorough and skeptical. Assume there ARE edge cases to find."""
        return base_prompt + adversarial_addition

    def prepare_contradiction_prompt(
        self,
        base_prompt: str,
        rule: str,
        existing_rules: List[str],
    ) -> str:
        adversarial_addition = """

**Adversarial Contradiction Detection:**
Actively try to find conflicts. Consider:
1. Can both rules fire simultaneously with conflicting conclusions?
2. Are there implicit assumptions that contradict existing rules?
3. Do variable bindings create unexpected interactions?
4. Could temporal or contextual factors create conflicts?"""
        return base_prompt + adversarial_addition


class CooperativeStrategy(CriticStrategy):
    """Provides constructive feedback with suggested improvements."""

    @property
    def strategy_type(self) -> CriticStrategyType:
        return CriticStrategyType.COOPERATIVE

    def prepare_edge_case_prompt(
        self,
        base_prompt: str,
        rule: str,
        context: Dict[str, Any],
    ) -> str:
        logger.debug("Preparing prompt with COOPERATIVE strategy")
        cooperative_addition = """

**Constructive Analysis Instructions:**
Identify edge cases while suggesting how to address them.
For each edge case found:
1. Explain why it matters
2. Suggest a concrete fix
3. Consider the trade-off between coverage and precision"""
        return base_prompt + cooperative_addition

    def prepare_contradiction_prompt(
        self,
        base_prompt: str,
        rule: str,
        existing_rules: List[str],
    ) -> str:
        cooperative_addition = """

**Constructive Contradiction Analysis:**
For any contradictions found:
1. Explain the semantic conflict clearly
2. Suggest which rule should take precedence
3. Propose a resolution that preserves both rules' intent"""
        return base_prompt + cooperative_addition


class SystematicStrategy(CriticStrategy):
    """Uses systematic checklist-based analysis."""

    @property
    def strategy_type(self) -> CriticStrategyType:
        return CriticStrategyType.SYSTEMATIC

    def prepare_edge_case_prompt(
        self,
        base_prompt: str,
        rule: str,
        context: Dict[str, Any],
    ) -> str:
        logger.debug("Preparing prompt with SYSTEMATIC strategy")
        systematic_addition = """

**Systematic Analysis Checklist:**
Check each category:
1. BOUNDARY CONDITIONS: Empty sets, single elements, maximum values
2. TYPE EDGE CASES: Null values, type mismatches, invalid formats
3. TEMPORAL: Time-based conditions, ordering dependencies
4. CONTEXTUAL: Jurisdiction variations, domain-specific exceptions
5. COMBINATORIAL: Multi-predicate interactions
6. NEGATION: Double negation, default assumptions"""
        return base_prompt + systematic_addition

    def prepare_contradiction_prompt(
        self,
        base_prompt: str,
        rule: str,
        existing_rules: List[str],
    ) -> str:
        systematic_addition = """

**Systematic Contradiction Checklist:**
1. DIRECT: Same head predicate, different conclusions
2. IMPLICIT: Rules that together lead to contradictions
3. PRECEDENCE: Conflicting specificity levels
4. NEGATION: Negated predicates vs positive assertions
5. TRANSITIVITY: Chains of implications leading to conflicts"""
        return base_prompt + systematic_addition


class DialecticalStrategy(CriticStrategy):
    """Uses thesis-antithesis-synthesis approach."""

    @property
    def strategy_type(self) -> CriticStrategyType:
        return CriticStrategyType.DIALECTICAL

    def prepare_edge_case_prompt(
        self,
        base_prompt: str,
        rule: str,
        context: Dict[str, Any],
    ) -> str:
        logger.debug("Preparing prompt with DIALECTICAL strategy")
        dialectical_addition = """

**Dialectical Analysis:**
1. THESIS: What does the rule claim to do?
2. ANTITHESIS: What counter-examples challenge this claim?
3. SYNTHESIS: How can the rule be refined to address counter-examples?

Engage in internal debate about the rule's validity."""
        return base_prompt + dialectical_addition

    def prepare_contradiction_prompt(
        self,
        base_prompt: str,
        rule: str,
        existing_rules: List[str],
    ) -> str:
        dialectical_addition = """

**Dialectical Contradiction Analysis:**
1. THESIS: Assume the new rule is correct
2. ANTITHESIS: Find existing rules that contradict this
3. SYNTHESIS: Propose a unified interpretation that resolves conflicts"""
        return base_prompt + dialectical_addition


def create_critic_strategy(strategy_type: CriticStrategyType) -> CriticStrategy:
    """Factory function to create critic strategy instances."""
    strategy_map: Dict[CriticStrategyType, Type[CriticStrategy]] = {
        CriticStrategyType.ADVERSARIAL: AdversarialStrategy,
        CriticStrategyType.COOPERATIVE: CooperativeStrategy,
        CriticStrategyType.SYSTEMATIC: SystematicStrategy,
        CriticStrategyType.DIALECTICAL: DialecticalStrategy,
    }
    return strategy_map[strategy_type]()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CriticConfig:
    """Configuration for the Critic LLM.

    Attributes:
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        strategy: Critic strategy type
        max_edge_cases: Maximum edge cases to find per rule
        max_contradictions: Maximum contradictions to report
        confidence_threshold: Minimum confidence to report findings
        max_retries: Maximum retry attempts
        retry_base_delay_seconds: Base delay for exponential backoff
        enable_cache: Enable caching
        cache_max_size: Maximum cache entries
    """

    model: str = "claude-3-5-haiku-20241022"
    temperature: float = 0.4  # Slightly higher for creative adversarial thinking
    max_tokens: int = 4096
    strategy: CriticStrategyType = CriticStrategyType.ADVERSARIAL
    max_edge_cases: int = 10
    max_contradictions: int = 10
    confidence_threshold: float = 0.3
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    enable_cache: bool = True
    cache_max_size: int = 100


# =============================================================================
# System Prompts
# =============================================================================


CRITIC_SYSTEM_PROMPT = """You are an expert ASP (Answer Set Programming) rule critic specialized in finding flaws, edge cases, and contradictions in formal logic rules.

Your task is to adversarially analyze ASP rules to identify:
1. Edge cases where the rule may produce incorrect results
2. Contradictions with existing knowledge
3. Overgeneralization or undergeneralization issues

**Critical Analysis Skills:**
1. Identify boundary conditions and corner cases
2. Find scenarios where the rule's assumptions break down
3. Detect logical contradictions with other rules
4. Assess whether the rule captures the intended semantics
5. Consider jurisdiction-specific variations
6. Evaluate variable safety and predicate usage

**ASP Syntax Understanding:**
- Rules have the form: head :- body.
- Variables are UPPERCASE (X, Y, Contract)
- Constants are lowercase (alice, contract1)
- Negation uses 'not' keyword
- Variable safety: all head variables must appear in positive body literals

**Analysis Standards:**
- Be thorough but focused
- Provide concrete examples for each finding
- Rate confidence in your findings
- Suggest fixes when possible
- Consider real-world applicability"""


EDGE_CASE_PROMPT_TEMPLATE = """Analyze this ASP rule for edge cases and potential failure modes.

**Rule to Analyze:**
```asp
{rule}
```

**Context:**
{context}

**Available Predicates:**
{predicates}

**Analysis Task:**
Find edge cases where this rule may:
1. Produce false positives (fires when it shouldn't)
2. Produce false negatives (doesn't fire when it should)
3. Have undefined behavior with certain inputs
4. Make assumptions that don't hold in edge scenarios

For each edge case, provide:
- Description of the edge case
- Concrete scenario/example
- How the rule fails
- Severity (low/medium/high/critical)
- Confidence in this finding
- Suggested fix (if applicable)

Generate your analysis:"""


CONTRADICTION_PROMPT_TEMPLATE = """Analyze this ASP rule for contradictions with existing rules.

**New Rule to Analyze:**
```asp
{rule}
```

**Existing Rules to Check Against:**
```asp
{existing_rules}
```

**Analysis Task:**
Find contradictions where:
1. The new rule directly conflicts with existing rules
2. Implicit contradictions arise from rule interactions
3. The new rule makes assumptions that conflict with existing knowledge
4. Variable bindings could cause inconsistent behavior

For each contradiction, provide:
- Description of the conflict
- The two conflicting rules
- Type of conflict (direct/implicit/contextual/semantic)
- Example trigger (inputs that expose the conflict)
- Resolution suggestion
- Confidence in this finding

Generate your analysis:"""


GENERALIZATION_PROMPT_TEMPLATE = """Assess this ASP rule's ability to generalize.

**Rule to Assess:**
```asp
{rule}
```

**Domain Context:**
{domain}

**Test Cases Available:**
{test_cases}

**Assessment Task:**
Evaluate:
1. Does the rule generalize beyond the training examples?
2. Is it overfitted to specific cases?
3. Is it too general (underfitted)?
4. What coverage does it have?

Provide:
- Generalization score (0.0-1.0)
- Coverage estimate
- Overfitting risk score
- Underfitting risk score
- Additional test cases needed
- Reasoning for your assessment

Generate your assessment:"""


# =============================================================================
# Abstract Base Class
# =============================================================================


class Critic(ABC):
    """Abstract base class for critic implementations.

    Defines the interface that all critic implementations must follow,
    enabling future extensibility for different critic approaches.
    """

    @abstractmethod
    def find_edge_cases(
        self,
        rule: str,
        context: Dict[str, Any],
    ) -> List[EdgeCase]:
        """Identify edge cases where the rule may fail."""
        pass

    @abstractmethod
    def detect_contradictions(
        self,
        rule: str,
        existing_rules: List[str],
    ) -> List[Contradiction]:
        """Find logical contradictions with existing knowledge base."""
        pass

    @abstractmethod
    def assess_generalization(
        self,
        rule: str,
        test_cases: List[Dict[str, Any]],
    ) -> GeneralizationAssessment:
        """Evaluate the rule's ability to generalize."""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        pass


# =============================================================================
# Main Implementation
# =============================================================================


class CriticLLM(Critic):
    """Specialized LLM for finding rule flaws.

    This class provides adversarial analysis of ASP rules, identifying
    edge cases, contradictions, and generalization issues to ensure
    rule quality before incorporation into the knowledge base.

    Features:
    - Strategy Pattern for different analysis approaches
    - Exponential backoff retry logic
    - LRU caching for repeated analyses
    - Comprehensive logging
    - Integration with dialectical validation pipeline

    Example:
        >>> from loft.neural.providers import AnthropicProvider
        >>> provider = AnthropicProvider(api_key, model="claude-3-5-haiku-20241022")
        >>> config = CriticConfig(strategy=CriticStrategyType.ADVERSARIAL)
        >>> critic = CriticLLM(provider, config)
        >>> edge_cases = critic.find_edge_cases(
        ...     rule="enforceable(C) :- contract(C), signed(C).",
        ...     context={"domain": "contracts"}
        ... )
        >>> for ec in edge_cases:
        ...     print(f"Edge case: {ec.description}")
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: Optional[CriticConfig] = None,
        strategy: Optional[CriticStrategy] = None,
    ):
        """Initialize the Critic LLM.

        Args:
            provider: LLM provider instance
            config: Configuration for analysis parameters
            strategy: Optional custom critic strategy (overrides config)

        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("provider cannot be None")

        self.provider = provider
        self.config = config or CriticConfig()
        self._llm = LLMInterface(provider, enable_cache=True, max_retries=3)

        # Thread-safe strategy access
        self._strategy_lock = threading.Lock()

        # Initialize strategy
        if strategy:
            self._strategy = strategy
            logger.info(f"Using custom strategy: {strategy.strategy_type.value}")
        else:
            self._strategy = create_critic_strategy(self.config.strategy)
            logger.info(f"Using config strategy: {self.config.strategy.value}")

        # Statistics tracking
        self._total_analyses = 0
        self._edge_case_analyses = 0
        self._contradiction_analyses = 0
        self._generalization_analyses = 0
        self._total_edge_cases_found = 0
        self._total_contradictions_found = 0
        self._cache_hits = 0
        self._total_retries = 0

        # Cache for repeated analyses
        self._cache: Dict[str, CriticResult] = {}
        self._cache_lock = threading.Lock()

        logger.info(
            f"Initialized CriticLLM: "
            f"strategy={self._strategy.strategy_type.value}, "
            f"model={self.config.model}, "
            f"cache={'enabled' if self.config.enable_cache else 'disabled'}"
        )

    def set_strategy(self, strategy: CriticStrategy) -> None:
        """Change the critic strategy at runtime.

        Thread-safe method to update the critic strategy.

        Args:
            strategy: New strategy to use for analysis

        Raises:
            ValueError: If strategy is None
        """
        if strategy is None:
            raise ValueError("strategy cannot be None")

        with self._strategy_lock:
            old_strategy = self._strategy.strategy_type.value
            self._strategy = strategy
            logger.info(f"Strategy changed: {old_strategy} -> {strategy.strategy_type.value}")

    def find_edge_cases(
        self,
        rule: str,
        context: Dict[str, Any],
    ) -> List[EdgeCase]:
        """Identify edge cases where the rule may fail.

        Args:
            rule: The ASP rule to analyze
            context: Context including domain, predicates, examples

        Returns:
            List of EdgeCase objects describing potential failure modes

        Raises:
            ValueError: If rule is empty
            CriticAnalysisError: If analysis fails after all retries
        """
        # Input validation
        if not rule or not rule.strip():
            raise ValueError("rule cannot be empty")

        start_time = time.time()
        self._total_analyses += 1
        self._edge_case_analyses += 1

        logger.debug(f"Starting edge case analysis for rule: {rule[:50]}...")

        # Check cache with thread safety
        cache_key = self._get_cache_key("edge_cases", rule, context)
        with self._cache_lock:
            if self.config.enable_cache and cache_key in self._cache:
                self._cache_hits += 1
                logger.debug("Cache hit for edge case analysis")
                return self._cache[cache_key].edge_cases

        # Get strategy with thread safety
        with self._strategy_lock:
            current_strategy = self._strategy

        # Prepare context strings
        predicates_str = (
            "\n".join(f"  - {p}" for p in context.get("predicates", [])) or "  (none specified)"
        )
        context_str = (
            "\n".join(f"  - {k}: {v}" for k, v in context.items() if k != "predicates")
            or "  (none specified)"
        )

        # Build prompt
        base_prompt = EDGE_CASE_PROMPT_TEMPLATE.format(
            rule=rule,
            context=context_str,
            predicates=predicates_str,
        )

        prompt = current_strategy.prepare_edge_case_prompt(base_prompt, rule, context)

        edge_cases: List[EdgeCase] = []

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Edge case analysis attempt {attempt + 1}")

                response = self._llm.query(
                    question=prompt,
                    output_schema=EdgeCaseAnalysisResult,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    system_prompt=CRITIC_SYSTEM_PROMPT,
                )

                # Validate response content
                if not hasattr(response, "content") or response.content is None:
                    raise LLMResponseParsingError("LLM response has no content")

                result = response.content
                if not isinstance(result, EdgeCaseAnalysisResult):
                    raise LLMResponseParsingError(
                        f"Expected EdgeCaseAnalysisResult, got {type(result).__name__}"
                    )

                for ec_schema in result.edge_cases[: self.config.max_edge_cases]:
                    if ec_schema.confidence >= self.config.confidence_threshold:
                        edge_cases.append(
                            EdgeCase(
                                description=ec_schema.description,
                                scenario=ec_schema.scenario,
                                failure_mode=ec_schema.failure_mode,
                                severity=ec_schema.severity,
                                confidence=ec_schema.confidence,
                                suggested_fix=ec_schema.suggested_fix,
                                related_predicates=list(ec_schema.related_predicates),
                            )
                        )

                self._total_edge_cases_found += len(edge_cases)
                logger.info(f"Found {len(edge_cases)} edge cases in rule")
                break

            except (ValidationError, LLMResponseParsingError) as e:
                # Parsing/validation errors - retry with backoff
                logger.warning(f"Edge case analysis attempt {attempt + 1} failed: {e}")
                self._total_retries += 1

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay_seconds * (2**attempt)
                    logger.debug(f"Waiting {delay:.1f}s before retry")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Edge case analysis failed after {self.config.max_retries} attempts"
                    )

            except (ConnectionError, TimeoutError, OSError) as e:
                # Network errors - retry with backoff
                logger.warning(f"Edge case analysis attempt {attempt + 1} failed (network): {e}")
                self._total_retries += 1

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay_seconds * (2**attempt)
                    logger.debug(f"Waiting {delay:.1f}s before retry")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Edge case analysis failed after {self.config.max_retries} attempts"
                    )

            except KeyboardInterrupt:
                # Don't retry on user interrupt
                raise

        # Cache result with thread safety
        if self.config.enable_cache and edge_cases:
            analysis_time_ms = (time.time() - start_time) * 1000
            with self._cache_lock:
                self._cache_result(cache_key, rule, edge_cases=edge_cases, time_ms=analysis_time_ms)

        return edge_cases

    def detect_contradictions(
        self,
        rule: str,
        existing_rules: List[str],
    ) -> List[Contradiction]:
        """Find logical contradictions with existing knowledge base.

        Args:
            rule: The new ASP rule to check
            existing_rules: List of existing ASP rules to check against

        Returns:
            List of Contradiction objects describing conflicts

        Raises:
            ValueError: If rule is empty
        """
        # Input validation
        if not rule or not rule.strip():
            raise ValueError("rule cannot be empty")

        start_time = time.time()
        self._total_analyses += 1
        self._contradiction_analyses += 1

        logger.debug(
            f"Starting contradiction analysis for rule: {rule[:50]}... "
            f"against {len(existing_rules)} existing rules"
        )

        # Check cache with thread safety (convert list to tuple for hashability)
        existing_rules_tuple = tuple(existing_rules) if existing_rules else ()
        cache_key = self._get_cache_key("contradictions", rule, {"existing": existing_rules_tuple})
        with self._cache_lock:
            if self.config.enable_cache and cache_key in self._cache:
                self._cache_hits += 1
                logger.debug("Cache hit for contradiction analysis")
                return self._cache[cache_key].contradictions

        # Get strategy with thread safety
        with self._strategy_lock:
            current_strategy = self._strategy

        # Build prompt
        existing_rules_str = "\n".join(existing_rules) if existing_rules else "(none)"
        base_prompt = CONTRADICTION_PROMPT_TEMPLATE.format(
            rule=rule,
            existing_rules=existing_rules_str,
        )

        prompt = current_strategy.prepare_contradiction_prompt(base_prompt, rule, existing_rules)

        contradictions: List[Contradiction] = []

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Contradiction analysis attempt {attempt + 1}")

                response = self._llm.query(
                    question=prompt,
                    output_schema=ContradictionAnalysisResult,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    system_prompt=CRITIC_SYSTEM_PROMPT,
                )

                # Validate response content
                if not hasattr(response, "content") or response.content is None:
                    raise LLMResponseParsingError("LLM response has no content")

                result = response.content
                if not isinstance(result, ContradictionAnalysisResult):
                    raise LLMResponseParsingError(
                        f"Expected ContradictionAnalysisResult, got {type(result).__name__}"
                    )

                for c_schema in result.contradictions[: self.config.max_contradictions]:
                    if c_schema.confidence >= self.config.confidence_threshold:
                        contradictions.append(
                            Contradiction(
                                description=c_schema.description,
                                rule1=c_schema.rule1,
                                rule2=c_schema.rule2,
                                conflict_type=c_schema.conflict_type,
                                example_trigger=c_schema.example_trigger,
                                resolution_suggestion=c_schema.resolution_suggestion,
                                confidence=c_schema.confidence,
                                affected_predicates=list(c_schema.affected_predicates),
                            )
                        )

                self._total_contradictions_found += len(contradictions)
                logger.info(
                    f"Found {len(contradictions)} contradictions "
                    f"(consistent={result.is_consistent})"
                )
                break

            except (ValidationError, LLMResponseParsingError) as e:
                # Parsing/validation errors - retry with backoff
                logger.warning(f"Contradiction analysis attempt {attempt + 1} failed: {e}")
                self._total_retries += 1

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay_seconds * (2**attempt)
                    logger.debug(f"Waiting {delay:.1f}s before retry")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Contradiction analysis failed after {self.config.max_retries} attempts"
                    )

            except (ConnectionError, TimeoutError, OSError) as e:
                # Network errors - retry with backoff
                logger.warning(
                    f"Contradiction analysis attempt {attempt + 1} failed (network): {e}"
                )
                self._total_retries += 1

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay_seconds * (2**attempt)
                    logger.debug(f"Waiting {delay:.1f}s before retry")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Contradiction analysis failed after {self.config.max_retries} attempts"
                    )

            except KeyboardInterrupt:
                # Don't retry on user interrupt
                raise

        # Cache result with thread safety
        if self.config.enable_cache:
            analysis_time_ms = (time.time() - start_time) * 1000
            with self._cache_lock:
                self._cache_result(
                    cache_key,
                    rule,
                    contradictions=contradictions,
                    time_ms=analysis_time_ms,
                )

        return contradictions

    def assess_generalization(
        self,
        rule: str,
        test_cases: List[Dict[str, Any]],
        domain: str = "legal",
    ) -> GeneralizationAssessment:
        """Evaluate the rule's ability to generalize.

        Args:
            rule: The ASP rule to assess
            test_cases: Test cases to evaluate against
            domain: Domain context

        Returns:
            GeneralizationAssessment with scores and recommendations

        Raises:
            ValueError: If rule is empty
        """
        # Input validation
        if not rule or not rule.strip():
            raise ValueError("rule cannot be empty")

        self._total_analyses += 1
        self._generalization_analyses += 1

        logger.debug(f"Starting generalization assessment for rule: {rule[:50]}...")

        # Build test cases string
        test_cases_str = (
            "\n".join(
                f"  Case {i + 1}: {tc.get('description', str(tc))}"
                for i, tc in enumerate(test_cases[:10])  # Limit to 10 cases
            )
            or "  (none provided)"
        )

        base_prompt = GENERALIZATION_PROMPT_TEMPLATE.format(
            rule=rule,
            domain=domain,
            test_cases=test_cases_str,
        )

        assessment = GeneralizationAssessment(
            generalization_score=0.5,
            coverage_estimate=0.5,
            overfitting_risk=0.5,
            underfitting_risk=0.5,
            confidence=0.0,
        )

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Generalization assessment attempt {attempt + 1}")

                response = self._llm.query(
                    question=base_prompt,
                    output_schema=GeneralizationSchema,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    system_prompt=CRITIC_SYSTEM_PROMPT,
                )

                # Validate response content
                if not hasattr(response, "content") or response.content is None:
                    raise LLMResponseParsingError("LLM response has no content")

                result = response.content
                if not isinstance(result, GeneralizationSchema):
                    raise LLMResponseParsingError(
                        f"Expected GeneralizationSchema, got {type(result).__name__}"
                    )

                assessment = GeneralizationAssessment(
                    generalization_score=result.generalization_score,
                    coverage_estimate=result.coverage_estimate,
                    overfitting_risk=result.overfitting_risk,
                    underfitting_risk=result.underfitting_risk,
                    test_cases_needed=list(result.test_cases_needed),
                    confidence=result.confidence,
                )

                logger.info(
                    f"Generalization assessment complete: "
                    f"score={assessment.generalization_score:.2f}, "
                    f"coverage={assessment.coverage_estimate:.2f}"
                )
                break

            except (ValidationError, LLMResponseParsingError) as e:
                # Parsing/validation errors - retry with backoff
                logger.warning(f"Generalization assessment attempt {attempt + 1} failed: {e}")
                self._total_retries += 1

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay_seconds * (2**attempt)
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Generalization assessment failed after {self.config.max_retries} attempts"
                    )

            except (ConnectionError, TimeoutError, OSError) as e:
                # Network errors - retry with backoff
                logger.warning(
                    f"Generalization assessment attempt {attempt + 1} failed (network): {e}"
                )
                self._total_retries += 1

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay_seconds * (2**attempt)
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Generalization assessment failed after {self.config.max_retries} attempts"
                    )

            except KeyboardInterrupt:
                # Don't retry on user interrupt
                raise

        return assessment

    def analyze_rule_comprehensive(
        self,
        rule: str,
        existing_rules: List[str],
        context: Dict[str, Any],
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> CriticResult:
        """Perform comprehensive analysis of a rule.

        Combines edge case detection, contradiction detection, and
        generalization assessment into a single comprehensive result.

        Args:
            rule: The ASP rule to analyze
            existing_rules: Existing rules to check for contradictions
            context: Context for edge case analysis
            test_cases: Optional test cases for generalization

        Returns:
            CriticResult with complete analysis
        """
        start_time = time.time()

        logger.info(f"Starting comprehensive analysis for rule: {rule[:50]}...")

        # Run all analyses
        edge_cases = self.find_edge_cases(rule, context)
        contradictions = self.detect_contradictions(rule, existing_rules)

        generalization = None
        if test_cases:
            generalization = self.assess_generalization(
                rule, test_cases, context.get("domain", "legal")
            )

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(edge_cases, contradictions, generalization)

        # Determine recommendation
        recommendation = self._determine_recommendation(quality_score, edge_cases, contradictions)

        analysis_time_ms = (time.time() - start_time) * 1000

        result = CriticResult(
            rule=rule,
            edge_cases=edge_cases,
            contradictions=contradictions,
            generalization=generalization,
            overall_quality_score=quality_score,
            recommendation=recommendation,
            analysis_time_ms=analysis_time_ms,
        )

        logger.info(
            f"Comprehensive analysis complete: "
            f"quality={quality_score:.2f}, recommendation={recommendation}, "
            f"edge_cases={len(edge_cases)}, contradictions={len(contradictions)}"
        )

        return result

    def _calculate_quality_score(
        self,
        edge_cases: List[EdgeCase],
        contradictions: List[Contradiction],
        generalization: Optional[GeneralizationAssessment],
    ) -> float:
        """Calculate overall quality score based on analysis results."""
        score = 1.0

        # Deduct for edge cases based on severity
        severity_weights = {"low": 0.02, "medium": 0.05, "high": 0.1, "critical": 0.2}
        for ec in edge_cases:
            deduction = severity_weights.get(ec.severity, 0.05) * ec.confidence
            score -= deduction

        # Deduct for contradictions
        for c in contradictions:
            deduction = 0.15 * c.confidence
            score -= deduction

        # Factor in generalization score
        if generalization:
            score *= 0.5 + 0.5 * generalization.generalization_score

        return max(0.0, min(1.0, score))

    def _determine_recommendation(
        self,
        quality_score: float,
        edge_cases: List[EdgeCase],
        contradictions: List[Contradiction],
    ) -> str:
        """Determine recommendation based on analysis."""
        critical_edge_cases = sum(1 for ec in edge_cases if ec.severity == "critical")
        high_confidence_contradictions = sum(1 for c in contradictions if c.confidence > 0.7)

        if critical_edge_cases > 0 or high_confidence_contradictions > 0:
            return "reject"
        elif quality_score < 0.5:
            return "reject"
        elif quality_score < 0.7 or len(edge_cases) > 3:
            return "revise"
        else:
            return "accept"

    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return {
            "total_analyses": self._total_analyses,
            "edge_case_analyses": self._edge_case_analyses,
            "contradiction_analyses": self._contradiction_analyses,
            "generalization_analyses": self._generalization_analyses,
            "total_edge_cases_found": self._total_edge_cases_found,
            "total_contradictions_found": self._total_contradictions_found,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                self._cache_hits / self._total_analyses if self._total_analyses > 0 else 0.0
            ),
            "total_retries": self._total_retries,
            "cache_size": len(self._cache),
            "model": self.config.model,
            "strategy": self._strategy.strategy_type.value,
        }

    def reset_statistics(self) -> None:
        """Reset analysis statistics."""
        self._total_analyses = 0
        self._edge_case_analyses = 0
        self._contradiction_analyses = 0
        self._generalization_analyses = 0
        self._total_edge_cases_found = 0
        self._total_contradictions_found = 0
        self._cache_hits = 0
        self._total_retries = 0
        logger.debug("Statistics reset")

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared ({cache_size} entries removed)")

    def _get_cache_key(
        self,
        analysis_type: str,
        rule: str,
        context: Any,
    ) -> str:
        """Generate a cache key for the given analysis.

        Uses JSON serialization for consistent, hashable representation
        of context objects that may contain lists or other unhashable types.
        """
        # Serialize context to JSON for consistent hashing
        # Use sort_keys for deterministic ordering, default=str for non-serializable types
        try:
            context_str = json.dumps(context, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback to str() if JSON serialization fails
            context_str = str(context)

        # Get strategy type with thread safety
        with self._strategy_lock:
            strategy_type = self._strategy.strategy_type.value

        key_parts = [
            analysis_type,
            rule,
            context_str,
            strategy_type,
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _cache_result(
        self,
        cache_key: str,
        rule: str,
        edge_cases: Optional[List[EdgeCase]] = None,
        contradictions: Optional[List[Contradiction]] = None,
        time_ms: float = 0.0,
    ) -> None:
        """Cache an analysis result."""
        if len(self._cache) >= self.config.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Cache eviction: removed {oldest_key[:16]}...")

        self._cache[cache_key] = CriticResult(
            rule=rule,
            edge_cases=edge_cases or [],
            contradictions=contradictions or [],
            analysis_time_ms=time_ms,
            from_cache=False,
        )
