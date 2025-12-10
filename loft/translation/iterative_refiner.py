"""
Iterative refinement for translation fidelity improvement.

This module implements an iterative refinement mechanism that uses LLM feedback
to improve ASP→NL translations until a fidelity threshold is met.

Issue #223: Iterative refinement loop for translation fidelity
"""

from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from loft.neural.llm_interface import LLMInterface

from loft.translation.asp_to_nl import asp_to_nl_statement
from loft.translation.quality import compute_quality_metrics, QualityMetrics


@dataclass
class RefinementMetrics:
    """Metrics captured during a single refinement iteration."""

    iteration: int
    fidelity_before: float
    fidelity_after: float
    improvement: float
    tokens_used: int = 0
    cost_usd: float = 0.0


@dataclass
class RefinementResult:
    """Result of iterative refinement process."""

    final_translation: str
    iterations_used: int
    initial_fidelity: float
    final_fidelity: float
    improvement: float
    converged: bool
    iteration_metrics: List[RefinementMetrics] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def cost_per_iteration(self) -> float:
        """Average cost per iteration."""
        if self.iterations_used == 0:
            return 0.0
        return self.total_cost_usd / self.iterations_used

    @property
    def improvement_per_iteration(self) -> float:
        """Average improvement per iteration."""
        if self.iterations_used == 0:
            return 0.0
        return self.improvement / self.iterations_used


class IterativeTranslationRefiner:
    """
    Refines ASP→NL translations iteratively until fidelity threshold is met.

    Uses LLM feedback to improve translations with strict cost controls:
    - Maximum iteration limit to prevent infinite loops
    - Cheap model (Haiku) for refinement to minimize costs
    - Improvement threshold to detect diminishing returns
    - Comprehensive cost tracking

    Example:
        >>> from loft.neural.llm_interface import LLMInterface
        >>> from loft.neural.providers import AnthropicProvider
        >>>
        >>> provider = AnthropicProvider(api_key="...", model="claude-3-5-haiku-20241022")
        >>> llm = LLMInterface(provider=provider)
        >>> refiner = IterativeTranslationRefiner(llm)
        >>>
        >>> result = refiner.refine(
        ...     original_nl="A contract is valid if it has consideration.",
        ...     asp_code="contract_valid(C) :- has_consideration(C)."
        ... )
        >>> print(f"Converged: {result.converged}, Fidelity: {result.final_fidelity:.2f}")
    """

    # Default refinement prompt template
    REFINEMENT_PROMPT_TEMPLATE = """Improve this translation to better match the original meaning.

Original statement: {original_nl}
Current translation: {current_translation}

Current quality metrics:
- Completeness: {completeness:.2f} (all predicates represented)
- Readability: {readability:.2f} (grammar and structure)
- Overall fidelity: {fidelity:.2f} (target: {target_fidelity:.2f})

Instructions:
1. Preserve ALL key legal terms from the original
2. Maintain the declarative sentence structure
3. Remove any content not in the original (hallucinations)
4. Keep the same meaning, just improve phrasing
5. Ensure all predicates from the ASP code are mentioned

ASP code for reference: {asp_code}

Improved translation (provide ONLY the improved sentence, no explanation):"""

    def __init__(
        self,
        llm_interface: "LLMInterface",
        max_iterations: int = 3,
        fidelity_threshold: float = 0.85,
        improvement_threshold: float = 0.05,
    ):
        """
        Initialize the iterative refiner.

        Args:
            llm_interface: LLM interface for refinement queries
            max_iterations: Maximum refinement iterations (default: 3)
            fidelity_threshold: Target fidelity score to achieve (default: 0.85)
            improvement_threshold: Minimum improvement to continue (default: 0.05)
        """
        self.llm = llm_interface
        self.max_iterations = max_iterations
        self.fidelity_threshold = fidelity_threshold
        self.improvement_threshold = improvement_threshold

    def refine(
        self,
        original_nl: str,
        asp_code: str,
        initial_translation: Optional[str] = None,
    ) -> RefinementResult:
        """
        Iteratively refine translation until fidelity threshold is met.

        The refinement loop terminates when:
        1. Fidelity threshold is reached (converged=True)
        2. Maximum iterations reached (converged=False)
        3. Improvement falls below threshold (diminishing returns)

        Args:
            original_nl: Original natural language text
            asp_code: ASP code that was generated from original_nl
            initial_translation: Optional initial ASP→NL translation
                               (if None, will generate using asp_to_nl_statement)

        Returns:
            RefinementResult with final translation and metrics
        """
        # Get initial translation if not provided
        current_translation = initial_translation or asp_to_nl_statement(asp_code)

        # Calculate initial fidelity
        initial_metrics = self._calculate_fidelity(original_nl, current_translation)
        initial_fidelity = initial_metrics.overall
        previous_fidelity = initial_fidelity

        logger.info(
            f"Starting iterative refinement: initial_fidelity={initial_fidelity:.2f}, "
            f"target={self.fidelity_threshold:.2f}"
        )

        iteration_metrics: List[RefinementMetrics] = []
        total_tokens = 0
        total_cost = 0.0

        for iteration in range(self.max_iterations):
            current_metrics = self._calculate_fidelity(original_nl, current_translation)
            current_fidelity = current_metrics.overall

            logger.debug(
                f"Iteration {iteration}: fidelity={current_fidelity:.2f}, "
                f"completeness={current_metrics.completeness:.2f}"
            )

            # Check if we've reached the target
            if current_fidelity >= self.fidelity_threshold:
                logger.info(
                    f"Converged at iteration {iteration}: "
                    f"fidelity={current_fidelity:.2f} >= {self.fidelity_threshold:.2f}"
                )
                return RefinementResult(
                    final_translation=current_translation,
                    iterations_used=iteration,
                    initial_fidelity=initial_fidelity,
                    final_fidelity=current_fidelity,
                    improvement=current_fidelity - initial_fidelity,
                    converged=True,
                    iteration_metrics=iteration_metrics,
                    total_tokens=total_tokens,
                    total_cost_usd=total_cost,
                )

            # Check for diminishing returns (after first iteration)
            if iteration > 0:
                improvement = current_fidelity - previous_fidelity
                if improvement < self.improvement_threshold:
                    logger.info(
                        f"Stopping due to diminishing returns at iteration {iteration}: "
                        f"improvement={improvement:.3f} < {self.improvement_threshold:.3f}"
                    )
                    return RefinementResult(
                        final_translation=current_translation,
                        iterations_used=iteration,
                        initial_fidelity=initial_fidelity,
                        final_fidelity=current_fidelity,
                        improvement=current_fidelity - initial_fidelity,
                        converged=False,
                        iteration_metrics=iteration_metrics,
                        total_tokens=total_tokens,
                        total_cost_usd=total_cost,
                    )

            # Request refinement from LLM
            refined_translation, tokens, cost = self._request_refinement(
                original_nl=original_nl,
                current_translation=current_translation,
                asp_code=asp_code,
                metrics=current_metrics,
            )

            # Track metrics for this iteration
            new_metrics = self._calculate_fidelity(original_nl, refined_translation)
            iteration_metric = RefinementMetrics(
                iteration=iteration,
                fidelity_before=current_fidelity,
                fidelity_after=new_metrics.overall,
                improvement=new_metrics.overall - current_fidelity,
                tokens_used=tokens,
                cost_usd=cost,
            )
            iteration_metrics.append(iteration_metric)

            total_tokens += tokens
            total_cost += cost
            previous_fidelity = current_fidelity
            current_translation = refined_translation

        # Max iterations reached
        final_metrics = self._calculate_fidelity(original_nl, current_translation)
        final_fidelity = final_metrics.overall

        logger.info(
            f"Max iterations ({self.max_iterations}) reached: "
            f"final_fidelity={final_fidelity:.2f}"
        )

        return RefinementResult(
            final_translation=current_translation,
            iterations_used=self.max_iterations,
            initial_fidelity=initial_fidelity,
            final_fidelity=final_fidelity,
            improvement=final_fidelity - initial_fidelity,
            converged=final_fidelity >= self.fidelity_threshold,
            iteration_metrics=iteration_metrics,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
        )

    def _calculate_fidelity(self, original_nl: str, translation: str) -> QualityMetrics:
        """
        Calculate fidelity metrics for a translation.

        Uses compute_quality_metrics which measures:
        - Completeness: Are all predicates represented?
        - Readability: Grammar and sentence structure
        - Fidelity: Semantic preservation
        - Overall: Combined score

        Args:
            original_nl: Original natural language
            translation: Current translation to evaluate

        Returns:
            QualityMetrics with all scores
        """
        # Note: compute_quality_metrics expects ASP text, but we're comparing
        # NL to NL, so we use original_nl as proxy for "expected content"
        return compute_quality_metrics(
            asp_text=original_nl,  # Using original NL as reference
            nl_text=translation,
            llm_interface=None,  # Skip LLM grammar check for speed
        )

    def _request_refinement(
        self,
        original_nl: str,
        current_translation: str,
        asp_code: str,
        metrics: QualityMetrics,
    ) -> tuple:
        """
        Request LLM to refine the translation.

        Args:
            original_nl: Original natural language text
            current_translation: Current translation to improve
            asp_code: ASP code for reference
            metrics: Current quality metrics

        Returns:
            Tuple of (refined_translation, tokens_used, cost_usd)
        """
        prompt = self.REFINEMENT_PROMPT_TEMPLATE.format(
            original_nl=original_nl,
            current_translation=current_translation,
            completeness=metrics.completeness,
            readability=metrics.readability,
            fidelity=metrics.overall,
            target_fidelity=self.fidelity_threshold,
            asp_code=asp_code,
        )

        try:
            response = self.llm.query(
                question=prompt,
                temperature=0.3,  # Lower temperature for more focused refinement
                max_tokens=256,  # Short response expected
            )

            # Extract the refined translation
            refined = response.raw_text.strip()

            # Clean up any extra formatting
            refined = self._clean_response(refined)

            # Get token/cost info from metadata
            tokens = response.metadata.tokens_total
            cost = response.metadata.cost_usd

            logger.debug(f"Refinement response: tokens={tokens}, cost=${cost:.6f}")

            return refined, tokens, cost

        except Exception as e:
            logger.error(f"Refinement request failed: {e}")
            # Return original on failure
            return current_translation, 0, 0.0

    def _clean_response(self, response: str) -> str:
        """
        Clean up LLM response to extract just the translation.

        Removes common prefixes/suffixes and formatting artifacts.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned translation text
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "Improved translation:",
            "Here is the improved translation:",
            "The improved translation is:",
            "Here's the improved version:",
        ]
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix) :].strip()

        # Remove quotes if present
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        if response.startswith("'") and response.endswith("'"):
            response = response[1:-1]

        # Ensure proper ending
        response = response.strip()
        if response and not response.endswith((".", "!", "?")):
            response += "."

        return response

    def refine_batch(
        self,
        examples: List[tuple],
    ) -> List[RefinementResult]:
        """
        Refine a batch of translations.

        Args:
            examples: List of (original_nl, asp_code) tuples

        Returns:
            List of RefinementResult for each example
        """
        results = []
        for original_nl, asp_code in examples:
            result = self.refine(original_nl, asp_code)
            results.append(result)
        return results

    def get_cost_estimate(
        self,
        num_examples: int,
        avg_iterations: float = 2.0,
    ) -> dict:
        """
        Estimate cost for refining a batch of examples.

        Based on typical token usage patterns:
        - ~200 input tokens per refinement prompt
        - ~50 output tokens per refinement response

        Args:
            num_examples: Number of examples to refine
            avg_iterations: Expected average iterations per example

        Returns:
            Dict with cost estimates
        """
        # Approximate token counts per iteration
        input_tokens_per_iter = 200
        output_tokens_per_iter = 50

        total_iterations = num_examples * avg_iterations

        # Haiku pricing (approximate)
        # Input: $0.25 per 1M tokens
        # Output: $1.25 per 1M tokens
        input_cost = (total_iterations * input_tokens_per_iter * 0.25) / 1_000_000
        output_cost = (total_iterations * output_tokens_per_iter * 1.25) / 1_000_000

        return {
            "num_examples": num_examples,
            "avg_iterations": avg_iterations,
            "total_iterations": total_iterations,
            "estimated_input_tokens": int(total_iterations * input_tokens_per_iter),
            "estimated_output_tokens": int(total_iterations * output_tokens_per_iter),
            "estimated_cost_usd": input_cost + output_cost,
            "cost_per_example": (input_cost + output_cost) / num_examples,
        }
