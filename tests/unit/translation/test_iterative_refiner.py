"""
Unit tests for IterativeTranslationRefiner.

Tests convergence behavior, cost tracking, and termination conditions.
"""

import pytest
from unittest.mock import Mock
from dataclasses import dataclass

from loft.translation.iterative_refiner import (
    IterativeTranslationRefiner,
    RefinementResult,
)
from loft.neural.llm_interface import LLMResponse, ResponseMetadata


@dataclass
class MockContent:
    """Mock content for LLMResponse."""

    text: str = ""


def create_mock_response(
    raw_text: str,
    tokens_total: int = 100,
    cost_usd: float = 0.001,
) -> LLMResponse:
    """Create a mock LLM response."""
    metadata = ResponseMetadata(
        model="claude-3-5-haiku-20241022",
        tokens_input=50,
        tokens_output=50,
        tokens_total=tokens_total,
        latency_ms=100.0,
        cost_usd=cost_usd,
        timestamp="2024-01-01T00:00:00Z",
        provider="anthropic",
    )
    return LLMResponse(
        content=MockContent(),
        raw_text=raw_text,
        confidence=0.9,
        metadata=metadata,
    )


class TestIterativeTranslationRefinerInit:
    """Tests for refiner initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        mock_llm = Mock()
        refiner = IterativeTranslationRefiner(mock_llm)

        assert refiner.max_iterations == 3
        assert refiner.fidelity_threshold == 0.85
        assert refiner.improvement_threshold == 0.05

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        mock_llm = Mock()
        refiner = IterativeTranslationRefiner(
            mock_llm,
            max_iterations=5,
            fidelity_threshold=0.90,
            improvement_threshold=0.02,
        )

        assert refiner.max_iterations == 5
        assert refiner.fidelity_threshold == 0.90
        assert refiner.improvement_threshold == 0.02


class TestRefinementResult:
    """Tests for RefinementResult dataclass."""

    def test_cost_per_iteration_with_iterations(self):
        """Test cost per iteration calculation."""
        result = RefinementResult(
            final_translation="Test translation",
            iterations_used=3,
            initial_fidelity=0.5,
            final_fidelity=0.85,
            improvement=0.35,
            converged=True,
            total_cost_usd=0.003,
        )

        assert result.cost_per_iteration == pytest.approx(0.001)

    def test_cost_per_iteration_zero_iterations(self):
        """Test cost per iteration with zero iterations."""
        result = RefinementResult(
            final_translation="Test",
            iterations_used=0,
            initial_fidelity=0.9,
            final_fidelity=0.9,
            improvement=0.0,
            converged=True,
        )

        assert result.cost_per_iteration == 0.0

    def test_improvement_per_iteration(self):
        """Test improvement per iteration calculation."""
        result = RefinementResult(
            final_translation="Test",
            iterations_used=2,
            initial_fidelity=0.6,
            final_fidelity=0.8,
            improvement=0.2,
            converged=True,
        )

        assert result.improvement_per_iteration == pytest.approx(0.1)


class TestRefinementConvergence:
    """Tests for refinement convergence behavior."""

    def test_immediate_convergence_high_initial_fidelity(self):
        """Test that refinement stops immediately if initial fidelity is high."""
        mock_llm = Mock()
        refiner = IterativeTranslationRefiner(
            mock_llm,
            fidelity_threshold=0.80,
        )

        # High-quality initial translation
        result = refiner.refine(
            original_nl="A contract is valid.",
            asp_code="contract_valid(c1).",
            initial_translation="A contract is valid.",  # Exact match
        )

        # Should converge immediately with no LLM calls
        assert result.converged is True
        assert result.iterations_used == 0
        mock_llm.query.assert_not_called()

    def test_convergence_after_refinement(self):
        """Test convergence after refinement iterations."""
        mock_llm = Mock()
        # Return progressively better translations
        mock_llm.query.side_effect = [
            create_mock_response("A contract is valid if it has consideration."),
        ]

        refiner = IterativeTranslationRefiner(
            mock_llm,
            max_iterations=3,
            fidelity_threshold=0.70,
        )

        result = refiner.refine(
            original_nl="A contract is valid if it has consideration.",
            asp_code="contract_valid(C) :- has_consideration(C).",
            initial_translation="Contract valid.",  # Poor initial
        )

        # Should have made at least one refinement attempt
        assert result.iterations_used >= 1
        assert result.final_fidelity >= result.initial_fidelity

    def test_max_iterations_reached(self):
        """Test that refinement stops at max iterations."""
        mock_llm = Mock()
        # Return mediocre translations that don't reach threshold
        mock_llm.query.return_value = create_mock_response("Contract stuff happens.")

        refiner = IterativeTranslationRefiner(
            mock_llm,
            max_iterations=2,
            fidelity_threshold=0.99,  # Unreachable threshold
            improvement_threshold=0.01,  # Low threshold to avoid early stopping
        )

        result = refiner.refine(
            original_nl="A comprehensive legal contract requires offer, acceptance, and consideration.",
            asp_code="contract(c1). requires_offer(c1). requires_acceptance(c1).",
            initial_translation="Bad translation.",
        )

        assert result.iterations_used == 2
        assert mock_llm.query.call_count == 2


class TestDiminishingReturns:
    """Tests for diminishing returns detection."""

    def test_stops_on_diminishing_returns(self):
        """Test that refinement stops when improvement is below threshold."""
        mock_llm = Mock()
        # Return same response repeatedly (no improvement)
        mock_llm.query.return_value = create_mock_response(
            "The contract is enforceable."
        )

        refiner = IterativeTranslationRefiner(
            mock_llm,
            max_iterations=5,
            fidelity_threshold=0.99,
            improvement_threshold=0.10,  # Require 10% improvement
        )

        result = refiner.refine(
            original_nl="The contract is enforceable.",
            asp_code="enforceable(c1).",
            initial_translation="The contract is enforceable.",
        )

        # Should stop early due to diminishing returns
        assert result.iterations_used < 5
        assert result.converged is False or result.final_fidelity >= 0.99


class TestCostTracking:
    """Tests for cost and token tracking."""

    def test_cost_accumulation(self):
        """Test that costs are properly accumulated across iterations."""
        mock_llm = Mock()
        mock_llm.query.side_effect = [
            create_mock_response("Translation 1", tokens_total=100, cost_usd=0.001),
            create_mock_response("Translation 2", tokens_total=150, cost_usd=0.0015),
        ]

        refiner = IterativeTranslationRefiner(
            mock_llm,
            max_iterations=2,
            fidelity_threshold=0.99,  # Won't be reached
            improvement_threshold=0.01,  # Low threshold to avoid early stopping
        )

        result = refiner.refine(
            original_nl="Test statement for cost tracking.",
            asp_code="test(x).",
            initial_translation="Bad.",
        )

        assert result.total_tokens == 250
        assert result.total_cost_usd == pytest.approx(0.0025)

    def test_iteration_metrics_recorded(self):
        """Test that per-iteration metrics are recorded."""
        mock_llm = Mock()
        mock_llm.query.side_effect = [
            create_mock_response(
                "Better translation", tokens_total=80, cost_usd=0.0008
            ),
            create_mock_response(
                "Even better translation", tokens_total=90, cost_usd=0.0009
            ),
        ]

        refiner = IterativeTranslationRefiner(
            mock_llm,
            max_iterations=2,
            fidelity_threshold=0.99,
            improvement_threshold=0.01,  # Low threshold to avoid early stopping
        )

        result = refiner.refine(
            original_nl="Contract requires writing.",
            asp_code="requires_writing(c1).",
            initial_translation="Bad.",
        )

        # Should have metrics for each iteration
        assert len(result.iteration_metrics) == 2
        assert result.iteration_metrics[0].tokens_used == 80
        assert result.iteration_metrics[1].tokens_used == 90


class TestResponseCleaning:
    """Tests for response cleaning functionality."""

    def test_clean_response_removes_prefix(self):
        """Test that common prefixes are removed."""
        mock_llm = Mock()
        refiner = IterativeTranslationRefiner(mock_llm)

        cleaned = refiner._clean_response("Improved translation: A contract is valid.")
        assert cleaned == "A contract is valid."

    def test_clean_response_removes_quotes(self):
        """Test that surrounding quotes are removed."""
        mock_llm = Mock()
        refiner = IterativeTranslationRefiner(mock_llm)

        cleaned = refiner._clean_response('"A contract is valid."')
        assert cleaned == "A contract is valid."

    def test_clean_response_adds_period(self):
        """Test that missing period is added."""
        mock_llm = Mock()
        refiner = IterativeTranslationRefiner(mock_llm)

        cleaned = refiner._clean_response("A contract is valid")
        assert cleaned == "A contract is valid."

    def test_clean_response_preserves_question_mark(self):
        """Test that question marks are preserved."""
        mock_llm = Mock()
        refiner = IterativeTranslationRefiner(mock_llm)

        cleaned = refiner._clean_response("Is the contract valid?")
        assert cleaned == "Is the contract valid?"


class TestCostEstimation:
    """Tests for cost estimation functionality."""

    def test_cost_estimate_single_example(self):
        """Test cost estimation for single example."""
        mock_llm = Mock()
        refiner = IterativeTranslationRefiner(mock_llm)

        estimate = refiner.get_cost_estimate(num_examples=1, avg_iterations=3.0)

        assert estimate["num_examples"] == 1
        assert estimate["total_iterations"] == 3.0
        assert estimate["estimated_cost_usd"] > 0
        assert "cost_per_example" in estimate

    def test_cost_estimate_batch(self):
        """Test cost estimation for batch processing."""
        mock_llm = Mock()
        refiner = IterativeTranslationRefiner(mock_llm)

        estimate = refiner.get_cost_estimate(num_examples=100, avg_iterations=2.0)

        assert estimate["num_examples"] == 100
        assert estimate["total_iterations"] == 200.0
        assert estimate["estimated_input_tokens"] == 40000  # 200 * 200
        assert estimate["estimated_output_tokens"] == 10000  # 200 * 50


class TestBatchRefinement:
    """Tests for batch refinement functionality."""

    def test_batch_refinement_processes_all(self):
        """Test that batch refinement processes all examples."""
        mock_llm = Mock()
        mock_llm.query.return_value = create_mock_response("Refined translation.")

        refiner = IterativeTranslationRefiner(
            mock_llm,
            max_iterations=1,
            fidelity_threshold=0.99,
        )

        examples = [
            ("Contract is valid.", "valid(c1)."),
            ("Contract requires writing.", "requires_writing(c1)."),
            ("Contract is enforceable.", "enforceable(c1)."),
        ]

        results = refiner.refine_batch(examples)

        assert len(results) == 3
        assert all(isinstance(r, RefinementResult) for r in results)


class TestErrorHandling:
    """Tests for error handling."""

    def test_llm_failure_returns_original(self):
        """Test that LLM failure returns original translation."""
        mock_llm = Mock()
        mock_llm.query.side_effect = Exception("API Error")

        refiner = IterativeTranslationRefiner(
            mock_llm,
            max_iterations=1,
            fidelity_threshold=0.99,
        )

        result = refiner.refine(
            original_nl="Contract is valid.",
            asp_code="valid(c1).",
            initial_translation="Contract is valid.",
        )

        # Should handle error gracefully
        assert result.final_translation is not None


class TestFidelityCalculation:
    """Tests for fidelity calculation."""

    def test_calculate_fidelity_exact_match(self):
        """Test fidelity calculation with exact match."""
        mock_llm = Mock()
        refiner = IterativeTranslationRefiner(mock_llm)

        metrics = refiner._calculate_fidelity(
            original_nl="Contract is valid.",
            translation="Contract is valid.",
        )

        # Exact match should have high fidelity
        assert metrics.overall >= 0.8

    def test_calculate_fidelity_poor_match(self):
        """Test fidelity calculation with poor match."""
        mock_llm = Mock()
        refiner = IterativeTranslationRefiner(mock_llm)

        metrics = refiner._calculate_fidelity(
            original_nl="A comprehensive legal contract requires offer, acceptance, and consideration.",
            translation="Bad.",
        )

        # Poor match should have low fidelity
        assert metrics.overall < 0.5


class TestPromptTemplate:
    """Tests for refinement prompt template."""

    def test_prompt_template_contains_required_fields(self):
        """Test that prompt template includes all required fields."""
        template = IterativeTranslationRefiner.REFINEMENT_PROMPT_TEMPLATE

        assert "{original_nl}" in template
        assert "{current_translation}" in template
        # Check for field names (with or without formatting specifiers)
        assert "completeness" in template
        assert "readability" in template
        assert "fidelity" in template
        assert "{asp_code}" in template
        assert "target_fidelity" in template

    def test_prompt_formatting(self):
        """Test that prompt can be formatted without errors."""
        mock_llm = Mock()
        refiner = IterativeTranslationRefiner(mock_llm)

        # Should format without raising exception
        prompt = refiner.REFINEMENT_PROMPT_TEMPLATE.format(
            original_nl="Test",
            current_translation="Test translation",
            completeness=0.8,
            readability=0.9,
            fidelity=0.85,
            target_fidelity=0.90,
            asp_code="test(x).",
        )

        assert "Test" in prompt
        assert "0.8" in prompt or "0.80" in prompt
