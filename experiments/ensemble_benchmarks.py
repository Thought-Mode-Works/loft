"""
Phase 6 Ensemble MVP Validation Benchmarks.

This module provides benchmarks to empirically validate the Phase 6 MVP criteria
from ROADMAP.md:

1. Specialized models outperform general-purpose LLMs on their tasks
2. Ensemble consensus improves accuracy by >20% over single LLM
3. Translator maintains >95% fidelity in bidirectional conversion
4. Meta-Reasoner generates actionable insights about system behavior
5. Cost/performance trade-offs are optimized

Usage:
    python experiments/ensemble_benchmarks.py --help
    python experiments/ensemble_benchmarks.py --dry-run
    python experiments/ensemble_benchmarks.py --output results/benchmark.json
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


# ============================================================================
# Data Classes for Benchmark Results
# ============================================================================


@dataclass
class AccuracyBenchmarkResult:
    """Result of ensemble vs single LLM accuracy comparison."""

    single_llm_accuracy: float
    ensemble_accuracy: float
    improvement_percentage: float
    mvp_target_met: bool  # >20% improvement
    test_cases_count: int
    single_llm_avg_time_ms: float
    ensemble_avg_time_ms: float
    voting_strategy_used: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FidelityBenchmarkResult:
    """Result of translator fidelity measurement."""

    average_fidelity: float
    min_fidelity: float
    max_fidelity: float
    mvp_target_met: bool  # >95% fidelity
    test_cases_count: int
    asp_to_nl_avg_time_ms: float
    nl_to_asp_avg_time_ms: float
    roundtrip_success_rate: float
    failed_roundtrips: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class InsightBenchmarkResult:
    """Result of meta-reasoner insight actionability scoring."""

    total_insights_generated: int
    actionable_insights_count: int
    actionability_percentage: float
    mvp_target_met: bool  # Actionable insights generated
    insight_types_distribution: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    high_priority_count: int = 0
    sample_insights: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CostBenchmarkResult:
    """Result of cost/performance optimization measurement."""

    total_tokens_used: int
    total_cost_usd: float
    cost_per_case_usd: float
    tokens_per_case: int
    avg_latency_ms: float
    p95_latency_ms: float
    mvp_target_met: bool  # Cost is optimized
    component_costs: Dict[str, float] = field(default_factory=dict)
    component_latencies: Dict[str, float] = field(default_factory=dict)


@dataclass
class MVPBenchmarkSummary:
    """Overall MVP validation summary."""

    timestamp: str
    run_id: str
    accuracy_result: Optional[AccuracyBenchmarkResult] = None
    fidelity_result: Optional[FidelityBenchmarkResult] = None
    insight_result: Optional[InsightBenchmarkResult] = None
    cost_result: Optional[CostBenchmarkResult] = None
    mvp_criteria_met: List[str] = field(default_factory=list)
    mvp_criteria_not_met: List[str] = field(default_factory=list)
    overall_mvp_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "mvp_criteria_met": self.mvp_criteria_met,
            "mvp_criteria_not_met": self.mvp_criteria_not_met,
            "overall_mvp_passed": self.overall_mvp_passed,
        }
        if self.accuracy_result:
            result["accuracy_benchmark"] = asdict(self.accuracy_result)
        if self.fidelity_result:
            result["fidelity_benchmark"] = asdict(self.fidelity_result)
        if self.insight_result:
            result["insight_benchmark"] = asdict(self.insight_result)
        if self.cost_result:
            result["cost_benchmark"] = asdict(self.cost_result)
        return result


# ============================================================================
# Test Case Generators
# ============================================================================


def generate_asp_generation_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for ASP rule generation benchmarks."""
    return [
        {
            "case_id": "sof_001",
            "principle": "A contract for the sale of goods over $500 must be in writing",
            "domain": "statute_of_frauds",
            "predicates": [
                "contract",
                "sale_of_goods",
                "value",
                "written",
                "enforceable",
            ],
            "expected_pattern": "enforceable",
        },
        {
            "case_id": "sof_002",
            "principle": "A contract that cannot be performed within one year must be in writing",
            "domain": "statute_of_frauds",
            "predicates": [
                "contract",
                "performance_period",
                "written",
                "enforceable",
            ],
            "expected_pattern": "enforceable",
        },
        {
            "case_id": "tort_001",
            "principle": "A person is liable for negligence if they breached a duty of care",
            "domain": "torts",
            "predicates": [
                "defendant",
                "duty_of_care",
                "breach",
                "causation",
                "liable",
            ],
            "expected_pattern": "liable",
        },
        {
            "case_id": "contract_001",
            "principle": "Consideration is required for a valid contract",
            "domain": "contracts",
            "predicates": ["contract", "offer", "acceptance", "consideration", "valid"],
            "expected_pattern": "valid",
        },
        {
            "case_id": "property_001",
            "principle": "Adverse possession requires continuous occupation for statutory period",
            "domain": "property",
            "predicates": [
                "possession",
                "continuous",
                "statutory_period",
                "adverse",
                "ownership",
            ],
            "expected_pattern": "ownership",
        },
    ]


def generate_translation_test_cases() -> List[Dict[str, str]]:
    """Generate test cases for translation fidelity benchmarks."""
    return [
        {
            "asp_rule": "enforceable(C) :- contract(C), written(C), consideration(C).",
            "domain": "contracts",
            "description": "Basic contract enforceability rule",
        },
        {
            "asp_rule": "liable(D) :- defendant(D), duty_of_care(D, P), breach(D, P), causation(D, P, H), harm(H).",
            "domain": "torts",
            "description": "Negligence liability rule",
        },
        {
            "asp_rule": "within_statute(C) :- contract(C), sale_of_goods(C), value(C, V), V > 500, written(C).",
            "domain": "statute_of_frauds",
            "description": "Statute of frauds goods value rule",
        },
        {
            "asp_rule": "adverse_possession(P, L) :- possession(P, L), continuous(P, L, T), T >= 10, hostile(P, L), open(P, L).",
            "domain": "property",
            "description": "Adverse possession requirements",
        },
        {
            "asp_rule": "valid_acceptance(A) :- acceptance(A), mirror_image(A, O), offer(O), not revoked(O).",
            "domain": "contracts",
            "description": "Valid acceptance requirements",
        },
    ]


def generate_failure_records_for_meta_reasoning() -> List[Dict[str, Any]]:
    """Generate simulated failure records for meta-reasoning benchmarks."""
    return [
        {
            "failure_id": "fail_001",
            "timestamp": datetime.now().isoformat(),
            "category": "syntax_error",
            "error_message": "Unsafe variable X in rule head",
            "domain": "contracts",
            "strategy_used": "chain_of_thought",
            "input_summary": "Contract enforceability rule",
            "confidence_before": 0.7,
        },
        {
            "failure_id": "fail_002",
            "timestamp": datetime.now().isoformat(),
            "category": "grounding_error",
            "error_message": "Program has no stable models",
            "domain": "torts",
            "strategy_used": "few_shot_learning",
            "input_summary": "Negligence liability rule",
            "confidence_before": 0.65,
        },
        {
            "failure_id": "fail_003",
            "timestamp": datetime.now().isoformat(),
            "category": "semantic_error",
            "error_message": "Rule contradicts existing knowledge base",
            "domain": "property",
            "strategy_used": "prompt_optimization",
            "input_summary": "Adverse possession rule",
            "confidence_before": 0.8,
        },
        {
            "failure_id": "fail_004",
            "timestamp": datetime.now().isoformat(),
            "category": "validation_error",
            "error_message": "Rule failed empirical validation on test cases",
            "domain": "statute_of_frauds",
            "strategy_used": "self_consistency",
            "input_summary": "Writing requirement rule",
            "confidence_before": 0.75,
        },
        {
            "failure_id": "fail_005",
            "timestamp": datetime.now().isoformat(),
            "category": "syntax_error",
            "error_message": "Embedded period in predicate name",
            "domain": "contracts",
            "strategy_used": "chain_of_thought",
            "input_summary": "Consideration rule",
            "confidence_before": 0.6,
        },
    ]


# ============================================================================
# Benchmark Suite Implementation
# ============================================================================


class EnsembleBenchmarkSuite:
    """Validates Phase 6 MVP criteria with empirical measurements."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        dry_run: bool = False,
    ):
        """
        Initialize benchmark suite.

        Args:
            model: LLM model to use for benchmarks
            dry_run: If True, use simulated results instead of real LLM calls
        """
        self.model = model
        self.dry_run = dry_run
        self._llm_interface: Optional[Any] = None
        self._orchestrator: Optional[Any] = None
        self._logic_generator: Optional[Any] = None
        self._translator: Optional[Any] = None
        self._meta_reasoner: Optional[Any] = None
        self._metrics_tracker: Optional[Any] = None

        if not dry_run:
            self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize neural ensemble components."""
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY='your-key-here' "
                "or use --dry-run for simulated benchmarks."
            )

        from loft.neural.llm_interface import LLMInterface
        from loft.neural.providers import AnthropicProvider

        provider = AnthropicProvider(api_key=api_key, model=self.model)
        self._llm_interface = LLMInterface(provider=provider)

        # Initialize metrics tracker
        from loft.autonomous.llm_metrics import (
            LLMMetricsTracker,
            set_global_metrics_tracker,
        )

        self._metrics_tracker = LLMMetricsTracker(model=self.model)
        set_global_metrics_tracker(self._metrics_tracker)

        # Initialize ensemble components (lazy loading)
        logger.info(f"Initialized benchmark suite with model: {self.model}")

    def _get_orchestrator(self) -> Any:
        """Get or create EnsembleOrchestrator (lazy init)."""
        if self._orchestrator is None and not self.dry_run:
            from loft.neural.ensemble.orchestrator import EnsembleOrchestrator

            self._orchestrator = EnsembleOrchestrator(llm_interface=self._llm_interface)
        return self._orchestrator

    def _get_logic_generator(self) -> Any:
        """Get or create LogicGeneratorLLM (lazy init)."""
        if self._logic_generator is None and not self.dry_run:
            from loft.neural.ensemble.logic_generator import LogicGeneratorLLM

            self._logic_generator = LogicGeneratorLLM(llm_interface=self._llm_interface)
        return self._logic_generator

    def _get_translator(self) -> Any:
        """Get or create TranslatorLLM (lazy init)."""
        if self._translator is None and not self.dry_run:
            from loft.neural.ensemble.translator import TranslatorLLM

            self._translator = TranslatorLLM(llm_interface=self._llm_interface)
        return self._translator

    def _get_meta_reasoner(self) -> Any:
        """Get or create MetaReasonerLLM (lazy init)."""
        if self._meta_reasoner is None and not self.dry_run:
            from loft.neural.ensemble.meta_reasoner import MetaReasonerLLM

            self._meta_reasoner = MetaReasonerLLM(llm_interface=self._llm_interface)
        return self._meta_reasoner

    def benchmark_ensemble_vs_single_llm(
        self,
        test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> AccuracyBenchmarkResult:
        """
        Compare ensemble accuracy vs single LLM accuracy.

        MVP target: >20% improvement

        Args:
            test_cases: List of test cases for ASP generation

        Returns:
            AccuracyBenchmarkResult with comparison metrics
        """
        logger.info("Running ensemble vs single LLM benchmark...")

        if test_cases is None:
            test_cases = generate_asp_generation_test_cases()

        if self.dry_run:
            # Simulated results for dry run
            logger.info("  [DRY RUN] Using simulated results")
            return AccuracyBenchmarkResult(
                single_llm_accuracy=0.65,
                ensemble_accuracy=0.82,
                improvement_percentage=26.15,
                mvp_target_met=True,
                test_cases_count=len(test_cases),
                single_llm_avg_time_ms=1200.0,
                ensemble_avg_time_ms=3500.0,
                voting_strategy_used="weighted",
                details={
                    "note": "Simulated results - use without --dry-run for real benchmarks"
                },
            )

        # Real benchmark implementation
        logic_gen = self._get_logic_generator()
        orchestrator = self._get_orchestrator()

        single_llm_successes = 0
        ensemble_successes = 0
        single_llm_times: List[float] = []
        ensemble_times: List[float] = []

        for case in test_cases:
            logger.debug(f"  Testing case: {case['case_id']}")

            # Single LLM test
            start = time.time()
            try:
                single_result = logic_gen.generate_rule(
                    principle=case["principle"],
                    domain=case["domain"],
                    available_predicates=case["predicates"],
                )
                single_llm_times.append((time.time() - start) * 1000)
                if single_result.is_valid:
                    single_llm_successes += 1
            except Exception as e:
                logger.warning(f"  Single LLM failed: {e}")
                single_llm_times.append((time.time() - start) * 1000)

            # Ensemble test (uses voting across strategies)
            start = time.time()
            try:
                from loft.neural.ensemble.orchestrator import TaskType

                ensemble_result = orchestrator.route_task(
                    task_type=TaskType.RULE_GENERATION,
                    input_data={
                        "principle": case["principle"],
                        "domain": case["domain"],
                        "predicates": case["predicates"],
                    },
                )
                ensemble_times.append((time.time() - start) * 1000)
                # Check if result is valid
                if ensemble_result.final_result and hasattr(
                    ensemble_result.final_result, "is_valid"
                ):
                    if ensemble_result.final_result.is_valid:
                        ensemble_successes += 1
                elif ensemble_result.voting_result:
                    if ensemble_result.voting_result.confidence > 0.6:
                        ensemble_successes += 1
            except Exception as e:
                logger.warning(f"  Ensemble failed: {e}")
                ensemble_times.append((time.time() - start) * 1000)

        # Calculate metrics
        total_cases = len(test_cases)
        single_accuracy = single_llm_successes / total_cases if total_cases > 0 else 0
        ensemble_accuracy = ensemble_successes / total_cases if total_cases > 0 else 0

        improvement = (
            ((ensemble_accuracy - single_accuracy) / single_accuracy * 100)
            if single_accuracy > 0
            else 0
        )

        return AccuracyBenchmarkResult(
            single_llm_accuracy=single_accuracy,
            ensemble_accuracy=ensemble_accuracy,
            improvement_percentage=improvement,
            mvp_target_met=improvement > 20.0,
            test_cases_count=total_cases,
            single_llm_avg_time_ms=(
                sum(single_llm_times) / len(single_llm_times) if single_llm_times else 0
            ),
            ensemble_avg_time_ms=(
                sum(ensemble_times) / len(ensemble_times) if ensemble_times else 0
            ),
            voting_strategy_used="weighted",
            details={
                "single_llm_successes": single_llm_successes,
                "ensemble_successes": ensemble_successes,
            },
        )

    def benchmark_translator_fidelity(
        self,
        test_cases: Optional[List[Dict[str, str]]] = None,
    ) -> FidelityBenchmarkResult:
        """
        Measure roundtrip fidelity for ASP<->NL translation.

        MVP target: >95% semantic similarity

        Args:
            test_cases: List of ASP rules to test

        Returns:
            FidelityBenchmarkResult with fidelity metrics
        """
        logger.info("Running translator fidelity benchmark...")

        if test_cases is None:
            test_cases = generate_translation_test_cases()

        if self.dry_run:
            logger.info("  [DRY RUN] Using simulated results")
            return FidelityBenchmarkResult(
                average_fidelity=0.962,
                min_fidelity=0.91,
                max_fidelity=0.99,
                mvp_target_met=True,
                test_cases_count=len(test_cases),
                asp_to_nl_avg_time_ms=800.0,
                nl_to_asp_avg_time_ms=950.0,
                roundtrip_success_rate=0.96,
                failed_roundtrips=[],
            )

        # Real benchmark
        translator = self._get_translator()

        fidelity_scores: List[float] = []
        asp_to_nl_times: List[float] = []
        nl_to_asp_times: List[float] = []
        failed_roundtrips: List[Dict[str, Any]] = []
        successful_roundtrips = 0

        for case in test_cases:
            logger.debug(f"  Testing rule: {case['asp_rule'][:50]}...")

            try:
                # ASP -> NL translation
                start = time.time()
                nl_result = translator.asp_to_natural_language(
                    asp_rule=case["asp_rule"],
                    domain=case["domain"],
                )
                asp_to_nl_times.append((time.time() - start) * 1000)

                # NL -> ASP translation
                start = time.time()
                asp_result = translator.natural_language_to_asp(
                    natural_language=nl_result.target,
                    domain=case["domain"],
                    available_predicates=[],  # Let it infer
                )
                nl_to_asp_times.append((time.time() - start) * 1000)

                # Validate roundtrip fidelity
                roundtrip_result = translator.validate_roundtrip(
                    original_asp=case["asp_rule"],
                    nl_translation=nl_result.target,
                    domain=case["domain"],
                )

                fidelity_scores.append(roundtrip_result.fidelity_score)
                if roundtrip_result.fidelity_score >= 0.8:
                    successful_roundtrips += 1
                else:
                    failed_roundtrips.append(
                        {
                            "original": case["asp_rule"],
                            "nl": nl_result.target,
                            "back_translated": asp_result.target,
                            "fidelity": roundtrip_result.fidelity_score,
                        }
                    )

            except Exception as e:
                logger.warning(f"  Translation failed: {e}")
                fidelity_scores.append(0.0)
                failed_roundtrips.append(
                    {
                        "original": case["asp_rule"],
                        "error": str(e),
                    }
                )

        # Calculate metrics
        avg_fidelity = (
            sum(fidelity_scores) / len(fidelity_scores) if fidelity_scores else 0
        )
        min_fidelity = min(fidelity_scores) if fidelity_scores else 0
        max_fidelity = max(fidelity_scores) if fidelity_scores else 0
        success_rate = (
            successful_roundtrips / len(test_cases) if len(test_cases) > 0 else 0
        )

        return FidelityBenchmarkResult(
            average_fidelity=avg_fidelity,
            min_fidelity=min_fidelity,
            max_fidelity=max_fidelity,
            mvp_target_met=avg_fidelity >= 0.95,
            test_cases_count=len(test_cases),
            asp_to_nl_avg_time_ms=(
                sum(asp_to_nl_times) / len(asp_to_nl_times) if asp_to_nl_times else 0
            ),
            nl_to_asp_avg_time_ms=(
                sum(nl_to_asp_times) / len(nl_to_asp_times) if nl_to_asp_times else 0
            ),
            roundtrip_success_rate=success_rate,
            failed_roundtrips=failed_roundtrips,
        )

    def benchmark_meta_reasoner_insights(
        self,
        failure_records: Optional[List[Dict[str, Any]]] = None,
    ) -> InsightBenchmarkResult:
        """
        Measure insight actionability from meta-reasoner.

        MVP target: Generates actionable insights

        Args:
            failure_records: List of failure records for analysis

        Returns:
            InsightBenchmarkResult with insight metrics
        """
        logger.info("Running meta-reasoner insight benchmark...")

        if failure_records is None:
            failure_records = generate_failure_records_for_meta_reasoning()

        if self.dry_run:
            logger.info("  [DRY RUN] Using simulated results")
            return InsightBenchmarkResult(
                total_insights_generated=8,
                actionable_insights_count=6,
                actionability_percentage=75.0,
                mvp_target_met=True,
                insight_types_distribution={
                    "pattern_identified": 3,
                    "root_cause": 2,
                    "strategy_recommendation": 2,
                    "prompt_improvement": 1,
                },
                avg_confidence=0.78,
                high_priority_count=2,
                sample_insights=[
                    {
                        "type": "pattern_identified",
                        "title": "Recurring unsafe variable errors",
                        "actionable": True,
                        "priority": "high",
                    },
                    {
                        "type": "strategy_recommendation",
                        "title": "Switch to few-shot for contracts domain",
                        "actionable": True,
                        "priority": "medium",
                    },
                ],
            )

        # Real benchmark
        meta_reasoner = self._get_meta_reasoner()

        try:
            # Convert failure records to FailureRecord objects
            from loft.neural.ensemble.meta_reasoner import (
                FailureRecord,
                FailureCategory,
            )

            failure_objs = []
            for record in failure_records:
                category = FailureCategory.UNKNOWN
                try:
                    category = FailureCategory(record.get("category", "unknown"))
                except ValueError:
                    pass

                failure_objs.append(
                    FailureRecord(
                        failure_id=record["failure_id"],
                        timestamp=record["timestamp"],
                        category=category,
                        error_message=record["error_message"],
                        domain=record.get("domain", "unknown"),
                        strategy_used=record.get("strategy_used"),
                        input_summary=record.get("input_summary"),
                        confidence_before=record.get("confidence_before", 0.5),
                    )
                )

            # Analyze reasoning patterns
            analysis_result = meta_reasoner.analyze_reasoning_patterns(
                failures=failure_objs,
                insights=[],
            )

            # Count insights by type and actionability
            insight_types: Dict[str, int] = {}
            actionable_count = 0
            high_priority_count = 0
            confidences: List[float] = []
            sample_insights: List[Dict[str, Any]] = []

            for insight in analysis_result.insights:
                insight_type = insight.insight_type.value
                insight_types[insight_type] = insight_types.get(insight_type, 0) + 1

                if insight.actionable:
                    actionable_count += 1

                if insight.priority == "high":
                    high_priority_count += 1

                confidences.append(insight.confidence)

                if len(sample_insights) < 3:
                    sample_insights.append(
                        {
                            "type": insight_type,
                            "title": insight.title,
                            "actionable": insight.actionable,
                            "priority": insight.priority,
                            "recommended_action": insight.recommended_action,
                        }
                    )

            total_insights = len(analysis_result.insights)
            actionability_pct = (
                (actionable_count / total_insights * 100) if total_insights > 0 else 0
            )
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return InsightBenchmarkResult(
                total_insights_generated=total_insights,
                actionable_insights_count=actionable_count,
                actionability_percentage=actionability_pct,
                mvp_target_met=actionable_count > 0,
                insight_types_distribution=insight_types,
                avg_confidence=avg_confidence,
                high_priority_count=high_priority_count,
                sample_insights=sample_insights,
            )

        except Exception as e:
            logger.error(f"Meta-reasoner benchmark failed: {e}")
            return InsightBenchmarkResult(
                total_insights_generated=0,
                actionable_insights_count=0,
                actionability_percentage=0.0,
                mvp_target_met=False,
                insight_types_distribution={},
                avg_confidence=0.0,
                high_priority_count=0,
                sample_insights=[{"error": str(e)}],
            )

    def benchmark_cost_performance(
        self,
        test_cases_count: int = 10,
    ) -> CostBenchmarkResult:
        """
        Measure cost/performance trade-offs.

        Args:
            test_cases_count: Number of test cases processed

        Returns:
            CostBenchmarkResult with cost/performance metrics
        """
        logger.info("Running cost/performance benchmark...")

        if self.dry_run:
            logger.info("  [DRY RUN] Using simulated results")
            return CostBenchmarkResult(
                total_tokens_used=15000,
                total_cost_usd=0.042,
                cost_per_case_usd=0.0042,
                tokens_per_case=1500,
                avg_latency_ms=1850.0,
                p95_latency_ms=3200.0,
                mvp_target_met=True,
                component_costs={
                    "logic_generator": 0.018,
                    "translator": 0.012,
                    "meta_reasoner": 0.008,
                    "critic": 0.004,
                },
                component_latencies={
                    "logic_generator": 1200.0,
                    "translator": 900.0,
                    "meta_reasoner": 1500.0,
                    "critic": 800.0,
                },
            )

        # Get metrics from tracker
        if self._metrics_tracker is None:
            return CostBenchmarkResult(
                total_tokens_used=0,
                total_cost_usd=0.0,
                cost_per_case_usd=0.0,
                tokens_per_case=0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                mvp_target_met=False,
                component_costs={},
                component_latencies={},
            )

        summary = self._metrics_tracker.get_metrics_summary()

        total_tokens = summary.get("total_tokens", 0)
        total_cost = summary.get("total_cost_usd", 0.0)
        avg_latency = summary.get("avg_latency_seconds", 0.0) * 1000

        cost_per_case = total_cost / test_cases_count if test_cases_count > 0 else 0
        tokens_per_case = (
            total_tokens // test_cases_count if test_cases_count > 0 else 0
        )

        return CostBenchmarkResult(
            total_tokens_used=total_tokens,
            total_cost_usd=total_cost,
            cost_per_case_usd=cost_per_case,
            tokens_per_case=tokens_per_case,
            avg_latency_ms=avg_latency,
            p95_latency_ms=avg_latency * 1.5,  # Estimate
            mvp_target_met=cost_per_case < 0.01,  # Under 1 cent per case
            component_costs={},
            component_latencies={},
        )

    def run_all_benchmarks(self) -> MVPBenchmarkSummary:
        """Run all MVP validation benchmarks."""
        logger.info("=" * 60)
        logger.info(" Phase 6 Ensemble MVP Validation Benchmarks")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("")

        run_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()

        summary = MVPBenchmarkSummary(
            timestamp=timestamp,
            run_id=run_id,
        )

        # Run each benchmark
        try:
            logger.info("[1/4] Ensemble vs Single LLM Accuracy")
            summary.accuracy_result = self.benchmark_ensemble_vs_single_llm()
            self._log_accuracy_result(summary.accuracy_result)
        except Exception as e:
            logger.error(f"Accuracy benchmark failed: {e}")

        try:
            logger.info("\n[2/4] Translator Fidelity")
            summary.fidelity_result = self.benchmark_translator_fidelity()
            self._log_fidelity_result(summary.fidelity_result)
        except Exception as e:
            logger.error(f"Fidelity benchmark failed: {e}")

        try:
            logger.info("\n[3/4] Meta-Reasoner Insights")
            summary.insight_result = self.benchmark_meta_reasoner_insights()
            self._log_insight_result(summary.insight_result)
        except Exception as e:
            logger.error(f"Insight benchmark failed: {e}")

        try:
            logger.info("\n[4/4] Cost/Performance")
            summary.cost_result = self.benchmark_cost_performance()
            self._log_cost_result(summary.cost_result)
        except Exception as e:
            logger.error(f"Cost benchmark failed: {e}")

        # Compile MVP summary
        self._compile_mvp_summary(summary)

        return summary

    def _log_accuracy_result(self, result: AccuracyBenchmarkResult) -> None:
        """Log accuracy benchmark result."""
        logger.info(f"  Single LLM accuracy: {result.single_llm_accuracy:.1%}")
        logger.info(f"  Ensemble accuracy: {result.ensemble_accuracy:.1%}")
        logger.info(f"  Improvement: {result.improvement_percentage:+.1f}%")
        logger.info(
            f"  MVP target (>20%): {'PASS' if result.mvp_target_met else 'FAIL'}"
        )

    def _log_fidelity_result(self, result: FidelityBenchmarkResult) -> None:
        """Log fidelity benchmark result."""
        logger.info(f"  Average fidelity: {result.average_fidelity:.1%}")
        logger.info(f"  Min/Max: {result.min_fidelity:.1%} / {result.max_fidelity:.1%}")
        logger.info(f"  Roundtrip success rate: {result.roundtrip_success_rate:.1%}")
        logger.info(
            f"  MVP target (>95%): {'PASS' if result.mvp_target_met else 'FAIL'}"
        )

    def _log_insight_result(self, result: InsightBenchmarkResult) -> None:
        """Log insight benchmark result."""
        logger.info(f"  Total insights: {result.total_insights_generated}")
        logger.info(f"  Actionable: {result.actionable_insights_count}")
        logger.info(f"  Actionability: {result.actionability_percentage:.1f}%")
        logger.info(
            f"  MVP target (actionable): {'PASS' if result.mvp_target_met else 'FAIL'}"
        )

    def _log_cost_result(self, result: CostBenchmarkResult) -> None:
        """Log cost benchmark result."""
        logger.info(f"  Total tokens: {result.total_tokens_used:,}")
        logger.info(f"  Total cost: ${result.total_cost_usd:.4f}")
        logger.info(f"  Cost per case: ${result.cost_per_case_usd:.4f}")
        logger.info(f"  Avg latency: {result.avg_latency_ms:.0f}ms")
        logger.info(
            f"  MVP target (optimized): {'PASS' if result.mvp_target_met else 'FAIL'}"
        )

    def _compile_mvp_summary(self, summary: MVPBenchmarkSummary) -> None:
        """Compile the final MVP summary."""
        criteria_met = []
        criteria_not_met = []

        # Check each criterion
        if summary.accuracy_result and summary.accuracy_result.mvp_target_met:
            criteria_met.append("accuracy_improvement_>20%")
        else:
            criteria_not_met.append("accuracy_improvement_>20%")

        if summary.fidelity_result and summary.fidelity_result.mvp_target_met:
            criteria_met.append("translator_fidelity_>95%")
        else:
            criteria_not_met.append("translator_fidelity_>95%")

        if summary.insight_result and summary.insight_result.mvp_target_met:
            criteria_met.append("actionable_insights")
        else:
            criteria_not_met.append("actionable_insights")

        if summary.cost_result and summary.cost_result.mvp_target_met:
            criteria_met.append("cost_optimized")
        else:
            criteria_not_met.append("cost_optimized")

        summary.mvp_criteria_met = criteria_met
        summary.mvp_criteria_not_met = criteria_not_met
        summary.overall_mvp_passed = len(criteria_not_met) == 0

        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info(" MVP Validation Summary")
        logger.info("=" * 60)
        logger.info(f"Criteria met: {criteria_met}")
        logger.info(f"Criteria not met: {criteria_not_met}")
        logger.info(
            f"Overall MVP: {'PASSED' if summary.overall_mvp_passed else 'FAILED'}"
        )


def main() -> None:
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Phase 6 Ensemble MVP Validation Benchmarks"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-haiku-20241022",
        help="LLM model to use (default: claude-3-5-haiku-20241022)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use simulated results instead of real LLM calls",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results JSON",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["accuracy", "fidelity", "insight", "cost", "all"],
        default="all",
        help="Which benchmark to run (default: all)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Run benchmarks
    suite = EnsembleBenchmarkSuite(
        model=args.model,
        dry_run=args.dry_run,
    )

    if args.benchmark == "all":
        summary = suite.run_all_benchmarks()
    else:
        # Run individual benchmark
        summary = MVPBenchmarkSummary(
            timestamp=datetime.now().isoformat(),
            run_id=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        if args.benchmark == "accuracy":
            summary.accuracy_result = suite.benchmark_ensemble_vs_single_llm()
        elif args.benchmark == "fidelity":
            summary.fidelity_result = suite.benchmark_translator_fidelity()
        elif args.benchmark == "insight":
            summary.insight_result = suite.benchmark_meta_reasoner_insights()
        elif args.benchmark == "cost":
            summary.cost_result = suite.benchmark_cost_performance()

    # Export results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info(f"\nResults exported to: {output_path}")
    else:
        # Print JSON to stdout
        print("\n" + json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
