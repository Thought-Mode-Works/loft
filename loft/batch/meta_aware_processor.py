"""
Meta-aware batch processor with strategy selection and failure analysis.

Integrates meta-reasoning components with batch processing to enable:
- Adaptive strategy selection based on case characteristics
- Failure pattern detection and analysis
- Prompt optimization during processing
- Autonomous improvement cycle integration

Issue #255: Phase 8 meta-reasoning batch integration.
"""

import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from loft.batch.schemas import BatchConfig, CaseResult, CaseStatus


@dataclass
class MetaAwareBatchConfig(BatchConfig):
    """Configuration for meta-aware batch processing."""

    # Meta-reasoning settings
    enable_strategy_selection: bool = True
    enable_failure_analysis: bool = True
    enable_prompt_optimization: bool = True

    # Adaptation thresholds
    min_failures_for_adaptation: int = 5
    failure_pattern_threshold: float = 0.5  # 50% of recent failures
    adaptation_window_size: int = 10  # Look at last N failures

    # Strategy settings
    default_strategy: str = "rule_based"
    strategy_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default strategy weights."""
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        if not self.strategy_weights:
            self.strategy_weights = {
                "checklist": 1.0,
                "causal_chain": 1.0,
                "balancing_test": 1.0,
                "rule_based": 1.0,
                "dialectical": 1.0,
                "analogical": 1.0,
            }


@dataclass
class FailurePattern:
    """Represents a detected failure pattern."""

    failure_type: str
    count: int
    cases: List[str]
    first_seen: datetime
    last_seen: datetime
    root_cause: Optional[str] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure_type": self.failure_type,
            "count": self.count,
            "cases": self.cases[:10],  # Limit to first 10
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "root_cause": self.root_cause,
            "recommendation": self.recommendation,
        }


@dataclass
class Adaptation:
    """Record of a strategy or prompt adaptation."""

    adaptation_type: str  # "strategy_change", "prompt_refinement", "threshold_adjust"
    timestamp: datetime
    trigger: str  # What triggered this adaptation
    changes: Dict[str, Any]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "adaptation_type": self.adaptation_type,
            "timestamp": self.timestamp.isoformat(),
            "trigger": self.trigger,
            "changes": self.changes,
            "reason": self.reason,
        }


@dataclass
class MetaProcessingResult:
    """Result from meta-aware case processing."""

    case_result: CaseResult
    strategy_used: str
    adaptation_triggered: bool = False
    failure_pattern_detected: Optional[FailurePattern] = None
    processing_insights: Dict[str, Any] = field(default_factory=dict)


class MetaAwareBatchProcessor:
    """
    Batch processor with meta-reasoning integration.

    Processes cases with adaptive strategy selection, failure analysis,
    and prompt optimization based on performance patterns.

    Example:
        >>> processor = MetaAwareBatchProcessor(
        ...     pipeline_processor=pipeline,
        ...     config=MetaAwareBatchConfig(),
        ... )
        >>> result = processor.process_case_with_meta(case)
        >>> print(f"Strategy used: {result.strategy_used}")
    """

    def __init__(
        self,
        pipeline_processor: Any,  # FullPipelineProcessor
        config: Optional[MetaAwareBatchConfig] = None,
        failure_analyzer: Optional[Any] = None,  # FailureAnalyzer
        strategy_selector: Optional[Any] = None,  # StrategySelector
        prompt_optimizer: Optional[Any] = None,  # PromptOptimizer
    ):
        """
        Initialize meta-aware batch processor.

        Args:
            pipeline_processor: The underlying pipeline processor
            config: Meta-aware batch configuration
            failure_analyzer: Optional failure analyzer component
            strategy_selector: Optional strategy selector component
            prompt_optimizer: Optional prompt optimizer component
        """
        self.pipeline = pipeline_processor
        self.config = config or MetaAwareBatchConfig()

        # Meta-reasoning components (lazy-loaded if not provided)
        self._failure_analyzer = failure_analyzer
        self._strategy_selector = strategy_selector
        self._prompt_optimizer = prompt_optimizer

        # Pattern tracking
        self.failure_patterns: List[FailurePattern] = []
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        self.adaptations: List[Adaptation] = []

        # Statistics
        self.cases_processed = 0
        self.successful_cases = 0
        self.failed_cases = 0

        logger.info("Initialized MetaAwareBatchProcessor")

    @property
    def failure_analyzer(self) -> Any:
        """Get or create failure analyzer."""
        if self._failure_analyzer is None:
            try:
                from loft.meta import create_failure_analyzer

                self._failure_analyzer = create_failure_analyzer()
            except ImportError:
                logger.warning("Could not import failure analyzer")
        return self._failure_analyzer

    @property
    def strategy_selector(self) -> Any:
        """Get or create strategy selector."""
        if self._strategy_selector is None:
            try:
                from loft.meta import create_selector

                self._strategy_selector = create_selector()
            except ImportError:
                logger.warning("Could not import strategy selector")
        return self._strategy_selector

    @property
    def prompt_optimizer(self) -> Any:
        """Get or create prompt optimizer."""
        if self._prompt_optimizer is None:
            try:
                from loft.meta import create_prompt_optimizer

                self._prompt_optimizer = create_prompt_optimizer()
            except ImportError:
                logger.warning("Could not import prompt optimizer")
        return self._prompt_optimizer

    def process_case_with_meta(
        self,
        case: Dict[str, Any],
        accumulated_rules: Optional[List[str]] = None,
    ) -> MetaProcessingResult:
        """
        Process case with meta-reasoning support.

        Args:
            case: Test case dictionary
            accumulated_rules: Previously generated rule IDs

        Returns:
            MetaProcessingResult with case result and meta-insights
        """
        accumulated_rules = accumulated_rules or []
        case_id = case.get("id", "unknown")

        logger.debug(f"Processing case {case_id} with meta-reasoning")

        # 1. Select strategy based on case characteristics
        strategy = self._select_strategy(case)

        # 2. Process case through pipeline
        start_time = time.time()
        case_result = self.pipeline.process_case(case, accumulated_rules)
        processing_time = time.time() - start_time

        self.cases_processed += 1

        # 3. Analyze result and detect patterns
        adaptation_triggered = False
        failure_pattern = None

        if case_result.status == CaseStatus.SUCCESS:
            self.successful_cases += 1
            self._update_strategy_performance(strategy, success=True)
        else:
            self.failed_cases += 1
            self._update_strategy_performance(strategy, success=False)

            # Analyze failure if enabled
            if self.config.enable_failure_analysis:
                failure_pattern = self._analyze_failure(case, case_result, strategy)

                if failure_pattern:
                    self.failure_patterns.append(failure_pattern)

                    # Check if adaptation is needed
                    if self._should_adapt():
                        self._adapt_strategies()
                        adaptation_triggered = True

        # 4. Build processing insights
        insights = {
            "strategy_selected": strategy,
            "processing_time_ms": processing_time * 1000,
            "cases_processed": self.cases_processed,
            "current_success_rate": (
                self.successful_cases / self.cases_processed
                if self.cases_processed > 0
                else 0.0
            ),
            "adaptations_count": len(self.adaptations),
        }

        return MetaProcessingResult(
            case_result=case_result,
            strategy_used=strategy,
            adaptation_triggered=adaptation_triggered,
            failure_pattern_detected=failure_pattern,
            processing_insights=insights,
        )

    def _select_strategy(self, case: Dict[str, Any]) -> str:
        """
        Select strategy for case processing.

        Args:
            case: Test case dictionary

        Returns:
            Strategy name to use
        """
        if not self.config.enable_strategy_selection:
            return self.config.default_strategy

        # Use strategy selector if available
        if self.strategy_selector:
            try:
                # Detect case type from case characteristics
                case_type = self._detect_case_type(case)

                # Get strategy recommendation
                selection = self.strategy_selector.select(
                    case_type=case_type,
                    historical_performance=self.strategy_performance,
                )

                if hasattr(selection, "strategy"):
                    return selection.strategy.value
                elif hasattr(selection, "name"):
                    return selection.name

            except Exception as e:
                logger.debug(f"Strategy selection failed: {e}")

        # Fall back to weighted random selection
        return self._weighted_strategy_selection()

    def _detect_case_type(self, case: Dict[str, Any]) -> str:
        """Detect case type from case data."""
        # Check for explicit case type
        if "case_type" in case:
            return case["case_type"]

        # Infer from domain
        domain = case.get("_domain", case.get("domain", "general"))

        # Infer from facts
        facts = case.get("asp_facts", case.get("facts", ""))
        if "contract" in facts.lower():
            return "contract"
        elif "tort" in facts.lower():
            return "tort"
        elif "property" in facts.lower():
            return "property"

        return domain

    def _weighted_strategy_selection(self) -> str:
        """Select strategy based on current weights."""
        import random

        weights = self.config.strategy_weights
        total_weight = sum(weights.values())

        if total_weight == 0:
            return self.config.default_strategy

        r = random.uniform(0, total_weight)
        cumulative = 0.0

        for strategy, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                return strategy

        return self.config.default_strategy

    def _update_strategy_performance(self, strategy: str, success: bool) -> None:
        """Update strategy performance tracking."""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                "successes": 0,
                "failures": 0,
                "total": 0,
                "success_rate": 0.0,
            }

        perf = self.strategy_performance[strategy]
        perf["total"] += 1

        if success:
            perf["successes"] += 1
        else:
            perf["failures"] += 1

        perf["success_rate"] = perf["successes"] / perf["total"]

    def _analyze_failure(
        self,
        case: Dict[str, Any],
        result: CaseResult,
        strategy: str,
    ) -> Optional[FailurePattern]:
        """
        Analyze a failure and extract pattern.

        Args:
            case: Failed case
            result: Case result
            strategy: Strategy that was used

        Returns:
            FailurePattern if detected
        """
        case_id = case.get("id", "unknown")
        error_message = result.error_message or "Unknown error"

        # Categorize failure type
        failure_type = self._categorize_failure(result)

        # Check if this pattern already exists
        for pattern in self.failure_patterns:
            if pattern.failure_type == failure_type:
                pattern.count += 1
                pattern.cases.append(case_id)
                pattern.last_seen = datetime.now()
                return pattern

        # Create new pattern
        pattern = FailurePattern(
            failure_type=failure_type,
            count=1,
            cases=[case_id],
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )

        # Use failure analyzer if available
        if self.failure_analyzer and self.config.enable_failure_analysis:
            try:
                # Analyze root cause
                analysis = self.failure_analyzer.analyze_failure(
                    error_message=error_message,
                    case_data=case,
                    strategy_used=strategy,
                )

                if hasattr(analysis, "root_cause"):
                    pattern.root_cause = str(analysis.root_cause)

                if hasattr(analysis, "recommendations") and analysis.recommendations:
                    pattern.recommendation = analysis.recommendations[0]

            except Exception as e:
                logger.debug(f"Failure analysis error: {e}")

        return pattern

    def _categorize_failure(self, result: CaseResult) -> str:
        """Categorize failure type from result."""
        if result.error_message:
            msg = result.error_message.lower()

            if "timeout" in msg:
                return "timeout"
            elif "validation" in msg:
                return "validation_failure"
            elif "generation" in msg:
                return "generation_failure"
            elif "parse" in msg or "syntax" in msg:
                return "syntax_error"
            elif "api" in msg or "rate" in msg:
                return "api_error"

        if result.rules_generated == 0:
            return "no_rules_generated"

        if result.rules_accepted == 0:
            return "no_rules_accepted"

        return "unknown_failure"

    def _should_adapt(self) -> bool:
        """Check if adaptation is needed based on failure patterns."""
        if len(self.failure_patterns) < self.config.min_failures_for_adaptation:
            return False

        # Get recent failures
        window_size = self.config.adaptation_window_size
        recent_failures = self.failure_patterns[-window_size:]

        # Count failure types
        failure_types = Counter(f.failure_type for f in recent_failures)

        # Check if any type exceeds threshold
        threshold_count = int(window_size * self.config.failure_pattern_threshold)

        return any(count >= threshold_count for count in failure_types.values())

    def _adapt_strategies(self) -> None:
        """Adapt strategies based on failure analysis."""
        logger.info("Triggering strategy adaptation")

        # Get most common recent failure type
        recent_failures = self.failure_patterns[-self.config.adaptation_window_size :]
        failure_types = Counter(f.failure_type for f in recent_failures)
        most_common = failure_types.most_common(1)

        if not most_common:
            return

        common_failure, count = most_common[0]

        # Determine adaptation based on failure type
        changes = {}
        reason = f"Repeated {common_failure} failures ({count} occurrences)"

        if common_failure == "timeout":
            # Reduce complexity by favoring simpler strategies
            changes = {"checklist": 1.5, "dialectical": 0.5}
            self._adjust_strategy_weights(changes)

        elif common_failure == "validation_failure":
            # Favor more rigorous strategies
            changes = {"dialectical": 1.5, "balancing_test": 1.3, "checklist": 0.8}
            self._adjust_strategy_weights(changes)

        elif common_failure == "no_rules_generated":
            # Favor more creative strategies
            changes = {"analogical": 1.5, "causal_chain": 1.3}
            self._adjust_strategy_weights(changes)

        # Record adaptation
        adaptation = Adaptation(
            adaptation_type="strategy_change",
            timestamp=datetime.now(),
            trigger=f"failure_pattern:{common_failure}",
            changes=changes,
            reason=reason,
        )
        self.adaptations.append(adaptation)

        logger.info(f"Adapted strategies: {changes}")

    def _adjust_strategy_weights(self, adjustments: Dict[str, float]) -> None:
        """Adjust strategy weights by multipliers."""
        for strategy, multiplier in adjustments.items():
            if strategy in self.config.strategy_weights:
                self.config.strategy_weights[strategy] *= multiplier

        # Normalize weights
        total = sum(self.config.strategy_weights.values())
        if total > 0:
            for strategy in self.config.strategy_weights:
                self.config.strategy_weights[strategy] /= total
                self.config.strategy_weights[strategy] *= len(
                    self.config.strategy_weights
                )

    def get_adaptations(self) -> List[Adaptation]:
        """Get list of adaptations made."""
        return self.adaptations.copy()

    def get_failure_summary(self) -> Dict[str, Any]:
        """Get summary of failure patterns."""
        failure_counts = Counter(f.failure_type for f in self.failure_patterns)

        return {
            "total_failures": len(self.failure_patterns),
            "failure_types": dict(failure_counts),
            "patterns": [p.to_dict() for p in self.failure_patterns[-10:]],
        }

    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of strategy performance."""
        return {
            "current_weights": self.config.strategy_weights.copy(),
            "performance": self.strategy_performance.copy(),
            "adaptations_count": len(self.adaptations),
        }

    def get_processor_function(
        self,
    ) -> Callable[[Dict[str, Any], List[str]], CaseResult]:
        """
        Get a processor function compatible with BatchLearningHarness.

        Returns:
            Callable that processes cases and returns CaseResult
        """

        def process(case: Dict[str, Any], accumulated_rules: List[str]) -> CaseResult:
            result = self.process_case_with_meta(case, accumulated_rules)
            return result.case_result

        return process


def create_meta_aware_processor(
    pipeline_processor: Any,
    config: Optional[MetaAwareBatchConfig] = None,
) -> MetaAwareBatchProcessor:
    """
    Factory function to create a meta-aware processor.

    Args:
        pipeline_processor: Underlying pipeline processor
        config: Optional configuration

    Returns:
        Configured MetaAwareBatchProcessor
    """
    return MetaAwareBatchProcessor(
        pipeline_processor=pipeline_processor,
        config=config,
    )
