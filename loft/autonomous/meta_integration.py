"""
Meta-Reasoning Integration for Autonomous Test Harness.

This module orchestrates all meta-reasoning components during autonomous
long-running experiments, coordinating improvement cycles, failure
analysis, strategy selection, and prompt optimization.

Classes:
- MetaReasoningOrchestrator: Main coordinator for meta-reasoning components
- ImprovementCycleManager: Manages improvement cycle execution
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loft.autonomous.config import AutonomousRunConfig, MetaReasoningConfig
from loft.autonomous.schemas import (
    CycleResult,
    CycleStatus,
    MetaReasoningState,
    RunMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class FailureAnalysisReport:
    """Report from failure analysis.

    Attributes:
        total_failures: Total failures analyzed
        patterns_identified: Number of patterns found
        pattern_descriptions: Descriptions of patterns
        recommended_actions: Suggested improvement actions
        domain_breakdown: Failures by domain
    """

    total_failures: int = 0
    patterns_identified: int = 0
    pattern_descriptions: List[str] = field(default_factory=list)
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)
    domain_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_failures": self.total_failures,
            "patterns_identified": self.patterns_identified,
            "pattern_descriptions": self.pattern_descriptions,
            "recommended_actions": self.recommended_actions,
            "domain_breakdown": self.domain_breakdown,
        }


@dataclass
class ImprovementCycleResult:
    """Result from running an improvement cycle.

    Attributes:
        cycle_number: Cycle number
        success: Whether cycle succeeded
        actions_taken: Number of actions executed
        improvements_made: Number of successful improvements
        accuracy_before: Accuracy before cycle
        accuracy_after: Accuracy after cycle
        failure_report: Failure analysis report
        error_message: Error if cycle failed
    """

    cycle_number: int
    success: bool
    actions_taken: int = 0
    improvements_made: int = 0
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    failure_report: Optional[FailureAnalysisReport] = None
    error_message: Optional[str] = None

    def to_cycle_result(self) -> CycleResult:
        """Convert to CycleResult schema."""
        return CycleResult(
            cycle_number=self.cycle_number,
            status=CycleStatus.COMPLETED if self.success else CycleStatus.FAILED,
            completed_at=datetime.now(),
            improvements_applied=self.improvements_made,
            accuracy_before=self.accuracy_before,
            accuracy_after=self.accuracy_after,
            failure_patterns=(
                self.failure_report.pattern_descriptions if self.failure_report else []
            ),
            error_message=self.error_message,
        )


class MetaReasoningOrchestrator:
    """Orchestrates meta-reasoning components for autonomous improvement.

    Coordinates the interaction between:
    - AutonomousImprover: Executes improvement cycles
    - PromptOptimizer: Optimizes prompt templates
    - FailureAnalyzer: Analyzes prediction failures
    - StrategySelector: Selects optimal strategies

    Attributes:
        config: Meta-reasoning configuration
        improver: AutonomousImprover instance (optional)
        prompt_optimizer: PromptOptimizer instance (optional)
        failure_analyzer: FailureAnalyzer instance (optional)
        strategy_selector: StrategySelector instance (optional)
    """

    def __init__(
        self,
        config: MetaReasoningConfig,
        improver: Optional[Any] = None,
        prompt_optimizer: Optional[Any] = None,
        failure_analyzer: Optional[Any] = None,
        strategy_selector: Optional[Any] = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Meta-reasoning configuration
            improver: Optional AutonomousImprover
            prompt_optimizer: Optional PromptOptimizer
            failure_analyzer: Optional FailureAnalyzer
            strategy_selector: Optional StrategySelector
        """
        self._config = config
        self._improver = improver
        self._prompt_optimizer = prompt_optimizer
        self._failure_analyzer = failure_analyzer
        self._strategy_selector = strategy_selector

        self._current_cycle = 0
        self._cycle_history: List[CycleResult] = []
        self._accumulated_metrics: RunMetrics = RunMetrics()

        # Failure pattern tracking from LLM processor (issue #169)
        self._processor_failure_patterns: Dict[str, int] = {}
        self._processor_failure_details: List[Dict[str, Any]] = []

        self._on_cycle_start: Optional[Callable[[int], None]] = None
        self._on_cycle_complete: Optional[Callable[[CycleResult], None]] = None
        self._on_improvement: Optional[Callable[[Dict[str, Any]], None]] = None

    @property
    def config(self) -> MetaReasoningConfig:
        """Get configuration."""
        return self._config

    @property
    def current_cycle(self) -> int:
        """Get current cycle number."""
        return self._current_cycle

    @property
    def cycle_history(self) -> List[CycleResult]:
        """Get cycle history."""
        return self._cycle_history.copy()

    def set_callbacks(
        self,
        on_cycle_start: Optional[Callable[[int], None]] = None,
        on_cycle_complete: Optional[Callable[[CycleResult], None]] = None,
        on_improvement: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Set callbacks for cycle events.

        Args:
            on_cycle_start: Called when a cycle starts
            on_cycle_complete: Called when a cycle completes
            on_improvement: Called when an improvement is applied
        """
        self._on_cycle_start = on_cycle_start
        self._on_cycle_complete = on_cycle_complete
        self._on_improvement = on_improvement

    def set_improver(self, improver: Any) -> None:
        """Set the AutonomousImprover instance.

        Args:
            improver: AutonomousImprover instance
        """
        self._improver = improver

    def set_prompt_optimizer(self, optimizer: Any) -> None:
        """Set the PromptOptimizer instance.

        Args:
            optimizer: PromptOptimizer instance
        """
        self._prompt_optimizer = optimizer

    def set_failure_analyzer(self, analyzer: Any) -> None:
        """Set the FailureAnalyzer instance.

        Args:
            analyzer: FailureAnalyzer instance
        """
        self._failure_analyzer = analyzer

    def set_strategy_selector(self, selector: Any) -> None:
        """Set the StrategySelector instance.

        Args:
            selector: StrategySelector instance
        """
        self._strategy_selector = selector

    def update_failure_patterns(
        self,
        patterns: Dict[str, int],
        details: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update failure patterns from LLM processor (issue #169).

        This method receives categorized failure patterns from LLMCaseProcessor
        to inform meta-reasoning decisions.

        Args:
            patterns: Dictionary mapping failure categories to counts
            details: Optional list of detailed failure information
        """
        # Merge patterns with existing
        for category, count in patterns.items():
            current_count = self._processor_failure_patterns.get(category, 0)
            self._processor_failure_patterns[category] = current_count + count

        if details:
            self._processor_failure_details.extend(details)

        logger.debug(
            f"Updated failure patterns: {len(patterns)} categories, "
            f"{sum(patterns.values())} new failures"
        )

    def get_processor_failure_patterns(self) -> Dict[str, int]:
        """Get accumulated failure patterns from LLM processor.

        Returns:
            Dictionary mapping failure categories to occurrence counts
        """
        return dict(self._processor_failure_patterns)

    def suggest_prompt_improvements(self) -> List[Dict[str, Any]]:
        """Suggest prompt improvements based on failure patterns (issue #169).

        Analyzes accumulated failure patterns to generate actionable
        suggestions for improving prompts and rule generation.

        Returns:
            List of improvement suggestions with category, description, and priority
        """
        suggestions = []

        if not self._processor_failure_patterns:
            return suggestions

        total_failures = sum(self._processor_failure_patterns.values())

        # Sort patterns by frequency
        sorted_patterns = sorted(
            self._processor_failure_patterns.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for category, count in sorted_patterns:
            percentage = (count / total_failures * 100) if total_failures > 0 else 0

            suggestion = {
                "category": category,
                "count": count,
                "percentage": round(percentage, 1),
                "priority": ("high" if percentage > 30 else "medium" if percentage > 10 else "low"),
            }

            # Generate specific recommendations based on category
            if category == "unsafe_variable":
                suggestion["description"] = (
                    "Variables appear only in negative literals. "
                    "Add explicit variable safety constraints to prompts."
                )
                suggestion["recommended_action"] = (
                    "Update GAP_FILLING prompt with variable safety rules"
                )
            elif category == "embedded_period":
                suggestion["description"] = (
                    "Periods appearing in fact atoms cause parsing errors. "
                    "Add period sanitization to fact preprocessing."
                )
                suggestion["recommended_action"] = "Sanitize periods in facts before ASP processing"
            elif category == "syntax_error":
                suggestion["description"] = (
                    "ASP syntax errors in generated rules. "
                    "Add syntax validation examples to prompts."
                )
                suggestion["recommended_action"] = (
                    "Include ASP syntax examples in rule generation prompts"
                )
            elif category == "invalid_arithmetic":
                suggestion["description"] = (
                    "Invalid arithmetic expressions in rules. "
                    "Restrict or guide arithmetic usage in prompts."
                )
                suggestion["recommended_action"] = "Add arithmetic constraints to rule templates"
            elif category == "grounding_error":
                suggestion["description"] = (
                    "Rules cannot be grounded due to infinite domains. "
                    "Add domain restriction guidance to prompts."
                )
                suggestion["recommended_action"] = "Include domain bounding examples in prompts"
            elif category == "json_parse_error":
                suggestion["description"] = (
                    "LLM responses not parseable as JSON. "
                    "Add stricter JSON formatting requirements."
                )
                suggestion["recommended_action"] = "Enforce JSON schema in prompt instructions"
            elif category == "validation_error":
                suggestion["description"] = (
                    "Generated rules fail validation checks. "
                    "Improve rule quality guidance in prompts."
                )
                suggestion["recommended_action"] = (
                    "Add validation criteria to rule generation prompts"
                )
            else:
                suggestion["description"] = f"Unknown failure category: {category}"
                suggestion["recommended_action"] = "Investigate and categorize these failures"

            suggestions.append(suggestion)

        return suggestions

    def clear_failure_patterns(self) -> None:
        """Clear accumulated failure patterns (e.g., after improvement cycle)."""
        self._processor_failure_patterns.clear()
        self._processor_failure_details.clear()
        logger.debug("Cleared processor failure patterns")

    def should_run_cycle(self, cases_since_last_cycle: int) -> bool:
        """Check if an improvement cycle should run.

        Args:
            cases_since_last_cycle: Cases processed since last cycle

        Returns:
            True if cycle should run
        """
        if not self._config.enable_autonomous_improvement:
            return False

        if cases_since_last_cycle < self._config.min_cases_for_analysis:
            return False

        return cases_since_last_cycle >= self._config.improvement_cycle_interval_cases

    def run_improvement_cycle(
        self,
        case_results: List[Dict[str, Any]],
        accumulated_rules: List[Dict[str, Any]],
        current_accuracy: float,
    ) -> ImprovementCycleResult:
        """Run a complete improvement cycle.

        Coordinates all meta-reasoning components to analyze results
        and apply improvements.

        Args:
            case_results: Results from recent cases
            accumulated_rules: Rules generated so far
            current_accuracy: Current accuracy metric

        Returns:
            ImprovementCycleResult with outcomes
        """
        self._current_cycle += 1
        cycle_number = self._current_cycle

        logger.info(f"Starting improvement cycle {cycle_number}")

        if self._on_cycle_start:
            self._on_cycle_start(cycle_number)

        cycle_result = CycleResult(
            cycle_number=cycle_number,
            status=CycleStatus.ANALYZING,
            started_at=datetime.now(),
            accuracy_before=current_accuracy,
        )

        try:
            failure_report = None
            improvements_made = 0
            actions_taken = 0

            if self._config.enable_failure_analysis and self._failure_analyzer:
                failure_report = self._analyze_failures(case_results)
                cycle_result.failure_patterns = failure_report.pattern_descriptions

            if self._config.enable_autonomous_improvement and self._improver:
                result = self._run_improver_cycle(case_results, failure_report, current_accuracy)
                actions_taken = result.get("actions_taken", 0)
                improvements_made = result.get("improvements_made", 0)

            if self._config.enable_prompt_optimization and self._prompt_optimizer:
                prompt_result = self._optimize_prompts(case_results)
                cycle_result.prompt_changes = prompt_result.get("changes", 0)

            if self._config.enable_strategy_selection and self._strategy_selector:
                strategy_result = self._update_strategies(case_results)
                cycle_result.strategy_changes = strategy_result.get("changes", 0)

            cycle_result.status = CycleStatus.COMPLETED
            cycle_result.completed_at = datetime.now()
            cycle_result.improvements_applied = improvements_made
            cycle_result.cases_processed = len(case_results)
            cycle_result.accuracy_after = current_accuracy

            self._cycle_history.append(cycle_result)

            if self._on_cycle_complete:
                self._on_cycle_complete(cycle_result)

            logger.info(
                f"Completed improvement cycle {cycle_number}: "
                f"{actions_taken} actions, {improvements_made} improvements"
            )

            return ImprovementCycleResult(
                cycle_number=cycle_number,
                success=True,
                actions_taken=actions_taken,
                improvements_made=improvements_made,
                accuracy_before=current_accuracy,
                accuracy_after=current_accuracy,
                failure_report=failure_report,
            )

        except Exception as e:
            logger.error(f"Improvement cycle {cycle_number} failed: {e}")

            cycle_result.status = CycleStatus.FAILED
            cycle_result.completed_at = datetime.now()
            cycle_result.error_message = str(e)
            self._cycle_history.append(cycle_result)

            if self._on_cycle_complete:
                self._on_cycle_complete(cycle_result)

            return ImprovementCycleResult(
                cycle_number=cycle_number,
                success=False,
                accuracy_before=current_accuracy,
                error_message=str(e),
            )

    def _analyze_failures(self, case_results: List[Dict[str, Any]]) -> FailureAnalysisReport:
        """Analyze failures from case results.

        Args:
            case_results: Results to analyze

        Returns:
            FailureAnalysisReport
        """
        failed_cases = [r for r in case_results if not r.get("correct", True)]

        if not failed_cases or not self._failure_analyzer:
            return FailureAnalysisReport(total_failures=len(failed_cases))

        domain_counts: Dict[str, int] = {}
        for case in failed_cases:
            domain = case.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        patterns: List[str] = []
        if hasattr(self._failure_analyzer, "identify_patterns"):
            try:
                pattern_results = self._failure_analyzer.identify_patterns(failed_cases)
                patterns = [p.get("description", str(p)) for p in pattern_results]
            except Exception as e:
                logger.warning(f"Pattern identification failed: {e}")

        return FailureAnalysisReport(
            total_failures=len(failed_cases),
            patterns_identified=len(patterns),
            pattern_descriptions=patterns[:10],
            domain_breakdown=domain_counts,
        )

    def _run_improver_cycle(
        self,
        case_results: List[Dict[str, Any]],
        failure_report: Optional[FailureAnalysisReport],
        current_accuracy: float,
    ) -> Dict[str, Any]:
        """Run the autonomous improver.

        Args:
            case_results: Case results
            failure_report: Failure analysis report
            current_accuracy: Current accuracy

        Returns:
            Dictionary with cycle outcomes
        """
        if not self._improver:
            return {"actions_taken": 0, "improvements_made": 0}

        try:
            if hasattr(self._improver, "run_cycle"):
                result = self._improver.run_cycle(
                    case_results=case_results,
                    current_accuracy=current_accuracy,
                    failure_patterns=(
                        failure_report.pattern_descriptions if failure_report else []
                    ),
                )
                return {
                    "actions_taken": result.get("actions_taken", 0),
                    "improvements_made": result.get("improvements_made", 0),
                }

            if hasattr(self._improver, "execute_improvements"):
                cycle_id = self._improver.start_improvement_cycle(goals=[])
                success = self._improver.execute_improvements(cycle_id)
                return {
                    "actions_taken": 1 if success else 0,
                    "improvements_made": 1 if success else 0,
                }

        except Exception as e:
            logger.warning(f"Improver cycle failed: {e}")

        return {"actions_taken": 0, "improvements_made": 0}

    def _optimize_prompts(self, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize prompts based on results.

        Args:
            case_results: Case results

        Returns:
            Dictionary with optimization outcomes
        """
        if not self._prompt_optimizer:
            return {"changes": 0}

        try:
            if hasattr(self._prompt_optimizer, "analyze_and_optimize"):
                result = self._prompt_optimizer.analyze_and_optimize(case_results)
                return {"changes": result.get("changes_made", 0)}

            if hasattr(self._prompt_optimizer, "record_result"):
                for result in case_results:
                    self._prompt_optimizer.record_result(
                        prompt_id=result.get("prompt_id", "default"),
                        success=result.get("correct", False),
                        context=result,
                    )
                return {"changes": 0}

        except Exception as e:
            logger.warning(f"Prompt optimization failed: {e}")

        return {"changes": 0}

    def _update_strategies(self, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update strategy selections based on results.

        Args:
            case_results: Case results

        Returns:
            Dictionary with strategy updates
        """
        if not self._strategy_selector:
            return {"changes": 0}

        try:
            changes = 0
            if hasattr(self._strategy_selector, "update_from_results"):
                changes = self._strategy_selector.update_from_results(case_results)
                return {"changes": changes}

            if hasattr(self._strategy_selector, "evaluator"):
                evaluator = self._strategy_selector.evaluator
                for result in case_results:
                    if hasattr(evaluator, "record_outcome"):
                        evaluator.record_outcome(
                            strategy_name=result.get("strategy", "default"),
                            domain=result.get("domain", "unknown"),
                            correct=result.get("correct", False),
                        )
                return {"changes": 0}

        except Exception as e:
            logger.warning(f"Strategy update failed: {e}")

        return {"changes": 0}

    def get_state_snapshot(self) -> MetaReasoningState:
        """Get serializable state snapshot.

        Returns:
            MetaReasoningState for checkpoint
        """
        improver_state = {}
        if self._improver and hasattr(self._improver, "get_state"):
            improver_state = self._improver.get_state()

        optimizer_state = {}
        if self._prompt_optimizer and hasattr(self._prompt_optimizer, "get_state"):
            optimizer_state = self._prompt_optimizer.get_state()

        analyzer_state = {}
        if self._failure_analyzer and hasattr(self._failure_analyzer, "get_state"):
            analyzer_state = self._failure_analyzer.get_state()

        selector_state = {}
        if self._strategy_selector and hasattr(self._strategy_selector, "get_state"):
            selector_state = self._strategy_selector.get_state()

        return MetaReasoningState(
            improver_state=improver_state,
            optimizer_state=optimizer_state,
            analyzer_state=analyzer_state,
            selector_state=selector_state,
        )

    def restore_from_state(self, state: MetaReasoningState) -> None:
        """Restore component state from snapshot.

        Args:
            state: MetaReasoningState to restore from
        """
        if self._improver and hasattr(self._improver, "restore_state"):
            self._improver.restore_state(state.improver_state)

        if self._prompt_optimizer and hasattr(self._prompt_optimizer, "restore_state"):
            self._prompt_optimizer.restore_state(state.optimizer_state)

        if self._failure_analyzer and hasattr(self._failure_analyzer, "restore_state"):
            self._failure_analyzer.restore_state(state.analyzer_state)

        if self._strategy_selector and hasattr(self._strategy_selector, "restore_state"):
            self._strategy_selector.restore_state(state.selector_state)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of meta-reasoning metrics.

        Returns:
            Dictionary with metrics summary
        """
        total_improvements = sum(cr.improvements_applied for cr in self._cycle_history)
        total_prompt_changes = sum(cr.prompt_changes for cr in self._cycle_history)
        total_strategy_changes = sum(cr.strategy_changes for cr in self._cycle_history)

        accuracy_trend = []
        for cr in self._cycle_history:
            accuracy_trend.append(
                {
                    "cycle": cr.cycle_number,
                    "before": cr.accuracy_before,
                    "after": cr.accuracy_after,
                    "delta": cr.accuracy_delta,
                }
            )

        return {
            "total_cycles": len(self._cycle_history),
            "total_improvements": total_improvements,
            "total_prompt_changes": total_prompt_changes,
            "total_strategy_changes": total_strategy_changes,
            "accuracy_trend": accuracy_trend,
            "failure_patterns": list(
                set(pattern for cr in self._cycle_history for pattern in cr.failure_patterns)
            ),
            # Processor failure patterns for meta-reasoning (issue #169)
            "processor_failure_patterns": dict(self._processor_failure_patterns),
            "total_processor_failures": sum(self._processor_failure_patterns.values()),
        }

    def reset(self) -> None:
        """Reset orchestrator state."""
        self._current_cycle = 0
        self._cycle_history.clear()
        self._accumulated_metrics = RunMetrics()


def create_orchestrator_from_config(
    config: AutonomousRunConfig,
) -> MetaReasoningOrchestrator:
    """Factory function to create an orchestrator from config.

    Args:
        config: Autonomous run configuration

    Returns:
        Configured MetaReasoningOrchestrator
    """
    return MetaReasoningOrchestrator(config=config.meta_reasoning)
