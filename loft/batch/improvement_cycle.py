"""
Batch improvement cycle management.

Manages improvement cycles during batch processing:
- Sets and tracks improvement goals
- Coordinates meta-reasoning components
- Evaluates cycle outcomes
- Generates improvement recommendations

Issue #255: Phase 8 meta-reasoning batch integration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from loft.batch.meta_state import MetaStateManager
from loft.batch.schemas import CaseResult, CaseStatus


@dataclass
class CycleGoal:
    """Goal for an improvement cycle."""

    goal_id: str
    metric: str  # "success_rate", "rules_generated", "validation_rate"
    target_value: float
    current_value: float = 0.0
    achieved: bool = False
    priority: int = 1  # 1 = highest

    def check_achieved(self, current: float) -> bool:
        """Check if goal is achieved."""
        self.current_value = current
        self.achieved = current >= self.target_value
        return self.achieved

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal_id": self.goal_id,
            "metric": self.metric,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "achieved": self.achieved,
            "priority": self.priority,
        }


@dataclass
class CycleMetrics:
    """Metrics collected during an improvement cycle."""

    cases_processed: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    rules_generated: int = 0
    rules_accepted: int = 0
    rules_rejected: int = 0

    # Derived metrics
    success_rate: float = 0.0
    generation_rate: float = 0.0
    acceptance_rate: float = 0.0

    # Timing
    total_time_ms: float = 0.0
    avg_case_time_ms: float = 0.0

    def compute_derived(self) -> None:
        """Compute derived metrics."""
        if self.cases_processed > 0:
            self.success_rate = self.successful_cases / self.cases_processed
            self.generation_rate = self.rules_generated / self.cases_processed
            self.avg_case_time_ms = self.total_time_ms / self.cases_processed

        if self.rules_generated > 0:
            self.acceptance_rate = self.rules_accepted / self.rules_generated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cases_processed": self.cases_processed,
            "successful_cases": self.successful_cases,
            "failed_cases": self.failed_cases,
            "rules_generated": self.rules_generated,
            "rules_accepted": self.rules_accepted,
            "rules_rejected": self.rules_rejected,
            "success_rate": self.success_rate,
            "generation_rate": self.generation_rate,
            "acceptance_rate": self.acceptance_rate,
            "total_time_ms": self.total_time_ms,
            "avg_case_time_ms": self.avg_case_time_ms,
        }


@dataclass
class GoalResults:
    """Results of goal evaluation."""

    all_achieved: bool = True
    achieved: List[CycleGoal] = field(default_factory=list)
    unmet: List[CycleGoal] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "all_achieved": self.all_achieved,
            "achieved": [g.to_dict() for g in self.achieved],
            "unmet": [g.to_dict() for g in self.unmet],
        }


@dataclass
class ImprovementCycleResult:
    """Result of a complete improvement cycle."""

    cycle_id: str
    cases_processed: int
    metrics: CycleMetrics
    goals: GoalResults
    adaptations: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "cases_processed": self.cases_processed,
            "metrics": self.metrics.to_dict(),
            "goals": self.goals.to_dict(),
            "adaptations": self.adaptations,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


class BatchImprovementCycle:
    """
    Manages improvement cycles during batch processing.

    Coordinates meta-reasoning components, tracks goals, and
    generates improvement recommendations based on cycle outcomes.

    Example:
        >>> cycle = BatchImprovementCycle(
        ...     meta_processor=processor,
        ...     state_manager=state_manager,
        ... )
        >>> result = cycle.run_improvement_cycle(cases, "cycle_001")
        >>> print(f"Success rate: {result.metrics.success_rate:.1%}")
    """

    def __init__(
        self,
        meta_processor: Any,  # MetaAwareBatchProcessor
        state_manager: Optional[MetaStateManager] = None,
        checkpoint_interval: int = 10,
    ):
        """
        Initialize improvement cycle.

        Args:
            meta_processor: Meta-aware batch processor
            state_manager: Optional meta-state manager
            checkpoint_interval: Cases between checkpoints
        """
        self.meta_processor = meta_processor
        self.state_manager = state_manager
        self.checkpoint_interval = checkpoint_interval

        # State
        self.current_goals: List[CycleGoal] = []
        self.cycle_history: List[ImprovementCycleResult] = []

    def set_goals(self, goals: List[CycleGoal]) -> None:
        """
        Set goals for the next improvement cycle.

        Args:
            goals: List of goals to achieve
        """
        self.current_goals = goals
        logger.info(f"Set {len(goals)} goals for improvement cycle")

    def create_default_goals(
        self,
        success_rate_target: float = 0.7,
        acceptance_rate_target: float = 0.5,
    ) -> List[CycleGoal]:
        """
        Create default improvement goals.

        Args:
            success_rate_target: Target success rate
            acceptance_rate_target: Target rule acceptance rate

        Returns:
            List of default goals
        """
        goals = [
            CycleGoal(
                goal_id="success_rate",
                metric="success_rate",
                target_value=success_rate_target,
                priority=1,
            ),
            CycleGoal(
                goal_id="acceptance_rate",
                metric="acceptance_rate",
                target_value=acceptance_rate_target,
                priority=2,
            ),
        ]

        return goals

    def run_improvement_cycle(
        self,
        cases: List[Dict[str, Any]],
        cycle_id: str,
        accumulated_rules: Optional[List[str]] = None,
    ) -> ImprovementCycleResult:
        """
        Run a complete improvement cycle on cases.

        Args:
            cases: Test cases to process
            cycle_id: Unique cycle identifier
            accumulated_rules: Previously generated rule IDs

        Returns:
            ImprovementCycleResult with metrics and recommendations
        """
        accumulated_rules = accumulated_rules or []

        logger.info(f"Starting improvement cycle {cycle_id} with {len(cases)} cases")

        # Ensure goals are set
        if not self.current_goals:
            self.current_goals = self.create_default_goals()

        # Initialize metrics
        metrics = CycleMetrics()
        start_time = datetime.now()
        results: List[CaseResult] = []

        # Process cases
        for i, case in enumerate(cases):
            try:
                result = self.meta_processor.process_case_with_meta(
                    case, accumulated_rules
                )
                case_result = result.case_result
                results.append(case_result)

                # Update metrics
                metrics.cases_processed += 1
                metrics.rules_generated += case_result.rules_generated
                metrics.rules_accepted += case_result.rules_accepted
                metrics.rules_rejected += case_result.rules_rejected

                if case_result.status == CaseStatus.SUCCESS:
                    metrics.successful_cases += 1
                else:
                    metrics.failed_cases += 1

                # Track accepted rules
                if case_result.generated_rule_ids:
                    accumulated_rules.extend(case_result.generated_rule_ids)

                # Checkpoint
                if (i + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(cycle_id, i + 1, metrics)

            except Exception as e:
                logger.error(f"Error processing case {case.get('id', i)}: {e}")
                metrics.failed_cases += 1
                metrics.cases_processed += 1

        # Compute timing
        end_time = datetime.now()
        metrics.total_time_ms = (end_time - start_time).total_seconds() * 1000
        metrics.compute_derived()

        # Evaluate goals
        goal_results = self._evaluate_goals(metrics)

        # Get adaptations from processor
        adaptations = []
        if hasattr(self.meta_processor, "get_adaptations"):
            adaptations = [
                a.to_dict() if hasattr(a, "to_dict") else a
                for a in self.meta_processor.get_adaptations()
            ]

        # Generate recommendations
        recommendations = self._generate_recommendations(goal_results, metrics)

        # Build result
        result = ImprovementCycleResult(
            cycle_id=cycle_id,
            cases_processed=metrics.cases_processed,
            metrics=metrics,
            goals=goal_results,
            adaptations=adaptations,
            recommendations=recommendations,
        )

        # Save to history
        self.cycle_history.append(result)

        # Update meta-state
        if self.state_manager:
            state = self.state_manager.load_or_create()
            state.update_from_processor(self.meta_processor)
            state.total_cycles_completed += 1
            self.state_manager.save(state)

        logger.info(
            f"Completed cycle {cycle_id}: "
            f"{metrics.success_rate:.1%} success, "
            f"{metrics.acceptance_rate:.1%} acceptance"
        )

        return result

    def _evaluate_goals(self, metrics: CycleMetrics) -> GoalResults:
        """Evaluate goals against cycle metrics."""
        results = GoalResults()

        for goal in self.current_goals:
            # Get metric value
            metric_value = getattr(metrics, goal.metric, None)

            if metric_value is not None:
                if goal.check_achieved(metric_value):
                    results.achieved.append(goal)
                else:
                    results.unmet.append(goal)
                    results.all_achieved = False
            else:
                logger.warning(f"Unknown metric: {goal.metric}")
                results.unmet.append(goal)
                results.all_achieved = False

        return results

    def _generate_recommendations(
        self,
        goal_results: GoalResults,
        metrics: CycleMetrics,
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        for goal in goal_results.unmet:
            gap = goal.target_value - goal.current_value

            if goal.metric == "success_rate":
                if gap > 0.3:
                    recommendations.append(
                        "Consider simplifying case complexity or adding more diverse training"
                    )
                elif gap > 0.1:
                    recommendations.append(
                        "Try adjusting strategy weights towards more successful patterns"
                    )
                else:
                    recommendations.append(
                        f"Success rate is {gap:.1%} below target - continue current approach"
                    )

            elif goal.metric == "acceptance_rate":
                if gap > 0.3:
                    recommendations.append(
                        "Validation criteria may be too strict - consider threshold adjustment"
                    )
                elif gap > 0.1:
                    recommendations.append(
                        "Improve rule generation quality through prompt refinement"
                    )
                else:
                    recommendations.append(
                        f"Acceptance rate is {gap:.1%} below target - minor tuning needed"
                    )

        # Add performance recommendation
        if metrics.avg_case_time_ms > 5000:
            recommendations.append(
                f"Average case time ({metrics.avg_case_time_ms:.0f}ms) is high - "
                "consider optimization"
            )

        # Add success recommendation if goals met
        if goal_results.all_achieved:
            recommendations.append(
                "All goals achieved - consider increasing targets for next cycle"
            )

        return recommendations

    def _save_checkpoint(
        self,
        cycle_id: str,
        case_num: int,
        metrics: CycleMetrics,
    ) -> None:
        """Save checkpoint during cycle."""
        if self.state_manager:
            state = self.state_manager.load_or_create()
            state.update_from_processor(self.meta_processor)
            checkpoint_id = f"{cycle_id}_{case_num:04d}"
            self.state_manager.checkpoint(state, checkpoint_id)
            logger.debug(f"Saved checkpoint {checkpoint_id}")

    def get_cycle_history(self) -> List[ImprovementCycleResult]:
        """Get history of completed cycles."""
        return self.cycle_history.copy()

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of improvement progress."""
        if not self.cycle_history:
            return {"cycles_completed": 0}

        latest = self.cycle_history[-1]
        first = self.cycle_history[0]

        improvement = {}
        if hasattr(first.metrics, "success_rate") and hasattr(
            latest.metrics, "success_rate"
        ):
            improvement["success_rate"] = (
                latest.metrics.success_rate - first.metrics.success_rate
            )

        return {
            "cycles_completed": len(self.cycle_history),
            "latest_success_rate": latest.metrics.success_rate,
            "latest_acceptance_rate": latest.metrics.acceptance_rate,
            "improvement": improvement,
            "total_cases_processed": sum(c.cases_processed for c in self.cycle_history),
            "total_rules_generated": sum(
                c.metrics.rules_generated for c in self.cycle_history
            ),
        }


def create_improvement_cycle(
    meta_processor: Any,
    state_manager: Optional[MetaStateManager] = None,
    checkpoint_interval: int = 10,
) -> BatchImprovementCycle:
    """
    Factory function to create an improvement cycle.

    Args:
        meta_processor: Meta-aware batch processor
        state_manager: Optional state manager
        checkpoint_interval: Cases between checkpoints

    Returns:
        Configured BatchImprovementCycle
    """
    return BatchImprovementCycle(
        meta_processor=meta_processor,
        state_manager=state_manager,
        checkpoint_interval=checkpoint_interval,
    )
