"""
Self-Improvement Module for Meta-Reasoning.

This module provides autonomous self-improvement capabilities:
- Measurable improvement goals
- Progress tracking over time
- Autonomous improvement cycles
- Safety mechanisms and rollback

Integrates with other meta-reasoning components to enable
the system to improve itself without human intervention.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import uuid


class GoalStatus(Enum):
    """Status of an improvement goal."""

    PENDING = "pending"  # Not yet started
    IN_PROGRESS = "in_progress"  # Actively working on
    ACHIEVED = "achieved"  # Target reached
    FAILED = "failed"  # Failed to achieve
    ABANDONED = "abandoned"  # Abandoned due to infeasibility


class CycleStatus(Enum):
    """Status of an improvement cycle."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"


class ActionType(Enum):
    """Types of improvement actions."""

    PROMPT_REFINEMENT = "prompt_refinement"
    STRATEGY_ADJUSTMENT = "strategy_adjustment"
    RULE_MODIFICATION = "rule_modification"
    THRESHOLD_TUNING = "threshold_tuning"
    VALIDATION_ENHANCEMENT = "validation_enhancement"


class MetricType(Enum):
    """Types of improvement metrics."""

    ACCURACY = "accuracy"
    RULE_ACCEPTANCE_RATE = "rule_acceptance_rate"
    PROMPT_EFFECTIVENESS = "prompt_effectiveness"
    ERROR_DIAGNOSIS_ACCURACY = "error_diagnosis_accuracy"
    STRATEGY_SELECTION_ACCURACY = "strategy_selection_accuracy"
    LATENCY = "latency"
    CONFIDENCE = "confidence"
    COVERAGE = "coverage"


@dataclass
class MetricValue:
    """A single metric measurement."""

    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


@dataclass
class ImprovementGoal:
    """
    A measurable improvement goal.

    Defines a concrete target for system improvement
    with progress tracking and deadline support.
    """

    goal_id: str
    metric_type: MetricType
    target_value: float
    baseline_value: float
    current_value: float = 0.0
    deadline: Optional[datetime] = None
    status: GoalStatus = GoalStatus.PENDING
    priority: int = 1  # 1 = highest
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    achieved_at: Optional[datetime] = None
    progress_history: List[MetricValue] = field(default_factory=list)

    @property
    def progress_percentage(self) -> float:
        """Calculate progress toward goal as percentage."""
        if self.target_value == self.baseline_value:
            return 100.0 if self.current_value >= self.target_value else 0.0

        progress = (self.current_value - self.baseline_value) / (
            self.target_value - self.baseline_value
        )
        return max(0.0, min(100.0, progress * 100))

    @property
    def is_achieved(self) -> bool:
        """Check if goal has been achieved."""
        if self.target_value > self.baseline_value:
            return self.current_value >= self.target_value
        return self.current_value <= self.target_value

    @property
    def is_overdue(self) -> bool:
        """Check if goal is past deadline."""
        if not self.deadline:
            return False
        return datetime.now() > self.deadline and not self.is_achieved

    def update_progress(self, new_value: float, context: Dict[str, Any] = None) -> None:
        """Update progress with new measurement."""
        self.current_value = new_value
        self.progress_history.append(
            MetricValue(
                metric_type=self.metric_type,
                value=new_value,
                context=context or {},
            )
        )

        if self.is_achieved and self.status != GoalStatus.ACHIEVED:
            self.status = GoalStatus.ACHIEVED
            self.achieved_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal_id": self.goal_id,
            "metric_type": self.metric_type.value,
            "target_value": self.target_value,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "progress_percentage": self.progress_percentage,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "status": self.status.value,
            "priority": self.priority,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "achieved_at": self.achieved_at.isoformat() if self.achieved_at else None,
            "is_achieved": self.is_achieved,
            "is_overdue": self.is_overdue,
        }


@dataclass
class ImprovementAction:
    """
    An action taken to improve the system.

    Records what was done, its impact, and whether
    it should be kept or rolled back.
    """

    action_id: str
    action_type: ActionType
    description: str
    target_component: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    executed_at: Optional[datetime] = None
    impact_measured: bool = False
    impact_value: float = 0.0
    success: bool = False
    rollback_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "target_component": self.target_component,
            "parameters": self.parameters,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "impact_measured": self.impact_measured,
            "impact_value": self.impact_value,
            "success": self.success,
        }


@dataclass
class CycleResults:
    """
    Results from an improvement cycle.

    Captures what was achieved, what failed,
    and the overall impact of the cycle.
    """

    goals_achieved: int = 0
    goals_failed: int = 0
    actions_executed: int = 0
    actions_successful: int = 0
    overall_improvement: float = 0.0
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goals_achieved": self.goals_achieved,
            "goals_failed": self.goals_failed,
            "actions_executed": self.actions_executed,
            "actions_successful": self.actions_successful,
            "overall_improvement": self.overall_improvement,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class ImprovementCycle:
    """
    An autonomous improvement cycle.

    Represents a complete cycle of observation, analysis,
    action, and evaluation without human intervention.
    """

    cycle_id: str
    started_at: datetime
    goals: List[ImprovementGoal]
    actions_taken: List[ImprovementAction] = field(default_factory=list)
    status: CycleStatus = CycleStatus.PENDING
    completed_at: Optional[datetime] = None
    results: Optional[CycleResults] = None
    iteration_number: int = 1
    parent_cycle_id: Optional[str] = None
    safety_violations: List[str] = field(default_factory=list)
    audit_log: List[Dict[str, Any]] = field(default_factory=list)

    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log an event in the audit trail."""
        self.audit_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "details": details,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at.isoformat(),
            "goals": [g.to_dict() for g in self.goals],
            "actions_taken": [a.to_dict() for a in self.actions_taken],
            "status": self.status.value,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "results": self.results.to_dict() if self.results else None,
            "iteration_number": self.iteration_number,
            "parent_cycle_id": self.parent_cycle_id,
            "safety_violations": self.safety_violations,
            "audit_log_length": len(self.audit_log),
        }


@dataclass
class ProgressReport:
    """Report on progress toward improvement goals."""

    report_id: str
    goal_id: str
    metric_type: MetricType
    baseline_value: float
    current_value: float
    target_value: float
    progress_percentage: float
    trend: str  # "improving", "stable", "declining"
    estimated_completion: Optional[datetime]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "goal_id": self.goal_id,
            "metric_type": self.metric_type.value,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "progress_percentage": self.progress_percentage,
            "trend": self.trend,
            "estimated_completion": (
                self.estimated_completion.isoformat()
                if self.estimated_completion
                else None
            ),
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class CycleEvaluation:
    """Evaluation of an improvement cycle's effectiveness."""

    evaluation_id: str
    cycle_id: str
    success: bool
    effectiveness_score: float  # 0-1
    goals_progress: Dict[str, float]  # goal_id -> progress %
    best_action: Optional[str]  # action_id
    worst_action: Optional[str]  # action_id
    lessons_learned: List[str]
    next_cycle_recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evaluation_id": self.evaluation_id,
            "cycle_id": self.cycle_id,
            "success": self.success,
            "effectiveness_score": self.effectiveness_score,
            "goals_progress": self.goals_progress,
            "best_action": self.best_action,
            "worst_action": self.worst_action,
            "lessons_learned": self.lessons_learned,
            "next_cycle_recommendations": self.next_cycle_recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class SafetyConfig:
    """Configuration for safety mechanisms."""

    max_accuracy_drop: float = 0.05  # 5% max drop before stop
    max_iterations_before_review: int = 10
    min_improvement_per_cycle: float = 0.01  # 1% minimum
    max_actions_per_cycle: int = 5
    require_rollback_data: bool = True
    auto_rollback_on_degradation: bool = True


class SelfImprovementTracker:
    """
    Tracks self-improvement metrics over time.

    Provides goal management, progress tracking,
    and improvement rate measurement.
    """

    def __init__(self):
        """Initialize the tracker."""
        self._goals: Dict[str, ImprovementGoal] = {}
        self._metrics_history: Dict[MetricType, List[MetricValue]] = defaultdict(list)
        self._improvement_rates: Dict[MetricType, List[float]] = defaultdict(list)

    def set_goal(self, goal: ImprovementGoal) -> None:
        """
        Set an improvement goal.

        Args:
            goal: The improvement goal to set
        """
        self._goals[goal.goal_id] = goal
        goal.status = GoalStatus.IN_PROGRESS

    def get_goal(self, goal_id: str) -> Optional[ImprovementGoal]:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    def get_all_goals(self) -> List[ImprovementGoal]:
        """Get all goals."""
        return list(self._goals.values())

    def get_active_goals(self) -> List[ImprovementGoal]:
        """Get goals that are in progress."""
        return [g for g in self._goals.values() if g.status == GoalStatus.IN_PROGRESS]

    def record_metric(
        self, metric_type: MetricType, value: float, context: Dict[str, Any] = None
    ) -> None:
        """
        Record a metric measurement.

        Args:
            metric_type: Type of metric
            value: Measured value
            context: Optional context
        """
        measurement = MetricValue(
            metric_type=metric_type, value=value, context=context or {}
        )
        self._metrics_history[metric_type].append(measurement)

        # Update related goals
        for goal in self._goals.values():
            if (
                goal.metric_type == metric_type
                and goal.status == GoalStatus.IN_PROGRESS
            ):
                goal.update_progress(value, context)

    def track_progress(self, goal_id: str) -> ProgressReport:
        """
        Track progress toward a goal.

        Args:
            goal_id: ID of the goal

        Returns:
            ProgressReport with current status

        Raises:
            ValueError: If goal not found
        """
        goal = self._goals.get(goal_id)
        if not goal:
            raise ValueError(f"Goal {goal_id} not found")

        # Calculate trend
        trend = self._calculate_trend(goal)

        # Estimate completion
        estimated_completion = self._estimate_completion(goal, trend)

        # Generate recommendations
        recommendations = self._generate_recommendations(goal, trend)

        return ProgressReport(
            report_id=f"progress_{uuid.uuid4().hex[:8]}",
            goal_id=goal_id,
            metric_type=goal.metric_type,
            baseline_value=goal.baseline_value,
            current_value=goal.current_value,
            target_value=goal.target_value,
            progress_percentage=goal.progress_percentage,
            trend=trend,
            estimated_completion=estimated_completion,
            recommendations=recommendations,
        )

    def measure_improvement_rate(
        self, metric_type: Optional[MetricType] = None
    ) -> float:
        """
        Measure overall improvement rate.

        Args:
            metric_type: Optional specific metric (all if None)

        Returns:
            Improvement rate as percentage per day
        """
        if metric_type:
            history = self._metrics_history.get(metric_type, [])
            return self._calculate_rate(history)

        # Calculate average across all metrics
        rates = []
        for mt, history in self._metrics_history.items():
            if len(history) >= 2:
                rates.append(self._calculate_rate(history))

        return sum(rates) / len(rates) if rates else 0.0

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics."""
        summary = {}
        for metric_type, history in self._metrics_history.items():
            if history:
                values = [m.value for m in history]
                summary[metric_type.value] = {
                    "current": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "average": sum(values) / len(values),
                    "measurements": len(values),
                }
        return summary

    def _calculate_trend(self, goal: ImprovementGoal) -> str:
        """Calculate progress trend for a goal."""
        if len(goal.progress_history) < 3:
            return "stable"

        recent = goal.progress_history[-3:]
        values = [m.value for m in recent]

        if values[-1] > values[0] * 1.02:  # >2% improvement
            return "improving"
        elif values[-1] < values[0] * 0.98:  # >2% decline
            return "declining"
        return "stable"

    def _calculate_rate(self, history: List[MetricValue]) -> float:
        """Calculate improvement rate from history."""
        if len(history) < 2:
            return 0.0

        first = history[0]
        last = history[-1]

        value_change = last.value - first.value
        time_delta = (last.timestamp - first.timestamp).total_seconds()

        if time_delta == 0 or first.value == 0:
            return 0.0

        # Return percentage change per day
        days = time_delta / 86400
        if days == 0:
            return 0.0

        return (value_change / first.value) * 100 / days

    def _estimate_completion(
        self, goal: ImprovementGoal, trend: str
    ) -> Optional[datetime]:
        """Estimate when goal will be achieved."""
        if goal.is_achieved:
            return goal.achieved_at

        if trend == "declining" or len(goal.progress_history) < 2:
            return None

        # Calculate rate of progress
        if len(goal.progress_history) >= 2:
            first = goal.progress_history[0]
            last = goal.progress_history[-1]

            value_change = last.value - first.value
            time_delta = (last.timestamp - first.timestamp).total_seconds()

            if value_change <= 0 or time_delta <= 0:
                return None

            remaining = goal.target_value - goal.current_value
            rate_per_second = value_change / time_delta

            if rate_per_second > 0:
                seconds_remaining = remaining / rate_per_second
                return datetime.now() + timedelta(seconds=seconds_remaining)

        return None

    def _generate_recommendations(self, goal: ImprovementGoal, trend: str) -> List[str]:
        """Generate recommendations for a goal."""
        recommendations = []

        if goal.is_achieved:
            recommendations.append(
                "Goal achieved! Consider setting a more ambitious target."
            )
            return recommendations

        if goal.is_overdue:
            recommendations.append(
                "Goal is overdue. Consider revising deadline or reassessing feasibility."
            )

        if trend == "declining":
            recommendations.append(
                "Performance is declining. Investigate recent changes and consider rollback."
            )
        elif trend == "stable" and goal.progress_percentage < 50:
            recommendations.append(
                "Progress has stalled. Consider trying different improvement strategies."
            )

        if goal.progress_percentage < 25:
            recommendations.append(
                "Limited progress. Focus more resources on this goal."
            )
        elif goal.progress_percentage > 75:
            recommendations.append(
                "Good progress! Continue current approach to achieve goal."
            )

        return recommendations


class AutonomousImprover:
    """
    Executes autonomous improvement cycles.

    Manages the complete cycle of observation, analysis,
    action, and evaluation without human intervention.
    """

    def __init__(
        self,
        tracker: SelfImprovementTracker,
        safety_config: Optional[SafetyConfig] = None,
    ):
        """
        Initialize the autonomous improver.

        Args:
            tracker: SelfImprovementTracker for metric tracking
            safety_config: Optional safety configuration
        """
        self.tracker = tracker
        self.safety = safety_config or SafetyConfig()

        self._cycles: Dict[str, ImprovementCycle] = {}
        self._action_handlers: Dict[ActionType, Callable] = {}
        self._rollback_handlers: Dict[ActionType, Callable] = {}
        self._current_cycle: Optional[str] = None

    def register_action_handler(
        self,
        action_type: ActionType,
        handler: Callable[[ImprovementAction], bool],
        rollback_handler: Optional[Callable[[ImprovementAction], bool]] = None,
    ) -> None:
        """
        Register a handler for an action type.

        Args:
            action_type: Type of action to handle
            handler: Function that executes the action
            rollback_handler: Optional function to rollback the action
        """
        self._action_handlers[action_type] = handler
        if rollback_handler:
            self._rollback_handlers[action_type] = rollback_handler

    def start_improvement_cycle(
        self,
        goals: List[ImprovementGoal],
        parent_cycle_id: Optional[str] = None,
    ) -> str:
        """
        Start a new improvement cycle.

        Args:
            goals: Goals to work toward
            parent_cycle_id: Optional parent cycle for chaining

        Returns:
            Cycle ID
        """
        cycle_id = f"cycle_{uuid.uuid4().hex[:8]}"

        # Determine iteration number
        iteration = 1
        if parent_cycle_id and parent_cycle_id in self._cycles:
            iteration = self._cycles[parent_cycle_id].iteration_number + 1

        cycle = ImprovementCycle(
            cycle_id=cycle_id,
            started_at=datetime.now(),
            goals=goals,
            status=CycleStatus.RUNNING,
            iteration_number=iteration,
            parent_cycle_id=parent_cycle_id,
        )

        cycle.log_event("cycle_started", {"goals": [g.goal_id for g in goals]})

        self._cycles[cycle_id] = cycle
        self._current_cycle = cycle_id

        # Register goals with tracker
        for goal in goals:
            self.tracker.set_goal(goal)

        return cycle_id

    def execute_improvements(self, cycle_id: str) -> bool:
        """
        Execute improvements autonomously.

        Args:
            cycle_id: ID of the cycle

        Returns:
            True if cycle completed successfully
        """
        cycle = self._cycles.get(cycle_id)
        if not cycle:
            raise ValueError(f"Cycle {cycle_id} not found")

        if cycle.status != CycleStatus.RUNNING:
            return False

        # Check safety limits
        if cycle.iteration_number > self.safety.max_iterations_before_review:
            cycle.safety_violations.append("Max iterations reached")
            cycle.status = CycleStatus.PAUSED
            cycle.log_event("safety_pause", {"reason": "max_iterations"})
            return False

        # Collect baseline metrics
        metrics_before = self._collect_metrics()

        # Generate and execute actions
        actions = self._generate_actions(cycle)
        actions_executed = 0
        actions_successful = 0

        for action in actions[: self.safety.max_actions_per_cycle]:
            cycle.log_event("action_started", {"action_id": action.action_id})

            success = self._execute_action(action)
            actions_executed += 1

            if success:
                actions_successful += 1
                cycle.actions_taken.append(action)
            else:
                cycle.log_event(
                    "action_failed",
                    {"action_id": action.action_id, "reason": "execution_failed"},
                )

        # Collect after metrics
        metrics_after = self._collect_metrics()

        # Check for degradation
        degradation = self._check_degradation(metrics_before, metrics_after)
        if degradation and self.safety.auto_rollback_on_degradation:
            cycle.safety_violations.append("Performance degradation detected")
            self._rollback_cycle(cycle)
            cycle.status = CycleStatus.ROLLED_BACK
            cycle.log_event("cycle_rolled_back", {"reason": "degradation"})
            return False

        # Calculate results
        cycle.results = self._calculate_results(
            cycle, metrics_before, metrics_after, actions_executed, actions_successful
        )

        cycle.completed_at = datetime.now()
        cycle.status = CycleStatus.COMPLETED
        cycle.log_event("cycle_completed", {"results": cycle.results.to_dict()})

        return True

    def evaluate_cycle(self, cycle_id: str) -> CycleEvaluation:
        """
        Evaluate cycle effectiveness.

        Args:
            cycle_id: ID of the cycle

        Returns:
            CycleEvaluation with analysis
        """
        cycle = self._cycles.get(cycle_id)
        if not cycle:
            raise ValueError(f"Cycle {cycle_id} not found")

        # Calculate goals progress
        goals_progress = {}
        for goal in cycle.goals:
            goals_progress[goal.goal_id] = goal.progress_percentage

        # Determine best and worst actions
        best_action = None
        worst_action = None
        best_impact = float("-inf")
        worst_impact = float("inf")

        for action in cycle.actions_taken:
            if action.impact_measured:
                if action.impact_value > best_impact:
                    best_impact = action.impact_value
                    best_action = action.action_id
                if action.impact_value < worst_impact:
                    worst_impact = action.impact_value
                    worst_action = action.action_id

        # Calculate effectiveness score
        if cycle.results:
            effectiveness = min(
                1.0,
                (cycle.results.goals_achieved / max(1, len(cycle.goals))) * 0.5
                + (
                    cycle.results.actions_successful
                    / max(1, cycle.results.actions_executed)
                )
                * 0.3
                + max(0, cycle.results.overall_improvement / 10) * 0.2,
            )
        else:
            effectiveness = 0.0

        # Generate lessons learned
        lessons = self._extract_lessons(cycle)

        # Generate recommendations for next cycle
        recommendations = self._generate_next_cycle_recommendations(
            cycle, effectiveness
        )

        return CycleEvaluation(
            evaluation_id=f"eval_{uuid.uuid4().hex[:8]}",
            cycle_id=cycle_id,
            success=cycle.status == CycleStatus.COMPLETED and effectiveness > 0.5,
            effectiveness_score=effectiveness,
            goals_progress=goals_progress,
            best_action=best_action,
            worst_action=worst_action,
            lessons_learned=lessons,
            next_cycle_recommendations=recommendations,
        )

    def get_cycle(self, cycle_id: str) -> Optional[ImprovementCycle]:
        """Get a cycle by ID."""
        return self._cycles.get(cycle_id)

    def get_all_cycles(self) -> List[ImprovementCycle]:
        """Get all cycles."""
        return list(self._cycles.values())

    def pause_cycle(self, cycle_id: str) -> bool:
        """Pause a running cycle."""
        cycle = self._cycles.get(cycle_id)
        if cycle and cycle.status == CycleStatus.RUNNING:
            cycle.status = CycleStatus.PAUSED
            cycle.log_event("cycle_paused", {})
            return True
        return False

    def resume_cycle(self, cycle_id: str) -> bool:
        """Resume a paused cycle."""
        cycle = self._cycles.get(cycle_id)
        if cycle and cycle.status == CycleStatus.PAUSED:
            cycle.status = CycleStatus.RUNNING
            cycle.log_event("cycle_resumed", {})
            return True
        return False

    def _generate_actions(self, cycle: ImprovementCycle) -> List[ImprovementAction]:
        """Generate improvement actions for a cycle."""
        actions = []

        for goal in cycle.goals:
            if goal.status != GoalStatus.IN_PROGRESS:
                continue

            # Generate action based on goal type
            action = self._create_action_for_goal(goal)
            if action:
                actions.append(action)

        return actions

    def _create_action_for_goal(
        self, goal: ImprovementGoal
    ) -> Optional[ImprovementAction]:
        """Create an action to improve a specific goal."""
        action_mapping = {
            MetricType.ACCURACY: ActionType.RULE_MODIFICATION,
            MetricType.RULE_ACCEPTANCE_RATE: ActionType.VALIDATION_ENHANCEMENT,
            MetricType.PROMPT_EFFECTIVENESS: ActionType.PROMPT_REFINEMENT,
            MetricType.ERROR_DIAGNOSIS_ACCURACY: ActionType.STRATEGY_ADJUSTMENT,
            MetricType.STRATEGY_SELECTION_ACCURACY: ActionType.STRATEGY_ADJUSTMENT,
            MetricType.LATENCY: ActionType.THRESHOLD_TUNING,
            MetricType.CONFIDENCE: ActionType.PROMPT_REFINEMENT,
            MetricType.COVERAGE: ActionType.RULE_MODIFICATION,
        }

        action_type = action_mapping.get(
            goal.metric_type, ActionType.STRATEGY_ADJUSTMENT
        )

        return ImprovementAction(
            action_id=f"action_{uuid.uuid4().hex[:8]}",
            action_type=action_type,
            description=f"Improve {goal.metric_type.value} from {goal.current_value:.2f} to {goal.target_value:.2f}",
            target_component=goal.metric_type.value,
            parameters={"goal_id": goal.goal_id, "target": goal.target_value},
        )

    def _execute_action(self, action: ImprovementAction) -> bool:
        """Execute a single action."""
        handler = self._action_handlers.get(action.action_type)

        if not handler:
            # No handler registered, simulate success
            action.executed_at = datetime.now()
            action.success = True
            return True

        try:
            action.executed_at = datetime.now()
            action.success = handler(action)
            return action.success
        except Exception:
            action.success = False
            return False

    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current metric values."""
        metrics = {}
        for metric_type, history in self.tracker._metrics_history.items():
            if history:
                metrics[metric_type.value] = history[-1].value
        return metrics

    def _check_degradation(
        self, before: Dict[str, float], after: Dict[str, float]
    ) -> bool:
        """Check if metrics degraded beyond threshold."""
        for metric, before_value in before.items():
            after_value = after.get(metric, before_value)
            if before_value > 0:
                drop = (before_value - after_value) / before_value
                if drop > self.safety.max_accuracy_drop:
                    return True
        return False

    def _rollback_cycle(self, cycle: ImprovementCycle) -> None:
        """Rollback all actions in a cycle."""
        for action in reversed(cycle.actions_taken):
            rollback_handler = self._rollback_handlers.get(action.action_type)
            if rollback_handler and action.rollback_data:
                try:
                    rollback_handler(action)
                    cycle.log_event(
                        "action_rolled_back", {"action_id": action.action_id}
                    )
                except Exception:
                    cycle.log_event("rollback_failed", {"action_id": action.action_id})

    def _calculate_results(
        self,
        cycle: ImprovementCycle,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        actions_executed: int,
        actions_successful: int,
    ) -> CycleResults:
        """Calculate cycle results."""
        goals_achieved = sum(1 for g in cycle.goals if g.is_achieved)
        goals_failed = sum(1 for g in cycle.goals if g.status == GoalStatus.FAILED)

        # Calculate overall improvement
        improvements = []
        for metric, before_value in metrics_before.items():
            after_value = metrics_after.get(metric, before_value)
            if before_value > 0:
                improvement = ((after_value - before_value) / before_value) * 100
                improvements.append(improvement)

        overall_improvement = (
            sum(improvements) / len(improvements) if improvements else 0.0
        )

        duration = (datetime.now() - cycle.started_at).total_seconds()

        return CycleResults(
            goals_achieved=goals_achieved,
            goals_failed=goals_failed,
            actions_executed=actions_executed,
            actions_successful=actions_successful,
            overall_improvement=overall_improvement,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            duration_seconds=duration,
        )

    def _extract_lessons(self, cycle: ImprovementCycle) -> List[str]:
        """Extract lessons learned from a cycle."""
        lessons = []

        if cycle.results:
            if cycle.results.overall_improvement > 5:
                lessons.append(
                    "Significant improvement achieved through current approach"
                )
            elif cycle.results.overall_improvement < 0:
                lessons.append("Actions led to degradation - review strategy")

            if cycle.results.actions_successful < cycle.results.actions_executed / 2:
                lessons.append("Low action success rate - improve action generation")

        if cycle.safety_violations:
            lessons.append(
                f"Safety violations occurred: {', '.join(cycle.safety_violations)}"
            )

        return lessons

    def _generate_next_cycle_recommendations(
        self, cycle: ImprovementCycle, effectiveness: float
    ) -> List[str]:
        """Generate recommendations for the next cycle."""
        recommendations = []

        if effectiveness < 0.3:
            recommendations.append("Consider different improvement strategies")
            recommendations.append("Review goal feasibility")
        elif effectiveness < 0.6:
            recommendations.append("Refine action generation based on what worked")
        else:
            recommendations.append("Continue current approach with minor adjustments")

        # Check for unachieved goals
        unachieved = [g for g in cycle.goals if not g.is_achieved]
        if unachieved:
            recommendations.append(
                f"Focus on {len(unachieved)} unachieved goals in next cycle"
            )

        return recommendations


# Factory functions


def create_improvement_goal(
    metric_type: MetricType,
    target_value: float,
    baseline_value: float,
    deadline_days: Optional[int] = None,
    description: str = "",
    priority: int = 1,
) -> ImprovementGoal:
    """
    Create an improvement goal.

    Args:
        metric_type: Type of metric to improve
        target_value: Target value to achieve
        baseline_value: Starting baseline value
        deadline_days: Optional days until deadline
        description: Description of the goal
        priority: Priority (1 = highest)

    Returns:
        Configured ImprovementGoal
    """
    deadline = None
    if deadline_days:
        deadline = datetime.now() + timedelta(days=deadline_days)

    return ImprovementGoal(
        goal_id=f"goal_{uuid.uuid4().hex[:8]}",
        metric_type=metric_type,
        target_value=target_value,
        baseline_value=baseline_value,
        current_value=baseline_value,
        deadline=deadline,
        description=description,
        priority=priority,
    )


def create_tracker() -> SelfImprovementTracker:
    """Create a SelfImprovementTracker instance."""
    return SelfImprovementTracker()


def create_improver(
    tracker: Optional[SelfImprovementTracker] = None,
    safety_config: Optional[SafetyConfig] = None,
) -> AutonomousImprover:
    """
    Create an AutonomousImprover instance.

    Args:
        tracker: Optional tracker (creates new if None)
        safety_config: Optional safety configuration

    Returns:
        Configured AutonomousImprover
    """
    if tracker is None:
        tracker = create_tracker()
    return AutonomousImprover(tracker, safety_config)


def create_default_goals(current_metrics: Dict[str, float]) -> List[ImprovementGoal]:
    """
    Create default improvement goals based on current metrics.

    Args:
        current_metrics: Current metric values

    Returns:
        List of improvement goals
    """
    goals = []

    # Accuracy goal: improve by 2%
    if "accuracy" in current_metrics:
        goals.append(
            create_improvement_goal(
                MetricType.ACCURACY,
                target_value=min(0.99, current_metrics["accuracy"] + 0.02),
                baseline_value=current_metrics["accuracy"],
                deadline_days=30,
                description="Improve prediction accuracy by 2%",
                priority=1,
            )
        )

    # Rule acceptance rate goal: target 70%
    if "rule_acceptance_rate" in current_metrics:
        goals.append(
            create_improvement_goal(
                MetricType.RULE_ACCEPTANCE_RATE,
                target_value=max(0.70, current_metrics["rule_acceptance_rate"] + 0.05),
                baseline_value=current_metrics["rule_acceptance_rate"],
                deadline_days=14,
                description="Improve rule acceptance rate to 70%+",
                priority=2,
            )
        )

    # Prompt effectiveness goal: improve by 15%
    if "prompt_effectiveness" in current_metrics:
        goals.append(
            create_improvement_goal(
                MetricType.PROMPT_EFFECTIVENESS,
                target_value=min(0.95, current_metrics["prompt_effectiveness"] + 0.15),
                baseline_value=current_metrics["prompt_effectiveness"],
                deadline_days=14,
                description="Improve prompt effectiveness by 15%",
                priority=2,
            )
        )

    return goals
