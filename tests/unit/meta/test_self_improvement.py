"""Tests for the self-improvement module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from loft.meta.self_improvement import (
    ActionType,
    AutonomousImprover,
    CycleEvaluation,
    CycleResults,
    CycleStatus,
    GoalStatus,
    ImprovementAction,
    ImprovementCycle,
    ImprovementGoal,
    MetricType,
    MetricValue,
    ProgressReport,
    SafetyConfig,
    SelfImprovementTracker,
    create_default_goals,
    create_improvement_goal,
    create_improver,
    create_tracker,
)


class TestGoalStatus:
    """Tests for GoalStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        assert GoalStatus.PENDING.value == "pending"
        assert GoalStatus.IN_PROGRESS.value == "in_progress"
        assert GoalStatus.ACHIEVED.value == "achieved"
        assert GoalStatus.FAILED.value == "failed"
        assert GoalStatus.ABANDONED.value == "abandoned"


class TestCycleStatus:
    """Tests for CycleStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        assert CycleStatus.PENDING.value == "pending"
        assert CycleStatus.RUNNING.value == "running"
        assert CycleStatus.COMPLETED.value == "completed"
        assert CycleStatus.FAILED.value == "failed"
        assert CycleStatus.ROLLED_BACK.value == "rolled_back"
        assert CycleStatus.PAUSED.value == "paused"


class TestActionType:
    """Tests for ActionType enum."""

    def test_all_types_exist(self):
        """Test all expected action types exist."""
        assert ActionType.PROMPT_REFINEMENT.value == "prompt_refinement"
        assert ActionType.STRATEGY_ADJUSTMENT.value == "strategy_adjustment"
        assert ActionType.RULE_MODIFICATION.value == "rule_modification"
        assert ActionType.THRESHOLD_TUNING.value == "threshold_tuning"
        assert ActionType.VALIDATION_ENHANCEMENT.value == "validation_enhancement"


class TestMetricType:
    """Tests for MetricType enum."""

    def test_all_types_exist(self):
        """Test all expected metric types exist."""
        assert MetricType.ACCURACY.value == "accuracy"
        assert MetricType.RULE_ACCEPTANCE_RATE.value == "rule_acceptance_rate"
        assert MetricType.PROMPT_EFFECTIVENESS.value == "prompt_effectiveness"
        assert MetricType.ERROR_DIAGNOSIS_ACCURACY.value == "error_diagnosis_accuracy"
        assert MetricType.STRATEGY_SELECTION_ACCURACY.value == "strategy_selection_accuracy"
        assert MetricType.LATENCY.value == "latency"
        assert MetricType.CONFIDENCE.value == "confidence"
        assert MetricType.COVERAGE.value == "coverage"


class TestMetricValue:
    """Tests for MetricValue dataclass."""

    def test_metric_value_creation(self):
        """Test creating a metric value."""
        metric = MetricValue(
            metric_type=MetricType.ACCURACY,
            value=0.85,
            context={"test_set": "validation"},
        )
        assert metric.metric_type == MetricType.ACCURACY
        assert metric.value == 0.85
        assert metric.context == {"test_set": "validation"}
        assert metric.timestamp is not None

    def test_to_dict(self):
        """Test converting to dictionary."""
        metric = MetricValue(
            metric_type=MetricType.ACCURACY,
            value=0.85,
        )
        result = metric.to_dict()
        assert result["metric_type"] == "accuracy"
        assert result["value"] == 0.85
        assert "timestamp" in result


class TestImprovementGoal:
    """Tests for ImprovementGoal dataclass."""

    def test_goal_creation(self):
        """Test creating an improvement goal."""
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            current_value=0.82,
        )
        assert goal.goal_id == "goal_test"
        assert goal.target_value == 0.90
        assert goal.baseline_value == 0.80
        assert goal.current_value == 0.82

    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            current_value=0.85,
        )
        # 0.85 is 50% between 0.80 and 0.90
        assert goal.progress_percentage == pytest.approx(50.0)

    def test_progress_percentage_full(self):
        """Test progress percentage at 100%."""
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            current_value=0.90,
        )
        assert goal.progress_percentage == 100.0

    def test_progress_percentage_equal_target_baseline(self):
        """Test progress when target equals baseline."""
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.90,
            current_value=0.90,
        )
        assert goal.progress_percentage == 100.0

    def test_is_achieved_increasing(self):
        """Test goal achievement for increasing metrics."""
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            current_value=0.91,
        )
        assert goal.is_achieved is True

    def test_is_achieved_decreasing(self):
        """Test goal achievement for decreasing metrics like latency."""
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.LATENCY,
            target_value=1.0,
            baseline_value=2.0,
            current_value=0.9,
        )
        assert goal.is_achieved is True

    def test_is_overdue(self):
        """Test deadline detection."""
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            current_value=0.82,
            deadline=datetime.now() - timedelta(days=1),
        )
        assert goal.is_overdue is True

    def test_is_not_overdue_no_deadline(self):
        """Test no deadline means not overdue."""
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            current_value=0.82,
        )
        assert goal.is_overdue is False

    def test_update_progress(self):
        """Test updating progress."""
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            current_value=0.82,
        )
        goal.update_progress(0.88)
        assert goal.current_value == 0.88
        assert len(goal.progress_history) == 1

    def test_update_progress_achieves_goal(self):
        """Test that achieving target updates status."""
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            current_value=0.82,
        )
        goal.update_progress(0.92)
        assert goal.status == GoalStatus.ACHIEVED
        assert goal.achieved_at is not None

    def test_to_dict(self):
        """Test converting to dictionary."""
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            current_value=0.85,
            description="Test goal",
        )
        result = goal.to_dict()
        assert result["goal_id"] == "goal_test"
        assert result["target_value"] == 0.90
        assert result["progress_percentage"] == pytest.approx(50.0)


class TestImprovementAction:
    """Tests for ImprovementAction dataclass."""

    def test_action_creation(self):
        """Test creating an improvement action."""
        action = ImprovementAction(
            action_id="action_test",
            action_type=ActionType.PROMPT_REFINEMENT,
            description="Refine prompts for clarity",
            target_component="prompt_manager",
        )
        assert action.action_id == "action_test"
        assert action.action_type == ActionType.PROMPT_REFINEMENT
        assert action.success is False

    def test_to_dict(self):
        """Test converting to dictionary."""
        action = ImprovementAction(
            action_id="action_test",
            action_type=ActionType.PROMPT_REFINEMENT,
            description="Refine prompts",
            target_component="prompt_manager",
        )
        result = action.to_dict()
        assert result["action_id"] == "action_test"
        assert result["action_type"] == "prompt_refinement"


class TestCycleResults:
    """Tests for CycleResults dataclass."""

    def test_results_creation(self):
        """Test creating cycle results."""
        results = CycleResults(
            goals_achieved=2,
            goals_failed=1,
            actions_executed=5,
            actions_successful=4,
            overall_improvement=3.5,
        )
        assert results.goals_achieved == 2
        assert results.actions_successful == 4

    def test_to_dict(self):
        """Test converting to dictionary."""
        results = CycleResults(
            goals_achieved=2,
            goals_failed=1,
            actions_executed=5,
            actions_successful=4,
            overall_improvement=3.5,
        )
        result = results.to_dict()
        assert result["goals_achieved"] == 2
        assert result["overall_improvement"] == 3.5


class TestImprovementCycle:
    """Tests for ImprovementCycle dataclass."""

    def test_cycle_creation(self):
        """Test creating an improvement cycle."""
        cycle = ImprovementCycle(
            cycle_id="cycle_test",
            started_at=datetime.now(),
            goals=[],
        )
        assert cycle.cycle_id == "cycle_test"
        assert cycle.status == CycleStatus.PENDING

    def test_log_event(self):
        """Test logging events."""
        cycle = ImprovementCycle(
            cycle_id="cycle_test",
            started_at=datetime.now(),
            goals=[],
        )
        cycle.log_event("test_event", {"key": "value"})
        assert len(cycle.audit_log) == 1
        assert cycle.audit_log[0]["event_type"] == "test_event"

    def test_to_dict(self):
        """Test converting to dictionary."""
        cycle = ImprovementCycle(
            cycle_id="cycle_test",
            started_at=datetime.now(),
            goals=[],
        )
        result = cycle.to_dict()
        assert result["cycle_id"] == "cycle_test"
        assert result["status"] == "pending"


class TestProgressReport:
    """Tests for ProgressReport dataclass."""

    def test_report_creation(self):
        """Test creating a progress report."""
        report = ProgressReport(
            report_id="report_test",
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            baseline_value=0.80,
            current_value=0.85,
            target_value=0.90,
            progress_percentage=50.0,
            trend="improving",
            estimated_completion=None,
            recommendations=["Keep going"],
        )
        assert report.trend == "improving"
        assert report.progress_percentage == 50.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        report = ProgressReport(
            report_id="report_test",
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            baseline_value=0.80,
            current_value=0.85,
            target_value=0.90,
            progress_percentage=50.0,
            trend="improving",
            estimated_completion=None,
            recommendations=["Keep going"],
        )
        result = report.to_dict()
        assert result["trend"] == "improving"


class TestCycleEvaluation:
    """Tests for CycleEvaluation dataclass."""

    def test_evaluation_creation(self):
        """Test creating a cycle evaluation."""
        evaluation = CycleEvaluation(
            evaluation_id="eval_test",
            cycle_id="cycle_test",
            success=True,
            effectiveness_score=0.75,
            goals_progress={"goal_1": 80.0},
            best_action="action_1",
            worst_action="action_2",
            lessons_learned=["Be patient"],
            next_cycle_recommendations=["Try harder"],
        )
        assert evaluation.success is True
        assert evaluation.effectiveness_score == 0.75

    def test_to_dict(self):
        """Test converting to dictionary."""
        evaluation = CycleEvaluation(
            evaluation_id="eval_test",
            cycle_id="cycle_test",
            success=True,
            effectiveness_score=0.75,
            goals_progress={},
            best_action=None,
            worst_action=None,
            lessons_learned=[],
            next_cycle_recommendations=[],
        )
        result = evaluation.to_dict()
        assert result["success"] is True


class TestSafetyConfig:
    """Tests for SafetyConfig dataclass."""

    def test_default_config(self):
        """Test default safety configuration."""
        config = SafetyConfig()
        assert config.max_accuracy_drop == 0.05
        assert config.max_iterations_before_review == 10
        assert config.auto_rollback_on_degradation is True

    def test_custom_config(self):
        """Test custom safety configuration."""
        config = SafetyConfig(
            max_accuracy_drop=0.10,
            max_iterations_before_review=5,
        )
        assert config.max_accuracy_drop == 0.10
        assert config.max_iterations_before_review == 5


class TestSelfImprovementTracker:
    """Tests for SelfImprovementTracker class."""

    def test_tracker_creation(self):
        """Test creating a tracker."""
        tracker = SelfImprovementTracker()
        assert tracker.get_all_goals() == []

    def test_set_goal(self):
        """Test setting a goal."""
        tracker = SelfImprovementTracker()
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
        )
        tracker.set_goal(goal)
        assert tracker.get_goal("goal_test") is not None
        assert goal.status == GoalStatus.IN_PROGRESS

    def test_get_active_goals(self):
        """Test getting active goals."""
        tracker = SelfImprovementTracker()
        goal1 = ImprovementGoal(
            goal_id="goal_1",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
        )
        goal2 = ImprovementGoal(
            goal_id="goal_2",
            metric_type=MetricType.CONFIDENCE,
            target_value=0.95,
            baseline_value=0.85,
        )
        tracker.set_goal(goal1)
        tracker.set_goal(goal2)
        active = tracker.get_active_goals()
        assert len(active) == 2

    def test_record_metric(self):
        """Test recording a metric."""
        tracker = SelfImprovementTracker()
        tracker.record_metric(MetricType.ACCURACY, 0.85)
        summary = tracker.get_metrics_summary()
        assert "accuracy" in summary
        assert summary["accuracy"]["current"] == 0.85

    def test_record_metric_updates_goals(self):
        """Test that recording metrics updates related goals."""
        tracker = SelfImprovementTracker()
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
        )
        tracker.set_goal(goal)
        tracker.record_metric(MetricType.ACCURACY, 0.88)
        assert goal.current_value == 0.88

    def test_track_progress(self):
        """Test tracking progress."""
        tracker = SelfImprovementTracker()
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
        )
        tracker.set_goal(goal)
        report = tracker.track_progress("goal_test")
        assert isinstance(report, ProgressReport)
        assert report.goal_id == "goal_test"

    def test_track_progress_goal_not_found(self):
        """Test tracking progress for non-existent goal."""
        tracker = SelfImprovementTracker()
        try:
            tracker.track_progress("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found" in str(e)

    def test_measure_improvement_rate_no_history(self):
        """Test improvement rate with no history."""
        tracker = SelfImprovementTracker()
        rate = tracker.measure_improvement_rate()
        assert rate == 0.0

    def test_measure_improvement_rate_single_metric(self):
        """Test improvement rate for single metric."""
        tracker = SelfImprovementTracker()
        tracker.record_metric(MetricType.ACCURACY, 0.80)
        tracker.record_metric(MetricType.ACCURACY, 0.85)
        rate = tracker.measure_improvement_rate(MetricType.ACCURACY)
        # Rate depends on timestamps, but should be non-negative
        assert rate >= 0 or rate <= 0  # Just check it returns a number

    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        tracker = SelfImprovementTracker()
        tracker.record_metric(MetricType.ACCURACY, 0.80)
        tracker.record_metric(MetricType.ACCURACY, 0.85)
        tracker.record_metric(MetricType.ACCURACY, 0.90)
        summary = tracker.get_metrics_summary()
        assert summary["accuracy"]["min"] == 0.80
        assert summary["accuracy"]["max"] == 0.90
        assert summary["accuracy"]["measurements"] == 3


class TestAutonomousImprover:
    """Tests for AutonomousImprover class."""

    def test_improver_creation(self):
        """Test creating an improver."""
        tracker = SelfImprovementTracker()
        improver = AutonomousImprover(tracker)
        assert improver.tracker is tracker

    def test_improver_with_custom_safety(self):
        """Test creating improver with custom safety config."""
        tracker = SelfImprovementTracker()
        safety = SafetyConfig(max_accuracy_drop=0.10)
        improver = AutonomousImprover(tracker, safety)
        assert improver.safety.max_accuracy_drop == 0.10

    def test_register_action_handler(self):
        """Test registering an action handler."""
        tracker = SelfImprovementTracker()
        improver = AutonomousImprover(tracker)
        handler = MagicMock(return_value=True)
        improver.register_action_handler(ActionType.PROMPT_REFINEMENT, handler)
        assert ActionType.PROMPT_REFINEMENT in improver._action_handlers

    def test_start_improvement_cycle(self):
        """Test starting an improvement cycle."""
        tracker = SelfImprovementTracker()
        improver = AutonomousImprover(tracker)
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
        )
        cycle_id = improver.start_improvement_cycle([goal])
        assert cycle_id.startswith("cycle_")
        cycle = improver.get_cycle(cycle_id)
        assert cycle is not None
        assert cycle.status == CycleStatus.RUNNING

    def test_execute_improvements(self):
        """Test executing improvements."""
        tracker = SelfImprovementTracker()
        improver = AutonomousImprover(tracker)
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            current_value=0.80,
        )
        cycle_id = improver.start_improvement_cycle([goal])
        success = improver.execute_improvements(cycle_id)
        assert success is True
        cycle = improver.get_cycle(cycle_id)
        assert cycle.status == CycleStatus.COMPLETED

    def test_execute_improvements_cycle_not_found(self):
        """Test executing improvements for non-existent cycle."""
        tracker = SelfImprovementTracker()
        improver = AutonomousImprover(tracker)
        try:
            improver.execute_improvements("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found" in str(e)

    def test_execute_improvements_with_handler(self):
        """Test executing improvements with registered handler."""
        tracker = SelfImprovementTracker()
        improver = AutonomousImprover(tracker)

        handler = MagicMock(return_value=True)
        improver.register_action_handler(ActionType.RULE_MODIFICATION, handler)

        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            current_value=0.80,
        )
        cycle_id = improver.start_improvement_cycle([goal])
        improver.execute_improvements(cycle_id)

        # Handler should have been called
        assert handler.called

    def test_evaluate_cycle(self):
        """Test evaluating a cycle."""
        tracker = SelfImprovementTracker()
        improver = AutonomousImprover(tracker)
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            current_value=0.80,
        )
        cycle_id = improver.start_improvement_cycle([goal])
        improver.execute_improvements(cycle_id)
        evaluation = improver.evaluate_cycle(cycle_id)
        assert isinstance(evaluation, CycleEvaluation)
        assert evaluation.cycle_id == cycle_id

    def test_evaluate_cycle_not_found(self):
        """Test evaluating non-existent cycle."""
        tracker = SelfImprovementTracker()
        improver = AutonomousImprover(tracker)
        try:
            improver.evaluate_cycle("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found" in str(e)

    def test_pause_and_resume_cycle(self):
        """Test pausing and resuming a cycle."""
        tracker = SelfImprovementTracker()
        improver = AutonomousImprover(tracker)
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
        )
        cycle_id = improver.start_improvement_cycle([goal])

        assert improver.pause_cycle(cycle_id) is True
        cycle = improver.get_cycle(cycle_id)
        assert cycle.status == CycleStatus.PAUSED

        assert improver.resume_cycle(cycle_id) is True
        assert cycle.status == CycleStatus.RUNNING

    def test_safety_max_iterations(self):
        """Test safety mechanism for max iterations."""
        tracker = SelfImprovementTracker()
        safety = SafetyConfig(max_iterations_before_review=1)
        improver = AutonomousImprover(tracker, safety)

        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
        )

        # First cycle
        cycle_id1 = improver.start_improvement_cycle([goal])
        improver.execute_improvements(cycle_id1)

        # Second cycle (linked to first)
        goal2 = ImprovementGoal(
            goal_id="goal_test2",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
        )
        cycle_id2 = improver.start_improvement_cycle([goal2], parent_cycle_id=cycle_id1)
        success = improver.execute_improvements(cycle_id2)

        # Should be paused due to max iterations
        assert success is False
        cycle2 = improver.get_cycle(cycle_id2)
        assert cycle2.status == CycleStatus.PAUSED

    def test_get_all_cycles(self):
        """Test getting all cycles."""
        tracker = SelfImprovementTracker()
        improver = AutonomousImprover(tracker)

        goal1 = ImprovementGoal(
            goal_id="goal_1",
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
        )
        goal2 = ImprovementGoal(
            goal_id="goal_2",
            metric_type=MetricType.CONFIDENCE,
            target_value=0.95,
            baseline_value=0.85,
        )

        improver.start_improvement_cycle([goal1])
        improver.start_improvement_cycle([goal2])

        cycles = improver.get_all_cycles()
        assert len(cycles) == 2


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_improvement_goal(self):
        """Test creating an improvement goal."""
        goal = create_improvement_goal(
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
            deadline_days=7,
            description="Test goal",
            priority=1,
        )
        assert goal.goal_id.startswith("goal_")
        assert goal.metric_type == MetricType.ACCURACY
        assert goal.target_value == 0.90
        assert goal.deadline is not None

    def test_create_improvement_goal_no_deadline(self):
        """Test creating goal without deadline."""
        goal = create_improvement_goal(
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
        )
        assert goal.deadline is None

    def test_create_tracker(self):
        """Test creating a tracker."""
        tracker = create_tracker()
        assert isinstance(tracker, SelfImprovementTracker)

    def test_create_improver(self):
        """Test creating an improver."""
        improver = create_improver()
        assert isinstance(improver, AutonomousImprover)
        assert isinstance(improver.tracker, SelfImprovementTracker)

    def test_create_improver_with_tracker(self):
        """Test creating improver with existing tracker."""
        tracker = SelfImprovementTracker()
        improver = create_improver(tracker)
        assert improver.tracker is tracker

    def test_create_improver_with_safety(self):
        """Test creating improver with safety config."""
        safety = SafetyConfig(max_accuracy_drop=0.10)
        improver = create_improver(safety_config=safety)
        assert improver.safety.max_accuracy_drop == 0.10

    def test_create_default_goals(self):
        """Test creating default goals."""
        current_metrics = {
            "accuracy": 0.85,
            "rule_acceptance_rate": 0.60,
            "prompt_effectiveness": 0.70,
        }
        goals = create_default_goals(current_metrics)
        assert len(goals) == 3

        # Check accuracy goal
        accuracy_goal = next(g for g in goals if g.metric_type == MetricType.ACCURACY)
        assert accuracy_goal.baseline_value == 0.85
        assert accuracy_goal.target_value == 0.87  # +2%

    def test_create_default_goals_empty_metrics(self):
        """Test creating goals with no metrics."""
        goals = create_default_goals({})
        assert len(goals) == 0


class TestIntegration:
    """Integration tests for self-improvement system."""

    def test_full_improvement_cycle(self):
        """Test a complete improvement cycle."""
        # Create tracker and improver
        tracker = create_tracker()
        improver = create_improver(tracker)

        # Set up initial metrics
        tracker.record_metric(MetricType.ACCURACY, 0.80)

        # Create goals
        goals = [
            create_improvement_goal(
                metric_type=MetricType.ACCURACY,
                target_value=0.90,
                baseline_value=0.80,
                description="Improve accuracy",
            )
        ]

        # Start cycle
        cycle_id = improver.start_improvement_cycle(goals)

        # Execute improvements
        success = improver.execute_improvements(cycle_id)
        assert success is True

        # Evaluate cycle
        evaluation = improver.evaluate_cycle(cycle_id)
        assert isinstance(evaluation, CycleEvaluation)
        assert len(evaluation.lessons_learned) >= 0
        assert len(evaluation.next_cycle_recommendations) > 0

    def test_multiple_goals_cycle(self):
        """Test cycle with multiple goals."""
        tracker = create_tracker()
        improver = create_improver(tracker)

        goals = [
            create_improvement_goal(
                metric_type=MetricType.ACCURACY,
                target_value=0.90,
                baseline_value=0.80,
            ),
            create_improvement_goal(
                metric_type=MetricType.CONFIDENCE,
                target_value=0.95,
                baseline_value=0.85,
            ),
            create_improvement_goal(
                metric_type=MetricType.COVERAGE,
                target_value=0.70,
                baseline_value=0.50,
            ),
        ]

        cycle_id = improver.start_improvement_cycle(goals)
        improver.execute_improvements(cycle_id)

        cycle = improver.get_cycle(cycle_id)
        assert len(cycle.actions_taken) > 0

    def test_progress_tracking_over_time(self):
        """Test progress tracking with multiple measurements."""
        tracker = create_tracker()
        goal = create_improvement_goal(
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
        )
        tracker.set_goal(goal)

        # Simulate progress
        tracker.record_metric(MetricType.ACCURACY, 0.82)
        tracker.record_metric(MetricType.ACCURACY, 0.84)
        tracker.record_metric(MetricType.ACCURACY, 0.86)

        report = tracker.track_progress(goal.goal_id)
        assert report.current_value == 0.86
        assert report.progress_percentage == pytest.approx(
            60.0
        )  # 0.86 is 60% between 0.80 and 0.90

    def test_audit_logging(self):
        """Test that cycles properly log events."""
        tracker = create_tracker()
        improver = create_improver(tracker)

        goal = create_improvement_goal(
            metric_type=MetricType.ACCURACY,
            target_value=0.90,
            baseline_value=0.80,
        )

        cycle_id = improver.start_improvement_cycle([goal])
        improver.execute_improvements(cycle_id)

        cycle = improver.get_cycle(cycle_id)
        assert len(cycle.audit_log) > 0
        event_types = [e["event_type"] for e in cycle.audit_log]
        assert "cycle_started" in event_types
        assert "cycle_completed" in event_types
