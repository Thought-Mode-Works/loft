"""
Unified Meta-Reasoning Dashboard and Reporting.

Provides a comprehensive view of meta-reasoning system health by aggregating
reports from all Phase 5 components (Observer, Strategy, Prompt, Failure,
Self-Improvement).
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from loft.meta.failure_analyzer import (
    FailureAnalyzer,
    RecommendationEngine,
)
from loft.meta.observer import ReasoningObserver
from loft.meta.prompt_optimizer import PromptOptimizer
from loft.meta.self_improvement import (
    AutonomousImprover,
    CycleStatus,
    GoalStatus,
    SelfImprovementTracker,
)
from loft.meta.strategy import StrategyEvaluator, StrategySelector


class HealthStatus(Enum):
    """Overall health status of the meta-reasoning system."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class TrendDirection(Enum):
    """Direction of a metric trend."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Severity level for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetaReasoningAlert:
    """An alert from the meta-reasoning system."""

    alert_id: str
    severity: AlertSeverity
    component: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "component": self.component,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaReasoningAlert":
        """Create from dictionary."""
        return cls(
            alert_id=data["alert_id"],
            severity=AlertSeverity(data["severity"]),
            component=data["component"],
            message=data["message"],
            details=data.get("details", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class ObserverSummary:
    """Summary of ReasoningObserver state."""

    total_chains_observed: int
    successful_chains: int
    failed_chains: int
    success_rate: float
    domains_observed: List[str]
    patterns_detected: int
    bottlenecks_identified: int
    average_chain_duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_chains_observed": self.total_chains_observed,
            "successful_chains": self.successful_chains,
            "failed_chains": self.failed_chains,
            "success_rate": self.success_rate,
            "domains_observed": self.domains_observed,
            "patterns_detected": self.patterns_detected,
            "bottlenecks_identified": self.bottlenecks_identified,
            "average_chain_duration_ms": self.average_chain_duration_ms,
        }


@dataclass
class StrategySummary:
    """Summary of Strategy system state."""

    strategies_available: int
    strategies_evaluated: int
    best_strategy: Optional[str]
    best_strategy_accuracy: float
    selections_made: int
    domains_covered: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategies_available": self.strategies_available,
            "strategies_evaluated": self.strategies_evaluated,
            "best_strategy": self.best_strategy,
            "best_strategy_accuracy": self.best_strategy_accuracy,
            "selections_made": self.selections_made,
            "domains_covered": self.domains_covered,
        }


@dataclass
class PromptSummary:
    """Summary of Prompt optimization state."""

    total_prompts: int
    active_prompts: int
    average_effectiveness: float
    best_prompt_category: Optional[str]
    active_ab_tests: int
    improvement_candidates: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_prompts": self.total_prompts,
            "active_prompts": self.active_prompts,
            "average_effectiveness": self.average_effectiveness,
            "best_prompt_category": self.best_prompt_category,
            "active_ab_tests": self.active_ab_tests,
            "improvement_candidates": self.improvement_candidates,
        }


@dataclass
class FailureSummary:
    """Summary of Failure analysis state."""

    total_errors_recorded: int
    patterns_identified: int
    root_causes_found: int
    recommendations_generated: int
    error_rate: float
    most_common_error_category: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_errors_recorded": self.total_errors_recorded,
            "patterns_identified": self.patterns_identified,
            "root_causes_found": self.root_causes_found,
            "recommendations_generated": self.recommendations_generated,
            "error_rate": self.error_rate,
            "most_common_error_category": self.most_common_error_category,
        }


@dataclass
class ImprovementSummary:
    """Summary of Self-improvement state."""

    active_goals: int
    completed_goals: int
    in_progress_goals: int
    total_cycles_run: int
    average_effectiveness: float
    actions_pending: int
    actions_completed: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "active_goals": self.active_goals,
            "completed_goals": self.completed_goals,
            "in_progress_goals": self.in_progress_goals,
            "total_cycles_run": self.total_cycles_run,
            "average_effectiveness": self.average_effectiveness,
            "actions_pending": self.actions_pending,
            "actions_completed": self.actions_completed,
        }


@dataclass
class TrendData:
    """Trend data for a metric over time."""

    metric_name: str
    direction: TrendDirection
    values: List[float]
    timestamps: List[datetime]
    change_percentage: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "direction": self.direction.value,
            "values": self.values,
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "change_percentage": self.change_percentage,
        }


@dataclass
class TrendReport:
    """Trend analysis report over a time period."""

    report_id: str
    start_date: datetime
    end_date: datetime
    accuracy_trend: TrendData
    error_rate_trend: TrendData
    improvement_effectiveness_trend: TrendData
    prompt_effectiveness_trend: TrendData
    overall_health_trend: TrendDirection
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "accuracy_trend": self.accuracy_trend.to_dict(),
            "error_rate_trend": self.error_rate_trend.to_dict(),
            "improvement_effectiveness_trend": (self.improvement_effectiveness_trend.to_dict()),
            "prompt_effectiveness_trend": self.prompt_effectiveness_trend.to_dict(),
            "overall_health_trend": self.overall_health_trend.value,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class MetaReasoningDashboard:
    """Unified view of meta-reasoning system health."""

    dashboard_id: str

    # Overall health
    system_health: HealthStatus
    overall_confidence: float

    # Component summaries
    observer_summary: ObserverSummary
    strategy_summary: StrategySummary
    prompt_summary: PromptSummary
    failure_summary: FailureSummary
    improvement_summary: ImprovementSummary

    # Cross-cutting metrics
    bottleneck_count: int
    active_improvements: int
    recent_failures: int
    prompt_effectiveness_trend: TrendDirection

    # Alerts
    alerts: List[MetaReasoningAlert]

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dashboard_id": self.dashboard_id,
            "system_health": self.system_health.value,
            "overall_confidence": self.overall_confidence,
            "observer_summary": self.observer_summary.to_dict(),
            "strategy_summary": self.strategy_summary.to_dict(),
            "prompt_summary": self.prompt_summary.to_dict(),
            "failure_summary": self.failure_summary.to_dict(),
            "improvement_summary": self.improvement_summary.to_dict(),
            "bottleneck_count": self.bottleneck_count,
            "active_improvements": self.active_improvements,
            "recent_failures": self.recent_failures,
            "prompt_effectiveness_trend": self.prompt_effectiveness_trend.value,
            "alerts": [alert.to_dict() for alert in self.alerts],
            "generated_at": self.generated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaReasoningDashboard":
        """Create from dictionary."""
        return cls(
            dashboard_id=data["dashboard_id"],
            system_health=HealthStatus(data["system_health"]),
            overall_confidence=data["overall_confidence"],
            observer_summary=ObserverSummary(**data["observer_summary"]),
            strategy_summary=StrategySummary(**data["strategy_summary"]),
            prompt_summary=PromptSummary(**data["prompt_summary"]),
            failure_summary=FailureSummary(**data["failure_summary"]),
            improvement_summary=ImprovementSummary(**data["improvement_summary"]),
            bottleneck_count=data["bottleneck_count"],
            active_improvements=data["active_improvements"],
            recent_failures=data["recent_failures"],
            prompt_effectiveness_trend=TrendDirection(data["prompt_effectiveness_trend"]),
            alerts=[MetaReasoningAlert.from_dict(a) for a in data.get("alerts", [])],
            generated_at=datetime.fromisoformat(data["generated_at"]),
        )

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Export to Markdown format."""
        return DashboardExporter.to_markdown(self)

    def to_html(self) -> str:
        """Export to HTML format."""
        return DashboardExporter.to_html(self)


class DashboardGenerator:
    """Generate unified meta-reasoning dashboard."""

    def __init__(
        self,
        observer: Optional[ReasoningObserver] = None,
        strategy_evaluator: Optional[StrategyEvaluator] = None,
        strategy_selector: Optional[StrategySelector] = None,
        prompt_optimizer: Optional[PromptOptimizer] = None,
        failure_analyzer: Optional[FailureAnalyzer] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
        improvement_tracker: Optional[SelfImprovementTracker] = None,
        improver: Optional[AutonomousImprover] = None,
    ):
        """Initialize the dashboard generator.

        Args:
            observer: ReasoningObserver for chain observations
            strategy_evaluator: StrategyEvaluator for strategy metrics
            strategy_selector: StrategySelector for strategy selection
            prompt_optimizer: PromptOptimizer for prompt metrics
            failure_analyzer: FailureAnalyzer for error analysis
            recommendation_engine: RecommendationEngine for recommendations
            improvement_tracker: SelfImprovementTracker for goals
            improver: AutonomousImprover for improvement cycles
        """
        self._observer = observer
        self._strategy_evaluator = strategy_evaluator
        self._strategy_selector = strategy_selector
        self._prompt_optimizer = prompt_optimizer
        self._failure_analyzer = failure_analyzer
        self._recommendation_engine = recommendation_engine
        self._improvement_tracker = improvement_tracker
        self._improver = improver

        # Alert thresholds
        self._error_rate_warning_threshold = 0.2
        self._error_rate_critical_threshold = 0.4
        self._accuracy_warning_threshold = 0.7
        self._accuracy_critical_threshold = 0.5

    def generate(self) -> MetaReasoningDashboard:
        """Generate current dashboard state.

        Returns:
            MetaReasoningDashboard with current system state
        """
        # Collect summaries from each component
        observer_summary = self._collect_observer_summary()
        strategy_summary = self._collect_strategy_summary()
        prompt_summary = self._collect_prompt_summary()
        failure_summary = self._collect_failure_summary()
        improvement_summary = self._collect_improvement_summary()

        # Calculate cross-cutting metrics
        bottleneck_count = observer_summary.bottlenecks_identified
        active_improvements = improvement_summary.actions_pending
        recent_failures = failure_summary.total_errors_recorded

        # Determine prompt effectiveness trend
        prompt_effectiveness_trend = self._calculate_prompt_trend()

        # Collect alerts
        alerts = self._collect_alerts(
            observer_summary,
            strategy_summary,
            prompt_summary,
            failure_summary,
            improvement_summary,
        )

        # Calculate overall health and confidence
        system_health = self._calculate_system_health(observer_summary, failure_summary, alerts)
        overall_confidence = self._calculate_overall_confidence(
            observer_summary, strategy_summary, failure_summary
        )

        return MetaReasoningDashboard(
            dashboard_id=f"dashboard_{uuid.uuid4().hex[:12]}",
            system_health=system_health,
            overall_confidence=overall_confidence,
            observer_summary=observer_summary,
            strategy_summary=strategy_summary,
            prompt_summary=prompt_summary,
            failure_summary=failure_summary,
            improvement_summary=improvement_summary,
            bottleneck_count=bottleneck_count,
            active_improvements=active_improvements,
            recent_failures=recent_failures,
            prompt_effectiveness_trend=prompt_effectiveness_trend,
            alerts=alerts,
        )

    def generate_trend_report(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> TrendReport:
        """Generate trend analysis over time period.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            TrendReport with trend data
        """
        # For now, generate placeholder trend data
        # In a full implementation, this would query historical data

        accuracy_trend = self._calculate_accuracy_trend(start_date, end_date)
        error_rate_trend = self._calculate_error_rate_trend(start_date, end_date)
        improvement_trend = self._calculate_improvement_trend(start_date, end_date)
        prompt_trend = self._calculate_prompt_trend_data(start_date, end_date)

        # Determine overall health trend
        overall_health_trend = self._determine_overall_trend(
            [
                accuracy_trend.direction,
                error_rate_trend.direction,
                improvement_trend.direction,
            ]
        )

        return TrendReport(
            report_id=f"trend_{uuid.uuid4().hex[:12]}",
            start_date=start_date,
            end_date=end_date,
            accuracy_trend=accuracy_trend,
            error_rate_trend=error_rate_trend,
            improvement_effectiveness_trend=improvement_trend,
            prompt_effectiveness_trend=prompt_trend,
            overall_health_trend=overall_health_trend,
        )

    def _collect_observer_summary(self) -> ObserverSummary:
        """Collect summary from ReasoningObserver."""
        if not self._observer:
            return ObserverSummary(
                total_chains_observed=0,
                successful_chains=0,
                failed_chains=0,
                success_rate=0.0,
                domains_observed=[],
                patterns_detected=0,
                bottlenecks_identified=0,
                average_chain_duration_ms=0.0,
            )

        chains = list(self._observer.chains.values())
        total = len(chains)
        successful = sum(1 for c in chains if c.overall_success)
        failed = total - successful
        success_rate = successful / total if total > 0 else 0.0

        domains = list(set(c.domain for c in chains if c.domain))
        patterns = len(self._observer.patterns)
        bottlenecks = len(getattr(self._observer, "_bottlenecks", []))

        # Calculate average duration
        durations = [c.total_duration_ms for c in chains if c.total_duration_ms]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return ObserverSummary(
            total_chains_observed=total,
            successful_chains=successful,
            failed_chains=failed,
            success_rate=success_rate,
            domains_observed=domains,
            patterns_detected=patterns,
            bottlenecks_identified=bottlenecks,
            average_chain_duration_ms=avg_duration,
        )

    def _collect_strategy_summary(self) -> StrategySummary:
        """Collect summary from Strategy components."""
        strategies_available = 0
        strategies_evaluated = 0
        best_strategy = None
        best_accuracy = 0.0
        selections_made = 0
        domains_covered: List[str] = []

        if self._strategy_evaluator:
            strategies = self._strategy_evaluator._strategies
            strategies_available = len(strategies)

            # Check for evaluated strategies
            results = self._strategy_evaluator._results
            strategies_evaluated = len(set(r.strategy_name for r in results.values() if results))

            # Find best strategy by accuracy
            strategy_accuracies: Dict[str, List[float]] = {}
            for result in results.values():
                if result.strategy_name not in strategy_accuracies:
                    strategy_accuracies[result.strategy_name] = []
                strategy_accuracies[result.strategy_name].append(1.0 if result.correct else 0.0)

            for name, accuracies in strategy_accuracies.items():
                avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
                if avg_acc > best_accuracy:
                    best_accuracy = avg_acc
                    best_strategy = name

            domains_covered = list(self._strategy_evaluator._domain_results.keys())

        if self._strategy_selector:
            selections_made = getattr(self._strategy_selector, "_selection_count", 0)

        return StrategySummary(
            strategies_available=strategies_available,
            strategies_evaluated=strategies_evaluated,
            best_strategy=best_strategy,
            best_strategy_accuracy=best_accuracy,
            selections_made=selections_made,
            domains_covered=domains_covered,
        )

    def _collect_prompt_summary(self) -> PromptSummary:
        """Collect summary from PromptOptimizer."""
        if not self._prompt_optimizer:
            return PromptSummary(
                total_prompts=0,
                active_prompts=0,
                average_effectiveness=0.0,
                best_prompt_category=None,
                active_ab_tests=0,
                improvement_candidates=0,
            )

        versions = self._prompt_optimizer._versions
        total_prompts = len(versions)
        active_prompts = sum(1 for v in versions.values() if v.is_active)

        # Calculate average effectiveness
        effectiveness_scores = []
        category_effectiveness: Dict[str, List[float]] = {}
        for version in versions.values():
            if version.metrics and version.metrics.success_count > 0:
                effectiveness = version.metrics.success_count / (
                    version.metrics.success_count + version.metrics.failure_count
                )
                effectiveness_scores.append(effectiveness)

                cat = version.category.value if version.category else "unknown"
                if cat not in category_effectiveness:
                    category_effectiveness[cat] = []
                category_effectiveness[cat].append(effectiveness)

        avg_effectiveness = (
            sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0.0
        )

        # Find best category
        best_category = None
        best_cat_effectiveness = 0.0
        for cat, scores in category_effectiveness.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            if avg > best_cat_effectiveness:
                best_cat_effectiveness = avg
                best_category = cat

        return PromptSummary(
            total_prompts=total_prompts,
            active_prompts=active_prompts,
            average_effectiveness=avg_effectiveness,
            best_prompt_category=best_category,
            active_ab_tests=0,  # Would need ABTester reference
            improvement_candidates=0,
        )

    def _collect_failure_summary(self) -> FailureSummary:
        """Collect summary from FailureAnalyzer."""
        if not self._failure_analyzer:
            return FailureSummary(
                total_errors_recorded=0,
                patterns_identified=0,
                root_causes_found=0,
                recommendations_generated=0,
                error_rate=0.0,
                most_common_error_category=None,
            )

        errors = self._failure_analyzer._errors
        total_errors = len(errors)

        # Count patterns
        patterns = getattr(self._failure_analyzer, "_patterns", {})
        patterns_count = len(patterns)

        # Count root causes
        analyses = getattr(self._failure_analyzer, "_analyses", {})
        root_causes = sum(
            len(a.root_causes) for a in analyses.values() if hasattr(a, "root_causes")
        )

        # Count recommendations
        recommendations_count = 0
        if self._recommendation_engine:
            recommendations_count = len(self._recommendation_engine._recommendations)

        # Find most common error category
        category_counts: Dict[str, int] = {}
        for error in errors.values():
            cat = error.category.value if error.category else "unknown"
            category_counts[cat] = category_counts.get(cat, 0) + 1

        most_common_category = None
        if category_counts:
            most_common_category = max(category_counts, key=category_counts.get)

        # Calculate error rate (errors / total observations)
        error_rate = 0.0
        if self._observer:
            total_chains = len(self._observer.chains)
            if total_chains > 0:
                error_rate = total_errors / total_chains

        return FailureSummary(
            total_errors_recorded=total_errors,
            patterns_identified=patterns_count,
            root_causes_found=root_causes,
            recommendations_generated=recommendations_count,
            error_rate=error_rate,
            most_common_error_category=most_common_category,
        )

    def _collect_improvement_summary(self) -> ImprovementSummary:
        """Collect summary from SelfImprovementTracker."""
        if not self._improvement_tracker:
            return ImprovementSummary(
                active_goals=0,
                completed_goals=0,
                in_progress_goals=0,
                total_cycles_run=0,
                average_effectiveness=0.0,
                actions_pending=0,
                actions_completed=0,
            )

        goals = self._improvement_tracker._goals
        # PENDING goals are considered "active" (not yet started)
        active_goals = sum(1 for g in goals.values() if g.status == GoalStatus.PENDING)
        completed_goals = sum(1 for g in goals.values() if g.status == GoalStatus.ACHIEVED)
        in_progress_goals = sum(1 for g in goals.values() if g.status == GoalStatus.IN_PROGRESS)

        # Count cycles
        cycles = getattr(self._improvement_tracker, "_cycles", {})
        total_cycles = len(cycles)

        # Calculate average effectiveness
        effectiveness_scores = []
        for cycle in cycles.values():
            if cycle.status == CycleStatus.COMPLETED and cycle.results:
                effectiveness_scores.append(cycle.results.overall_improvement)

        avg_effectiveness = (
            sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0.0
        )

        # Count actions
        actions = getattr(self._improvement_tracker, "_actions", {})
        actions_pending = sum(
            1 for a in actions.values() if getattr(a, "status", None) == "pending"
        )
        actions_completed = sum(
            1 for a in actions.values() if getattr(a, "status", None) == "completed"
        )

        return ImprovementSummary(
            active_goals=active_goals,
            completed_goals=completed_goals,
            in_progress_goals=in_progress_goals,
            total_cycles_run=total_cycles,
            average_effectiveness=avg_effectiveness,
            actions_pending=actions_pending,
            actions_completed=actions_completed,
        )

    def _calculate_prompt_trend(self) -> TrendDirection:
        """Calculate prompt effectiveness trend direction."""
        if not self._prompt_optimizer:
            return TrendDirection.UNKNOWN

        # For now, return stable - in a full implementation,
        # this would analyze historical effectiveness data
        return TrendDirection.STABLE

    def _collect_alerts(
        self,
        observer_summary: ObserverSummary,
        strategy_summary: StrategySummary,
        prompt_summary: PromptSummary,
        failure_summary: FailureSummary,
        improvement_summary: ImprovementSummary,
    ) -> List[MetaReasoningAlert]:
        """Collect alerts based on current state."""
        alerts: List[MetaReasoningAlert] = []

        # Check error rate
        if failure_summary.error_rate >= self._error_rate_critical_threshold:
            alerts.append(
                MetaReasoningAlert(
                    alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                    severity=AlertSeverity.CRITICAL,
                    component="failure_analyzer",
                    message=f"Critical error rate: {failure_summary.error_rate:.1%}",
                    details={"error_rate": failure_summary.error_rate},
                )
            )
        elif failure_summary.error_rate >= self._error_rate_warning_threshold:
            alerts.append(
                MetaReasoningAlert(
                    alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                    severity=AlertSeverity.WARNING,
                    component="failure_analyzer",
                    message=f"Elevated error rate: {failure_summary.error_rate:.1%}",
                    details={"error_rate": failure_summary.error_rate},
                )
            )

        # Check accuracy
        if observer_summary.success_rate < self._accuracy_critical_threshold:
            alerts.append(
                MetaReasoningAlert(
                    alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                    severity=AlertSeverity.CRITICAL,
                    component="observer",
                    message=(f"Critical accuracy drop: {observer_summary.success_rate:.1%}"),
                    details={"success_rate": observer_summary.success_rate},
                )
            )
        elif observer_summary.success_rate < self._accuracy_warning_threshold:
            alerts.append(
                MetaReasoningAlert(
                    alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                    severity=AlertSeverity.WARNING,
                    component="observer",
                    message=f"Low accuracy: {observer_summary.success_rate:.1%}",
                    details={"success_rate": observer_summary.success_rate},
                )
            )

        # Check for bottlenecks
        if observer_summary.bottlenecks_identified > 5:
            alerts.append(
                MetaReasoningAlert(
                    alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                    severity=AlertSeverity.WARNING,
                    component="observer",
                    message=(f"{observer_summary.bottlenecks_identified} bottlenecks identified"),
                    details={"bottleneck_count": observer_summary.bottlenecks_identified},
                )
            )

        # Check for pending improvements
        if improvement_summary.actions_pending > 10:
            alerts.append(
                MetaReasoningAlert(
                    alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                    severity=AlertSeverity.INFO,
                    component="self_improvement",
                    message=(f"{improvement_summary.actions_pending} improvement actions pending"),
                    details={"pending_actions": improvement_summary.actions_pending},
                )
            )

        return alerts

    def _calculate_system_health(
        self,
        observer_summary: ObserverSummary,
        failure_summary: FailureSummary,
        alerts: List[MetaReasoningAlert],
    ) -> HealthStatus:
        """Calculate overall system health status."""
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            return HealthStatus.CRITICAL

        # Check for warning alerts
        warning_alerts = [a for a in alerts if a.severity == AlertSeverity.WARNING]
        if len(warning_alerts) >= 2:
            return HealthStatus.WARNING

        # Check key metrics
        if observer_summary.total_chains_observed == 0:
            return HealthStatus.UNKNOWN

        if (
            observer_summary.success_rate >= 0.8
            and failure_summary.error_rate < self._error_rate_warning_threshold
        ):
            return HealthStatus.HEALTHY

        return HealthStatus.WARNING

    def _calculate_overall_confidence(
        self,
        observer_summary: ObserverSummary,
        strategy_summary: StrategySummary,
        failure_summary: FailureSummary,
    ) -> float:
        """Calculate overall system confidence score (0-1)."""
        factors = []

        # Success rate contributes to confidence
        if observer_summary.total_chains_observed > 0:
            factors.append(observer_summary.success_rate)

        # Strategy accuracy contributes
        if strategy_summary.best_strategy_accuracy > 0:
            factors.append(strategy_summary.best_strategy_accuracy)

        # Low error rate contributes (invert error rate)
        factors.append(1.0 - min(failure_summary.error_rate, 1.0))

        if not factors:
            return 0.5

        return sum(factors) / len(factors)

    def _calculate_accuracy_trend(self, start_date: datetime, end_date: datetime) -> TrendData:
        """Calculate accuracy trend over time period."""
        # Placeholder implementation - would query historical data
        return TrendData(
            metric_name="accuracy",
            direction=TrendDirection.STABLE,
            values=[0.8, 0.82, 0.81, 0.83],
            timestamps=[
                start_date,
                start_date + (end_date - start_date) / 3,
                start_date + 2 * (end_date - start_date) / 3,
                end_date,
            ],
            change_percentage=3.75,
        )

    def _calculate_error_rate_trend(self, start_date: datetime, end_date: datetime) -> TrendData:
        """Calculate error rate trend over time period."""
        return TrendData(
            metric_name="error_rate",
            direction=TrendDirection.STABLE,
            values=[0.15, 0.14, 0.13, 0.12],
            timestamps=[
                start_date,
                start_date + (end_date - start_date) / 3,
                start_date + 2 * (end_date - start_date) / 3,
                end_date,
            ],
            change_percentage=-20.0,
        )

    def _calculate_improvement_trend(self, start_date: datetime, end_date: datetime) -> TrendData:
        """Calculate improvement effectiveness trend."""
        return TrendData(
            metric_name="improvement_effectiveness",
            direction=TrendDirection.IMPROVING,
            values=[0.6, 0.65, 0.7, 0.75],
            timestamps=[
                start_date,
                start_date + (end_date - start_date) / 3,
                start_date + 2 * (end_date - start_date) / 3,
                end_date,
            ],
            change_percentage=25.0,
        )

    def _calculate_prompt_trend_data(self, start_date: datetime, end_date: datetime) -> TrendData:
        """Calculate prompt effectiveness trend data."""
        return TrendData(
            metric_name="prompt_effectiveness",
            direction=TrendDirection.STABLE,
            values=[0.75, 0.76, 0.75, 0.77],
            timestamps=[
                start_date,
                start_date + (end_date - start_date) / 3,
                start_date + 2 * (end_date - start_date) / 3,
                end_date,
            ],
            change_percentage=2.67,
        )

    def _determine_overall_trend(self, trends: List[TrendDirection]) -> TrendDirection:
        """Determine overall trend from multiple trend directions."""
        improving = sum(1 for t in trends if t == TrendDirection.IMPROVING)
        declining = sum(1 for t in trends if t == TrendDirection.DECLINING)

        if improving > declining:
            return TrendDirection.IMPROVING
        elif declining > improving:
            return TrendDirection.DECLINING
        else:
            return TrendDirection.STABLE


class DashboardExporter:
    """Export dashboard to various formats."""

    @staticmethod
    def to_json(dashboard: MetaReasoningDashboard, indent: int = 2) -> str:
        """Export dashboard to JSON string."""
        return json.dumps(dashboard.to_dict(), indent=indent)

    @staticmethod
    def to_markdown(dashboard: MetaReasoningDashboard) -> str:
        """Export dashboard to Markdown format."""
        lines = []
        lines.append("# Meta-Reasoning Dashboard")
        lines.append("")
        lines.append(f"**Generated:** {dashboard.generated_at.isoformat()}")
        lines.append(f"**Dashboard ID:** {dashboard.dashboard_id}")
        lines.append("")

        # Overall Health
        lines.append("## System Health")
        lines.append("")
        lines.append(f"- **Status:** {dashboard.system_health.value.upper()}")
        lines.append(f"- **Confidence:** {dashboard.overall_confidence:.1%}")
        lines.append("")

        # Alerts
        if dashboard.alerts:
            lines.append("## Alerts")
            lines.append("")
            for alert in dashboard.alerts:
                severity_icon = {
                    AlertSeverity.INFO: "INFO",
                    AlertSeverity.WARNING: "WARNING",
                    AlertSeverity.ERROR: "ERROR",
                    AlertSeverity.CRITICAL: "CRITICAL",
                }
                lines.append(
                    f"- **[{severity_icon[alert.severity]}]** ({alert.component}) {alert.message}"
                )
            lines.append("")

        # Cross-cutting metrics
        lines.append("## Key Metrics")
        lines.append("")
        lines.append(f"- **Bottlenecks:** {dashboard.bottleneck_count}")
        lines.append(f"- **Active Improvements:** {dashboard.active_improvements}")
        lines.append(f"- **Recent Failures:** {dashboard.recent_failures}")
        lines.append(
            f"- **Prompt Effectiveness Trend:** {dashboard.prompt_effectiveness_trend.value}"
        )
        lines.append("")

        # Observer Summary
        obs = dashboard.observer_summary
        lines.append("## Observer Summary")
        lines.append("")
        lines.append(f"- Total Chains: {obs.total_chains_observed}")
        lines.append(f"- Success Rate: {obs.success_rate:.1%}")
        lines.append(f"- Patterns Detected: {obs.patterns_detected}")
        lines.append(f"- Domains: {', '.join(obs.domains_observed) or 'None'}")
        lines.append("")

        # Strategy Summary
        strat = dashboard.strategy_summary
        lines.append("## Strategy Summary")
        lines.append("")
        lines.append(f"- Strategies Available: {strat.strategies_available}")
        lines.append(f"- Best Strategy: {strat.best_strategy or 'N/A'}")
        lines.append(f"- Best Accuracy: {strat.best_strategy_accuracy:.1%}")
        lines.append(f"- Selections Made: {strat.selections_made}")
        lines.append("")

        # Prompt Summary
        prompt = dashboard.prompt_summary
        lines.append("## Prompt Summary")
        lines.append("")
        lines.append(f"- Total Prompts: {prompt.total_prompts}")
        lines.append(f"- Active Prompts: {prompt.active_prompts}")
        lines.append(f"- Average Effectiveness: {prompt.average_effectiveness:.1%}")
        lines.append(f"- Best Category: {prompt.best_prompt_category or 'N/A'}")
        lines.append("")

        # Failure Summary
        fail = dashboard.failure_summary
        lines.append("## Failure Summary")
        lines.append("")
        lines.append(f"- Total Errors: {fail.total_errors_recorded}")
        lines.append(f"- Patterns Identified: {fail.patterns_identified}")
        lines.append(f"- Error Rate: {fail.error_rate:.1%}")
        lines.append(f"- Most Common Category: {fail.most_common_error_category or 'N/A'}")
        lines.append("")

        # Improvement Summary
        imp = dashboard.improvement_summary
        lines.append("## Improvement Summary")
        lines.append("")
        lines.append(f"- Active Goals: {imp.active_goals}")
        lines.append(f"- Completed Goals: {imp.completed_goals}")
        lines.append(f"- Total Cycles: {imp.total_cycles_run}")
        lines.append(f"- Average Effectiveness: {imp.average_effectiveness:.1%}")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_html(dashboard: MetaReasoningDashboard) -> str:
        """Export dashboard to HTML format."""
        html_parts = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html>")
        html_parts.append("<head>")
        html_parts.append("<title>Meta-Reasoning Dashboard</title>")
        html_parts.append("<style>")
        html_parts.append(
            """
            body { font-family: Arial, sans-serif; margin: 20px; }
            .dashboard { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .health-healthy { color: green; }
            .health-warning { color: orange; }
            .health-critical { color: red; }
            .health-unknown { color: gray; }
            .section { margin-bottom: 20px; padding: 15px;
                       border: 1px solid #ddd; border-radius: 5px; }
            .section h2 { margin-top: 0; color: #333; }
            .metrics { display: flex; flex-wrap: wrap; gap: 10px; }
            .metric { background: #f5f5f5; padding: 10px;
                      border-radius: 3px; min-width: 150px; }
            .metric-label { font-size: 12px; color: #666; }
            .metric-value { font-size: 20px; font-weight: bold; }
            .alert { padding: 10px; margin: 5px 0; border-radius: 3px; }
            .alert-info { background: #e3f2fd; }
            .alert-warning { background: #fff3e0; }
            .alert-error { background: #ffebee; }
            .alert-critical { background: #ffcdd2; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            """
        )
        html_parts.append("</style>")
        html_parts.append("</head>")
        html_parts.append("<body>")
        html_parts.append('<div class="dashboard">')

        # Header
        html_parts.append('<div class="header">')
        html_parts.append("<h1>Meta-Reasoning Dashboard</h1>")
        html_parts.append(
            f"<p>Generated: {dashboard.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>"
        )
        html_parts.append("</div>")

        # Health Status
        health_class = f"health-{dashboard.system_health.value}"
        html_parts.append('<div class="section">')
        html_parts.append("<h2>System Health</h2>")
        html_parts.append('<div class="metrics">')
        html_parts.append('<div class="metric">')
        html_parts.append('<div class="metric-label">Status</div>')
        html_parts.append(
            f'<div class="metric-value {health_class}">'
            f"{dashboard.system_health.value.upper()}</div>"
        )
        html_parts.append("</div>")
        html_parts.append('<div class="metric">')
        html_parts.append('<div class="metric-label">Confidence</div>')
        html_parts.append(f'<div class="metric-value">{dashboard.overall_confidence:.1%}</div>')
        html_parts.append("</div>")
        html_parts.append("</div>")
        html_parts.append("</div>")

        # Alerts
        if dashboard.alerts:
            html_parts.append('<div class="section">')
            html_parts.append("<h2>Alerts</h2>")
            for alert in dashboard.alerts:
                alert_class = f"alert-{alert.severity.value}"
                html_parts.append(f'<div class="alert {alert_class}">')
                html_parts.append(
                    f"<strong>[{alert.severity.value.upper()}]</strong> "
                    f"({alert.component}) {alert.message}"
                )
                html_parts.append("</div>")
            html_parts.append("</div>")

        # Key Metrics
        html_parts.append('<div class="section">')
        html_parts.append("<h2>Key Metrics</h2>")
        html_parts.append('<div class="metrics">')
        for label, value in [
            ("Bottlenecks", dashboard.bottleneck_count),
            ("Active Improvements", dashboard.active_improvements),
            ("Recent Failures", dashboard.recent_failures),
        ]:
            html_parts.append('<div class="metric">')
            html_parts.append(f'<div class="metric-label">{label}</div>')
            html_parts.append(f'<div class="metric-value">{value}</div>')
            html_parts.append("</div>")
        html_parts.append("</div>")
        html_parts.append("</div>")

        # Component Summaries
        summaries = [
            ("Observer", dashboard.observer_summary.to_dict()),
            ("Strategy", dashboard.strategy_summary.to_dict()),
            ("Prompt", dashboard.prompt_summary.to_dict()),
            ("Failure", dashboard.failure_summary.to_dict()),
            ("Improvement", dashboard.improvement_summary.to_dict()),
        ]

        for title, summary_dict in summaries:
            html_parts.append('<div class="section">')
            html_parts.append(f"<h2>{title} Summary</h2>")
            html_parts.append("<table>")
            for key, value in summary_dict.items():
                display_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    if value <= 1.0 and "rate" in key.lower():
                        display_value = f"{value:.1%}"
                    else:
                        display_value = f"{value:.2f}"
                elif isinstance(value, list):
                    display_value = ", ".join(str(v) for v in value) or "None"
                else:
                    display_value = str(value) if value is not None else "N/A"
                html_parts.append(f"<tr><td>{display_key}</td><td>{display_value}</td></tr>")
            html_parts.append("</table>")
            html_parts.append("</div>")

        html_parts.append("</div>")
        html_parts.append("</body>")
        html_parts.append("</html>")

        return "\n".join(html_parts)


def create_dashboard_generator(
    observer: Optional[ReasoningObserver] = None,
    strategy_evaluator: Optional[StrategyEvaluator] = None,
    strategy_selector: Optional[StrategySelector] = None,
    prompt_optimizer: Optional[PromptOptimizer] = None,
    failure_analyzer: Optional[FailureAnalyzer] = None,
    recommendation_engine: Optional[RecommendationEngine] = None,
    improvement_tracker: Optional[SelfImprovementTracker] = None,
    improver: Optional[AutonomousImprover] = None,
) -> DashboardGenerator:
    """Factory function to create a DashboardGenerator.

    Args:
        observer: ReasoningObserver for chain observations
        strategy_evaluator: StrategyEvaluator for strategy metrics
        strategy_selector: StrategySelector for strategy selection
        prompt_optimizer: PromptOptimizer for prompt metrics
        failure_analyzer: FailureAnalyzer for error analysis
        recommendation_engine: RecommendationEngine for recommendations
        improvement_tracker: SelfImprovementTracker for goals
        improver: AutonomousImprover for improvement cycles

    Returns:
        New DashboardGenerator instance
    """
    return DashboardGenerator(
        observer=observer,
        strategy_evaluator=strategy_evaluator,
        strategy_selector=strategy_selector,
        prompt_optimizer=prompt_optimizer,
        failure_analyzer=failure_analyzer,
        recommendation_engine=recommendation_engine,
        improvement_tracker=improvement_tracker,
        improver=improver,
    )
