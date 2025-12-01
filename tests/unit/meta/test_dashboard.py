"""Tests for meta-reasoning dashboard and reporting."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from loft.meta.dashboard import (
    AlertSeverity,
    DashboardExporter,
    DashboardGenerator,
    FailureSummary,
    HealthStatus,
    ImprovementSummary,
    MetaReasoningAlert,
    MetaReasoningDashboard,
    ObserverSummary,
    PromptSummary,
    StrategySummary,
    TrendData,
    TrendDirection,
    TrendReport,
    create_dashboard_generator,
)
from loft.meta.failure_analyzer import ErrorCategory
from loft.meta.self_improvement import GoalStatus


# Helper functions for creating test objects
def create_test_observer_summary(
    total_chains: int = 100,
    successful: int = 85,
    success_rate: float = 0.85,
) -> ObserverSummary:
    """Create a test observer summary."""
    return ObserverSummary(
        total_chains_observed=total_chains,
        successful_chains=successful,
        failed_chains=total_chains - successful,
        success_rate=success_rate,
        domains_observed=["contracts", "torts"],
        patterns_detected=5,
        bottlenecks_identified=2,
        average_chain_duration_ms=150.5,
    )


def create_test_strategy_summary() -> StrategySummary:
    """Create a test strategy summary."""
    return StrategySummary(
        strategies_available=6,
        strategies_evaluated=4,
        best_strategy="RuleBasedStrategy",
        best_strategy_accuracy=0.88,
        selections_made=50,
        domains_covered=["contracts", "torts", "property"],
    )


def create_test_prompt_summary() -> PromptSummary:
    """Create a test prompt summary."""
    return PromptSummary(
        total_prompts=10,
        active_prompts=8,
        average_effectiveness=0.82,
        best_prompt_category="rule_extraction",
        active_ab_tests=2,
        improvement_candidates=3,
    )


def create_test_failure_summary(
    error_rate: float = 0.15,
) -> FailureSummary:
    """Create a test failure summary."""
    return FailureSummary(
        total_errors_recorded=15,
        patterns_identified=3,
        root_causes_found=8,
        recommendations_generated=5,
        error_rate=error_rate,
        most_common_error_category="LOGIC_ERROR",
    )


def create_test_improvement_summary() -> ImprovementSummary:
    """Create a test improvement summary."""
    return ImprovementSummary(
        active_goals=3,
        completed_goals=2,
        in_progress_goals=1,
        total_cycles_run=5,
        average_effectiveness=0.72,
        actions_pending=4,
        actions_completed=8,
    )


def create_test_dashboard(
    health: HealthStatus = HealthStatus.HEALTHY,
    confidence: float = 0.85,
) -> MetaReasoningDashboard:
    """Create a test dashboard."""
    return MetaReasoningDashboard(
        dashboard_id="dashboard_test123",
        system_health=health,
        overall_confidence=confidence,
        observer_summary=create_test_observer_summary(),
        strategy_summary=create_test_strategy_summary(),
        prompt_summary=create_test_prompt_summary(),
        failure_summary=create_test_failure_summary(),
        improvement_summary=create_test_improvement_summary(),
        bottleneck_count=2,
        active_improvements=4,
        recent_failures=15,
        prompt_effectiveness_trend=TrendDirection.STABLE,
        alerts=[],
    )


def create_test_alert(
    severity: AlertSeverity = AlertSeverity.WARNING,
    component: str = "observer",
    message: str = "Test alert",
) -> MetaReasoningAlert:
    """Create a test alert."""
    return MetaReasoningAlert(
        alert_id="alert_test123",
        severity=severity,
        component=component,
        message=message,
        details={"test_key": "test_value"},
    )


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Test that all health statuses exist."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestTrendDirection:
    """Tests for TrendDirection enum."""

    def test_trend_direction_values(self):
        """Test that all trend directions exist."""
        assert TrendDirection.IMPROVING.value == "improving"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.DECLINING.value == "declining"
        assert TrendDirection.UNKNOWN.value == "unknown"


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_alert_severity_values(self):
        """Test that all alert severities exist."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestMetaReasoningAlert:
    """Tests for MetaReasoningAlert dataclass."""

    def test_alert_creation(self):
        """Test creating an alert."""
        alert = create_test_alert()
        assert alert.alert_id == "alert_test123"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.component == "observer"
        assert alert.message == "Test alert"
        assert alert.details == {"test_key": "test_value"}

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = create_test_alert()
        data = alert.to_dict()

        assert data["alert_id"] == "alert_test123"
        assert data["severity"] == "warning"
        assert data["component"] == "observer"
        assert data["message"] == "Test alert"
        assert data["details"] == {"test_key": "test_value"}
        assert "timestamp" in data

    def test_alert_from_dict(self):
        """Test creating alert from dictionary."""
        original = create_test_alert()
        data = original.to_dict()
        restored = MetaReasoningAlert.from_dict(data)

        assert restored.alert_id == original.alert_id
        assert restored.severity == original.severity
        assert restored.component == original.component
        assert restored.message == original.message


class TestObserverSummary:
    """Tests for ObserverSummary dataclass."""

    def test_observer_summary_creation(self):
        """Test creating observer summary."""
        summary = create_test_observer_summary()
        assert summary.total_chains_observed == 100
        assert summary.successful_chains == 85
        assert summary.failed_chains == 15
        assert summary.success_rate == 0.85

    def test_observer_summary_to_dict(self):
        """Test converting to dictionary."""
        summary = create_test_observer_summary()
        data = summary.to_dict()

        assert data["total_chains_observed"] == 100
        assert data["success_rate"] == 0.85
        assert "contracts" in data["domains_observed"]


class TestStrategySummary:
    """Tests for StrategySummary dataclass."""

    def test_strategy_summary_creation(self):
        """Test creating strategy summary."""
        summary = create_test_strategy_summary()
        assert summary.strategies_available == 6
        assert summary.best_strategy == "RuleBasedStrategy"

    def test_strategy_summary_to_dict(self):
        """Test converting to dictionary."""
        summary = create_test_strategy_summary()
        data = summary.to_dict()

        assert data["strategies_available"] == 6
        assert data["best_strategy_accuracy"] == 0.88


class TestPromptSummary:
    """Tests for PromptSummary dataclass."""

    def test_prompt_summary_creation(self):
        """Test creating prompt summary."""
        summary = create_test_prompt_summary()
        assert summary.total_prompts == 10
        assert summary.average_effectiveness == 0.82

    def test_prompt_summary_to_dict(self):
        """Test converting to dictionary."""
        summary = create_test_prompt_summary()
        data = summary.to_dict()

        assert data["total_prompts"] == 10
        assert data["best_prompt_category"] == "rule_extraction"


class TestFailureSummary:
    """Tests for FailureSummary dataclass."""

    def test_failure_summary_creation(self):
        """Test creating failure summary."""
        summary = create_test_failure_summary()
        assert summary.total_errors_recorded == 15
        assert summary.error_rate == 0.15

    def test_failure_summary_to_dict(self):
        """Test converting to dictionary."""
        summary = create_test_failure_summary()
        data = summary.to_dict()

        assert data["total_errors_recorded"] == 15
        assert data["patterns_identified"] == 3


class TestImprovementSummary:
    """Tests for ImprovementSummary dataclass."""

    def test_improvement_summary_creation(self):
        """Test creating improvement summary."""
        summary = create_test_improvement_summary()
        assert summary.active_goals == 3
        assert summary.total_cycles_run == 5

    def test_improvement_summary_to_dict(self):
        """Test converting to dictionary."""
        summary = create_test_improvement_summary()
        data = summary.to_dict()

        assert data["active_goals"] == 3
        assert data["average_effectiveness"] == 0.72


class TestTrendData:
    """Tests for TrendData dataclass."""

    def test_trend_data_creation(self):
        """Test creating trend data."""
        now = datetime.now()
        trend = TrendData(
            metric_name="accuracy",
            direction=TrendDirection.IMPROVING,
            values=[0.8, 0.82, 0.85],
            timestamps=[now, now + timedelta(days=1), now + timedelta(days=2)],
            change_percentage=6.25,
        )

        assert trend.metric_name == "accuracy"
        assert trend.direction == TrendDirection.IMPROVING
        assert len(trend.values) == 3

    def test_trend_data_to_dict(self):
        """Test converting to dictionary."""
        now = datetime.now()
        trend = TrendData(
            metric_name="accuracy",
            direction=TrendDirection.IMPROVING,
            values=[0.8, 0.85],
            timestamps=[now, now + timedelta(days=1)],
            change_percentage=6.25,
        )

        data = trend.to_dict()
        assert data["metric_name"] == "accuracy"
        assert data["direction"] == "improving"
        assert data["change_percentage"] == 6.25


class TestTrendReport:
    """Tests for TrendReport dataclass."""

    def test_trend_report_creation(self):
        """Test creating trend report."""
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()

        accuracy_trend = TrendData(
            metric_name="accuracy",
            direction=TrendDirection.IMPROVING,
            values=[0.8, 0.85],
            timestamps=[start, end],
            change_percentage=6.25,
        )

        report = TrendReport(
            report_id="trend_test123",
            start_date=start,
            end_date=end,
            accuracy_trend=accuracy_trend,
            error_rate_trend=accuracy_trend,
            improvement_effectiveness_trend=accuracy_trend,
            prompt_effectiveness_trend=accuracy_trend,
            overall_health_trend=TrendDirection.IMPROVING,
        )

        assert report.report_id == "trend_test123"
        assert report.overall_health_trend == TrendDirection.IMPROVING

    def test_trend_report_to_dict(self):
        """Test converting to dictionary."""
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()

        accuracy_trend = TrendData(
            metric_name="accuracy",
            direction=TrendDirection.STABLE,
            values=[0.8, 0.8],
            timestamps=[start, end],
            change_percentage=0.0,
        )

        report = TrendReport(
            report_id="trend_test123",
            start_date=start,
            end_date=end,
            accuracy_trend=accuracy_trend,
            error_rate_trend=accuracy_trend,
            improvement_effectiveness_trend=accuracy_trend,
            prompt_effectiveness_trend=accuracy_trend,
            overall_health_trend=TrendDirection.STABLE,
        )

        data = report.to_dict()
        assert data["report_id"] == "trend_test123"
        assert data["overall_health_trend"] == "stable"
        assert "accuracy_trend" in data


class TestMetaReasoningDashboard:
    """Tests for MetaReasoningDashboard dataclass."""

    def test_dashboard_creation(self):
        """Test creating a dashboard."""
        dashboard = create_test_dashboard()
        assert dashboard.dashboard_id == "dashboard_test123"
        assert dashboard.system_health == HealthStatus.HEALTHY
        assert dashboard.overall_confidence == 0.85

    def test_dashboard_to_dict(self):
        """Test converting dashboard to dictionary."""
        dashboard = create_test_dashboard()
        data = dashboard.to_dict()

        assert data["dashboard_id"] == "dashboard_test123"
        assert data["system_health"] == "healthy"
        assert data["overall_confidence"] == 0.85
        assert "observer_summary" in data
        assert "strategy_summary" in data
        assert "prompt_summary" in data
        assert "failure_summary" in data
        assert "improvement_summary" in data

    def test_dashboard_from_dict(self):
        """Test creating dashboard from dictionary."""
        original = create_test_dashboard()
        data = original.to_dict()
        restored = MetaReasoningDashboard.from_dict(data)

        assert restored.dashboard_id == original.dashboard_id
        assert restored.system_health == original.system_health
        assert restored.overall_confidence == original.overall_confidence

    def test_dashboard_to_json(self):
        """Test exporting dashboard to JSON."""
        dashboard = create_test_dashboard()
        json_str = dashboard.to_json()

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["dashboard_id"] == "dashboard_test123"

    def test_dashboard_to_markdown(self):
        """Test exporting dashboard to Markdown."""
        dashboard = create_test_dashboard()
        markdown = dashboard.to_markdown()

        assert "# Meta-Reasoning Dashboard" in markdown
        assert "## System Health" in markdown
        assert "## Observer Summary" in markdown
        assert "## Strategy Summary" in markdown

    def test_dashboard_to_html(self):
        """Test exporting dashboard to HTML."""
        dashboard = create_test_dashboard()
        html = dashboard.to_html()

        assert "<!DOCTYPE html>" in html
        assert "<title>Meta-Reasoning Dashboard</title>" in html
        assert "System Health" in html

    def test_dashboard_with_alerts(self):
        """Test dashboard with alerts."""
        alert = create_test_alert(severity=AlertSeverity.CRITICAL)
        dashboard = MetaReasoningDashboard(
            dashboard_id="dashboard_alerts",
            system_health=HealthStatus.CRITICAL,
            overall_confidence=0.5,
            observer_summary=create_test_observer_summary(),
            strategy_summary=create_test_strategy_summary(),
            prompt_summary=create_test_prompt_summary(),
            failure_summary=create_test_failure_summary(),
            improvement_summary=create_test_improvement_summary(),
            bottleneck_count=10,
            active_improvements=0,
            recent_failures=50,
            prompt_effectiveness_trend=TrendDirection.DECLINING,
            alerts=[alert],
        )

        assert len(dashboard.alerts) == 1
        assert dashboard.alerts[0].severity == AlertSeverity.CRITICAL

        data = dashboard.to_dict()
        assert len(data["alerts"]) == 1


class TestDashboardGenerator:
    """Tests for DashboardGenerator class."""

    @pytest.fixture
    def mock_observer(self):
        """Create mock observer."""
        observer = MagicMock()
        observer.chains = {}
        observer.patterns = {}
        return observer

    @pytest.fixture
    def mock_strategy_evaluator(self):
        """Create mock strategy evaluator."""
        evaluator = MagicMock()
        evaluator._strategies = {}
        evaluator._results = {}
        evaluator._domain_results = {}
        return evaluator

    @pytest.fixture
    def mock_prompt_optimizer(self):
        """Create mock prompt optimizer."""
        optimizer = MagicMock()
        optimizer._versions = {}
        return optimizer

    @pytest.fixture
    def mock_failure_analyzer(self):
        """Create mock failure analyzer."""
        analyzer = MagicMock()
        analyzer._errors = {}
        analyzer._patterns = {}
        analyzer._analyses = {}
        return analyzer

    @pytest.fixture
    def mock_improvement_tracker(self):
        """Create mock improvement tracker."""
        tracker = MagicMock()
        tracker._goals = {}
        tracker._cycles = {}
        tracker._actions = {}
        return tracker

    def test_generator_creation(self):
        """Test creating a dashboard generator."""
        generator = DashboardGenerator()
        assert generator._observer is None
        assert generator._strategy_evaluator is None

    def test_generator_with_components(
        self, mock_observer, mock_strategy_evaluator, mock_prompt_optimizer
    ):
        """Test creating generator with components."""
        generator = DashboardGenerator(
            observer=mock_observer,
            strategy_evaluator=mock_strategy_evaluator,
            prompt_optimizer=mock_prompt_optimizer,
        )

        assert generator._observer is mock_observer
        assert generator._strategy_evaluator is mock_strategy_evaluator
        assert generator._prompt_optimizer is mock_prompt_optimizer

    def test_generate_empty_dashboard(self):
        """Test generating dashboard with no components."""
        generator = DashboardGenerator()
        dashboard = generator.generate()

        assert dashboard.dashboard_id.startswith("dashboard_")
        # With 0 observations, system generates critical alert for 0% accuracy
        assert dashboard.system_health in [HealthStatus.UNKNOWN, HealthStatus.CRITICAL]
        assert dashboard.observer_summary.total_chains_observed == 0

    def test_generate_with_observer(self, mock_observer):
        """Test generating dashboard with observer data."""
        # Setup mock chains
        chain1 = MagicMock()
        chain1.overall_success = True
        chain1.domain = "contracts"
        chain1.total_duration_ms = 100.0

        chain2 = MagicMock()
        chain2.overall_success = False
        chain2.domain = "torts"
        chain2.total_duration_ms = 200.0

        mock_observer.chains = {"chain1": chain1, "chain2": chain2}
        mock_observer.patterns = {"p1": MagicMock()}

        generator = DashboardGenerator(observer=mock_observer)
        dashboard = generator.generate()

        assert dashboard.observer_summary.total_chains_observed == 2
        assert dashboard.observer_summary.successful_chains == 1
        assert dashboard.observer_summary.failed_chains == 1
        assert dashboard.observer_summary.success_rate == 0.5

    def test_generate_with_failure_analyzer(self, mock_observer, mock_failure_analyzer):
        """Test generating dashboard with failure data."""
        # Setup mock errors
        error1 = MagicMock()
        error1.category = ErrorCategory.INFERENCE_ERROR

        error2 = MagicMock()
        error2.category = ErrorCategory.INFERENCE_ERROR

        mock_failure_analyzer._errors = {"e1": error1, "e2": error2}

        # Setup observer for error rate calculation
        mock_observer.chains = {"c1": MagicMock(), "c2": MagicMock()}
        mock_observer.patterns = {}

        generator = DashboardGenerator(
            observer=mock_observer,
            failure_analyzer=mock_failure_analyzer,
        )
        dashboard = generator.generate()

        assert dashboard.failure_summary.total_errors_recorded == 2
        assert dashboard.failure_summary.most_common_error_category == "inference_error"

    def test_generate_with_improvement_tracker(self, mock_improvement_tracker):
        """Test generating dashboard with improvement data."""
        # Setup mock goals
        goal1 = MagicMock()
        goal1.status = GoalStatus.IN_PROGRESS

        goal2 = MagicMock()
        goal2.status = GoalStatus.ACHIEVED

        mock_improvement_tracker._goals = {"g1": goal1, "g2": goal2}
        mock_improvement_tracker._cycles = {}
        mock_improvement_tracker._actions = {}

        generator = DashboardGenerator(improvement_tracker=mock_improvement_tracker)
        dashboard = generator.generate()

        assert dashboard.improvement_summary.in_progress_goals == 1
        assert dashboard.improvement_summary.completed_goals == 1

    def test_generate_trend_report(self):
        """Test generating trend report."""
        generator = DashboardGenerator()
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()

        report = generator.generate_trend_report(start, end)

        assert report.report_id.startswith("trend_")
        assert report.start_date == start
        assert report.end_date == end
        assert report.accuracy_trend is not None
        assert report.error_rate_trend is not None

    def test_alert_generation_critical_error_rate(self, mock_failure_analyzer):
        """Test that critical error rate triggers alert."""
        # High error rate scenario
        errors = {f"e{i}": MagicMock() for i in range(50)}
        for e in errors.values():
            e.category = ErrorCategory.INFERENCE_ERROR
        mock_failure_analyzer._errors = errors

        # Create observer with 100 chains (50 errors = 50% error rate)
        mock_observer = MagicMock()
        chains = {}
        for i in range(100):
            chain = MagicMock()
            chain.overall_success = i < 50  # 50% success
            chain.domain = "test"
            chain.total_duration_ms = 100.0
            chains[f"c{i}"] = chain
        mock_observer.chains = chains
        mock_observer.patterns = {}

        generator = DashboardGenerator(
            observer=mock_observer,
            failure_analyzer=mock_failure_analyzer,
        )
        dashboard = generator.generate()

        # Should have critical alert for high error rate
        critical_alerts = [a for a in dashboard.alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) >= 1
        assert dashboard.system_health == HealthStatus.CRITICAL

    def test_alert_generation_low_accuracy(self, mock_observer):
        """Test that low accuracy triggers alert."""
        # Setup chains with low success rate
        chains = {}
        for i in range(10):
            chain = MagicMock()
            chain.overall_success = i < 3  # Only 30% success
            chain.domain = "test"
            chain.total_duration_ms = 100.0
            chains[f"c{i}"] = chain

        mock_observer.chains = chains
        mock_observer.patterns = {}

        generator = DashboardGenerator(observer=mock_observer)
        dashboard = generator.generate()

        # Should have warning/critical alert for low accuracy
        accuracy_alerts = [a for a in dashboard.alerts if "accuracy" in a.message.lower()]
        assert len(accuracy_alerts) >= 1

    def test_system_health_calculation(self, mock_observer):
        """Test system health calculation."""
        # Healthy scenario
        chains = {}
        for i in range(10):
            chain = MagicMock()
            chain.overall_success = i < 9  # 90% success
            chain.domain = "test"
            chain.total_duration_ms = 100.0
            chains[f"c{i}"] = chain

        mock_observer.chains = chains
        mock_observer.patterns = {}

        generator = DashboardGenerator(observer=mock_observer)
        dashboard = generator.generate()

        # High success rate, no critical issues = healthy
        assert dashboard.system_health in [HealthStatus.HEALTHY, HealthStatus.WARNING]

    def test_confidence_calculation(self, mock_observer, mock_strategy_evaluator):
        """Test overall confidence calculation."""
        # Setup high success rate
        chains = {}
        for i in range(10):
            chain = MagicMock()
            chain.overall_success = i < 9  # 90% success
            chain.domain = "test"
            chain.total_duration_ms = 100.0
            chains[f"c{i}"] = chain

        mock_observer.chains = chains
        mock_observer.patterns = {}

        generator = DashboardGenerator(observer=mock_observer)
        dashboard = generator.generate()

        # Should have reasonable confidence
        assert 0.0 <= dashboard.overall_confidence <= 1.0


class TestDashboardExporter:
    """Tests for DashboardExporter class."""

    def test_to_json(self):
        """Test JSON export."""
        dashboard = create_test_dashboard()
        json_str = DashboardExporter.to_json(dashboard)

        data = json.loads(json_str)
        assert data["dashboard_id"] == "dashboard_test123"
        assert data["system_health"] == "healthy"

    def test_to_json_with_indent(self):
        """Test JSON export with custom indent."""
        dashboard = create_test_dashboard()
        json_str = DashboardExporter.to_json(dashboard, indent=4)

        assert "    " in json_str  # 4-space indent

    def test_to_markdown(self):
        """Test Markdown export."""
        dashboard = create_test_dashboard()
        markdown = DashboardExporter.to_markdown(dashboard)

        # Check structure
        assert "# Meta-Reasoning Dashboard" in markdown
        assert "## System Health" in markdown
        assert "## Key Metrics" in markdown
        assert "## Observer Summary" in markdown
        assert "## Strategy Summary" in markdown
        assert "## Prompt Summary" in markdown
        assert "## Failure Summary" in markdown
        assert "## Improvement Summary" in markdown

        # Check content
        assert "HEALTHY" in markdown
        assert "85.0%" in markdown  # Success rate

    def test_to_markdown_with_alerts(self):
        """Test Markdown export with alerts."""
        alert = create_test_alert(severity=AlertSeverity.WARNING)
        dashboard = MetaReasoningDashboard(
            dashboard_id="dashboard_alerts",
            system_health=HealthStatus.WARNING,
            overall_confidence=0.7,
            observer_summary=create_test_observer_summary(),
            strategy_summary=create_test_strategy_summary(),
            prompt_summary=create_test_prompt_summary(),
            failure_summary=create_test_failure_summary(),
            improvement_summary=create_test_improvement_summary(),
            bottleneck_count=5,
            active_improvements=3,
            recent_failures=20,
            prompt_effectiveness_trend=TrendDirection.STABLE,
            alerts=[alert],
        )

        markdown = DashboardExporter.to_markdown(dashboard)
        assert "## Alerts" in markdown
        assert "[WARNING]" in markdown
        assert "Test alert" in markdown

    def test_to_html(self):
        """Test HTML export."""
        dashboard = create_test_dashboard()
        html = DashboardExporter.to_html(dashboard)

        # Check HTML structure
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "</html>" in html

        # Check content
        assert "<title>Meta-Reasoning Dashboard</title>" in html
        assert "System Health" in html
        assert "HEALTHY" in html

    def test_to_html_with_alerts(self):
        """Test HTML export with alerts."""
        alert = create_test_alert(severity=AlertSeverity.ERROR)
        dashboard = MetaReasoningDashboard(
            dashboard_id="dashboard_alerts",
            system_health=HealthStatus.WARNING,
            overall_confidence=0.6,
            observer_summary=create_test_observer_summary(),
            strategy_summary=create_test_strategy_summary(),
            prompt_summary=create_test_prompt_summary(),
            failure_summary=create_test_failure_summary(),
            improvement_summary=create_test_improvement_summary(),
            bottleneck_count=8,
            active_improvements=2,
            recent_failures=30,
            prompt_effectiveness_trend=TrendDirection.DECLINING,
            alerts=[alert],
        )

        html = DashboardExporter.to_html(dashboard)
        assert "alert-error" in html
        assert "Test alert" in html

    def test_to_html_styling(self):
        """Test HTML export includes styling."""
        dashboard = create_test_dashboard()
        html = DashboardExporter.to_html(dashboard)

        # Check CSS is included
        assert "<style>" in html
        assert "font-family" in html
        assert ".dashboard" in html
        assert ".metric" in html


class TestCreateDashboardGenerator:
    """Tests for factory function."""

    def test_create_empty_generator(self):
        """Test creating generator without components."""
        generator = create_dashboard_generator()
        assert isinstance(generator, DashboardGenerator)

    def test_create_with_components(self):
        """Test creating generator with components."""
        mock_observer = MagicMock()
        mock_evaluator = MagicMock()

        generator = create_dashboard_generator(
            observer=mock_observer,
            strategy_evaluator=mock_evaluator,
        )

        assert generator._observer is mock_observer
        assert generator._strategy_evaluator is mock_evaluator


class TestDashboardIntegration:
    """Integration tests for dashboard functionality."""

    def test_full_dashboard_workflow(self):
        """Test complete dashboard generation workflow."""
        # Create generator
        generator = create_dashboard_generator()

        # Generate dashboard
        dashboard = generator.generate()

        # Export to all formats
        json_export = dashboard.to_json()
        markdown_export = dashboard.to_markdown()
        html_export = dashboard.to_html()

        # Verify exports are valid
        assert json.loads(json_export)  # Valid JSON
        assert "# Meta-Reasoning Dashboard" in markdown_export
        assert "<!DOCTYPE html>" in html_export

    def test_dashboard_trend_report_workflow(self):
        """Test trend report generation workflow."""
        generator = create_dashboard_generator()
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()

        report = generator.generate_trend_report(start, end)

        # Verify report structure
        assert report.start_date == start
        assert report.end_date == end
        assert report.accuracy_trend.direction in TrendDirection
        assert report.overall_health_trend in TrendDirection

        # Export to dict
        data = report.to_dict()
        assert "accuracy_trend" in data
        assert "overall_health_trend" in data

    def test_dashboard_roundtrip_serialization(self):
        """Test serialization roundtrip."""
        original = create_test_dashboard()

        # Serialize and deserialize
        json_str = original.to_json()
        data = json.loads(json_str)
        restored = MetaReasoningDashboard.from_dict(data)

        # Verify data integrity
        assert restored.dashboard_id == original.dashboard_id
        assert restored.system_health == original.system_health
        assert restored.overall_confidence == original.overall_confidence
        assert (
            restored.observer_summary.total_chains_observed
            == original.observer_summary.total_chains_observed
        )

    def test_multiple_alert_severities(self):
        """Test dashboard with multiple alert severities."""
        alerts = [
            create_test_alert(severity=AlertSeverity.INFO, message="Info alert"),
            create_test_alert(severity=AlertSeverity.WARNING, message="Warning alert"),
            create_test_alert(severity=AlertSeverity.ERROR, message="Error alert"),
            create_test_alert(severity=AlertSeverity.CRITICAL, message="Critical alert"),
        ]

        dashboard = MetaReasoningDashboard(
            dashboard_id="dashboard_multi_alerts",
            system_health=HealthStatus.CRITICAL,
            overall_confidence=0.3,
            observer_summary=create_test_observer_summary(),
            strategy_summary=create_test_strategy_summary(),
            prompt_summary=create_test_prompt_summary(),
            failure_summary=create_test_failure_summary(),
            improvement_summary=create_test_improvement_summary(),
            bottleneck_count=15,
            active_improvements=0,
            recent_failures=100,
            prompt_effectiveness_trend=TrendDirection.DECLINING,
            alerts=alerts,
        )

        assert len(dashboard.alerts) == 4

        # Verify all severities in exports
        markdown = dashboard.to_markdown()
        assert "[INFO]" in markdown
        assert "[WARNING]" in markdown
        assert "[ERROR]" in markdown
        assert "[CRITICAL]" in markdown

        html = dashboard.to_html()
        assert "alert-info" in html
        assert "alert-warning" in html
        assert "alert-error" in html
        assert "alert-critical" in html
