"""
Tests for meta-reasoning integration adapters.

Tests the adapters that enable cross-component data flow.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from loft.meta.adapters import (
    EventDrivenIntegration,
    FailureToImprovementAdapter,
    ObserverToFailureAdapter,
    create_event_driven_integration,
    create_failure_to_improvement_adapter,
    create_observer_to_failure_adapter,
)
from loft.meta.event_bus import (
    ComponentType,
    EventType,
    MetaReasoningEventBus,
    create_event,
)
from loft.meta.failure_analyzer import (
    FailureAnalyzer,
    Recommendation,
    RecommendationCategory,
    RecommendationEngine,
)
from loft.meta.schemas import ImprovementPriority
from loft.meta.observer import ReasoningObserver
from loft.meta.schemas import ReasoningChain, ReasoningStep, ReasoningStepType
from loft.meta.self_improvement import (
    ActionType,
    ImprovementGoal,
    MetricType,
    SelfImprovementTracker,
)


def create_test_chain(
    chain_id: str = "chain_001",
    case_id: str = "case_001",
    domain: str = "contracts",
    success: bool = False,
    prediction: str = "plaintiff",
    ground_truth: str = "defendant",
) -> ReasoningChain:
    """Create a test reasoning chain."""
    now = datetime.now()
    step = ReasoningStep(
        step_id="step_001",
        step_type=ReasoningStepType.TRANSLATION,
        description="Test step",
        input_data={"input": "test"},
        output_data={"output": "test"},
        started_at=now,
        completed_at=now,
        success=success,
    )
    return ReasoningChain(
        chain_id=chain_id,
        case_id=case_id,
        domain=domain,
        steps=[step],
        prediction=prediction,
        ground_truth=ground_truth,
        overall_success=success,
        started_at=now,
        completed_at=now,
    )


def create_test_recommendation(
    recommendation_id: str = "rec_001",
    priority: ImprovementPriority = ImprovementPriority.MEDIUM,
    category: RecommendationCategory = RecommendationCategory.RULE_MODIFICATION,
) -> Recommendation:
    """Create a test recommendation."""
    return Recommendation(
        recommendation_id=recommendation_id,
        category=category,
        title="Test Recommendation",
        description="Test recommendation description",
        priority=priority,
        expected_impact=0.15,  # 15% improvement
        implementation_steps=["Step 1", "Step 2"],
    )


def create_test_goal(goal_id: str = "goal_001") -> ImprovementGoal:
    """Create a test improvement goal."""
    return ImprovementGoal(
        goal_id=goal_id,
        metric_type=MetricType.ACCURACY,
        target_value=0.9,
        baseline_value=0.7,
        current_value=0.8,
        description="Test goal description",
    )


class TestObserverToFailureAdapter:
    """Tests for ObserverToFailureAdapter class."""

    @pytest.fixture
    def observer(self):
        """Create a mock observer."""
        observer = MagicMock(spec=ReasoningObserver)
        return observer

    @pytest.fixture
    def analyzer(self):
        """Create a mock analyzer."""
        analyzer = MagicMock(spec=FailureAnalyzer)
        return analyzer

    @pytest.fixture
    def event_bus(self):
        """Create an event bus."""
        return MetaReasoningEventBus()

    def test_adapter_creation(self, observer, analyzer):
        """Test creating an adapter."""
        adapter = ObserverToFailureAdapter(observer, analyzer)
        assert adapter.observer is observer
        assert adapter.analyzer is analyzer
        assert adapter.processed_count == 0

    def test_adapter_with_event_bus(self, observer, analyzer, event_bus):
        """Test creating adapter with event bus."""
        adapter = ObserverToFailureAdapter(observer, analyzer, event_bus)
        assert adapter._event_bus is event_bus

    def test_process_failed_chains_basic(self, observer, analyzer):
        """Test processing failed chains."""
        failed_chain = create_test_chain(success=False)
        observer.chains = {failed_chain.chain_id: failed_chain}

        adapter = ObserverToFailureAdapter(observer, analyzer)
        errors = adapter.process_failed_chains()

        assert len(errors) == 1
        assert adapter.processed_count == 1
        analyzer.record_error.assert_called_once()

    def test_process_failed_chains_skips_successful(self, observer, analyzer):
        """Test that successful chains are skipped."""
        successful_chain = create_test_chain(chain_id="success", success=True)
        failed_chain = create_test_chain(chain_id="failed", success=False)
        observer.chains = {
            successful_chain.chain_id: successful_chain,
            failed_chain.chain_id: failed_chain,
        }

        adapter = ObserverToFailureAdapter(observer, analyzer)
        errors = adapter.process_failed_chains()

        assert len(errors) == 1
        assert adapter.processed_count == 1

    def test_process_failed_chains_with_domain_filter(self, observer, analyzer):
        """Test filtering by domain."""
        adapter = ObserverToFailureAdapter(observer, analyzer)
        observer.get_chains_by_domain.return_value = []

        adapter.process_failed_chains(domain="contracts")
        observer.get_chains_by_domain.assert_called_with("contracts")

    def test_process_failed_chains_with_limit(self, observer, analyzer):
        """Test limiting processed chains."""
        chains = [
            create_test_chain(chain_id=f"chain_{i}", success=False) for i in range(5)
        ]
        observer.chains = {chain.chain_id: chain for chain in chains}

        adapter = ObserverToFailureAdapter(observer, analyzer)
        errors = adapter.process_failed_chains(limit=2)

        assert len(errors) == 2
        assert adapter.processed_count == 2

    def test_process_failed_chains_skips_already_processed(self, observer, analyzer):
        """Test that already processed chains are skipped."""
        chain = create_test_chain(success=False)
        observer.chains = {chain.chain_id: chain}

        adapter = ObserverToFailureAdapter(observer, analyzer)

        # Process first time
        errors_1 = adapter.process_failed_chains()
        assert len(errors_1) == 1

        # Process again - should skip
        errors_2 = adapter.process_failed_chains()
        assert len(errors_2) == 0
        assert adapter.processed_count == 1

    def test_process_failed_chains_publishes_events(
        self, observer, analyzer, event_bus
    ):
        """Test that events are published when event bus is connected."""
        failed_chain = create_test_chain(success=False)
        observer.chains = {failed_chain.chain_id: failed_chain}

        received_events = []

        def handler(event):
            received_events.append(event)

        event_bus.subscribe(EventType.FAILURE_RECORDED, handler)

        adapter = ObserverToFailureAdapter(observer, analyzer, event_bus)
        adapter.process_failed_chains()

        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.FAILURE_RECORDED

    def test_get_unprocessed_failures(self, observer, analyzer):
        """Test getting unprocessed failed chains."""
        chains = [
            create_test_chain(chain_id="chain_1", success=False),
            create_test_chain(chain_id="chain_2", success=False),
        ]
        observer.chains = {chain.chain_id: chain for chain in chains}

        adapter = ObserverToFailureAdapter(observer, analyzer)

        # All should be unprocessed initially
        unprocessed = adapter.get_unprocessed_failures()
        assert len(unprocessed) == 2

        # Process one
        adapter.process_failed_chains(limit=1)

        # Check unprocessed again
        unprocessed = adapter.get_unprocessed_failures()
        assert len(unprocessed) == 1

    def test_reset_processed_tracking(self, observer, analyzer):
        """Test resetting processed chain tracking."""
        chains = [create_test_chain(success=False)]
        observer.chains = {chain.chain_id: chain for chain in chains}

        adapter = ObserverToFailureAdapter(observer, analyzer)
        adapter.process_failed_chains()
        assert adapter.processed_count == 1

        cleared = adapter.reset_processed_tracking()
        assert cleared == 1
        assert adapter.processed_count == 0


class TestFailureToImprovementAdapter:
    """Tests for FailureToImprovementAdapter class."""

    @pytest.fixture
    def recommendation_engine(self):
        """Create a mock recommendation engine."""
        engine = MagicMock(spec=RecommendationEngine)
        return engine

    @pytest.fixture
    def tracker(self):
        """Create a mock tracker."""
        tracker = MagicMock(spec=SelfImprovementTracker)
        return tracker

    @pytest.fixture
    def event_bus(self):
        """Create an event bus."""
        return MetaReasoningEventBus()

    @pytest.fixture
    def goal(self):
        """Create a test goal."""
        return create_test_goal()

    def test_adapter_creation(self, recommendation_engine, tracker):
        """Test creating an adapter."""
        adapter = FailureToImprovementAdapter(recommendation_engine, tracker)
        assert adapter.recommendation_engine is recommendation_engine
        assert adapter.tracker is tracker
        assert adapter.conversion_count == 0

    def test_adapter_with_event_bus(self, recommendation_engine, tracker, event_bus):
        """Test creating adapter with event bus."""
        adapter = FailureToImprovementAdapter(recommendation_engine, tracker, event_bus)
        assert adapter._event_bus is event_bus

    def test_convert_recommendations_to_actions(
        self, recommendation_engine, tracker, goal
    ):
        """Test converting recommendations to actions."""
        recommendation = create_test_recommendation()
        recommendation_engine._recommendations = {
            recommendation.recommendation_id: recommendation
        }

        adapter = FailureToImprovementAdapter(recommendation_engine, tracker)
        actions = adapter.convert_recommendations_to_actions(goal)

        assert len(actions) == 1
        assert adapter.conversion_count == 1
        assert actions[0].action_id == f"act_{recommendation.recommendation_id}"

    def test_convert_recommendations_with_priority_filter(
        self, recommendation_engine, tracker, goal
    ):
        """Test converting multiple recommendations."""
        low_priority_rec = create_test_recommendation(
            recommendation_id="rec_low", priority=ImprovementPriority.LOW
        )
        high_priority_rec = create_test_recommendation(
            recommendation_id="rec_high", priority=ImprovementPriority.HIGH
        )
        recommendation_engine._recommendations = {
            low_priority_rec.recommendation_id: low_priority_rec,
            high_priority_rec.recommendation_id: high_priority_rec,
        }

        adapter = FailureToImprovementAdapter(recommendation_engine, tracker)
        actions = adapter.convert_recommendations_to_actions(goal)

        # Both should be converted (no priority filtering in current implementation)
        assert len(actions) == 2

    def test_convert_recommendations_with_limit(
        self, recommendation_engine, tracker, goal
    ):
        """Test limiting conversions."""
        recommendations = [
            create_test_recommendation(recommendation_id=f"rec_{i}") for i in range(5)
        ]
        recommendation_engine._recommendations = {
            rec.recommendation_id: rec for rec in recommendations
        }

        adapter = FailureToImprovementAdapter(recommendation_engine, tracker)
        actions = adapter.convert_recommendations_to_actions(goal, limit=2)

        assert len(actions) == 2

    def test_convert_recommendations_skips_already_converted(
        self, recommendation_engine, tracker, goal
    ):
        """Test that already converted recommendations are skipped."""
        recommendation = create_test_recommendation()
        recommendation_engine._recommendations = {
            recommendation.recommendation_id: recommendation
        }

        adapter = FailureToImprovementAdapter(recommendation_engine, tracker)

        # Convert first time
        actions_1 = adapter.convert_recommendations_to_actions(goal)
        assert len(actions_1) == 1

        # Convert again - should skip
        actions_2 = adapter.convert_recommendations_to_actions(goal)
        assert len(actions_2) == 0
        assert adapter.conversion_count == 1

    def test_action_type_mapping(self, recommendation_engine, tracker, goal):
        """Test that recommendation categories map to action types."""
        test_cases = [
            (RecommendationCategory.RULE_ADDITION, ActionType.RULE_MODIFICATION),
            (RecommendationCategory.RULE_MODIFICATION, ActionType.RULE_MODIFICATION),
            (RecommendationCategory.PROMPT_IMPROVEMENT, ActionType.PROMPT_REFINEMENT),
            (
                RecommendationCategory.VALIDATION_ENHANCEMENT,
                ActionType.VALIDATION_ENHANCEMENT,
            ),
            (RecommendationCategory.DOMAIN_EXPANSION, ActionType.STRATEGY_ADJUSTMENT),
            (RecommendationCategory.DATA_AUGMENTATION, ActionType.STRATEGY_ADJUSTMENT),
            (
                RecommendationCategory.STRATEGY_ADJUSTMENT,
                ActionType.STRATEGY_ADJUSTMENT,
            ),
        ]

        for category, expected_action_type in test_cases:
            adapter = FailureToImprovementAdapter(recommendation_engine, tracker)
            recommendation = create_test_recommendation(
                recommendation_id=f"rec_{category.value}", category=category
            )
            recommendation_engine._recommendations = {
                recommendation.recommendation_id: recommendation
            }

            actions = adapter.convert_recommendations_to_actions(goal)
            assert len(actions) == 1
            assert actions[0].action_type == expected_action_type

    def test_risk_level_calculation(self, recommendation_engine, tracker, goal):
        """Test risk level calculation based on priority."""
        test_cases = [
            (ImprovementPriority.CRITICAL, "high"),
            (ImprovementPriority.HIGH, "high"),
            (ImprovementPriority.MEDIUM, "medium"),
            (ImprovementPriority.LOW, "low"),
        ]

        for priority, expected_risk in test_cases:
            adapter = FailureToImprovementAdapter(recommendation_engine, tracker)
            recommendation = create_test_recommendation(
                recommendation_id=f"rec_{priority.value}", priority=priority
            )
            recommendation_engine._recommendations = {
                recommendation.recommendation_id: recommendation
            }

            actions = adapter.convert_recommendations_to_actions(goal)
            assert len(actions) == 1
            assert actions[0].parameters["risk_level"] == expected_risk

    def test_requires_approval_for_high_priority(
        self, recommendation_engine, tracker, goal
    ):
        """Test that high priority actions require approval."""
        high_priority_rec = create_test_recommendation(
            priority=ImprovementPriority.HIGH
        )
        low_priority_rec = create_test_recommendation(
            recommendation_id="rec_low", priority=ImprovementPriority.LOW
        )
        recommendation_engine._recommendations = {
            high_priority_rec.recommendation_id: high_priority_rec,
            low_priority_rec.recommendation_id: low_priority_rec,
        }

        adapter = FailureToImprovementAdapter(recommendation_engine, tracker)
        actions = adapter.convert_recommendations_to_actions(goal)

        assert len(actions) == 2
        high_action = next(a for a in actions if "rec_001" in a.action_id)
        low_action = next(a for a in actions if "rec_low" in a.action_id)

        assert high_action.parameters["requires_approval"] is True
        assert low_action.parameters["requires_approval"] is False

    def test_get_action_for_recommendation(self, recommendation_engine, tracker, goal):
        """Test looking up action ID for recommendation."""
        recommendation = create_test_recommendation()
        recommendation_engine._recommendations = {
            recommendation.recommendation_id: recommendation
        }

        adapter = FailureToImprovementAdapter(recommendation_engine, tracker)
        adapter.convert_recommendations_to_actions(goal)

        action_id = adapter.get_action_for_recommendation(
            recommendation.recommendation_id
        )
        assert action_id is not None
        assert action_id == f"act_{recommendation.recommendation_id}"

        # Non-existent recommendation
        assert adapter.get_action_for_recommendation("nonexistent") is None

    def test_reset_conversion_tracking(self, recommendation_engine, tracker, goal):
        """Test resetting conversion tracking."""
        recommendation = create_test_recommendation()
        recommendation_engine._recommendations = {
            recommendation.recommendation_id: recommendation
        }

        adapter = FailureToImprovementAdapter(recommendation_engine, tracker)
        adapter.convert_recommendations_to_actions(goal)
        assert adapter.conversion_count == 1

        cleared = adapter.reset_conversion_tracking()
        assert cleared == 1
        assert adapter.conversion_count == 0


class TestEventDrivenIntegration:
    """Tests for EventDrivenIntegration class."""

    @pytest.fixture
    def event_bus(self):
        """Create an event bus."""
        return MetaReasoningEventBus()

    @pytest.fixture
    def observer(self):
        """Create a mock observer."""
        return MagicMock(spec=ReasoningObserver)

    @pytest.fixture
    def analyzer(self):
        """Create a mock analyzer."""
        return MagicMock(spec=FailureAnalyzer)

    @pytest.fixture
    def recommendation_engine(self):
        """Create a mock recommendation engine."""
        return MagicMock(spec=RecommendationEngine)

    @pytest.fixture
    def tracker(self):
        """Create a mock tracker."""
        return MagicMock(spec=SelfImprovementTracker)

    def test_integration_creation(self, event_bus):
        """Test creating integration coordinator."""
        integration = EventDrivenIntegration(event_bus)
        assert integration.event_bus is event_bus
        assert integration.subscription_count == 0

    def test_setup_observer_to_failure_flow(self, event_bus, observer, analyzer):
        """Test setting up observer to failure flow."""
        integration = EventDrivenIntegration(event_bus)
        adapter = integration.setup_observer_to_failure_flow(observer, analyzer)

        assert isinstance(adapter, ObserverToFailureAdapter)
        assert adapter.observer is observer
        assert adapter.analyzer is analyzer
        assert integration.subscription_count == 1

    def test_setup_observer_to_failure_flow_no_auto(
        self, event_bus, observer, analyzer
    ):
        """Test setting up flow without auto-processing."""
        integration = EventDrivenIntegration(event_bus)
        adapter = integration.setup_observer_to_failure_flow(
            observer, analyzer, auto_process=False
        )

        assert isinstance(adapter, ObserverToFailureAdapter)
        assert integration.subscription_count == 0

    def test_setup_failure_to_improvement_flow(
        self, event_bus, recommendation_engine, tracker
    ):
        """Test setting up failure to improvement flow."""
        goal = create_test_goal()
        integration = EventDrivenIntegration(event_bus)
        adapter = integration.setup_failure_to_improvement_flow(
            recommendation_engine, tracker, goal=goal
        )

        assert isinstance(adapter, FailureToImprovementAdapter)
        assert integration.subscription_count == 1

    def test_setup_failure_to_improvement_flow_no_auto(
        self, event_bus, recommendation_engine, tracker
    ):
        """Test setting up flow without auto-conversion."""
        integration = EventDrivenIntegration(event_bus)
        adapter = integration.setup_failure_to_improvement_flow(
            recommendation_engine, tracker, auto_convert=False
        )

        assert isinstance(adapter, FailureToImprovementAdapter)
        assert integration.subscription_count == 0

    def test_get_adapter(self, event_bus, observer, analyzer):
        """Test getting adapters by name."""
        integration = EventDrivenIntegration(event_bus)

        # No adapters yet
        assert integration.get_adapter("observer_to_failure") is None

        # Setup adapter
        adapter = integration.setup_observer_to_failure_flow(observer, analyzer)

        # Now should be retrievable
        retrieved = integration.get_adapter("observer_to_failure")
        assert retrieved is adapter

    def test_teardown(self, event_bus, observer, analyzer):
        """Test tearing down integration."""
        integration = EventDrivenIntegration(event_bus)
        integration.setup_observer_to_failure_flow(observer, analyzer)
        assert integration.subscription_count == 1

        removed = integration.teardown()
        assert removed == 1
        assert integration.subscription_count == 0
        assert integration.get_adapter("observer_to_failure") is None


class TestFactoryFunctions:
    """Tests for adapter factory functions."""

    def test_create_observer_to_failure_adapter(self):
        """Test factory function for ObserverToFailureAdapter."""
        observer = MagicMock(spec=ReasoningObserver)
        analyzer = MagicMock(spec=FailureAnalyzer)

        adapter = create_observer_to_failure_adapter(observer, analyzer)

        assert isinstance(adapter, ObserverToFailureAdapter)
        assert adapter.observer is observer
        assert adapter.analyzer is analyzer

    def test_create_observer_to_failure_adapter_with_bus(self):
        """Test factory with event bus."""
        observer = MagicMock(spec=ReasoningObserver)
        analyzer = MagicMock(spec=FailureAnalyzer)
        bus = MetaReasoningEventBus()

        adapter = create_observer_to_failure_adapter(observer, analyzer, bus)

        assert adapter._event_bus is bus

    def test_create_failure_to_improvement_adapter(self):
        """Test factory function for FailureToImprovementAdapter."""
        engine = MagicMock(spec=RecommendationEngine)
        tracker = MagicMock(spec=SelfImprovementTracker)

        adapter = create_failure_to_improvement_adapter(engine, tracker)

        assert isinstance(adapter, FailureToImprovementAdapter)
        assert adapter.recommendation_engine is engine
        assert adapter.tracker is tracker

    def test_create_failure_to_improvement_adapter_with_bus(self):
        """Test factory with event bus."""
        engine = MagicMock(spec=RecommendationEngine)
        tracker = MagicMock(spec=SelfImprovementTracker)
        bus = MetaReasoningEventBus()

        adapter = create_failure_to_improvement_adapter(engine, tracker, bus)

        assert adapter._event_bus is bus

    def test_create_event_driven_integration(self):
        """Test factory function for EventDrivenIntegration."""
        bus = MetaReasoningEventBus()

        integration = create_event_driven_integration(bus)

        assert isinstance(integration, EventDrivenIntegration)
        assert integration.event_bus is bus


class TestAdaptersIntegration:
    """Integration tests for adapters working together."""

    def test_full_pipeline_flow(self):
        """Test the full pipeline from observer to improvement."""
        # Create components
        observer = ReasoningObserver()
        analyzer = FailureAnalyzer()
        engine = RecommendationEngine(analyzer)
        tracker = SelfImprovementTracker()
        bus = MetaReasoningEventBus()

        # Create adapters
        o2f_adapter = ObserverToFailureAdapter(observer, analyzer, bus)
        # Verify adapters can be created with same bus
        FailureToImprovementAdapter(engine, tracker, bus)

        # Record a failed chain in the observer using observe_reasoning_chain
        failed_chain = create_test_chain(success=False)
        observer.observe_reasoning_chain(
            case_id=failed_chain.case_id,
            domain=failed_chain.domain,
            steps=failed_chain.steps,
            prediction=failed_chain.prediction,
            ground_truth=failed_chain.ground_truth,
        )

        # Process failed chains
        errors = o2f_adapter.process_failed_chains()
        assert len(errors) == 1

        # The analyzer now has an error recorded
        # In a real scenario, we would analyze patterns and generate recommendations

    def test_event_driven_workflow(self):
        """Test event-driven workflow between components."""
        bus = MetaReasoningEventBus()
        events_received = []

        def capture_events(event):
            events_received.append(event)

        # Subscribe to all relevant events
        bus.subscribe(EventType.CHAIN_OBSERVED, capture_events)
        bus.subscribe(EventType.FAILURE_RECORDED, capture_events)
        bus.subscribe(EventType.RECOMMENDATION_GENERATED, capture_events)

        # Simulate workflow events
        chain_event = create_event(
            EventType.CHAIN_OBSERVED,
            ComponentType.OBSERVER,
            {"chain_id": "chain_001", "overall_success": False},
        )
        bus.publish(chain_event)

        failure_event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"error_id": "err_001"},
            correlation_id="chain_001",
        )
        bus.publish(failure_event)

        rec_event = create_event(
            EventType.RECOMMENDATION_GENERATED,
            ComponentType.RECOMMENDATION_ENGINE,
            {"recommendation_id": "rec_001"},
            correlation_id="chain_001",
        )
        bus.publish(rec_event)

        # All events should be received
        assert len(events_received) == 3

        # Check event correlation
        history = bus.get_history()
        correlated = [e for e in history if e.correlation_id == "chain_001"]
        assert len(correlated) == 2  # failure and rec events have correlation
