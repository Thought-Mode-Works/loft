"""
Tests for meta-reasoning event bus.

Tests the publish/subscribe event system for cross-component communication.
"""

from datetime import datetime

import pytest

from loft.meta.event_bus import (
    ComponentType,
    EventType,
    MetaReasoningEvent,
    MetaReasoningEventBus,
    Subscription,
    create_event,
    create_event_bus,
    get_global_event_bus,
    reset_global_event_bus,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_observer_events_exist(self):
        """Test that observer event types exist."""
        assert EventType.CHAIN_OBSERVED.value == "chain_observed"
        assert EventType.PATTERN_DETECTED.value == "pattern_detected"
        assert EventType.BOTTLENECK_DETECTED.value == "bottleneck_detected"

    def test_strategy_events_exist(self):
        """Test that strategy event types exist."""
        assert EventType.STRATEGY_SELECTED.value == "strategy_selected"
        assert EventType.STRATEGY_CHANGED.value == "strategy_changed"
        assert EventType.STRATEGY_EVALUATED.value == "strategy_evaluated"

    def test_prompt_events_exist(self):
        """Test that prompt event types exist."""
        assert EventType.PROMPT_REGISTERED.value == "prompt_registered"
        assert EventType.PROMPT_UPDATED.value == "prompt_updated"
        assert EventType.PROMPT_RESULT_RECORDED.value == "prompt_result_recorded"
        assert EventType.AB_TEST_COMPLETED.value == "ab_test_completed"

    def test_failure_events_exist(self):
        """Test that failure event types exist."""
        assert EventType.FAILURE_RECORDED.value == "failure_recorded"
        assert EventType.FAILURE_PATTERN_FOUND.value == "failure_pattern_found"
        assert EventType.DIAGNOSIS_CREATED.value == "diagnosis_created"
        assert EventType.RECOMMENDATION_GENERATED.value == "recommendation_generated"

    def test_self_improvement_events_exist(self):
        """Test that self-improvement event types exist."""
        assert EventType.IMPROVEMENT_CYCLE_STARTED.value == "improvement_cycle_started"
        assert EventType.IMPROVEMENT_CYCLE_COMPLETED.value == "improvement_cycle_completed"
        assert EventType.IMPROVEMENT_APPLIED.value == "improvement_applied"
        assert EventType.GOAL_STATUS_CHANGED.value == "goal_status_changed"
        assert EventType.METRIC_RECORDED.value == "metric_recorded"

    def test_system_events_exist(self):
        """Test that system event types exist."""
        assert EventType.COMPONENT_INITIALIZED.value == "component_initialized"
        assert EventType.COMPONENT_ERROR.value == "component_error"


class TestComponentType:
    """Tests for ComponentType enum."""

    def test_all_components_exist(self):
        """Test that all expected component types exist."""
        assert ComponentType.OBSERVER.value == "observer"
        assert ComponentType.META_REASONER.value == "meta_reasoner"
        assert ComponentType.STRATEGY_EVALUATOR.value == "strategy_evaluator"
        assert ComponentType.STRATEGY_SELECTOR.value == "strategy_selector"
        assert ComponentType.PROMPT_OPTIMIZER.value == "prompt_optimizer"
        assert ComponentType.AB_TESTER.value == "ab_tester"
        assert ComponentType.FAILURE_ANALYZER.value == "failure_analyzer"
        assert ComponentType.RECOMMENDATION_ENGINE.value == "recommendation_engine"
        assert ComponentType.SELF_IMPROVEMENT_TRACKER.value == "self_improvement_tracker"
        assert ComponentType.AUTONOMOUS_IMPROVER.value == "autonomous_improver"
        assert ComponentType.EVENT_BUS.value == "event_bus"


class TestMetaReasoningEvent:
    """Tests for MetaReasoningEvent dataclass."""

    def test_event_creation(self):
        """Test creating an event with required fields."""
        event = MetaReasoningEvent(
            event_id="evt_123",
            event_type=EventType.FAILURE_RECORDED,
            source_component=ComponentType.FAILURE_ANALYZER,
            timestamp=datetime.now(),
            payload={"error_id": "err_001"},
        )
        assert event.event_id == "evt_123"
        assert event.event_type == EventType.FAILURE_RECORDED
        assert event.source_component == ComponentType.FAILURE_ANALYZER
        assert event.payload == {"error_id": "err_001"}
        assert event.correlation_id is None
        assert event.metadata == {}

    def test_event_with_optional_fields(self):
        """Test creating an event with optional fields."""
        event = MetaReasoningEvent(
            event_id="evt_456",
            event_type=EventType.STRATEGY_SELECTED,
            source_component=ComponentType.STRATEGY_SELECTOR,
            timestamp=datetime.now(),
            payload={"strategy": "rule_based"},
            correlation_id="corr_789",
            metadata={"version": "1.0"},
        )
        assert event.correlation_id == "corr_789"
        assert event.metadata == {"version": "1.0"}

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        timestamp = datetime.now()
        event = MetaReasoningEvent(
            event_id="evt_789",
            event_type=EventType.CHAIN_OBSERVED,
            source_component=ComponentType.OBSERVER,
            timestamp=timestamp,
            payload={"chain_id": "chain_001"},
            correlation_id="corr_123",
            metadata={"priority": "high"},
        )
        result = event.to_dict()
        assert result["event_id"] == "evt_789"
        assert result["event_type"] == "chain_observed"
        assert result["source_component"] == "observer"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["payload"] == {"chain_id": "chain_001"}
        assert result["correlation_id"] == "corr_123"
        assert result["metadata"] == {"priority": "high"}

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        timestamp = datetime.now()
        data = {
            "event_id": "evt_999",
            "event_type": "failure_recorded",
            "source_component": "failure_analyzer",
            "timestamp": timestamp.isoformat(),
            "payload": {"error_count": 5},
            "correlation_id": "corr_456",
            "metadata": {"batch": True},
        }
        event = MetaReasoningEvent.from_dict(data)
        assert event.event_id == "evt_999"
        assert event.event_type == EventType.FAILURE_RECORDED
        assert event.source_component == ComponentType.FAILURE_ANALYZER
        assert event.payload == {"error_count": 5}
        assert event.correlation_id == "corr_456"
        assert event.metadata == {"batch": True}

    def test_event_roundtrip(self):
        """Test that to_dict/from_dict roundtrip preserves data."""
        original = MetaReasoningEvent(
            event_id="evt_roundtrip",
            event_type=EventType.IMPROVEMENT_APPLIED,
            source_component=ComponentType.AUTONOMOUS_IMPROVER,
            timestamp=datetime.now(),
            payload={"improvement_id": "imp_001"},
            correlation_id="corr_roundtrip",
            metadata={"tested": True},
        )
        data = original.to_dict()
        restored = MetaReasoningEvent.from_dict(data)
        assert restored.event_id == original.event_id
        assert restored.event_type == original.event_type
        assert restored.source_component == original.source_component
        assert restored.payload == original.payload
        assert restored.correlation_id == original.correlation_id
        assert restored.metadata == original.metadata


class TestSubscription:
    """Tests for Subscription dataclass."""

    def test_subscription_creation(self):
        """Test creating a subscription."""

        def handler(event: MetaReasoningEvent) -> None:
            pass

        sub = Subscription(
            subscription_id="sub_123",
            event_type=EventType.FAILURE_RECORDED,
            handler=handler,
        )
        assert sub.subscription_id == "sub_123"
        assert sub.event_type == EventType.FAILURE_RECORDED
        assert sub.handler == handler
        assert sub.component is None
        assert sub.filter_fn is None

    def test_subscription_with_component(self):
        """Test subscription with component identifier."""

        def handler(event: MetaReasoningEvent) -> None:
            pass

        sub = Subscription(
            subscription_id="sub_456",
            event_type=EventType.STRATEGY_SELECTED,
            handler=handler,
            component=ComponentType.STRATEGY_EVALUATOR,
        )
        assert sub.component == ComponentType.STRATEGY_EVALUATOR

    def test_subscription_with_filter(self):
        """Test subscription with filter function."""

        def handler(event: MetaReasoningEvent) -> None:
            pass

        def filter_fn(event: MetaReasoningEvent) -> bool:
            return event.payload.get("priority") == "high"

        sub = Subscription(
            subscription_id="sub_789",
            event_type=EventType.FAILURE_RECORDED,
            handler=handler,
            filter_fn=filter_fn,
        )
        assert sub.filter_fn is not None


class TestMetaReasoningEventBus:
    """Tests for MetaReasoningEventBus class."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh event bus for testing."""
        return MetaReasoningEventBus()

    def test_event_bus_creation(self, event_bus):
        """Test creating an event bus."""
        assert event_bus.get_subscription_count() == 0
        assert event_bus.get_history() == []
        assert not event_bus.is_paused

    def test_event_bus_with_custom_history(self):
        """Test creating event bus with custom max history."""
        bus = MetaReasoningEventBus(max_history=10)
        assert bus._max_history == 10

    def test_subscribe_basic(self, event_bus):
        """Test basic subscription."""
        received_events = []

        def handler(event: MetaReasoningEvent) -> None:
            received_events.append(event)

        sub_id = event_bus.subscribe(EventType.FAILURE_RECORDED, handler)
        assert sub_id.startswith("sub_")
        assert event_bus.get_subscription_count() == 1
        assert event_bus.get_subscription_count(EventType.FAILURE_RECORDED) == 1

    def test_subscribe_with_component(self, event_bus):
        """Test subscription with component identifier."""

        def handler(event: MetaReasoningEvent) -> None:
            pass

        sub_id = event_bus.subscribe(
            EventType.STRATEGY_SELECTED,
            handler,
            component=ComponentType.STRATEGY_EVALUATOR,
        )
        assert sub_id.startswith("sub_")

    def test_subscribe_with_filter(self, event_bus):
        """Test subscription with filter function."""
        received_events = []

        def handler(event: MetaReasoningEvent) -> None:
            received_events.append(event)

        def filter_high_priority(event: MetaReasoningEvent) -> bool:
            return event.payload.get("priority") == "high"

        event_bus.subscribe(
            EventType.FAILURE_RECORDED,
            handler,
            filter_fn=filter_high_priority,
        )

        # Publish high priority event
        high_event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"priority": "high"},
        )
        event_bus.publish(high_event)

        # Publish low priority event
        low_event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"priority": "low"},
        )
        event_bus.publish(low_event)

        # Only high priority should be received
        assert len(received_events) == 1
        assert received_events[0].payload["priority"] == "high"

    def test_subscribe_all(self, event_bus):
        """Test subscribing to all event types."""
        received_events = []

        def handler(event: MetaReasoningEvent) -> None:
            received_events.append(event)

        sub_ids = event_bus.subscribe_all(handler)
        assert len(sub_ids) == len(EventType)
        assert event_bus.get_subscription_count() == len(EventType)

    def test_unsubscribe(self, event_bus):
        """Test unsubscribing."""

        def handler(event: MetaReasoningEvent) -> None:
            pass

        sub_id = event_bus.subscribe(EventType.FAILURE_RECORDED, handler)
        assert event_bus.get_subscription_count() == 1

        result = event_bus.unsubscribe(sub_id)
        assert result is True
        assert event_bus.get_subscription_count() == 0

    def test_unsubscribe_nonexistent(self, event_bus):
        """Test unsubscribing with invalid ID."""
        result = event_bus.unsubscribe("nonexistent_id")
        assert result is False

    def test_publish_basic(self, event_bus):
        """Test basic event publishing."""
        received_events = []

        def handler(event: MetaReasoningEvent) -> None:
            received_events.append(event)

        event_bus.subscribe(EventType.FAILURE_RECORDED, handler)

        event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"error_id": "err_001"},
        )
        handlers_called = event_bus.publish(event)

        assert handlers_called == 1
        assert len(received_events) == 1
        assert received_events[0].payload["error_id"] == "err_001"

    def test_publish_to_multiple_subscribers(self, event_bus):
        """Test publishing to multiple subscribers."""
        received_1 = []
        received_2 = []

        def handler_1(event: MetaReasoningEvent) -> None:
            received_1.append(event)

        def handler_2(event: MetaReasoningEvent) -> None:
            received_2.append(event)

        event_bus.subscribe(EventType.FAILURE_RECORDED, handler_1)
        event_bus.subscribe(EventType.FAILURE_RECORDED, handler_2)

        event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"test": True},
        )
        handlers_called = event_bus.publish(event)

        assert handlers_called == 2
        assert len(received_1) == 1
        assert len(received_2) == 1

    def test_publish_no_subscribers(self, event_bus):
        """Test publishing with no subscribers."""
        event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"test": True},
        )
        handlers_called = event_bus.publish(event)
        assert handlers_called == 0

    def test_publish_wrong_event_type(self, event_bus):
        """Test that handlers only receive correct event types."""
        received_events = []

        def handler(event: MetaReasoningEvent) -> None:
            received_events.append(event)

        event_bus.subscribe(EventType.FAILURE_RECORDED, handler)

        # Publish different event type
        event = create_event(
            EventType.STRATEGY_SELECTED,
            ComponentType.STRATEGY_SELECTOR,
            {"test": True},
        )
        event_bus.publish(event)

        assert len(received_events) == 0

    def test_publish_handler_error_continues(self, event_bus):
        """Test that handler errors don't stop other handlers."""
        received_events = []

        def error_handler(event: MetaReasoningEvent) -> None:
            raise RuntimeError("Handler error")

        def normal_handler(event: MetaReasoningEvent) -> None:
            received_events.append(event)

        event_bus.subscribe(EventType.FAILURE_RECORDED, error_handler)
        event_bus.subscribe(EventType.FAILURE_RECORDED, normal_handler)

        event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"test": True},
        )
        handlers_called = event_bus.publish(event)

        # Error handler counted but normal handler still runs
        assert handlers_called == 1  # Only successful handler counted
        assert len(received_events) == 1

    def test_get_history(self, event_bus):
        """Test getting event history."""
        event_1 = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"event": 1},
        )
        event_2 = create_event(
            EventType.STRATEGY_SELECTED,
            ComponentType.STRATEGY_SELECTOR,
            {"event": 2},
        )

        event_bus.publish(event_1)
        event_bus.publish(event_2)

        history = event_bus.get_history()
        assert len(history) == 2

    def test_get_history_by_event_type(self, event_bus):
        """Test filtering history by event type."""
        event_1 = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"event": 1},
        )
        event_2 = create_event(
            EventType.STRATEGY_SELECTED,
            ComponentType.STRATEGY_SELECTOR,
            {"event": 2},
        )

        event_bus.publish(event_1)
        event_bus.publish(event_2)

        history = event_bus.get_history(event_type=EventType.FAILURE_RECORDED)
        assert len(history) == 1
        assert history[0].event_type == EventType.FAILURE_RECORDED

    def test_get_history_by_component(self, event_bus):
        """Test filtering history by source component."""
        event_1 = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"event": 1},
        )
        event_2 = create_event(
            EventType.STRATEGY_SELECTED,
            ComponentType.STRATEGY_SELECTOR,
            {"event": 2},
        )

        event_bus.publish(event_1)
        event_bus.publish(event_2)

        history = event_bus.get_history(source_component=ComponentType.FAILURE_ANALYZER)
        assert len(history) == 1
        assert history[0].source_component == ComponentType.FAILURE_ANALYZER

    def test_get_history_since(self, event_bus):
        """Test filtering history by timestamp."""
        old_event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"event": "old"},
        )
        event_bus.publish(old_event)

        cutoff_time = datetime.now()

        new_event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"event": "new"},
        )
        event_bus.publish(new_event)

        history = event_bus.get_history(since=cutoff_time)
        assert len(history) == 1
        assert history[0].payload["event"] == "new"

    def test_get_history_with_limit(self, event_bus):
        """Test limiting history results."""
        for i in range(5):
            event = create_event(
                EventType.FAILURE_RECORDED,
                ComponentType.FAILURE_ANALYZER,
                {"event": i},
            )
            event_bus.publish(event)

        history = event_bus.get_history(limit=3)
        assert len(history) == 3
        # Should get last 3 events
        assert history[0].payload["event"] == 2
        assert history[1].payload["event"] == 3
        assert history[2].payload["event"] == 4

    def test_history_max_limit(self):
        """Test that history respects max limit."""
        bus = MetaReasoningEventBus(max_history=3)

        for i in range(5):
            event = create_event(
                EventType.FAILURE_RECORDED,
                ComponentType.FAILURE_ANALYZER,
                {"event": i},
            )
            bus.publish(event)

        history = bus.get_history()
        assert len(history) == 3
        # Should have last 3 events
        assert history[0].payload["event"] == 2

    def test_clear_history(self, event_bus):
        """Test clearing history."""
        for i in range(3):
            event = create_event(
                EventType.FAILURE_RECORDED,
                ComponentType.FAILURE_ANALYZER,
                {"event": i},
            )
            event_bus.publish(event)

        count = event_bus.clear_history()
        assert count == 3
        assert len(event_bus.get_history()) == 0

    def test_pause_and_resume(self, event_bus):
        """Test pausing and resuming event delivery."""
        received_events = []

        def handler(event: MetaReasoningEvent) -> None:
            received_events.append(event)

        event_bus.subscribe(EventType.FAILURE_RECORDED, handler)

        # Pause the bus
        event_bus.pause()
        assert event_bus.is_paused

        # Publish while paused
        event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"paused": True},
        )
        handlers_called = event_bus.publish(event)

        assert handlers_called == 0
        assert len(received_events) == 0

        # Resume and publish
        event_bus.resume()
        assert not event_bus.is_paused

        event_2 = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"resumed": True},
        )
        handlers_called = event_bus.publish(event_2)

        assert handlers_called == 1
        assert len(received_events) == 1


class TestCreateEvent:
    """Tests for create_event factory function."""

    def test_create_event_basic(self):
        """Test creating an event with factory function."""
        event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"error_id": "err_001"},
        )
        assert event.event_id.startswith("evt_")
        assert event.event_type == EventType.FAILURE_RECORDED
        assert event.source_component == ComponentType.FAILURE_ANALYZER
        assert event.payload == {"error_id": "err_001"}
        assert event.correlation_id is None
        assert event.metadata == {}
        assert isinstance(event.timestamp, datetime)

    def test_create_event_with_correlation_id(self):
        """Test creating event with correlation ID."""
        event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"test": True},
            correlation_id="corr_123",
        )
        assert event.correlation_id == "corr_123"

    def test_create_event_with_metadata(self):
        """Test creating event with metadata."""
        event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"test": True},
            metadata={"version": "1.0", "priority": "high"},
        )
        assert event.metadata == {"version": "1.0", "priority": "high"}


class TestCreateEventBus:
    """Tests for create_event_bus factory function."""

    def test_create_event_bus_default(self):
        """Test creating event bus with defaults."""
        bus = create_event_bus()
        assert isinstance(bus, MetaReasoningEventBus)
        assert bus._max_history == 1000

    def test_create_event_bus_custom_history(self):
        """Test creating event bus with custom history."""
        bus = create_event_bus(max_history=500)
        assert bus._max_history == 500


class TestGlobalEventBus:
    """Tests for global event bus functions."""

    def test_get_global_event_bus(self):
        """Test getting global event bus."""
        reset_global_event_bus()
        bus = get_global_event_bus()
        assert isinstance(bus, MetaReasoningEventBus)

    def test_global_event_bus_singleton(self):
        """Test that global event bus is a singleton."""
        reset_global_event_bus()
        bus_1 = get_global_event_bus()
        bus_2 = get_global_event_bus()
        assert bus_1 is bus_2

    def test_reset_global_event_bus(self):
        """Test resetting global event bus."""
        reset_global_event_bus()
        bus_1 = get_global_event_bus()
        reset_global_event_bus()
        bus_2 = get_global_event_bus()
        assert bus_1 is not bus_2


class TestEventBusIntegration:
    """Integration tests for event bus with multiple components."""

    def test_multi_component_communication(self):
        """Test communication between multiple components via event bus."""
        bus = MetaReasoningEventBus()
        observer_events = []
        analyzer_events = []

        def observer_handler(event: MetaReasoningEvent) -> None:
            observer_events.append(event)

        def analyzer_handler(event: MetaReasoningEvent) -> None:
            analyzer_events.append(event)

        # Observer subscribes to strategy events
        bus.subscribe(
            EventType.STRATEGY_SELECTED,
            observer_handler,
            component=ComponentType.OBSERVER,
        )

        # Analyzer subscribes to failure events
        bus.subscribe(
            EventType.FAILURE_RECORDED,
            analyzer_handler,
            component=ComponentType.FAILURE_ANALYZER,
        )

        # Selector publishes strategy event
        strategy_event = create_event(
            EventType.STRATEGY_SELECTED,
            ComponentType.STRATEGY_SELECTOR,
            {"strategy": "rule_based"},
        )
        bus.publish(strategy_event)

        # Observer publishes failure event
        failure_event = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.OBSERVER,
            {"chain_id": "chain_001"},
        )
        bus.publish(failure_event)

        assert len(observer_events) == 1
        assert observer_events[0].payload["strategy"] == "rule_based"
        assert len(analyzer_events) == 1
        assert analyzer_events[0].payload["chain_id"] == "chain_001"

    def test_event_correlation(self):
        """Test correlating related events."""
        bus = MetaReasoningEventBus()
        correlation_id = "workflow_001"

        # Publish related events with same correlation ID
        event_1 = create_event(
            EventType.CHAIN_OBSERVED,
            ComponentType.OBSERVER,
            {"chain_id": "chain_001"},
            correlation_id=correlation_id,
        )
        event_2 = create_event(
            EventType.FAILURE_RECORDED,
            ComponentType.FAILURE_ANALYZER,
            {"error_id": "err_001"},
            correlation_id=correlation_id,
        )
        event_3 = create_event(
            EventType.RECOMMENDATION_GENERATED,
            ComponentType.RECOMMENDATION_ENGINE,
            {"recommendation_id": "rec_001"},
            correlation_id=correlation_id,
        )

        bus.publish(event_1)
        bus.publish(event_2)
        bus.publish(event_3)

        # All events should be in history
        history = bus.get_history()
        correlated = [e for e in history if e.correlation_id == correlation_id]
        assert len(correlated) == 3
