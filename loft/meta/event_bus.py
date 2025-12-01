"""
Event bus for cross-component communication in meta-reasoning.

Provides a publish/subscribe event system that enables loose coupling
between meta-reasoning components (Observer, Strategy, Prompt, Failure, Self-Improvement).
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(Enum):
    """Types of events that can be published in the meta-reasoning system."""

    # Observer events
    CHAIN_OBSERVED = "chain_observed"
    PATTERN_DETECTED = "pattern_detected"
    BOTTLENECK_DETECTED = "bottleneck_detected"

    # Strategy events
    STRATEGY_SELECTED = "strategy_selected"
    STRATEGY_CHANGED = "strategy_changed"
    STRATEGY_EVALUATED = "strategy_evaluated"

    # Prompt events
    PROMPT_REGISTERED = "prompt_registered"
    PROMPT_UPDATED = "prompt_updated"
    PROMPT_RESULT_RECORDED = "prompt_result_recorded"
    AB_TEST_COMPLETED = "ab_test_completed"

    # Failure events
    FAILURE_RECORDED = "failure_recorded"
    FAILURE_PATTERN_FOUND = "failure_pattern_found"
    DIAGNOSIS_CREATED = "diagnosis_created"
    RECOMMENDATION_GENERATED = "recommendation_generated"

    # Self-Improvement events
    IMPROVEMENT_CYCLE_STARTED = "improvement_cycle_started"
    IMPROVEMENT_CYCLE_COMPLETED = "improvement_cycle_completed"
    IMPROVEMENT_APPLIED = "improvement_applied"
    GOAL_STATUS_CHANGED = "goal_status_changed"
    METRIC_RECORDED = "metric_recorded"

    # System events
    COMPONENT_INITIALIZED = "component_initialized"
    COMPONENT_ERROR = "component_error"


class ComponentType(Enum):
    """Types of meta-reasoning components."""

    OBSERVER = "observer"
    META_REASONER = "meta_reasoner"
    STRATEGY_EVALUATOR = "strategy_evaluator"
    STRATEGY_SELECTOR = "strategy_selector"
    PROMPT_OPTIMIZER = "prompt_optimizer"
    AB_TESTER = "ab_tester"
    FAILURE_ANALYZER = "failure_analyzer"
    RECOMMENDATION_ENGINE = "recommendation_engine"
    SELF_IMPROVEMENT_TRACKER = "self_improvement_tracker"
    AUTONOMOUS_IMPROVER = "autonomous_improver"
    EVENT_BUS = "event_bus"


@dataclass
class MetaReasoningEvent:
    """An event published by a meta-reasoning component.

    Events carry information about significant occurrences in the
    meta-reasoning system, enabling components to react to changes
    in other parts of the system without direct coupling.
    """

    event_id: str
    event_type: EventType
    source_component: ComponentType
    timestamp: datetime
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source_component": self.source_component.value,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaReasoningEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            source_component=ComponentType(data["source_component"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            payload=data.get("payload", {}),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )


# Type alias for event handlers
EventHandler = Callable[[MetaReasoningEvent], None]


@dataclass
class Subscription:
    """A subscription to events."""

    subscription_id: str
    event_type: EventType
    handler: EventHandler
    component: Optional[ComponentType] = None
    filter_fn: Optional[Callable[[MetaReasoningEvent], bool]] = None
    created_at: datetime = field(default_factory=datetime.now)


class MetaReasoningEventBus:
    """Central event bus for meta-reasoning component communication.

    The event bus enables loose coupling between components by providing
    a publish/subscribe mechanism for events. Components can:
    - Publish events when significant things happen
    - Subscribe to events they care about
    - Filter events based on criteria
    - Access event history for debugging/auditing

    Example:
        >>> bus = MetaReasoningEventBus()
        >>> def handle_failure(event):
        ...     print(f"Failure recorded: {event.payload}")
        >>> bus.subscribe(EventType.FAILURE_RECORDED, handle_failure)
        >>> bus.publish(create_event(
        ...     EventType.FAILURE_RECORDED,
        ...     ComponentType.FAILURE_ANALYZER,
        ...     {"error_id": "err_001", "domain": "contracts"}
        ... ))
    """

    def __init__(self, max_history: int = 1000):
        """Initialize the event bus.

        Args:
            max_history: Maximum number of events to keep in history
        """
        self._subscriptions: Dict[EventType, List[Subscription]] = defaultdict(list)
        self._history: List[MetaReasoningEvent] = []
        self._max_history = max_history
        self._paused = False

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        component: Optional[ComponentType] = None,
        filter_fn: Optional[Callable[[MetaReasoningEvent], bool]] = None,
    ) -> str:
        """Subscribe to events of a specific type.

        Args:
            event_type: The type of events to subscribe to
            handler: Function to call when event is published
            component: Optional component identifier for the subscriber
            filter_fn: Optional function to filter events (return True to receive)

        Returns:
            Subscription ID that can be used to unsubscribe
        """
        subscription_id = f"sub_{uuid.uuid4().hex[:12]}"
        subscription = Subscription(
            subscription_id=subscription_id,
            event_type=event_type,
            handler=handler,
            component=component,
            filter_fn=filter_fn,
        )
        self._subscriptions[event_type].append(subscription)
        return subscription_id

    def subscribe_all(
        self,
        handler: EventHandler,
        component: Optional[ComponentType] = None,
        filter_fn: Optional[Callable[[MetaReasoningEvent], bool]] = None,
    ) -> List[str]:
        """Subscribe to all event types.

        Args:
            handler: Function to call when any event is published
            component: Optional component identifier for the subscriber
            filter_fn: Optional function to filter events

        Returns:
            List of subscription IDs
        """
        subscription_ids = []
        for event_type in EventType:
            sub_id = self.subscribe(event_type, handler, component, filter_fn)
            subscription_ids.append(sub_id)
        return subscription_ids

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.

        Args:
            subscription_id: The subscription ID returned from subscribe()

        Returns:
            True if subscription was found and removed
        """
        for event_type, subscriptions in self._subscriptions.items():
            for i, sub in enumerate(subscriptions):
                if sub.subscription_id == subscription_id:
                    subscriptions.pop(i)
                    return True
        return False

    def publish(self, event: MetaReasoningEvent) -> int:
        """Publish an event to all subscribers.

        Args:
            event: The event to publish

        Returns:
            Number of handlers that received the event
        """
        if self._paused:
            return 0

        # Add to history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Notify subscribers
        handlers_called = 0
        subscriptions = self._subscriptions.get(event.event_type, [])

        for subscription in subscriptions:
            # Apply filter if present
            if subscription.filter_fn is not None:
                if not subscription.filter_fn(event):
                    continue

            try:
                subscription.handler(event)
                handlers_called += 1
            except Exception:
                # Log error but don't stop other handlers
                # In production, this would go to a proper logger
                pass

        return handlers_called

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        source_component: Optional[ComponentType] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[MetaReasoningEvent]:
        """Get event history for auditing/debugging.

        Args:
            event_type: Filter by event type
            source_component: Filter by source component
            since: Only events after this timestamp
            limit: Maximum number of events to return

        Returns:
            List of events matching criteria
        """
        events = self._history

        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        if source_component is not None:
            events = [e for e in events if e.source_component == source_component]

        if since is not None:
            events = [e for e in events if e.timestamp >= since]

        if limit is not None:
            events = events[-limit:]

        return events

    def get_subscription_count(self, event_type: Optional[EventType] = None) -> int:
        """Get the number of active subscriptions.

        Args:
            event_type: Optionally filter by event type

        Returns:
            Number of subscriptions
        """
        if event_type is not None:
            return len(self._subscriptions.get(event_type, []))
        return sum(len(subs) for subs in self._subscriptions.values())

    def clear_history(self) -> int:
        """Clear event history.

        Returns:
            Number of events cleared
        """
        count = len(self._history)
        self._history.clear()
        return count

    def pause(self) -> None:
        """Pause event delivery (events will be dropped)."""
        self._paused = True

    def resume(self) -> None:
        """Resume event delivery."""
        self._paused = False

    @property
    def is_paused(self) -> bool:
        """Check if event delivery is paused."""
        return self._paused


def create_event(
    event_type: EventType,
    source_component: ComponentType,
    payload: Dict[str, Any],
    correlation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MetaReasoningEvent:
    """Factory function to create a MetaReasoningEvent.

    Args:
        event_type: Type of the event
        source_component: Component publishing the event
        payload: Event data
        correlation_id: Optional ID to correlate related events
        metadata: Optional additional metadata

    Returns:
        New MetaReasoningEvent instance
    """
    return MetaReasoningEvent(
        event_id=f"evt_{uuid.uuid4().hex[:12]}",
        event_type=event_type,
        source_component=source_component,
        timestamp=datetime.now(),
        payload=payload,
        correlation_id=correlation_id,
        metadata=metadata or {},
    )


def create_event_bus(max_history: int = 1000) -> MetaReasoningEventBus:
    """Factory function to create a MetaReasoningEventBus.

    Args:
        max_history: Maximum number of events to keep in history

    Returns:
        New MetaReasoningEventBus instance
    """
    return MetaReasoningEventBus(max_history=max_history)


# Singleton event bus for global access (optional usage pattern)
_global_event_bus: Optional[MetaReasoningEventBus] = None


def get_global_event_bus() -> MetaReasoningEventBus:
    """Get the global event bus instance.

    Creates one if it doesn't exist. This enables a simple
    global pub/sub pattern for components that don't want to
    manage event bus references explicitly.

    Returns:
        The global MetaReasoningEventBus instance
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = MetaReasoningEventBus()
    return _global_event_bus


def reset_global_event_bus() -> None:
    """Reset the global event bus (mainly for testing)."""
    global _global_event_bus
    _global_event_bus = None
