"""
Integration adapters for cross-component data flow in meta-reasoning.

These adapters provide standardized interfaces for passing data between
meta-reasoning components, ensuring loose coupling and type safety.
"""

from typing import Any, Dict, List, Optional

from loft.meta.event_bus import (
    ComponentType,
    EventType,
    MetaReasoningEvent,
    MetaReasoningEventBus,
    create_event,
)
from loft.meta.failure_analyzer import (
    FailureAnalyzer,
    PredictionError,
    Recommendation,
    RecommendationEngine,
    create_prediction_error_from_chain,
)
from loft.meta.observer import ReasoningObserver
from loft.meta.schemas import ReasoningChain
from loft.meta.self_improvement import (
    ActionType,
    ImprovementAction,
    ImprovementGoal,
    SelfImprovementTracker,
)


class ObserverToFailureAdapter:
    """Adapts ReasoningObserver output for FailureAnalyzer input.

    This adapter bridges the gap between the observation system and
    failure analysis by converting ReasoningChain data into
    PredictionError objects suitable for failure analysis.

    Example:
        >>> observer = ReasoningObserver()
        >>> analyzer = FailureAnalyzer()
        >>> adapter = ObserverToFailureAdapter(observer, analyzer)
        >>> errors = adapter.process_failed_chains()
        >>> print(f"Processed {len(errors)} failures")
    """

    def __init__(
        self,
        observer: ReasoningObserver,
        analyzer: FailureAnalyzer,
        event_bus: Optional[MetaReasoningEventBus] = None,
    ):
        """Initialize the adapter.

        Args:
            observer: The ReasoningObserver providing chain data
            analyzer: The FailureAnalyzer to receive error data
            event_bus: Optional event bus for publishing events
        """
        self._observer = observer
        self._analyzer = analyzer
        self._event_bus = event_bus
        self._processed_chain_ids: set = set()

    @property
    def observer(self) -> ReasoningObserver:
        """Get the connected observer."""
        return self._observer

    @property
    def analyzer(self) -> FailureAnalyzer:
        """Get the connected analyzer."""
        return self._analyzer

    @property
    def processed_count(self) -> int:
        """Get count of processed chains."""
        return len(self._processed_chain_ids)

    def process_failed_chains(
        self,
        domain: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[PredictionError]:
        """Process failed chains from observer and send to analyzer.

        Args:
            domain: Optional domain filter
            limit: Maximum number of chains to process

        Returns:
            List of PredictionError objects created
        """
        errors_created = []

        # Get chains from observer - use appropriate method based on domain filter
        if domain:
            chains = self._observer.get_chains_by_domain(domain)
        else:
            chains = list(self._observer.chains.values())

        # Filter to failed chains not yet processed
        failed_chains = [
            chain
            for chain in chains
            if not chain.overall_success and chain.chain_id not in self._processed_chain_ids
        ]

        if limit:
            failed_chains = failed_chains[:limit]

        for chain in failed_chains:
            error = self._convert_chain_to_error(chain)
            if error:
                self._analyzer.record_error(error)
                errors_created.append(error)
                self._processed_chain_ids.add(chain.chain_id)

                # Publish event if bus is connected
                if self._event_bus:
                    event = create_event(
                        EventType.FAILURE_RECORDED,
                        ComponentType.FAILURE_ANALYZER,
                        {
                            "error_id": error.error_id,
                            "chain_id": chain.chain_id,
                            "domain": chain.domain,
                            "predicted": error.predicted,
                            "actual": error.actual,
                        },
                        correlation_id=chain.chain_id,
                    )
                    self._event_bus.publish(event)

        return errors_created

    def _convert_chain_to_error(
        self,
        chain: ReasoningChain,
    ) -> Optional[PredictionError]:
        """Convert a ReasoningChain to a PredictionError.

        Args:
            chain: The failed chain to convert

        Returns:
            PredictionError or None if conversion fails
        """
        return create_prediction_error_from_chain(
            chain=chain,
            expected_output=chain.ground_truth,
            actual_output=chain.prediction,
        )

    def get_unprocessed_failures(self, domain: Optional[str] = None) -> List[ReasoningChain]:
        """Get failed chains that haven't been processed yet.

        Args:
            domain: Optional domain filter

        Returns:
            List of unprocessed failed chains
        """
        if domain:
            chains = self._observer.get_chains_by_domain(domain)
        else:
            chains = list(self._observer.chains.values())
        return [
            chain
            for chain in chains
            if not chain.overall_success and chain.chain_id not in self._processed_chain_ids
        ]

    def reset_processed_tracking(self) -> int:
        """Reset the set of processed chain IDs.

        Returns:
            Number of IDs that were cleared
        """
        count = len(self._processed_chain_ids)
        self._processed_chain_ids.clear()
        return count


class FailureToImprovementAdapter:
    """Adapts FailureAnalyzer output for SelfImprovement input.

    This adapter converts failure analysis recommendations into
    improvement actions that can be tracked and executed by the
    self-improvement system.

    Example:
        >>> analyzer = FailureAnalyzer()
        >>> tracker = SelfImprovementTracker()
        >>> adapter = FailureToImprovementAdapter(analyzer, tracker)
        >>> actions = adapter.convert_recommendations_to_actions(goal)
        >>> print(f"Created {len(actions)} improvement actions")
    """

    def __init__(
        self,
        recommendation_engine: RecommendationEngine,
        tracker: SelfImprovementTracker,
        event_bus: Optional[MetaReasoningEventBus] = None,
    ):
        """Initialize the adapter.

        Args:
            recommendation_engine: The engine providing recommendations
            tracker: The tracker to receive improvement actions
            event_bus: Optional event bus for publishing events
        """
        self._recommendation_engine = recommendation_engine
        self._tracker = tracker
        self._event_bus = event_bus
        self._recommendation_to_action_map: Dict[str, str] = {}

    @property
    def recommendation_engine(self) -> RecommendationEngine:
        """Get the connected recommendation engine."""
        return self._recommendation_engine

    @property
    def tracker(self) -> SelfImprovementTracker:
        """Get the connected tracker."""
        return self._tracker

    @property
    def conversion_count(self) -> int:
        """Get count of converted recommendations."""
        return len(self._recommendation_to_action_map)

    def convert_recommendations_to_actions(
        self,
        goal: ImprovementGoal,
        min_priority: float = 0.0,
        limit: Optional[int] = None,
    ) -> List[ImprovementAction]:
        """Convert recommendations to improvement actions for a goal.

        Args:
            goal: The improvement goal to associate actions with
            min_priority: Minimum priority threshold (0-1)
            limit: Maximum number of recommendations to convert

        Returns:
            List of ImprovementAction objects created
        """
        actions_created = []

        # Get recommendations from engine's internal storage
        all_recommendations = list(self._recommendation_engine._recommendations.values())

        # Note: min_priority parameter is kept for API compatibility but not used
        # since Recommendation uses an enum for priority. All recommendations are processed.
        recommendations = all_recommendations

        # Apply limit
        if limit:
            recommendations = recommendations[:limit]

        for rec in recommendations:
            # Skip if already converted
            if rec.recommendation_id in self._recommendation_to_action_map:
                continue

            action = self._convert_recommendation_to_action(rec, goal)
            if action:
                actions_created.append(action)
                self._recommendation_to_action_map[rec.recommendation_id] = action.action_id

                # Publish event if bus is connected
                if self._event_bus:
                    event = create_event(
                        EventType.RECOMMENDATION_GENERATED,
                        ComponentType.RECOMMENDATION_ENGINE,
                        {
                            "recommendation_id": rec.recommendation_id,
                            "action_id": action.action_id,
                            "goal_id": goal.goal_id,
                            "action_type": action.action_type.value,
                        },
                        correlation_id=goal.goal_id,
                    )
                    self._event_bus.publish(event)

        return actions_created

    def _convert_recommendation_to_action(
        self,
        recommendation: Recommendation,
        goal: ImprovementGoal,
    ) -> Optional[ImprovementAction]:
        """Convert a single recommendation to an improvement action.

        Args:
            recommendation: The recommendation to convert
            goal: The goal to associate with

        Returns:
            ImprovementAction or None if conversion fails
        """
        # Map recommendation category to action type
        action_type = self._map_category_to_action_type(recommendation)

        # Determine if approval is required based on priority
        from loft.meta.schemas import ImprovementPriority

        requires_approval = recommendation.priority in (
            ImprovementPriority.CRITICAL,
            ImprovementPriority.HIGH,
        )

        return ImprovementAction(
            action_id=f"act_{recommendation.recommendation_id}",
            action_type=action_type,
            description=recommendation.description,
            target_component=recommendation.title,  # Use title as component name
            parameters={
                "recommendation_id": recommendation.recommendation_id,
                "priority": recommendation.priority.value,
                "category": recommendation.category.value,
                "implementation_steps": recommendation.implementation_steps,
                "expected_impact": recommendation.expected_impact,
                "risk_level": self._calculate_risk_level(recommendation),
                "requires_approval": requires_approval,
            },
        )

    def _map_category_to_action_type(self, recommendation: Recommendation) -> ActionType:
        """Map a recommendation category to an action type.

        Args:
            recommendation: The recommendation

        Returns:
            Appropriate ActionType
        """
        from loft.meta.failure_analyzer import RecommendationCategory

        category_to_action = {
            RecommendationCategory.RULE_ADDITION: ActionType.RULE_MODIFICATION,
            RecommendationCategory.RULE_MODIFICATION: ActionType.RULE_MODIFICATION,
            RecommendationCategory.PROMPT_IMPROVEMENT: ActionType.PROMPT_REFINEMENT,
            RecommendationCategory.VALIDATION_ENHANCEMENT: ActionType.VALIDATION_ENHANCEMENT,
            RecommendationCategory.DOMAIN_EXPANSION: ActionType.STRATEGY_ADJUSTMENT,
            RecommendationCategory.DATA_AUGMENTATION: ActionType.STRATEGY_ADJUSTMENT,
            RecommendationCategory.STRATEGY_ADJUSTMENT: ActionType.STRATEGY_ADJUSTMENT,
        }

        return category_to_action.get(recommendation.category, ActionType.STRATEGY_ADJUSTMENT)

    def _calculate_risk_level(self, recommendation: Recommendation) -> str:
        """Calculate risk level for an action based on recommendation.

        Args:
            recommendation: The source recommendation

        Returns:
            Risk level: "low", "medium", or "high"
        """
        from loft.meta.schemas import ImprovementPriority

        # Higher priority recommendations typically have higher risk
        if recommendation.priority in (ImprovementPriority.CRITICAL, ImprovementPriority.HIGH):
            return "high"
        elif recommendation.priority == ImprovementPriority.MEDIUM:
            return "medium"
        else:
            return "low"

    def get_action_for_recommendation(self, recommendation_id: str) -> Optional[str]:
        """Get the action ID for a converted recommendation.

        Args:
            recommendation_id: The recommendation ID to look up

        Returns:
            Action ID or None if not converted
        """
        return self._recommendation_to_action_map.get(recommendation_id)

    def reset_conversion_tracking(self) -> int:
        """Reset the recommendation to action mapping.

        Returns:
            Number of mappings that were cleared
        """
        count = len(self._recommendation_to_action_map)
        self._recommendation_to_action_map.clear()
        return count


class EventDrivenIntegration:
    """Coordinates event-driven integration between all meta-reasoning components.

    This class provides a high-level interface for setting up event-driven
    communication between Observer, FailureAnalyzer, and SelfImprovement
    components using the MetaReasoningEventBus.

    Example:
        >>> bus = MetaReasoningEventBus()
        >>> integration = EventDrivenIntegration(bus)
        >>> integration.setup_observer_to_failure_flow(observer, analyzer)
        >>> integration.setup_failure_to_improvement_flow(engine, tracker)
    """

    def __init__(self, event_bus: MetaReasoningEventBus):
        """Initialize the integration coordinator.

        Args:
            event_bus: The event bus to use for communication
        """
        self._event_bus = event_bus
        self._subscription_ids: List[str] = []
        self._adapters: Dict[str, Any] = {}

    @property
    def event_bus(self) -> MetaReasoningEventBus:
        """Get the event bus."""
        return self._event_bus

    @property
    def subscription_count(self) -> int:
        """Get the number of active subscriptions."""
        return len(self._subscription_ids)

    def setup_observer_to_failure_flow(
        self,
        observer: ReasoningObserver,
        analyzer: FailureAnalyzer,
        auto_process: bool = True,
    ) -> ObserverToFailureAdapter:
        """Set up event flow from Observer to FailureAnalyzer.

        Args:
            observer: The ReasoningObserver
            analyzer: The FailureAnalyzer
            auto_process: If True, automatically process chains on CHAIN_OBSERVED events

        Returns:
            The created adapter
        """
        adapter = ObserverToFailureAdapter(observer, analyzer, self._event_bus)
        self._adapters["observer_to_failure"] = adapter

        if auto_process:
            # Subscribe to chain observed events
            def handle_chain_observed(event: MetaReasoningEvent) -> None:
                if not event.payload.get("overall_success", True):
                    adapter.process_failed_chains(limit=1)

            sub_id = self._event_bus.subscribe(
                EventType.CHAIN_OBSERVED,
                handle_chain_observed,
                component=ComponentType.FAILURE_ANALYZER,
            )
            self._subscription_ids.append(sub_id)

        return adapter

    def setup_failure_to_improvement_flow(
        self,
        recommendation_engine: RecommendationEngine,
        tracker: SelfImprovementTracker,
        goal: Optional[ImprovementGoal] = None,
        auto_convert: bool = True,
    ) -> FailureToImprovementAdapter:
        """Set up event flow from FailureAnalyzer to SelfImprovement.

        Args:
            recommendation_engine: The RecommendationEngine
            tracker: The SelfImprovementTracker
            goal: Optional default goal for converted actions
            auto_convert: If True, automatically convert on RECOMMENDATION_GENERATED events

        Returns:
            The created adapter
        """
        adapter = FailureToImprovementAdapter(
            recommendation_engine, tracker, self._event_bus
        )
        self._adapters["failure_to_improvement"] = adapter

        if auto_convert and goal:
            # Subscribe to recommendation events
            def handle_recommendation(event: MetaReasoningEvent) -> None:
                adapter.convert_recommendations_to_actions(goal, limit=1)

            sub_id = self._event_bus.subscribe(
                EventType.RECOMMENDATION_GENERATED,
                handle_recommendation,
                component=ComponentType.SELF_IMPROVEMENT_TRACKER,
            )
            self._subscription_ids.append(sub_id)

        return adapter

    def get_adapter(self, name: str) -> Optional[Any]:
        """Get an adapter by name.

        Args:
            name: Adapter name ("observer_to_failure" or "failure_to_improvement")

        Returns:
            The adapter or None if not found
        """
        return self._adapters.get(name)

    def teardown(self) -> int:
        """Remove all subscriptions and clear adapters.

        Returns:
            Number of subscriptions removed
        """
        count = 0
        for sub_id in self._subscription_ids:
            if self._event_bus.unsubscribe(sub_id):
                count += 1
        self._subscription_ids.clear()
        self._adapters.clear()
        return count


def create_observer_to_failure_adapter(
    observer: ReasoningObserver,
    analyzer: FailureAnalyzer,
    event_bus: Optional[MetaReasoningEventBus] = None,
) -> ObserverToFailureAdapter:
    """Factory function to create an ObserverToFailureAdapter.

    Args:
        observer: The ReasoningObserver providing chain data
        analyzer: The FailureAnalyzer to receive error data
        event_bus: Optional event bus for publishing events

    Returns:
        New ObserverToFailureAdapter instance
    """
    return ObserverToFailureAdapter(observer, analyzer, event_bus)


def create_failure_to_improvement_adapter(
    recommendation_engine: RecommendationEngine,
    tracker: SelfImprovementTracker,
    event_bus: Optional[MetaReasoningEventBus] = None,
) -> FailureToImprovementAdapter:
    """Factory function to create a FailureToImprovementAdapter.

    Args:
        recommendation_engine: The engine providing recommendations
        tracker: The tracker to receive improvement actions
        event_bus: Optional event bus for publishing events

    Returns:
        New FailureToImprovementAdapter instance
    """
    return FailureToImprovementAdapter(recommendation_engine, tracker, event_bus)


def create_event_driven_integration(
    event_bus: MetaReasoningEventBus,
) -> EventDrivenIntegration:
    """Factory function to create an EventDrivenIntegration.

    Args:
        event_bus: The event bus to use for communication

    Returns:
        New EventDrivenIntegration instance
    """
    return EventDrivenIntegration(event_bus)
