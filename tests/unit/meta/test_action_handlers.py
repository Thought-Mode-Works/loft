"""Tests for the action handlers module."""

import pytest

from loft.meta.action_handlers import (
    HandlerResult,
    PromptRefinementHandler,
    StrategyAdjustmentHandler,
    create_prompt_refinement_handler,
    create_prompt_refinement_rollback_handler,
    create_strategy_adjustment_handler,
    create_strategy_adjustment_rollback_handler,
    register_real_handlers,
)
from loft.meta.prompt_optimizer import PromptOptimizer, PromptVersion, PromptCategory
from loft.meta.self_improvement import (
    ActionType,
    AutonomousImprover,
    ImprovementAction,
    ImprovementGoal,
    MetricType,
    SelfImprovementTracker,
)
from loft.meta.strategy import StrategySelector, create_evaluator


class TestHandlerResult:
    """Tests for HandlerResult dataclass."""

    def test_basic_creation(self):
        """Test basic creation of HandlerResult."""
        result = HandlerResult(success=True, message="Test message")
        assert result.success is True
        assert result.message == "Test message"
        assert result.rollback_data is None
        assert result.verified is False

    def test_with_rollback_data(self):
        """Test creation with rollback data."""
        rollback = {"previous_version": 1}
        result = HandlerResult(
            success=True, message="Applied", rollback_data=rollback, verified=True
        )
        assert result.rollback_data == rollback
        assert result.verified is True


class TestPromptRefinementHandler:
    """Tests for PromptRefinementHandler."""

    @pytest.fixture
    def optimizer(self):
        """Create a PromptOptimizer with test prompts."""
        opt = PromptOptimizer()
        # Register base prompt
        base_prompt = PromptVersion(
            prompt_id="test_prompt",
            version=1,
            template="Original template",
            category=PromptCategory.RULE_GENERATION,
        )
        opt.register_prompt(base_prompt)
        return opt

    @pytest.fixture
    def handler(self, optimizer):
        """Create a PromptRefinementHandler."""
        return PromptRefinementHandler(optimizer)

    def test_optimizer_property(self, handler, optimizer):
        """Test optimizer property returns the optimizer."""
        assert handler.optimizer is optimizer

    def test_apply_winning_version_success(self, handler, optimizer):
        """Test applying a winning version successfully."""
        # Explicitly set version 1 as active before creating v2
        optimizer.set_active_version("test_prompt", 1)

        # Create a second version (use full_id for original)
        optimizer.create_new_version(
            original_prompt_id="test_prompt_v1",
            new_template="Improved template",
            modification_reason="A/B test winner",
        )

        # Create action to apply version 2
        action = ImprovementAction(
            action_id="action_001",
            action_type=ActionType.PROMPT_REFINEMENT,
            description="Apply winning version",
            target_component="prompt_optimizer",
            parameters={
                "target_prompt_id": "test_prompt",
                "winning_version": 2,
            },
        )

        # Execute
        success = handler.execute(action)

        assert success is True
        assert action.success is True
        assert action.rollback_data is not None
        assert action.rollback_data["previous_version"] == 1
        assert action.executed_at is not None
        assert optimizer.get_active_version("test_prompt") == 2

    def test_apply_winning_version_missing_params(self, handler):
        """Test applying version with missing parameters."""
        action = ImprovementAction(
            action_id="action_002",
            action_type=ActionType.PROMPT_REFINEMENT,
            description="Apply winning version",
            target_component="prompt_optimizer",
            parameters={},
        )

        success = handler.execute(action)

        assert success is False
        assert action.success is False

    def test_apply_winning_version_nonexistent(self, handler):
        """Test applying a version that doesn't exist."""
        action = ImprovementAction(
            action_id="action_003",
            action_type=ActionType.PROMPT_REFINEMENT,
            description="Apply winning version",
            target_component="prompt_optimizer",
            parameters={
                "target_prompt_id": "test_prompt",
                "winning_version": 999,
            },
        )

        success = handler.execute(action)

        assert success is False

    def test_create_new_version_success(self, handler, optimizer):
        """Test creating a new version successfully."""
        action = ImprovementAction(
            action_id="action_004",
            action_type=ActionType.PROMPT_REFINEMENT,
            description="Create new version",
            target_component="prompt_optimizer",
            parameters={
                "original_prompt_id": "test_prompt_v1",
                "new_template": "Brand new template",
                "modification_reason": "LLM suggested improvement",
            },
        )

        success = handler.execute(action)

        assert success is True
        assert action.success is True
        assert action.rollback_data is not None
        assert "new_version_id" in action.rollback_data

    def test_create_new_version_missing_params(self, handler):
        """Test creating version with missing parameters."""
        action = ImprovementAction(
            action_id="action_005",
            action_type=ActionType.PROMPT_REFINEMENT,
            description="Create new version",
            target_component="prompt_optimizer",
            parameters={"original_prompt_id": "test_prompt"},
        )

        success = handler.execute(action)

        assert success is False

    def test_create_new_version_invalid_prompt(self, handler):
        """Test creating version for nonexistent prompt."""
        action = ImprovementAction(
            action_id="action_006",
            action_type=ActionType.PROMPT_REFINEMENT,
            description="Create new version",
            target_component="prompt_optimizer",
            parameters={
                "original_prompt_id": "nonexistent",
                "new_template": "New template",
            },
        )

        success = handler.execute(action)

        assert success is False

    def test_rollback_winning_version(self, handler, optimizer):
        """Test rolling back a winning version application."""
        # Explicitly set version 1 as active first
        optimizer.set_active_version("test_prompt", 1)

        # Create second version (use full_id for original)
        optimizer.create_new_version(
            original_prompt_id="test_prompt_v1",
            new_template="Improved template",
            modification_reason="Test",
        )

        action = ImprovementAction(
            action_id="action_007",
            action_type=ActionType.PROMPT_REFINEMENT,
            description="Apply winning version",
            target_component="prompt_optimizer",
            parameters={
                "target_prompt_id": "test_prompt",
                "winning_version": 2,
            },
        )

        handler.execute(action)
        assert optimizer.get_active_version("test_prompt") == 2

        # Rollback
        rollback_success = handler.rollback(action)

        assert rollback_success is True
        assert optimizer.get_active_version("test_prompt") == 1

    def test_rollback_no_data(self, handler):
        """Test rollback with no rollback data."""
        action = ImprovementAction(
            action_id="action_008",
            action_type=ActionType.PROMPT_REFINEMENT,
            description="Test",
            target_component="prompt_optimizer",
            parameters={},
        )

        rollback_success = handler.rollback(action)

        assert rollback_success is False

    def test_rollback_create_version(self, handler, optimizer):
        """Test rollback of create version (returns True, version persists)."""
        action = ImprovementAction(
            action_id="action_009",
            action_type=ActionType.PROMPT_REFINEMENT,
            description="Create new version",
            target_component="prompt_optimizer",
            parameters={
                "original_prompt_id": "test_prompt_v1",
                "new_template": "New template",
            },
        )

        handler.execute(action)
        rollback_success = handler.rollback(action)

        # For create_new_version, rollback returns True but version persists
        assert rollback_success is True

    def test_unknown_action_type(self, handler):
        """Test handling unknown action subtype."""
        action = ImprovementAction(
            action_id="action_010",
            action_type=ActionType.PROMPT_REFINEMENT,
            description="Unknown action",
            target_component="prompt_optimizer",
            parameters={"unknown_key": "value"},
        )

        success = handler.execute(action)

        assert success is False


class TestStrategyAdjustmentHandler:
    """Tests for StrategyAdjustmentHandler."""

    @pytest.fixture
    def selector(self):
        """Create a StrategySelector with test configuration."""
        evaluator = create_evaluator()
        selector = StrategySelector(evaluator)
        selector.set_domain_default("contracts", "checklist")
        return selector

    @pytest.fixture
    def handler(self, selector):
        """Create a StrategyAdjustmentHandler."""
        return StrategyAdjustmentHandler(selector)

    def test_selector_property(self, handler, selector):
        """Test selector property returns the selector."""
        assert handler.selector is selector

    def test_execute_with_recommended_strategy(self, handler, selector):
        """Test executing strategy adjustment with recommended_strategy."""
        action = ImprovementAction(
            action_id="action_011",
            action_type=ActionType.STRATEGY_ADJUSTMENT,
            description="Update domain default",
            target_component="strategy_selector",
            parameters={
                "domain": "torts",
                "recommended_strategy": "causal_chain",
            },
        )

        success = handler.execute(action)

        assert success is True
        assert action.success is True
        assert action.rollback_data is not None
        assert action.rollback_data["domain"] == "torts"
        assert action.rollback_data["new_default"] == "causal_chain"
        assert selector.get_domain_default("torts") == "causal_chain"

    def test_execute_with_new_default(self, handler, selector):
        """Test executing strategy adjustment with new_default."""
        action = ImprovementAction(
            action_id="action_012",
            action_type=ActionType.STRATEGY_ADJUSTMENT,
            description="Update domain default",
            target_component="strategy_selector",
            parameters={
                "domain": "property",
                "new_default": "rule_based",
            },
        )

        success = handler.execute(action)

        assert success is True
        assert selector.get_domain_default("property") == "rule_based"

    def test_execute_updates_existing_default(self, handler, selector):
        """Test executing updates an existing default."""
        action = ImprovementAction(
            action_id="action_013",
            action_type=ActionType.STRATEGY_ADJUSTMENT,
            description="Update contracts default",
            target_component="strategy_selector",
            parameters={
                "domain": "contracts",
                "recommended_strategy": "balancing_test",
            },
        )

        success = handler.execute(action)

        assert success is True
        assert action.rollback_data["previous_default"] == "checklist"
        assert selector.get_domain_default("contracts") == "balancing_test"

    def test_execute_missing_domain(self, handler):
        """Test executing with missing domain."""
        action = ImprovementAction(
            action_id="action_014",
            action_type=ActionType.STRATEGY_ADJUSTMENT,
            description="Missing domain",
            target_component="strategy_selector",
            parameters={"recommended_strategy": "causal_chain"},
        )

        success = handler.execute(action)

        assert success is False

    def test_execute_missing_strategy(self, handler):
        """Test executing with missing strategy."""
        action = ImprovementAction(
            action_id="action_015",
            action_type=ActionType.STRATEGY_ADJUSTMENT,
            description="Missing strategy",
            target_component="strategy_selector",
            parameters={"domain": "contracts"},
        )

        success = handler.execute(action)

        assert success is False

    def test_rollback_success(self, handler, selector):
        """Test rolling back a strategy adjustment."""
        action = ImprovementAction(
            action_id="action_016",
            action_type=ActionType.STRATEGY_ADJUSTMENT,
            description="Update contracts default",
            target_component="strategy_selector",
            parameters={
                "domain": "contracts",
                "recommended_strategy": "causal_chain",
            },
        )

        handler.execute(action)
        assert selector.get_domain_default("contracts") == "causal_chain"

        rollback_success = handler.rollback(action)

        assert rollback_success is True
        assert selector.get_domain_default("contracts") == "checklist"

    def test_rollback_no_data(self, handler):
        """Test rollback with no rollback data."""
        action = ImprovementAction(
            action_id="action_017",
            action_type=ActionType.STRATEGY_ADJUSTMENT,
            description="Test",
            target_component="strategy_selector",
            parameters={},
        )

        rollback_success = handler.rollback(action)

        assert rollback_success is False

    def test_rollback_missing_domain(self, handler):
        """Test rollback with missing domain in rollback data."""
        action = ImprovementAction(
            action_id="action_018",
            action_type=ActionType.STRATEGY_ADJUSTMENT,
            description="Test",
            target_component="strategy_selector",
            parameters={},
        )
        action.rollback_data = {"previous_default": "checklist"}

        rollback_success = handler.rollback(action)

        assert rollback_success is False


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_prompt_refinement_handler(self):
        """Test creating prompt refinement handler function."""
        optimizer = PromptOptimizer()
        handler_fn = create_prompt_refinement_handler(optimizer)

        assert callable(handler_fn)

    def test_create_prompt_refinement_rollback_handler(self):
        """Test creating prompt refinement rollback handler function."""
        optimizer = PromptOptimizer()
        rollback_fn = create_prompt_refinement_rollback_handler(optimizer)

        assert callable(rollback_fn)

    def test_create_strategy_adjustment_handler(self):
        """Test creating strategy adjustment handler function."""
        evaluator = create_evaluator()
        selector = StrategySelector(evaluator)
        handler_fn = create_strategy_adjustment_handler(selector)

        assert callable(handler_fn)

    def test_create_strategy_adjustment_rollback_handler(self):
        """Test creating strategy adjustment rollback handler function."""
        evaluator = create_evaluator()
        selector = StrategySelector(evaluator)
        rollback_fn = create_strategy_adjustment_rollback_handler(selector)

        assert callable(rollback_fn)


class TestRegisterRealHandlers:
    """Tests for register_real_handlers function."""

    @pytest.fixture
    def improver(self):
        """Create an AutonomousImprover."""
        tracker = SelfImprovementTracker()
        goal = ImprovementGoal(
            goal_id="goal_test",
            metric_type=MetricType.ACCURACY,
            target_value=0.9,
            baseline_value=0.8,
            description="Test goal",
        )
        tracker.set_goal(goal)
        return AutonomousImprover(tracker=tracker)

    @pytest.fixture
    def optimizer(self):
        """Create a PromptOptimizer."""
        return PromptOptimizer()

    @pytest.fixture
    def selector(self):
        """Create a StrategySelector."""
        evaluator = create_evaluator()
        return StrategySelector(evaluator)

    def test_register_prompt_handler_only(self, improver, optimizer):
        """Test registering only prompt handler."""
        result = register_real_handlers(improver, optimizer=optimizer)

        assert result == {"prompt_refinement": True}

    def test_register_strategy_handler_only(self, improver, selector):
        """Test registering only strategy handler."""
        result = register_real_handlers(improver, selector=selector)

        assert result == {"strategy_adjustment": True}

    def test_register_both_handlers(self, improver, optimizer, selector):
        """Test registering both handlers."""
        result = register_real_handlers(
            improver, optimizer=optimizer, selector=selector
        )

        assert result == {"prompt_refinement": True, "strategy_adjustment": True}

    def test_register_no_handlers(self, improver):
        """Test registering no handlers."""
        result = register_real_handlers(improver)

        assert result == {}


class TestPromptOptimizerActiveVersion:
    """Tests for PromptOptimizer active version methods."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with test prompts."""
        opt = PromptOptimizer()
        base = PromptVersion(
            prompt_id="test_prompt",
            version=1,
            template="V1 template",
            category=PromptCategory.RULE_GENERATION,
        )
        opt.register_prompt(base)
        return opt

    def test_get_active_version_returns_latest(self, optimizer):
        """Test get_active_version returns latest by default."""
        assert optimizer.get_active_version("test_prompt") == 1

    def test_get_active_version_nonexistent(self, optimizer):
        """Test get_active_version for nonexistent prompt."""
        assert optimizer.get_active_version("nonexistent") is None

    def test_set_active_version_success(self, optimizer):
        """Test setting active version."""
        optimizer.create_new_version(
            original_prompt_id="test_prompt_v1",
            new_template="V2 template",
            modification_reason="Test",
        )

        success = optimizer.set_active_version("test_prompt", 2)

        assert success is True
        assert optimizer.get_active_version("test_prompt") == 2

    def test_set_active_version_nonexistent(self, optimizer):
        """Test setting active version for nonexistent version."""
        success = optimizer.set_active_version("test_prompt", 999)

        assert success is False

    def test_get_active_prompt(self, optimizer):
        """Test getting active prompt version."""
        prompt = optimizer.get_active_prompt("test_prompt")

        assert prompt is not None
        assert prompt.version == 1
        assert prompt.template == "V1 template"

    def test_get_active_prompt_after_set(self, optimizer):
        """Test getting active prompt after setting version."""
        optimizer.create_new_version(
            original_prompt_id="test_prompt_v1",
            new_template="V2 template",
            modification_reason="Test",
        )
        optimizer.set_active_version("test_prompt", 2)

        prompt = optimizer.get_active_prompt("test_prompt")

        assert prompt is not None
        assert prompt.version == 2
        assert prompt.template == "V2 template"

    def test_get_active_prompt_nonexistent(self, optimizer):
        """Test getting active prompt for nonexistent prompt."""
        prompt = optimizer.get_active_prompt("nonexistent")

        assert prompt is None


class TestStrategySelectorGetDomainDefault:
    """Tests for StrategySelector.get_domain_default method."""

    @pytest.fixture
    def selector(self):
        """Create a StrategySelector."""
        evaluator = create_evaluator()
        return StrategySelector(evaluator)

    def test_get_domain_default_nonexistent(self, selector):
        """Test getting default for domain with no default."""
        assert selector.get_domain_default("unknown") is None

    def test_get_domain_default_after_set(self, selector):
        """Test getting default after setting it."""
        selector.set_domain_default("contracts", "checklist")

        assert selector.get_domain_default("contracts") == "checklist"

    def test_get_domain_default_multiple_domains(self, selector):
        """Test getting defaults for multiple domains."""
        selector.set_domain_default("contracts", "checklist")
        selector.set_domain_default("torts", "causal_chain")
        selector.set_domain_default("property", "rule_based")

        assert selector.get_domain_default("contracts") == "checklist"
        assert selector.get_domain_default("torts") == "causal_chain"
        assert selector.get_domain_default("property") == "rule_based"
