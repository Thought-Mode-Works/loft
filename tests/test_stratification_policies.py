"""
Unit tests for stratification policy enhancements.

Tests new policy features: cooldowns and dependency constraints.
"""

from loft.symbolic.stratification import (
    MODIFICATION_POLICIES,
    StratificationLevel,
    get_policy,
    infer_stratification_level,
)


class TestPolicyEnhancements:
    """Test new policy features added in Phase 3.2."""

    def test_all_policies_have_cooldowns(self):
        """Test all policies have cooldown configured."""
        for level, policy in MODIFICATION_POLICIES.items():
            assert hasattr(policy, "modification_cooldown_hours")
            assert policy.modification_cooldown_hours >= 0

    def test_all_policies_have_dependencies(self):
        """Test all policies have can_depend_on configured."""
        for level, policy in MODIFICATION_POLICIES.items():
            assert hasattr(policy, "can_depend_on")
            assert isinstance(policy.can_depend_on, set)
            assert len(policy.can_depend_on) > 0

    def test_constitutional_strictest_policy(self):
        """Test constitutional has the strictest policy."""
        const_policy = get_policy(StratificationLevel.CONSTITUTIONAL)

        assert const_policy.autonomous_allowed is False
        assert const_policy.confidence_threshold == 1.0
        assert const_policy.modification_cooldown_hours == float("inf")
        assert StratificationLevel.CONSTITUTIONAL in const_policy.can_depend_on
        assert const_policy.max_modifications_per_session == 0

    def test_strategic_policy(self):
        """Test strategic layer policy."""
        strat_policy = get_policy(StratificationLevel.STRATEGIC)

        assert strat_policy.autonomous_allowed is True
        assert strat_policy.confidence_threshold == 0.90
        assert strat_policy.modification_cooldown_hours == 24.0
        assert StratificationLevel.CONSTITUTIONAL in strat_policy.can_depend_on
        assert StratificationLevel.STRATEGIC in strat_policy.can_depend_on
        # Strategic cannot depend on tactical or operational
        assert StratificationLevel.TACTICAL not in strat_policy.can_depend_on
        assert StratificationLevel.OPERATIONAL not in strat_policy.can_depend_on

    def test_tactical_policy(self):
        """Test tactical layer policy."""
        tact_policy = get_policy(StratificationLevel.TACTICAL)

        assert tact_policy.autonomous_allowed is True
        assert tact_policy.confidence_threshold == 0.80
        assert tact_policy.modification_cooldown_hours == 1.0
        assert StratificationLevel.CONSTITUTIONAL in tact_policy.can_depend_on
        assert StratificationLevel.STRATEGIC in tact_policy.can_depend_on
        assert StratificationLevel.TACTICAL in tact_policy.can_depend_on
        # Tactical cannot depend on operational
        assert StratificationLevel.OPERATIONAL not in tact_policy.can_depend_on

    def test_operational_most_permissive(self):
        """Test operational has the most permissive policy."""
        op_policy = get_policy(StratificationLevel.OPERATIONAL)

        assert op_policy.autonomous_allowed is True
        assert op_policy.confidence_threshold == 0.70
        assert op_policy.modification_cooldown_hours == 0.0  # No cooldown
        # Operational can depend on all layers
        assert len(op_policy.can_depend_on) == 4
        assert StratificationLevel.CONSTITUTIONAL in op_policy.can_depend_on
        assert StratificationLevel.STRATEGIC in op_policy.can_depend_on
        assert StratificationLevel.TACTICAL in op_policy.can_depend_on
        assert StratificationLevel.OPERATIONAL in op_policy.can_depend_on

    def test_dependency_hierarchy(self):
        """Test dependency hierarchy is enforced in policies."""
        # More stable layers should have fewer allowed dependencies
        const_deps = len(get_policy(StratificationLevel.CONSTITUTIONAL).can_depend_on)
        strat_deps = len(get_policy(StratificationLevel.STRATEGIC).can_depend_on)
        tact_deps = len(get_policy(StratificationLevel.TACTICAL).can_depend_on)
        op_deps = len(get_policy(StratificationLevel.OPERATIONAL).can_depend_on)

        # Constitutional has fewest dependencies
        assert const_deps == 1  # Only itself

        # Dependencies increase as we go down
        assert const_deps < strat_deps < tact_deps <= op_deps

    def test_cooldown_hierarchy(self):
        """Test cooldown periods decrease as we go down layers."""
        const_cooldown = get_policy(StratificationLevel.CONSTITUTIONAL).modification_cooldown_hours
        strat_cooldown = get_policy(StratificationLevel.STRATEGIC).modification_cooldown_hours
        tact_cooldown = get_policy(StratificationLevel.TACTICAL).modification_cooldown_hours
        op_cooldown = get_policy(StratificationLevel.OPERATIONAL).modification_cooldown_hours

        # Constitutional never allows modification
        assert const_cooldown == float("inf")

        # Strategic has longest finite cooldown (24h)
        assert strat_cooldown == 24.0

        # Tactical shorter (1h)
        assert tact_cooldown == 1.0

        # Operational has no cooldown
        assert op_cooldown == 0.0


class TestModificationPolicy:
    """Test ModificationPolicy class methods."""

    def test_allows_modification_high_confidence_autonomous(self):
        """Test that high confidence autonomous modifications are allowed for autonomous layers."""
        policy = get_policy(StratificationLevel.TACTICAL)
        assert policy.allows_modification(confidence=0.9, is_autonomous=True)

    def test_rejects_low_confidence(self):
        """Test that low confidence modifications are rejected."""
        policy = get_policy(StratificationLevel.TACTICAL)
        assert not policy.allows_modification(confidence=0.5, is_autonomous=True)

    def test_rejects_autonomous_for_constitutional(self):
        """Test that autonomous modifications are rejected for constitutional layer."""
        policy = get_policy(StratificationLevel.CONSTITUTIONAL)
        assert not policy.allows_modification(confidence=1.0, is_autonomous=True)

    def test_allows_non_autonomous_modification(self):
        """Test that non-autonomous modifications are allowed if confidence is sufficient."""
        policy = get_policy(StratificationLevel.CONSTITUTIONAL)
        assert policy.allows_modification(confidence=1.0, is_autonomous=False)

    def test_summary_format(self):
        """Test that summary generates readable output."""
        policy = get_policy(StratificationLevel.TACTICAL)
        summary = policy.summary()

        assert "TACTICAL" in summary
        assert "Autonomous:" in summary
        assert "Confidence Threshold:" in summary
        assert "0.80" in summary

    def test_summary_includes_all_fields(self):
        """Test that summary includes all policy fields."""
        policy = get_policy(StratificationLevel.STRATEGIC)
        summary = policy.summary()

        assert "Human Approval" in summary
        assert "Max Modifications" in summary
        assert "Regression Tests" in summary


class TestStratificationLevel:
    """Test StratificationLevel enum functionality."""

    def test_level_comparison_operational_less_than_tactical(self):
        """Test that operational < tactical."""
        assert StratificationLevel.OPERATIONAL < StratificationLevel.TACTICAL

    def test_level_comparison_tactical_less_than_strategic(self):
        """Test that tactical < strategic."""
        assert StratificationLevel.TACTICAL < StratificationLevel.STRATEGIC

    def test_level_comparison_strategic_less_than_constitutional(self):
        """Test that strategic < constitutional."""
        assert StratificationLevel.STRATEGIC < StratificationLevel.CONSTITUTIONAL

    def test_level_comparison_transitive(self):
        """Test that operational is less than all higher levels."""
        assert StratificationLevel.OPERATIONAL < StratificationLevel.TACTICAL
        assert StratificationLevel.OPERATIONAL < StratificationLevel.STRATEGIC
        assert StratificationLevel.OPERATIONAL < StratificationLevel.CONSTITUTIONAL

    def test_level_string_values(self):
        """Test that enum values are correct strings."""
        assert StratificationLevel.CONSTITUTIONAL.value == "constitutional"
        assert StratificationLevel.STRATEGIC.value == "strategic"
        assert StratificationLevel.TACTICAL.value == "tactical"
        assert StratificationLevel.OPERATIONAL.value == "operational"


class TestInferStratificationLevel:
    """Test stratification level inference from rule text."""

    def test_infer_constitutional_from_fundamental(self):
        """Test that 'fundamental' keyword triggers constitutional."""
        rule = "fundamental(X) :- entity(X)."
        assert infer_stratification_level(rule) == StratificationLevel.CONSTITUTIONAL

    def test_infer_constitutional_from_constitutional(self):
        """Test that 'constitutional' keyword triggers constitutional."""
        rule = "constitutional_principle(justice)."
        assert infer_stratification_level(rule) == StratificationLevel.CONSTITUTIONAL

    def test_infer_strategic_from_policy(self):
        """Test that 'policy' keyword triggers strategic."""
        rule = "policy(X) :- rule(X), valid(X)."
        assert infer_stratification_level(rule) == StratificationLevel.STRATEGIC

    def test_infer_strategic_from_must(self):
        """Test that 'must' keyword triggers strategic."""
        rule = "action_must_be_logged(X) :- action(X)."
        assert infer_stratification_level(rule) == StratificationLevel.STRATEGIC

    def test_infer_operational_from_cache(self):
        """Test that 'cache' keyword triggers operational."""
        rule = "cache_result(X, Y) :- compute(X, Y)."
        assert infer_stratification_level(rule) == StratificationLevel.OPERATIONAL

    def test_infer_operational_from_optimize(self):
        """Test that 'optimize' keyword triggers operational."""
        rule = "optimize_query(X) :- query(X), slow(X)."
        assert infer_stratification_level(rule) == StratificationLevel.OPERATIONAL

    def test_infer_tactical_default(self):
        """Test that rules default to tactical when no keywords match."""
        rule = "process(X) :- input(X), valid(X)."
        assert infer_stratification_level(rule) == StratificationLevel.TACTICAL

    def test_infer_case_insensitive(self):
        """Test that inference is case-insensitive."""
        rule1 = "FUNDAMENTAL(X) :- entity(X)."
        rule2 = "Fundamental(X) :- entity(X)."
        assert infer_stratification_level(rule1) == StratificationLevel.CONSTITUTIONAL
        assert infer_stratification_level(rule2) == StratificationLevel.CONSTITUTIONAL
