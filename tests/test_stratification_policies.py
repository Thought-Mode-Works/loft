"""
Unit tests for stratification policy enhancements.

Tests new policy features: cooldowns and dependency constraints.
"""


from loft.symbolic.stratification import (
    MODIFICATION_POLICIES,
    StratificationLevel,
    get_policy,
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
