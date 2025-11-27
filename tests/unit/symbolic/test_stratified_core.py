"""
Unit tests for stratified ASP core.

Tests rule management, policy enforcement, and modification tracking.
"""

import pytest
from datetime import datetime
from loft.symbolic.stratified_core import StratifiedASPCore
from loft.symbolic.asp_rule import ASPRule, RuleMetadata
from loft.symbolic.stratification import StratificationLevel


def create_test_metadata(provenance: str = "test") -> RuleMetadata:
    """Create test metadata for rules."""
    return RuleMetadata(
        provenance=provenance,
        timestamp=datetime.utcnow().isoformat(),
        validation_score=1.0,
        author="test",
        tags=["test"],
        notes="Test rule",
    )


def create_test_rule(
    rule_id: str,
    asp_text: str,
    level: StratificationLevel,
    confidence: float,
) -> ASPRule:
    """Create a test ASP rule with proper metadata."""
    return ASPRule(
        rule_id=rule_id,
        asp_text=asp_text,
        stratification_level=level,
        confidence=confidence,
        metadata=create_test_metadata(),
    )


class TestStratifiedASPCore:
    """Tests for StratifiedASPCore class."""

    @pytest.fixture
    def core(self):
        """Create a stratified ASP core instance."""
        return StratifiedASPCore()

    def test_initialization(self, core):
        """Test core initialization."""
        assert core is not None
        # Should have no rules initially
        for level in StratificationLevel:
            rules = core.get_rules_by_layer(level)
            assert len(rules) == 0

    def test_add_constitutional_rule_with_bypass(self, core):
        """Test adding constitutional rule with bypass."""
        rule = create_test_rule(
            rule_id="const_1",
            asp_text="fundamental(X) :- entity(X).",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
        )

        result = core.add_rule(rule, StratificationLevel.CONSTITUTIONAL, bypass_checks=True)

        assert result.success
        assert result.rule_id == "const_1"
        const_rules = core.get_rules_by_layer(StratificationLevel.CONSTITUTIONAL)
        assert len(const_rules) == 1

    def test_add_tactical_rule(self, core):
        """Test adding tactical rule."""
        rule = create_test_rule(
            rule_id="tact_1",
            asp_text="process(X) :- input(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.85,
        )

        result = core.add_rule(rule, StratificationLevel.TACTICAL)

        assert result.success
        tact_rules = core.get_rules_by_layer(StratificationLevel.TACTICAL)
        assert len(tact_rules) == 1

    def test_reject_low_confidence_rule(self, core):
        """Test that ASPRule validates confidence on creation."""
        # ASPRule validates confidence in __post_init__, so we expect ValueError
        with pytest.raises(ValueError, match="tactical level requires confidence"):
            create_test_rule(
                rule_id="bad_1",
                asp_text="process(X) :- input(X).",
                level=StratificationLevel.TACTICAL,
                confidence=0.5,  # Below 0.8 threshold
            )

    def test_reject_autonomous_constitutional_modification(self, core):
        """Test that autonomous constitutional modifications are rejected."""
        rule = create_test_rule(
            rule_id="const_2",
            asp_text="fundamental(truth).",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
        )

        result = core.add_rule(rule, StratificationLevel.CONSTITUTIONAL, is_autonomous=True)

        assert not result.success
        assert "autonomous" in result.reason.lower() or "not allowed" in result.reason.lower()

    def test_get_rules_by_layer(self, core):
        """Test retrieving rules by stratification layer."""
        # Add rules to different layers
        tactical_rule = create_test_rule(
            rule_id="tact_1",
            asp_text="process(X) :- input(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.85,
        )
        strategic_rule = create_test_rule(
            rule_id="strat_1",
            asp_text="policy(X) :- rule(X).",
            level=StratificationLevel.STRATEGIC,
            confidence=0.95,
        )

        core.add_rule(tactical_rule, StratificationLevel.TACTICAL)
        core.add_rule(strategic_rule, StratificationLevel.STRATEGIC)

        # Check tactical layer
        tact_rules = core.get_rules_by_layer(StratificationLevel.TACTICAL)
        assert len(tact_rules) == 1
        assert tact_rules[0].rule_id == "tact_1"

        # Check strategic layer
        strat_rules = core.get_rules_by_layer(StratificationLevel.STRATEGIC)
        assert len(strat_rules) == 1
        assert strat_rules[0].rule_id == "strat_1"

    def test_get_all_rules(self, core):
        """Test retrieving all rules across layers."""
        # Add multiple rules
        for i in range(3):
            rule = create_test_rule(
                rule_id=f"op_{i}",
                asp_text=f"op_{i}(X) :- input(X).",
                level=StratificationLevel.OPERATIONAL,
                confidence=0.75,
            )
            core.add_rule(rule, StratificationLevel.OPERATIONAL)

        all_rules = core.get_all_rules()
        assert len(all_rules) == 3

    def test_modification_stats_tracking(self, core):
        """Test that modification stats are tracked correctly."""
        rule = create_test_rule(
            rule_id="tact_1",
            asp_text="process(X) :- input(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.85,
        )

        core.add_rule(rule, StratificationLevel.TACTICAL)
        stats = core.get_modification_stats()

        # Tactical should have 1 modification
        assert "tactical" in stats
        assert stats["tactical"].total_modifications == 1
        assert stats["tactical"].rules_current == 1
        assert stats["tactical"].last_modification is not None

    def test_layer_mismatch_warning(self, core):
        """Test warning when rule's level doesn't match target layer."""
        # Create a tactical rule but try to add to strategic layer
        rule = create_test_rule(
            rule_id="tact_1",
            asp_text="process(X) :- input(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.85,
        )

        # This should still succeed with bypass, but log a warning
        result = core.add_rule(rule, StratificationLevel.STRATEGIC, bypass_checks=True)
        assert result.success

    def test_multiple_rules_same_layer(self, core):
        """Test adding multiple rules to the same layer."""
        for i in range(5):
            rule = create_test_rule(
                rule_id=f"op_{i}",
                asp_text=f"op_{i}(X) :- input(X).",
                level=StratificationLevel.OPERATIONAL,
                confidence=0.75,
            )
            result = core.add_rule(rule, StratificationLevel.OPERATIONAL)
            assert result.success

        op_rules = core.get_rules_by_layer(StratificationLevel.OPERATIONAL)
        assert len(op_rules) == 5

    def test_duplicate_rule_id(self, core):
        """Test that duplicate rule IDs are handled."""
        rule1 = create_test_rule(
            rule_id="dup_1",
            asp_text="process(X) :- input(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.85,
        )
        rule2 = create_test_rule(
            rule_id="dup_1",  # Same ID
            asp_text="other(X) :- thing(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.85,
        )

        core.add_rule(rule1, StratificationLevel.TACTICAL)
        core.add_rule(rule2, StratificationLevel.TACTICAL)

        # Both should succeed (or second should replace first, depending on implementation)
        # At minimum, we should have at least one rule
        tact_rules = core.get_rules_by_layer(StratificationLevel.TACTICAL)
        assert len(tact_rules) >= 1

    def test_empty_layer_stats(self, core):
        """Test stats for layers with no rules."""
        stats = core.get_modification_stats()

        for level_name in ["constitutional", "strategic", "tactical", "operational"]:
            assert level_name in stats
            assert stats[level_name].total_modifications == 0
            assert stats[level_name].rules_current == 0
            assert stats[level_name].last_modification is None

    def test_predicate_layer_lookup(self, core):
        """Test finding which layer defines a predicate."""
        # Add a rule that defines a predicate
        rule = create_test_rule(
            rule_id="strat_1",
            asp_text="validated(X) :- input(X), check(X).",
            level=StratificationLevel.STRATEGIC,
            confidence=0.95,
        )

        core.add_rule(rule, StratificationLevel.STRATEGIC)

        # The core should be able to find that 'validated' is in strategic layer
        pred_layer = core._find_predicate_layer("validated")
        assert pred_layer == StratificationLevel.STRATEGIC

    def test_predicate_not_found(self, core):
        """Test finding a predicate that doesn't exist."""
        pred_layer = core._find_predicate_layer("nonexistent_predicate")
        assert pred_layer is None

    def test_add_rule_result_structure(self, core):
        """Test AddRuleResult structure."""
        rule = create_test_rule(
            rule_id="test_1",
            asp_text="test(X) :- input(X).",
            level=StratificationLevel.OPERATIONAL,
            confidence=0.75,
        )

        result = core.add_rule(rule, StratificationLevel.OPERATIONAL)

        assert hasattr(result, "success")
        assert hasattr(result, "rule_id")
        assert hasattr(result, "reason")
        assert isinstance(result.success, bool)

    def test_high_confidence_strategic_rule(self, core):
        """Test adding strategic rule with high confidence."""
        rule = create_test_rule(
            rule_id="strat_1",
            asp_text="policy(X) :- must_comply(X).",
            level=StratificationLevel.STRATEGIC,
            confidence=0.95,
        )

        result = core.add_rule(rule, StratificationLevel.STRATEGIC)
        assert result.success

    def test_min_confidence_operational(self, core):
        """Test operational layer accepts minimum confidence (0.7)."""
        rule = create_test_rule(
            rule_id="op_1",
            asp_text="cache(X) :- compute(X).",
            level=StratificationLevel.OPERATIONAL,
            confidence=0.70,
        )

        result = core.add_rule(rule, StratificationLevel.OPERATIONAL)
        assert result.success

    def test_get_layer_stats_after_additions(self, core):
        """Test modification stats after adding multiple rules."""
        # Use operational layer (no cooldown) to add multiple rules quickly
        for i in range(3):
            rule = create_test_rule(
                rule_id=f"op_{i}",
                asp_text=f"rule_{i}(X) :- input(X).",
                level=StratificationLevel.OPERATIONAL,
                confidence=0.75,
            )
            core.add_rule(rule, StratificationLevel.OPERATIONAL)

        stats = core.get_modification_stats()
        op_stats = stats["operational"]

        assert op_stats.total_modifications == 3
        assert op_stats.rules_current == 3

    def test_cooldown_enforcement(self, core):
        """Test that cooldown periods are enforced."""
        # Add first tactical rule (should succeed)
        rule1 = create_test_rule(
            rule_id="tact_1",
            asp_text="rule_1(X) :- input(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.85,
        )
        result1 = core.add_rule(rule1, StratificationLevel.TACTICAL)
        assert result1.success

        # Try to add second tactical rule immediately (should fail due to cooldown)
        rule2 = create_test_rule(
            rule_id="tact_2",
            asp_text="rule_2(X) :- input(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.85,
        )
        result2 = core.add_rule(rule2, StratificationLevel.TACTICAL)
        assert not result2.success
        assert "cooldown" in result2.reason.lower() or "remaining" in result2.reason.lower()
