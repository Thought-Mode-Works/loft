"""
Integration tests for ASPRule with stratified core.

Tests that ASPRule now works seamlessly with StratifiedASPCore.
"""

from datetime import datetime

from loft.symbolic.asp_rule import ASPRule, RuleMetadata
from loft.symbolic.stratification import StratificationLevel
from loft.symbolic.stratified_core import StratifiedASPCore


class TestASPRuleIntegration:
    """Test ASPRule integrates with stratified core."""

    def test_asprule_extracts_predicates_automatically(self):
        """Test that ASPRule automatically extracts predicates."""
        rule = ASPRule(
            rule_id="test_1",
            asp_text="valid_contract(C) :- signed(C), consideration(C).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.85,
            metadata=RuleMetadata(
                provenance="test",
                timestamp=datetime.now().isoformat(),
            ),
        )

        # Should auto-extract predicates
        assert "valid_contract" in rule.new_predicates
        assert "signed" in rule.predicates_used
        assert "consideration" in rule.predicates_used

    def test_asprule_with_stratified_core(self):
        """Test ASPRule works with StratifiedASPCore."""
        core = StratifiedASPCore()

        rule = ASPRule(
            rule_id="test_rule",
            asp_text="enforceable(C) :- valid(C), not void(C).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.85,
            metadata=RuleMetadata(
                provenance="llm",
                timestamp=datetime.now().isoformat(),
            ),
        )

        # Should be able to add rule to core
        result = core.add_rule(
            rule, target_layer=StratificationLevel.TACTICAL, is_autonomous=True
        )

        assert result.success is True
        assert len(core.get_rules_by_layer(StratificationLevel.TACTICAL)) == 1

    def test_dependency_validation_with_real_rules(self):
        """Test dependency validation works with real ASPRules."""
        core = StratifiedASPCore()

        # Add base rule at tactical level
        base_rule = ASPRule(
            rule_id="base",
            asp_text="base_pred(X) :- input(X).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.85,
            metadata=RuleMetadata(
                provenance="test", timestamp=datetime.now().isoformat()
            ),
        )

        result1 = core.add_rule(
            base_rule, target_layer=StratificationLevel.TACTICAL, is_autonomous=True
        )
        assert result1.success is True

        # Try to add strategic rule that depends on tactical (should fail)
        strategic_rule = ASPRule(
            rule_id="strategic",
            asp_text="high_level(X) :- base_pred(X).",
            stratification_level=StratificationLevel.STRATEGIC,
            confidence=0.95,
            metadata=RuleMetadata(
                provenance="test", timestamp=datetime.now().isoformat()
            ),
        )

        result2 = core.add_rule(
            strategic_rule,
            target_layer=StratificationLevel.STRATEGIC,
            is_autonomous=True,
        )

        # Should fail due to dependency violation
        assert result2.success is False
        assert "violation" in result2.reason.lower()

    def test_fact_predicate_extraction(self):
        """Test predicate extraction from facts (no body)."""
        fact = ASPRule(
            rule_id="fact_1",
            asp_text="contract(c1).",
            stratification_level=StratificationLevel.OPERATIONAL,
            confidence=0.75,
            metadata=RuleMetadata(
                provenance="test", timestamp=datetime.now().isoformat()
            ),
        )

        assert "contract" in fact.new_predicates
        assert len(fact.predicates_used) == 0  # Facts have no body

    def test_complex_rule_predicate_extraction(self):
        """Test predicate extraction from complex rules."""
        rule = ASPRule(
            rule_id="complex",
            asp_text="enforceable(C) :- contract(C), signed(C, P), party(P), not void(C).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.85,
            metadata=RuleMetadata(
                provenance="test", timestamp=datetime.now().isoformat()
            ),
        )

        assert "enforceable" in rule.new_predicates
        assert "contract" in rule.predicates_used
        assert "signed" in rule.predicates_used
        assert "party" in rule.predicates_used
        assert "void" in rule.predicates_used

    def test_stratification_level_consistency(self):
        """Test StratificationLevel is consistent across modules."""
        # Should be able to use StratificationLevel from either import
        from loft.symbolic.asp_rule import ASPRule
        from loft.symbolic.stratification import StratificationLevel as StratLevel

        rule = ASPRule(
            rule_id="test",
            asp_text="test(X) :- input(X).",
            stratification_level=StratLevel.TACTICAL,
            confidence=0.85,
            metadata=RuleMetadata(
                provenance="test", timestamp=datetime.now().isoformat()
            ),
        )

        # Should work without issues
        assert rule.stratification_level == StratLevel.TACTICAL
        assert rule.stratification_level.value == "tactical"
