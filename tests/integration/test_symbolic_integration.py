"""
Integration tests for symbolic core.

Tests end-to-end scenarios with realistic legal cases.
"""

from datetime import datetime
from loft.symbolic import (
    ASPCore,
    ASPRule,
    StratificationLevel,
    RuleMetadata,
    create_statute_of_frauds_rules,
    create_contract_basics_rules,
)


class TestStatuteOfFraudsScenarios:
    """Integration tests for statute of frauds scenarios."""

    def test_land_sale_with_signed_writing(self) -> None:
        """Test land sale that satisfies statute of frauds."""
        core = ASPCore()

        # Load statute of frauds rules
        sof_program = create_statute_of_frauds_rules()
        for rule in sof_program.rules:
            core.add_rule(rule)

        # Add facts for a valid land sale
        facts = [
            "contract(land_deal_1).",
            "land_sale(land_deal_1).",
            "writing(deed_1).",
            "signed(deed_1, buyer).",
            "signed(deed_1, seller).",
            "party_to_contract(land_deal_1, buyer).",
            "party_to_contract(land_deal_1, seller).",
            "references(deed_1, land_deal_1).",
            "has_term(deed_1, parties).",
            "has_term(deed_1, subject_matter).",
            "has_term(deed_1, consideration).",
        ]
        core.add_facts(facts)

        # Load and query
        core.load_rules()
        result = core.query("satisfies_statute_of_frauds")

        # Should satisfy statute of frauds
        assert result.satisfiable
        assert len(result.symbols) > 0
        assert any("land_deal_1" in str(s) for s in result.symbols)

    def test_land_sale_without_writing(self) -> None:
        """Test land sale without writing (should not satisfy)."""
        core = ASPCore()

        # Load statute of frauds rules
        sof_program = create_statute_of_frauds_rules()
        for rule in sof_program.rules:
            core.add_rule(rule)

        # Add facts for invalid land sale (no writing)
        facts = [
            "contract(land_deal_2).",
            "land_sale(land_deal_2).",
            # No writing facts
        ]
        core.add_facts(facts)

        core.load_rules()
        result = core.query("satisfies_statute_of_frauds")

        # Should NOT satisfy statute of frauds
        assert result.satisfiable  # Program is consistent
        # But no satisfies_statute_of_frauds(land_deal_2) in answer sets
        matching = [s for s in result.symbols if "land_deal_2" in str(s)]
        assert len(matching) == 0

    def test_goods_sale_under_500(self) -> None:
        """Test goods sale under $500 (not within statute)."""
        core = ASPCore()

        sof_program = create_statute_of_frauds_rules()
        for rule in sof_program.rules:
            core.add_rule(rule)

        # Add facts
        facts = [
            "contract(goods_sale_1).",
            "goods_sale(goods_sale_1, 400).",  # Under $500
        ]
        core.add_facts(facts)

        core.load_rules()

        # Query if within statute
        result = core.query("within_statute")

        # Should NOT be within statute (<$500)
        assert result.satisfiable
        matching = [s for s in result.symbols if "goods_sale_1" in str(s)]
        assert len(matching) == 0

    def test_goods_sale_over_500_with_writing(self) -> None:
        """Test goods sale over $500 with signed writing."""
        core = ASPCore()

        sof_program = create_statute_of_frauds_rules()
        for rule in sof_program.rules:
            core.add_rule(rule)

        facts = [
            "contract(goods_sale_2).",
            "goods_sale(goods_sale_2, 750).",  # Over $500
            "writing(invoice_1).",
            "signed(invoice_1, merchant).",
            "party_to_contract(goods_sale_2, merchant).",
            "references(invoice_1, goods_sale_2).",
            "has_term(invoice_1, parties).",
            "has_term(invoice_1, subject_matter).",
            "has_term(invoice_1, consideration).",
        ]
        core.add_facts(facts)

        core.load_rules()
        result = core.query("satisfies_statute_of_frauds")

        assert result.satisfiable
        assert any("goods_sale_2" in str(s) for s in result.symbols)

    def test_part_performance_exception(self) -> None:
        """Test part performance exception to statute of frauds."""
        core = ASPCore()

        sof_program = create_statute_of_frauds_rules()
        for rule in sof_program.rules:
            core.add_rule(rule)

        facts = [
            "contract(land_deal_3).",
            "land_sale(land_deal_3).",
            # No signed writing, but part performance
            "part_performance(land_deal_3).",
        ]
        core.add_facts(facts)

        core.load_rules()
        result = core.query("satisfies_statute_of_frauds")

        # Should satisfy via exception
        assert result.satisfiable
        assert any("land_deal_3" in str(s) for s in result.symbols)


class TestEnforceabilityReasoning:
    """Tests for contract enforceability with default reasoning."""

    def test_default_enforceability(self) -> None:
        """Test default enforceability (not proven unenforceable)."""
        core = ASPCore()

        sof_program = create_statute_of_frauds_rules()
        for rule in sof_program.rules:
            core.add_rule(rule)

        # Contract not within statute - should be enforceable by default
        facts = [
            "contract(simple_contract).",
            # Not a land sale, goods sale, etc.
        ]
        core.add_facts(facts)

        core.load_rules()
        result = core.query("enforceable")

        assert result.satisfiable
        assert any("simple_contract" in str(s) for s in result.symbols)

    def test_unenforceable_within_statute_no_writing(self) -> None:
        """Test unenforceability when within statute without writing."""
        core = ASPCore()

        sof_program = create_statute_of_frauds_rules()
        for rule in sof_program.rules:
            core.add_rule(rule)

        facts = [
            "contract(bad_land_deal).",
            "land_sale(bad_land_deal).",
            # No writing - within statute but doesn't satisfy
        ]
        core.add_facts(facts)

        core.load_rules()
        result = core.query("unenforceable")

        assert result.satisfiable
        assert any("bad_land_deal" in str(s) for s in result.symbols)


class TestContractFormation:
    """Tests for basic contract formation rules."""

    def test_valid_contract_formation(self) -> None:
        """Test valid contract with all required elements."""
        core = ASPCore()

        basics_program = create_contract_basics_rules()
        for rule in basics_program.rules:
            core.add_rule(rule)

        facts = [
            "contract(deal_1).",
            "mutual_assent(deal_1).",
            "consideration(deal_1).",
            "legal_capacity(deal_1).",
            "legal_purpose(deal_1).",
        ]
        core.add_facts(facts)

        core.load_rules()
        result = core.query("valid_contract")

        assert result.satisfiable
        assert any("deal_1" in str(s) for s in result.symbols)

    def test_invalid_contract_no_consideration(self) -> None:
        """Test invalid contract missing consideration."""
        core = ASPCore()

        basics_program = create_contract_basics_rules()
        for rule in basics_program.rules:
            core.add_rule(rule)

        facts = [
            "contract(bad_deal).",
            "mutual_assent(bad_deal).",
            # Missing consideration
            "legal_capacity(bad_deal).",
            "legal_purpose(bad_deal).",
        ]
        core.add_facts(facts)

        core.load_rules()
        result = core.query("valid_contract")

        # Should be satisfiable but no valid_contract(bad_deal)
        assert result.satisfiable
        matching = [s for s in result.symbols if "bad_deal" in str(s)]
        assert len(matching) == 0


class TestStratifiedReasoning:
    """Tests for stratified reasoning across layers."""

    def test_multi_layer_reasoning(self) -> None:
        """Test reasoning across multiple stratification layers."""
        core = ASPCore()

        # Add constitutional rule
        core.add_rule(
            ASPRule(
                "const_1",
                "fundamental_right(X) :- human(X).",
                StratificationLevel.CONSTITUTIONAL,
                1.0,
                RuleMetadata("human", datetime.utcnow().isoformat()),
            )
        )

        # Add strategic rule
        core.add_rule(
            ASPRule(
                "strat_1",
                "protected(X) :- fundamental_right(X).",
                StratificationLevel.STRATEGIC,
                0.95,
                RuleMetadata("human", datetime.utcnow().isoformat()),
            )
        )

        # Add tactical rule
        core.add_rule(
            ASPRule(
                "tac_1",
                "requires_strict_scrutiny(Law) :- affects(Law, X), protected(X).",
                StratificationLevel.TACTICAL,
                0.90,
                RuleMetadata("llm", datetime.utcnow().isoformat()),
            )
        )

        facts = [
            "human(john).",
            "affects(law_1, john).",
        ]
        core.add_facts(facts)

        core.load_rules()

        # Query across layers
        result = core.query("requires_strict_scrutiny")

        assert result.satisfiable
        assert any("law_1" in str(s) for s in result.symbols)


class TestConsistencyChecking:
    """Tests for consistency checking with contradictions."""

    def test_consistent_program(self) -> None:
        """Test consistent ASP program."""
        core = ASPCore()

        core.add_facts(["fact(a).", "fact(b)."])
        core.add_rule(
            ASPRule(
                "r1",
                "derived(X) :- fact(X).",
                StratificationLevel.OPERATIONAL,
                0.8,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )

        core.load_rules()

        assert core.check_consistency()
        assert core.count_answer_sets() > 0

    def test_inconsistent_program_with_constraint(self) -> None:
        """Test inconsistent program with unsatisfiable constraint."""
        core = ASPCore()

        core.add_facts(["bad_thing."])
        core.add_rule(
            ASPRule(
                "constraint",
                ":- bad_thing.",  # Constraint violated
                StratificationLevel.TACTICAL,
                0.95,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )

        core.load_rules()

        # Should be unsatisfiable
        assert not core.check_consistency()


class TestPerformance:
    """Performance tests for ASP core."""

    def test_large_rule_set_performance(self) -> None:
        """Test performance with 100+ rules."""
        import time

        core = ASPCore()

        # Add 100 simple rules
        for i in range(100):
            core.add_rule(
                ASPRule(
                    f"rule_{i}",
                    f"derived_{i}(X) :- fact_{i}(X).",
                    StratificationLevel.OPERATIONAL,
                    0.7,
                    RuleMetadata("test", datetime.utcnow().isoformat()),
                )
            )

        # Add some facts
        for i in range(20):
            core.add_facts([f"fact_{i}(a)."])

        # Measure load time
        start = time.time()
        core.load_rules()
        load_time = time.time() - start

        # Measure query time
        start = time.time()
        result = core.query()
        query_time = time.time() - start

        # Should be fast enough (<2s total)
        total_time = load_time + query_time
        assert total_time < 2.0
        assert result.satisfiable

    def test_query_performance(self) -> None:
        """Test query performance."""
        import time

        core = ASPCore()

        # Create moderate complexity program
        sof_program = create_statute_of_frauds_rules()
        for rule in sof_program.rules:
            core.add_rule(rule)

        facts = [
            "contract(c1).",
            "land_sale(c1).",
            "writing(w1).",
            "signed(w1, p1).",
            "party_to_contract(c1, p1).",
            "references(w1, c1).",
            "has_term(w1, parties).",
            "has_term(w1, subject_matter).",
            "has_term(w1, consideration).",
        ]
        core.add_facts(facts)

        core.load_rules()

        # Measure query time
        start = time.time()
        for _ in range(10):  # Run 10 queries
            core.query("satisfies_statute_of_frauds")
        elapsed = time.time() - start

        # Should be fast (<1s for 10 queries)
        assert elapsed < 1.0


class TestCompositionality:
    """Tests for compositional operations."""

    def test_compose_multiple_programs(self) -> None:
        """Test composing multiple ASP programs."""
        core = ASPCore()

        # Load statute of frauds rules
        sof = create_statute_of_frauds_rules()
        for rule in sof.rules:
            core.add_rule(rule)

        # Load contract basics
        basics = create_contract_basics_rules()
        for rule in basics.rules:
            core.add_rule(rule)

        # Should work together
        facts = [
            "contract(c1).",
            "land_sale(c1).",
            "mutual_assent(c1).",
        ]
        core.add_facts(facts)

        core.load_rules()

        # Both rule sets should be active
        result_sof = core.query("within_statute")
        result_basic = core.query("mutual_assent")

        assert result_sof.satisfiable
        assert result_basic.satisfiable
