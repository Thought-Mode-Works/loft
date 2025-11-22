"""
Unit tests for symbolic core components.

Tests ASPRule, ASPProgram, StratifiedASPCore, ASPCore, and legal primitives.
"""

import pytest
from datetime import datetime
from loft.symbolic import (
    ASPRule,
    StratificationLevel,
    RuleMetadata,
    ASPProgram,
    StratifiedASPCore,
    ASPCore,
    create_rule_id,
    compose_programs,
    create_statute_of_frauds_rules,
    create_contract_basics_rules,
    create_meta_reasoning_rules,
)


class TestRuleMetadata:
    """Tests for RuleMetadata class."""

    def test_metadata_creation(self) -> None:
        """Test creating metadata."""
        metadata = RuleMetadata(
            provenance="llm",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=0.95,
            author="test_author",
            tags=["test", "contract"],
            notes="Test notes",
        )

        assert metadata.provenance == "llm"
        assert metadata.validation_score == 0.95
        assert "test" in metadata.tags

    def test_metadata_serialization(self) -> None:
        """Test metadata to/from dict."""
        metadata = RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=1.0,
        )

        data = metadata.to_dict()
        restored = RuleMetadata.from_dict(data)

        assert restored.provenance == metadata.provenance
        assert restored.validation_score == metadata.validation_score


class TestASPRule:
    """Tests for ASPRule class."""

    def test_rule_creation(self) -> None:
        """Test creating an ASP rule."""
        rule = ASPRule(
            rule_id="test_rule_1",
            asp_text="contract(c1).",
            stratification_level=StratificationLevel.OPERATIONAL,
            confidence=0.9,
            metadata=RuleMetadata(
                provenance="llm",
                timestamp=datetime.utcnow().isoformat(),
            ),
        )

        assert rule.rule_id == "test_rule_1"
        assert rule.confidence == 0.9
        assert rule.stratification_level == StratificationLevel.OPERATIONAL

    def test_rule_confidence_validation(self) -> None:
        """Test confidence validation."""
        # Invalid confidence range
        with pytest.raises(ValueError, match="Confidence must be between"):
            ASPRule(
                rule_id="test",
                asp_text="fact.",
                stratification_level=StratificationLevel.OPERATIONAL,
                confidence=1.5,  # Invalid
                metadata=RuleMetadata("test", datetime.utcnow().isoformat()),
            )

        # Confidence too low for stratification level
        with pytest.raises(ValueError, match="requires confidence"):
            ASPRule(
                rule_id="test",
                asp_text="fact.",
                stratification_level=StratificationLevel.STRATEGIC,  # Requires >0.9
                confidence=0.7,  # Too low
                metadata=RuleMetadata("test", datetime.utcnow().isoformat()),
            )

    def test_rule_to_clingo(self) -> None:
        """Test converting rule to Clingo format."""
        rule = ASPRule(
            rule_id="test_rule",
            asp_text="enforceable(C) :- contract(C), not unenforceable(C).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.85,
            metadata=RuleMetadata("human", datetime.utcnow().isoformat()),
        )

        clingo_text = rule.to_clingo()

        assert "test_rule" in clingo_text
        assert "tactical" in clingo_text
        assert "0.85" in clingo_text
        assert "enforceable(C)" in clingo_text

    def test_rule_serialization(self) -> None:
        """Test rule to/from dict."""
        rule = ASPRule(
            rule_id="r1",
            asp_text="test(X).",
            stratification_level=StratificationLevel.OPERATIONAL,
            confidence=0.8,
            metadata=RuleMetadata("llm", datetime.utcnow().isoformat()),
        )

        data = rule.to_dict()
        restored = ASPRule.from_dict(data)

        assert restored.rule_id == rule.rule_id
        assert restored.asp_text == rule.asp_text
        assert restored.stratification_level == rule.stratification_level

    def test_rule_is_fact(self) -> None:
        """Test detecting facts."""
        fact = ASPRule(
            rule_id="f1",
            asp_text="contract(c1).",
            stratification_level=StratificationLevel.OPERATIONAL,
            confidence=0.9,
            metadata=RuleMetadata("system", datetime.utcnow().isoformat()),
        )

        rule = ASPRule(
            rule_id="r1",
            asp_text="valid(X) :- contract(X).",
            stratification_level=StratificationLevel.OPERATIONAL,
            confidence=0.9,
            metadata=RuleMetadata("system", datetime.utcnow().isoformat()),
        )

        assert fact.is_fact()
        assert not rule.is_fact()

    def test_rule_is_constraint(self) -> None:
        """Test detecting constraints."""
        constraint = ASPRule(
            rule_id="c1",
            asp_text=":- contract(C), invalid(C).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.95,
            metadata=RuleMetadata("human", datetime.utcnow().isoformat()),
        )

        rule = ASPRule(
            rule_id="r1",
            asp_text="valid(X) :- contract(X).",
            stratification_level=StratificationLevel.OPERATIONAL,
            confidence=0.9,
            metadata=RuleMetadata("system", datetime.utcnow().isoformat()),
        )

        assert constraint.is_constraint()
        assert not rule.is_constraint()

    def test_rule_is_choice_rule(self) -> None:
        """Test detecting choice rules."""
        choice = ASPRule(
            rule_id="ch1",
            asp_text="{signed(W, P) : party(P)} :- writing(W).",
            stratification_level=StratificationLevel.OPERATIONAL,
            confidence=0.8,
            metadata=RuleMetadata("llm", datetime.utcnow().isoformat()),
        )

        rule = ASPRule(
            rule_id="r1",
            asp_text="valid(X) :- contract(X).",
            stratification_level=StratificationLevel.OPERATIONAL,
            confidence=0.9,
            metadata=RuleMetadata("system", datetime.utcnow().isoformat()),
        )

        assert choice.is_choice_rule()
        assert not rule.is_choice_rule()

    def test_extract_predicates(self) -> None:
        """Test predicate extraction."""
        rule = ASPRule(
            rule_id="r1",
            asp_text="enforceable(C) :- contract(C), signed(C), not invalid(C).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.9,
            metadata=RuleMetadata("human", datetime.utcnow().isoformat()),
        )

        predicates = rule.extract_predicates()

        assert "enforceable" in predicates
        assert "contract" in predicates
        assert "signed" in predicates
        assert "invalid" in predicates


class TestASPProgram:
    """Tests for ASPProgram class."""

    def test_program_creation(self) -> None:
        """Test creating an ASP program."""
        program = ASPProgram(
            name="test_program",
            description="Test program",
        )

        assert program.name == "test_program"
        assert len(program.rules) == 0
        assert len(program.facts) == 0

    def test_add_rule(self) -> None:
        """Test adding rules to program."""
        program = ASPProgram()

        rule = ASPRule(
            rule_id="r1",
            asp_text="test(X).",
            stratification_level=StratificationLevel.OPERATIONAL,
            confidence=0.8,
            metadata=RuleMetadata("test", datetime.utcnow().isoformat()),
        )

        program.add_rule(rule)
        assert len(program.rules) == 1

    def test_add_fact(self) -> None:
        """Test adding facts to program."""
        program = ASPProgram()

        program.add_fact("contract(c1)")
        program.add_fact("party(john).")  # Already has period

        assert len(program.facts) == 2
        assert all(f.endswith(".") for f in program.facts)

    def test_program_to_asp(self) -> None:
        """Test converting program to ASP text."""
        program = ASPProgram(name="test")

        program.add_fact("contract(c1).")
        program.add_rule(
            ASPRule(
                "r1",
                "valid(X) :- contract(X).",
                StratificationLevel.OPERATIONAL,
                0.9,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )

        asp_text = program.to_asp()

        assert "contract(c1)." in asp_text
        assert "valid(X)" in asp_text
        assert "Facts" in asp_text

    def test_get_rules_by_level(self) -> None:
        """Test filtering rules by level."""
        program = ASPProgram()

        program.add_rule(
            ASPRule(
                "r1",
                "tactical_rule.",
                StratificationLevel.TACTICAL,
                0.9,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )
        program.add_rule(
            ASPRule(
                "r2",
                "operational_rule.",
                StratificationLevel.OPERATIONAL,
                0.8,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )

        tactical = program.get_rules_by_level(StratificationLevel.TACTICAL)
        assert len(tactical) == 1

    def test_program_serialization(self) -> None:
        """Test program to/from JSON."""
        program = ASPProgram(name="test")
        program.add_fact("fact1.")
        program.add_rule(
            ASPRule(
                "r1",
                "rule1.",
                StratificationLevel.OPERATIONAL,
                0.8,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )

        json_str = program.to_json()
        restored = ASPProgram.from_json(json_str)

        assert restored.name == program.name
        assert len(restored.rules) == len(program.rules)
        assert len(restored.facts) == len(program.facts)


class TestStratifiedASPCore:
    """Tests for StratifiedASPCore class."""

    def test_stratified_core_creation(self) -> None:
        """Test creating stratified core."""
        core = StratifiedASPCore()

        assert core.constitutional is not None
        assert core.strategic is not None
        assert core.tactical is not None
        assert core.operational is not None

    def test_add_rule_to_correct_layer(self) -> None:
        """Test adding rule to appropriate layer."""
        core = StratifiedASPCore()

        rule = ASPRule(
            "r1",
            "test.",
            StratificationLevel.TACTICAL,
            0.85,
            RuleMetadata("test", datetime.utcnow().isoformat()),
        )

        core.add_rule(rule)

        assert len(core.tactical.rules) == 1
        assert len(core.operational.rules) == 0

    def test_get_full_program(self) -> None:
        """Test composing all layers."""
        core = StratifiedASPCore()

        # Add rules to different layers
        core.add_rule(
            ASPRule(
                "r1",
                "const_rule.",
                StratificationLevel.CONSTITUTIONAL,
                1.0,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )
        core.add_rule(
            ASPRule(
                "r2",
                "tactical_rule.",
                StratificationLevel.TACTICAL,
                0.85,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )

        full_program = core.get_full_program()

        assert "CONSTITUTIONAL" in full_program
        assert "TACTICAL" in full_program
        assert "const_rule" in full_program
        assert "tactical_rule" in full_program

    def test_get_all_rules(self) -> None:
        """Test getting all rules from all layers."""
        core = StratifiedASPCore()

        core.add_rule(
            ASPRule(
                "r1",
                "rule1.",
                StratificationLevel.CONSTITUTIONAL,
                1.0,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )
        core.add_rule(
            ASPRule(
                "r2",
                "rule2.",
                StratificationLevel.OPERATIONAL,
                0.7,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )

        all_rules = core.get_all_rules()
        assert len(all_rules) == 2

    def test_stratified_core_serialization(self) -> None:
        """Test stratified core to/from dict."""
        core = StratifiedASPCore()
        core.add_rule(
            ASPRule(
                "r1",
                "rule.",
                StratificationLevel.TACTICAL,
                0.9,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )

        data = core.to_dict()
        restored = StratifiedASPCore.from_dict(data)

        assert len(restored.get_all_rules()) == len(core.get_all_rules())


class TestComposePrograms:
    """Tests for program composition."""

    def test_compose_two_programs(self) -> None:
        """Test composing two ASP programs."""
        prog1 = ASPProgram(name="prog1")
        prog1.add_fact("fact1.")
        prog1.add_rule(
            ASPRule(
                "r1",
                "rule1.",
                StratificationLevel.OPERATIONAL,
                0.8,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )

        prog2 = ASPProgram(name="prog2")
        prog2.add_fact("fact2.")

        composed = compose_programs(prog1, prog2)

        assert len(composed.rules) == 1
        assert len(composed.facts) == 2
        assert "prog1+prog2" in composed.name


class TestASPCore:
    """Tests for ASPCore class."""

    def test_asp_core_creation(self) -> None:
        """Test creating ASP core."""
        core = ASPCore()
        assert core is not None
        assert not core._loaded

    def test_load_simple_program(self) -> None:
        """Test loading a simple ASP program."""
        core = ASPCore()

        # Add a simple fact
        core.add_facts(["contract(c1)."])

        # Load rules
        core.load_rules()

        assert core._loaded
        assert core.check_consistency()

    def test_query_simple(self) -> None:
        """Test querying ASP program."""
        core = ASPCore()

        # Add facts
        core.add_facts(["contract(c1).", "contract(c2)."])

        # Add rule
        core.add_rule(
            ASPRule(
                "r1",
                "valid(X) :- contract(X).",
                StratificationLevel.OPERATIONAL,
                0.9,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )

        core.load_rules()

        # Query
        result = core.query("valid")

        assert result.satisfiable
        assert len(result.symbols) >= 2  # valid(c1) and valid(c2)

    def test_check_consistency(self) -> None:
        """Test consistency checking."""
        core = ASPCore()

        # Consistent program
        core.add_facts(["fact(a)."])
        core.load_rules()

        assert core.check_consistency()

    def test_count_answer_sets(self) -> None:
        """Test counting answer sets."""
        core = ASPCore()

        core.add_facts(["contract(c1)."])
        core.load_rules()

        count = core.count_answer_sets()
        assert count >= 1

    def test_get_program_text(self) -> None:
        """Test getting program text."""
        core = ASPCore()

        core.add_rule(
            ASPRule(
                "r1",
                "test_rule.",
                StratificationLevel.OPERATIONAL,
                0.8,
                RuleMetadata("test", datetime.utcnow().isoformat()),
            )
        )

        program_text = core.get_program_text()
        assert "test_rule" in program_text


class TestLegalPrimitives:
    """Tests for legal domain primitives."""

    def test_statute_of_frauds_rules(self) -> None:
        """Test statute of frauds rule creation."""
        program = create_statute_of_frauds_rules()

        assert program.name == "statute_of_frauds"
        assert len(program.rules) > 0

        # Check for key rules
        asp_text = program.to_asp()
        assert "satisfies_statute_of_frauds" in asp_text
        assert "within_statute" in asp_text
        assert "exception_applies" in asp_text

    def test_contract_basics_rules(self) -> None:
        """Test contract basics rules."""
        program = create_contract_basics_rules()

        assert program.name == "contract_basics"
        assert len(program.rules) > 0

        asp_text = program.to_asp()
        assert "valid_contract" in asp_text
        assert "mutual_assent" in asp_text
        assert "consideration" in asp_text

    def test_meta_reasoning_rules(self) -> None:
        """Test meta-reasoning rules."""
        program = create_meta_reasoning_rules()

        assert program.name == "meta_reasoning"
        assert len(program.rules) > 0

        asp_text = program.to_asp()
        assert "missing_rule" in asp_text
        assert "low_confidence_rule" in asp_text


class TestCreateRuleId:
    """Tests for rule ID generation."""

    def test_create_unique_ids(self) -> None:
        """Test that generated IDs are unique."""
        id1 = create_rule_id()
        id2 = create_rule_id()

        assert id1 != id2
        assert id1.startswith("rule_")
        assert len(id1) > 5
