"""
Test data generators for consistency testing.

Provides both deterministic fixtures and property-based generators
using hypothesis for generating test rules and core states.
"""

from typing import Optional
from datetime import datetime
from hypothesis import strategies as st
from ..version_control import Rule, CoreState, StratificationLevel, create_state_id


# Hypothesis strategies for generating test data


@st.composite
def stratification_level_strategy(draw) -> StratificationLevel:  # type: ignore[no-untyped-def]
    """Generate a random stratification level."""
    return draw(st.sampled_from(list(StratificationLevel)))  # type: ignore[no-any-return]


@st.composite
def predicate_name_strategy(draw) -> str:  # type: ignore[no-untyped-def]
    """Generate a valid predicate name."""
    # Start with lowercase letter, followed by alphanumeric or underscore
    first_char = draw(st.sampled_from("abcdefghijklmnopqrstuvwxyz"))
    rest = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", max_size=10))
    return first_char + rest  # type: ignore[no-any-return]


@st.composite
def variable_name_strategy(draw) -> str:  # type: ignore[no-untyped-def]
    """Generate a valid variable name (uppercase)."""
    first_char = draw(st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    rest = draw(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", max_size=5))
    return first_char + rest  # type: ignore[no-any-return]


@st.composite
def asp_fact_strategy(draw) -> str:  # type: ignore[no-untyped-def]
    """Generate a simple ASP fact."""
    predicate = draw(predicate_name_strategy())

    # Either unary predicate or predicate with arguments
    use_args = draw(st.booleans())

    if use_args:
        # Generate 1-3 arguments
        num_args = draw(st.integers(min_value=1, max_value=3))
        args = []
        for _ in range(num_args):
            # Argument can be a constant or variable
            is_var = draw(st.booleans())
            if is_var:
                args.append(draw(variable_name_strategy()))
            else:
                args.append(draw(st.sampled_from(["a", "b", "c", "1", "2", "3"])))

        return f"{predicate}({', '.join(args)})."
    else:
        return f"{predicate}."


@st.composite
def asp_rule_strategy(draw) -> str:  # type: ignore[no-untyped-def]
    """Generate a simple ASP rule with head and body."""
    head_predicate = draw(predicate_name_strategy())
    body_predicate = draw(predicate_name_strategy())

    # Ensure head and body are different
    while body_predicate == head_predicate:
        body_predicate = draw(predicate_name_strategy())

    # Generate rule: head :- body.
    use_negation = draw(st.booleans())
    negation = "-" if use_negation else ""

    return f"{head_predicate} :- {negation}{body_predicate}."


@st.composite
def asp_rule_content_strategy(draw) -> str:  # type: ignore[no-untyped-def]
    """Generate ASP rule content (fact or rule)."""
    # 60% rules, 40% facts
    is_rule = draw(st.booleans())

    if is_rule:
        return draw(asp_rule_strategy())  # type: ignore[no-any-return]
    else:
        return draw(asp_fact_strategy())  # type: ignore[no-any-return]


@st.composite
def rule_strategy(  # type: ignore[no-untyped-def, misc]
    draw,
    content: Optional[str] = None,
    level: Optional[StratificationLevel] = None,
) -> Rule:
    """
    Generate a random Rule.

    Args:
        content: Optional fixed content
        level: Optional fixed stratification level
    """
    rule_id = f"rule_{draw(st.integers(min_value=0, max_value=999999))}"

    if content is None:
        content = draw(asp_rule_content_strategy())

    if level is None:
        level = draw(stratification_level_strategy())

    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    provenance = draw(st.sampled_from(["llm", "human", "validation", "system"]))

    return Rule(
        rule_id=rule_id,
        content=content,
        level=level,
        confidence=confidence,
        provenance=provenance,
        timestamp=datetime.utcnow().isoformat(),
    )


@st.composite
def core_state_strategy(  # type: ignore[no-untyped-def, misc]
    draw,
    min_rules: int = 0,
    max_rules: int = 10,
) -> CoreState:
    """
    Generate a random CoreState.

    Args:
        min_rules: Minimum number of rules
        max_rules: Maximum number of rules
    """
    num_rules = draw(st.integers(min_value=min_rules, max_value=max_rules))
    rules = [draw(rule_strategy()) for _ in range(num_rules)]

    # Generate configuration
    config_keys = draw(st.lists(st.text(min_size=1, max_size=10), max_size=5))
    configuration = {
        key: draw(st.one_of(st.text(), st.integers(), st.floats(), st.booleans()))
        for key in config_keys
    }

    # Generate metrics
    metric_keys = draw(st.lists(st.text(min_size=1, max_size=10), max_size=5))
    metrics = {key: draw(st.floats(min_value=0.0, max_value=1.0)) for key in metric_keys}

    return CoreState(
        state_id=create_state_id(),
        timestamp=datetime.utcnow().isoformat(),
        rules=rules,
        configuration=configuration,
        metrics=metrics,
    )


# Deterministic test fixtures


class TestFixtures:
    """Deterministic test fixtures for consistency testing."""

    @staticmethod
    def empty_state() -> CoreState:
        """Create an empty core state."""
        return CoreState(
            state_id=create_state_id(),
            timestamp=datetime.utcnow().isoformat(),
            rules=[],
            configuration={},
            metrics={},
        )

    @staticmethod
    def simple_consistent_state() -> CoreState:
        """Create a simple consistent state with no conflicts."""
        rules = [
            Rule(
                rule_id="r1",
                content="animal(dog).",
                level=StratificationLevel.OPERATIONAL,
                confidence=1.0,
                provenance="human",
                timestamp=datetime.utcnow().isoformat(),
            ),
            Rule(
                rule_id="r2",
                content="mammal(X) :- animal(X).",
                level=StratificationLevel.TACTICAL,
                confidence=0.9,
                provenance="llm",
                timestamp=datetime.utcnow().isoformat(),
            ),
            Rule(
                rule_id="r3",
                content="living_thing(X) :- mammal(X).",
                level=StratificationLevel.STRATEGIC,
                confidence=0.95,
                provenance="llm",
                timestamp=datetime.utcnow().isoformat(),
            ),
        ]

        return CoreState(
            state_id=create_state_id(),
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={"test": "value"},
            metrics={"consistency": 1.0},
        )

    @staticmethod
    def contradictory_state() -> CoreState:
        """Create a state with contradictory rules."""
        rules = [
            Rule(
                rule_id="r1",
                content="alive(x).",
                level=StratificationLevel.OPERATIONAL,
                confidence=1.0,
                provenance="human",
                timestamp=datetime.utcnow().isoformat(),
            ),
            Rule(
                rule_id="r2",
                content="-alive(x).",
                level=StratificationLevel.OPERATIONAL,
                confidence=1.0,
                provenance="human",
                timestamp=datetime.utcnow().isoformat(),
            ),
        ]

        return CoreState(
            state_id=create_state_id(),
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={},
            metrics={},
        )

    @staticmethod
    def incomplete_state() -> CoreState:
        """Create a state with incomplete rules (undefined predicates)."""
        rules = [
            Rule(
                rule_id="r1",
                content="conclusion(X) :- premise(X).",
                level=StratificationLevel.TACTICAL,
                confidence=0.9,
                provenance="llm",
                timestamp=datetime.utcnow().isoformat(),
            ),
            # premise(X) is never defined
        ]

        return CoreState(
            state_id=create_state_id(),
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={},
            metrics={},
        )

    @staticmethod
    def incoherent_state() -> CoreState:
        """Create a state with incoherent stratification."""
        rules = [
            Rule(
                rule_id="r1",
                content="tactical_fact(a).",
                level=StratificationLevel.TACTICAL,
                confidence=0.8,
                provenance="llm",
                timestamp=datetime.utcnow().isoformat(),
            ),
            Rule(
                rule_id="r2",
                content="const_rule(X) :- tactical_fact(X).",
                level=StratificationLevel.CONSTITUTIONAL,
                confidence=1.0,
                provenance="human",
                timestamp=datetime.utcnow().isoformat(),
            ),
        ]

        return CoreState(
            state_id=create_state_id(),
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={},
            metrics={},
        )

    @staticmethod
    def circular_dependency_state() -> CoreState:
        """Create a state with circular dependencies."""
        rules = [
            Rule(
                rule_id="r1",
                content="a(X) :- b(X).",
                level=StratificationLevel.TACTICAL,
                confidence=0.9,
                provenance="llm",
                timestamp=datetime.utcnow().isoformat(),
            ),
            Rule(
                rule_id="r2",
                content="b(X) :- c(X).",
                level=StratificationLevel.TACTICAL,
                confidence=0.9,
                provenance="llm",
                timestamp=datetime.utcnow().isoformat(),
            ),
            Rule(
                rule_id="r3",
                content="c(X) :- a(X).",
                level=StratificationLevel.TACTICAL,
                confidence=0.9,
                provenance="llm",
                timestamp=datetime.utcnow().isoformat(),
            ),
        ]

        return CoreState(
            state_id=create_state_id(),
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={},
            metrics={},
        )
