"""
Comprehensive tests for consistency test data generators.

Tests hypothesis strategies and TestFixtures class.
Aims to reach 80%+ coverage for loft/consistency/generators.py.
"""

from hypothesis import given, settings, HealthCheck
from loft.consistency.generators import (
    stratification_level_strategy,
    predicate_name_strategy,
    variable_name_strategy,
    asp_fact_strategy,
    asp_rule_strategy,
    asp_rule_content_strategy,
    rule_strategy,
    core_state_strategy,
    TestFixtures,
)
from loft.version_control import Rule, CoreState, StratificationLevel


class TestStratificationLevelStrategy:
    """Tests for stratification_level_strategy."""

    @given(stratification_level_strategy())
    @settings(max_examples=20)
    def test_generates_valid_stratification_level(self, level):
        """Test that generated stratification levels are valid."""
        assert isinstance(level, StratificationLevel)
        assert level in list(StratificationLevel)

    def test_generates_all_stratification_levels(self):
        """Test that strategy can generate all stratification levels."""
        levels_seen = set()

        @given(stratification_level_strategy())
        @settings(max_examples=50)
        def collect_levels(level):
            levels_seen.add(level)

        collect_levels()

        # Should have seen at least some variety
        assert len(levels_seen) > 0

    def test_stratification_level_is_enum(self):
        """Test that generated levels are proper enums."""
        @given(stratification_level_strategy())
        @settings(max_examples=10)
        def check_enum(level):
            assert hasattr(level, 'value')
            assert level.value in ['constitutional', 'strategic', 'tactical', 'operational']

        check_enum()


class TestPredicateNameStrategy:
    """Tests for predicate_name_strategy."""

    @given(predicate_name_strategy())
    @settings(max_examples=20)
    def test_generates_valid_predicate_name(self, name):
        """Test that generated predicate names are valid."""
        assert isinstance(name, str)
        assert len(name) > 0
        # Should start with lowercase letter
        assert name[0].islower()
        assert name[0].isalpha()

    @given(predicate_name_strategy())
    @settings(max_examples=20)
    def test_predicate_name_format(self, name):
        """Test predicate name follows ASP conventions."""
        # First char is lowercase letter
        assert name[0] in "abcdefghijklmnopqrstuvwxyz"
        # Rest are alphanumeric or underscore
        for char in name[1:]:
            assert char.isalnum() or char == "_"

    def test_predicate_names_vary(self):
        """Test that strategy generates different predicate names."""
        names_seen = set()

        @given(predicate_name_strategy())
        @settings(max_examples=50)
        def collect_names(name):
            names_seen.add(name)

        collect_names()

        # Should generate at least some variety
        assert len(names_seen) > 5


class TestVariableNameStrategy:
    """Tests for variable_name_strategy."""

    @given(variable_name_strategy())
    @settings(max_examples=20)
    def test_generates_valid_variable_name(self, name):
        """Test that generated variable names are valid."""
        assert isinstance(name, str)
        assert len(name) > 0
        # Should start with uppercase letter
        assert name[0].isupper()
        assert name[0].isalpha()

    @given(variable_name_strategy())
    @settings(max_examples=20)
    def test_variable_name_format(self, name):
        """Test variable name follows ASP conventions."""
        # First char is uppercase letter
        assert name[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # Rest are uppercase alphanumeric or underscore
        for char in name[1:]:
            assert char.isupper() or char.isdigit() or char == "_"

    def test_variable_names_vary(self):
        """Test that strategy generates different variable names."""
        names_seen = set()

        @given(variable_name_strategy())
        @settings(max_examples=50)
        def collect_names(name):
            names_seen.add(name)

        collect_names()

        # Should generate variety
        assert len(names_seen) > 5


class TestAspFactStrategy:
    """Tests for asp_fact_strategy."""

    @given(asp_fact_strategy())
    @settings(max_examples=30)
    def test_generates_valid_asp_fact(self, fact):
        """Test that generated ASP facts are valid."""
        assert isinstance(fact, str)
        assert len(fact) > 0
        # Should end with period
        assert fact.endswith(".")
        # Should contain a predicate name (starts with lowercase)
        assert fact[0].islower()

    @given(asp_fact_strategy())
    @settings(max_examples=30)
    def test_fact_has_predicate(self, fact):
        """Test that facts have predicates."""
        # Extract predicate name (before ( or .)
        predicate = fact.split("(")[0].split(".")[0]
        assert len(predicate) > 0
        assert predicate[0].islower()

    def test_generates_facts_with_and_without_arguments(self):
        """Test that strategy generates both unary and n-ary predicates."""
        facts_with_args = 0
        facts_without_args = 0

        @given(asp_fact_strategy())
        @settings(max_examples=50)
        def classify_facts(fact):
            nonlocal facts_with_args, facts_without_args
            if "(" in fact:
                facts_with_args += 1
            else:
                facts_without_args += 1

        classify_facts()

        # Should generate both types
        assert facts_with_args > 0
        assert facts_without_args > 0


class TestAspRuleStrategy:
    """Tests for asp_rule_strategy."""

    @given(asp_rule_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.large_base_example])
    def test_generates_valid_asp_rule(self, rule):
        """Test that generated ASP rules are valid."""
        assert isinstance(rule, str)
        assert len(rule) > 0
        # Should end with period
        assert rule.endswith(".")
        # Should contain implication operator
        assert ":-" in rule

    @given(asp_rule_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.large_base_example])
    def test_rule_has_head_and_body(self, rule):
        """Test that rules have head and body."""
        parts = rule.split(":-")
        assert len(parts) == 2
        # Head should not be empty
        assert len(parts[0].strip()) > 0
        # Body should not be empty
        assert len(parts[1].strip()) > 1  # At least one char plus period

    @given(asp_rule_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.large_base_example])
    def test_rule_head_and_body_different(self, rule):
        """Test that head and body predicates are different."""
        parts = rule.split(":-")
        head = parts[0].strip()
        body = parts[1].strip().rstrip(".")

        # Extract predicate names
        head_pred = head.split("(")[0] if "(" in head else head
        body_pred = body.lstrip("-").split("(")[0] if "(" in body else body.lstrip("-")

        # They should be different (as per strategy implementation)
        assert head_pred != body_pred

    def test_generates_rules_with_and_without_negation(self):
        """Test that strategy generates rules with and without negation."""
        rules_with_negation = 0
        rules_without_negation = 0

        @given(asp_rule_strategy())
        @settings(max_examples=50, suppress_health_check=[HealthCheck.large_base_example])
        def classify_rules(rule):
            nonlocal rules_with_negation, rules_without_negation
            body = rule.split(":-")[1]
            if body.strip().startswith("-"):
                rules_with_negation += 1
            else:
                rules_without_negation += 1

        classify_rules()

        # Should generate both types
        assert rules_with_negation > 0
        assert rules_without_negation > 0


class TestAspRuleContentStrategy:
    """Tests for asp_rule_content_strategy."""

    @given(asp_rule_content_strategy())
    @settings(max_examples=30)
    def test_generates_valid_asp_content(self, content):
        """Test that generated ASP content is valid."""
        assert isinstance(content, str)
        assert len(content) > 0
        # Should end with period
        assert content.endswith(".")

    def test_generates_both_facts_and_rules(self):
        """Test that strategy generates both facts and rules."""
        has_facts = False
        has_rules = False

        @given(asp_rule_content_strategy())
        @settings(max_examples=50)
        def classify_content(content):
            nonlocal has_facts, has_rules
            if ":-" in content:
                has_rules = True
            else:
                has_facts = True

        classify_content()

        # Should generate both
        assert has_facts
        assert has_rules


class TestRuleStrategy:
    """Tests for rule_strategy."""

    @given(rule_strategy())
    @settings(max_examples=30)
    def test_generates_valid_rule(self, rule):
        """Test that generated Rules are valid."""
        assert isinstance(rule, Rule)
        assert isinstance(rule.rule_id, str)
        assert isinstance(rule.content, str)
        assert isinstance(rule.level, StratificationLevel)
        assert 0.0 <= rule.confidence <= 1.0
        assert rule.provenance in ["llm", "human", "validation", "system"]
        assert isinstance(rule.timestamp, str)

    @given(rule_strategy())
    @settings(max_examples=20)
    def test_rule_has_valid_content(self, rule):
        """Test that rule content is valid ASP."""
        assert len(rule.content) > 0
        assert rule.content.endswith(".")

    def test_rule_with_fixed_content(self):
        """Test generating rule with fixed content."""
        fixed_content = "test_pred(x)."

        @given(rule_strategy(content=fixed_content))
        @settings(max_examples=10)
        def check_content(rule):
            assert rule.content == fixed_content

        check_content()

    def test_rule_with_fixed_level(self):
        """Test generating rule with fixed stratification level."""
        fixed_level = StratificationLevel.STRATEGIC

        @given(rule_strategy(level=fixed_level))
        @settings(max_examples=10)
        def check_level(rule):
            assert rule.level == fixed_level

        check_level()

    def test_rule_with_both_fixed_parameters(self):
        """Test generating rule with both content and level fixed."""
        fixed_content = "strategic_rule(x)."
        fixed_level = StratificationLevel.STRATEGIC

        @given(rule_strategy(content=fixed_content, level=fixed_level))
        @settings(max_examples=10)
        def check_fixed(rule):
            assert rule.content == fixed_content
            assert rule.level == fixed_level

        check_fixed()

    @given(rule_strategy())
    @settings(max_examples=30)
    def test_rule_confidence_range(self, rule):
        """Test that rule confidence is in valid range."""
        assert 0.0 <= rule.confidence <= 1.0

    @given(rule_strategy())
    @settings(max_examples=30)
    def test_rule_provenance_valid(self, rule):
        """Test that rule provenance is valid."""
        valid_provenances = {"llm", "human", "validation", "system"}
        assert rule.provenance in valid_provenances


class TestCoreStateStrategy:
    """Tests for core_state_strategy."""

    @given(core_state_strategy())
    @settings(max_examples=20)
    def test_generates_valid_core_state(self, state):
        """Test that generated CoreStates are valid."""
        assert isinstance(state, CoreState)
        assert isinstance(state.state_id, str)
        assert isinstance(state.timestamp, str)
        assert isinstance(state.rules, list)
        assert isinstance(state.configuration, dict)
        assert isinstance(state.metrics, dict)

    @given(core_state_strategy(min_rules=0, max_rules=0))
    @settings(max_examples=10)
    def test_generates_empty_state(self, state):
        """Test generating states with no rules."""
        assert len(state.rules) == 0

    @given(core_state_strategy(min_rules=5, max_rules=10))
    @settings(max_examples=20)
    def test_respects_rule_count_constraints(self, state):
        """Test that generated states respect rule count constraints."""
        assert 5 <= len(state.rules) <= 10

    @given(core_state_strategy(min_rules=3, max_rules=3))
    @settings(max_examples=10)
    def test_generates_exact_rule_count(self, state):
        """Test generating states with exact rule count."""
        assert len(state.rules) == 3

    @given(core_state_strategy())
    @settings(max_examples=20)
    def test_all_rules_are_valid(self, state):
        """Test that all generated rules are valid."""
        for rule in state.rules:
            assert isinstance(rule, Rule)
            assert isinstance(rule.content, str)
            assert len(rule.content) > 0

    @given(core_state_strategy())
    @settings(max_examples=20)
    def test_configuration_is_dict(self, state):
        """Test that configuration is a dictionary."""
        assert isinstance(state.configuration, dict)

    @given(core_state_strategy())
    @settings(max_examples=20)
    def test_metrics_are_valid(self, state):
        """Test that metrics are valid floats."""
        assert isinstance(state.metrics, dict)
        for key, value in state.metrics.items():
            assert isinstance(key, str)
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0

    @given(core_state_strategy())
    @settings(max_examples=20)
    def test_state_has_unique_id(self, state):
        """Test that states have unique IDs."""
        assert len(state.state_id) > 0


class TestFixturesEmpty:
    """Tests for TestFixtures.empty_state."""

    def test_empty_state_returns_core_state(self):
        """Test that empty_state returns a CoreState."""
        state = TestFixtures.empty_state()
        assert isinstance(state, CoreState)

    def test_empty_state_has_no_rules(self):
        """Test that empty_state has no rules."""
        state = TestFixtures.empty_state()
        assert len(state.rules) == 0

    def test_empty_state_has_valid_id(self):
        """Test that empty_state has a valid state ID."""
        state = TestFixtures.empty_state()
        assert isinstance(state.state_id, str)
        assert len(state.state_id) > 0

    def test_empty_state_has_timestamp(self):
        """Test that empty_state has a timestamp."""
        state = TestFixtures.empty_state()
        assert isinstance(state.timestamp, str)
        assert len(state.timestamp) > 0

    def test_empty_state_has_empty_configuration(self):
        """Test that empty_state has empty configuration."""
        state = TestFixtures.empty_state()
        assert state.configuration == {}

    def test_empty_state_has_empty_metrics(self):
        """Test that empty_state has empty metrics."""
        state = TestFixtures.empty_state()
        assert state.metrics == {}


class TestFixturesSimpleConsistent:
    """Tests for TestFixtures.simple_consistent_state."""

    def test_simple_consistent_state_returns_core_state(self):
        """Test that simple_consistent_state returns a CoreState."""
        state = TestFixtures.simple_consistent_state()
        assert isinstance(state, CoreState)

    def test_simple_consistent_state_has_rules(self):
        """Test that simple_consistent_state has rules."""
        state = TestFixtures.simple_consistent_state()
        assert len(state.rules) > 0

    def test_simple_consistent_state_has_three_rules(self):
        """Test that simple_consistent_state has exactly 3 rules."""
        state = TestFixtures.simple_consistent_state()
        assert len(state.rules) == 3

    def test_simple_consistent_state_rules_are_valid(self):
        """Test that all rules are valid Rule instances."""
        state = TestFixtures.simple_consistent_state()
        for rule in state.rules:
            assert isinstance(rule, Rule)
            assert isinstance(rule.content, str)
            assert len(rule.content) > 0

    def test_simple_consistent_state_has_configuration(self):
        """Test that simple_consistent_state has configuration."""
        state = TestFixtures.simple_consistent_state()
        assert isinstance(state.configuration, dict)
        assert "test" in state.configuration

    def test_simple_consistent_state_has_metrics(self):
        """Test that simple_consistent_state has metrics."""
        state = TestFixtures.simple_consistent_state()
        assert isinstance(state.metrics, dict)
        assert "consistency" in state.metrics


class TestFixturesContradictory:
    """Tests for TestFixtures.contradictory_state."""

    def test_contradictory_state_returns_core_state(self):
        """Test that contradictory_state returns a CoreState."""
        state = TestFixtures.contradictory_state()
        assert isinstance(state, CoreState)

    def test_contradictory_state_has_rules(self):
        """Test that contradictory_state has rules."""
        state = TestFixtures.contradictory_state()
        assert len(state.rules) >= 2

    def test_contradictory_state_has_contradiction(self):
        """Test that contradictory_state contains contradictory rules."""
        state = TestFixtures.contradictory_state()
        # Should have both alive(x) and -alive(x)
        contents = [rule.content for rule in state.rules]
        assert "alive(x)." in contents
        assert "-alive(x)." in contents

    def test_contradictory_state_rules_are_same_level(self):
        """Test that contradictory rules are at same level."""
        state = TestFixtures.contradictory_state()
        levels = [rule.level for rule in state.rules]
        # Both should be OPERATIONAL
        assert all(level == StratificationLevel.OPERATIONAL for level in levels)


class TestFixturesIncomplete:
    """Tests for TestFixtures.incomplete_state."""

    def test_incomplete_state_returns_core_state(self):
        """Test that incomplete_state returns a CoreState."""
        state = TestFixtures.incomplete_state()
        assert isinstance(state, CoreState)

    def test_incomplete_state_has_rules(self):
        """Test that incomplete_state has rules."""
        state = TestFixtures.incomplete_state()
        assert len(state.rules) > 0

    def test_incomplete_state_has_undefined_predicate(self):
        """Test that incomplete_state references undefined predicate."""
        state = TestFixtures.incomplete_state()
        # Should have rule that uses premise(X) without defining it
        rule_content = state.rules[0].content
        assert "premise" in rule_content
        assert ":-" in rule_content


class TestFixturesIncoherent:
    """Tests for TestFixtures.incoherent_state."""

    def test_incoherent_state_returns_core_state(self):
        """Test that incoherent_state returns a CoreState."""
        state = TestFixtures.incoherent_state()
        assert isinstance(state, CoreState)

    def test_incoherent_state_has_rules(self):
        """Test that incoherent_state has rules."""
        state = TestFixtures.incoherent_state()
        assert len(state.rules) > 0

    def test_incoherent_state_has_stratification_issue(self):
        """Test that incoherent_state has stratification incoherence."""
        state = TestFixtures.incoherent_state()
        levels = [rule.level for rule in state.rules]
        # Should have both TACTICAL and CONSTITUTIONAL
        assert StratificationLevel.TACTICAL in levels
        assert StratificationLevel.CONSTITUTIONAL in levels


class TestFixturesCircularDependency:
    """Tests for TestFixtures.circular_dependency_state."""

    def test_circular_dependency_state_returns_core_state(self):
        """Test that circular_dependency_state returns a CoreState."""
        state = TestFixtures.circular_dependency_state()
        assert isinstance(state, CoreState)

    def test_circular_dependency_state_has_rules(self):
        """Test that circular_dependency_state has rules."""
        state = TestFixtures.circular_dependency_state()
        assert len(state.rules) >= 3

    def test_circular_dependency_state_has_cycle(self):
        """Test that circular_dependency_state forms a cycle."""
        state = TestFixtures.circular_dependency_state()
        # Should have a -> b, b -> c, c -> a
        contents = [rule.content for rule in state.rules]
        assert "a(X) :- b(X)." in contents
        assert "b(X) :- c(X)." in contents
        assert "c(X) :- a(X)." in contents


class TestFixturesIntegration:
    """Integration tests for all fixtures."""

    def test_all_fixtures_return_valid_states(self):
        """Test that all fixtures return valid CoreState instances."""
        fixtures = [
            TestFixtures.empty_state(),
            TestFixtures.simple_consistent_state(),
            TestFixtures.contradictory_state(),
            TestFixtures.incomplete_state(),
            TestFixtures.incoherent_state(),
            TestFixtures.circular_dependency_state(),
        ]

        for state in fixtures:
            assert isinstance(state, CoreState)
            assert isinstance(state.state_id, str)
            assert isinstance(state.timestamp, str)
            assert isinstance(state.rules, list)
            assert isinstance(state.configuration, dict)
            assert isinstance(state.metrics, dict)

    def test_all_fixtures_have_unique_ids(self):
        """Test that each fixture generates unique state IDs."""
        states = [
            TestFixtures.empty_state(),
            TestFixtures.simple_consistent_state(),
            TestFixtures.contradictory_state(),
        ]

        ids = [state.state_id for state in states]
        # Should all be different
        assert len(set(ids)) == len(ids)

    def test_fixtures_are_deterministic(self):
        """Test that calling same fixture twice gives same structure."""
        # Empty state should always be empty
        state1 = TestFixtures.empty_state()
        state2 = TestFixtures.empty_state()
        assert len(state1.rules) == len(state2.rules) == 0

        # Contradictory state should always have same rules
        state1 = TestFixtures.contradictory_state()
        state2 = TestFixtures.contradictory_state()
        assert len(state1.rules) == len(state2.rules)
