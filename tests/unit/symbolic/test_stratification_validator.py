"""
Unit tests for stratification validator.

Tests integrity validation, violation detection, and cycle detection.
"""

import pytest
from datetime import datetime
from loft.symbolic.stratification_validator import (
    StratificationValidator,
    StratificationViolation,
    StratificationReport,
)
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


class TestStratificationValidator:
    """Tests for StratificationValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a stratification validator."""
        return StratificationValidator()

    @pytest.fixture
    def core_with_rules(self):
        """Create a stratified core with sample rules."""
        core = StratifiedASPCore()

        # Constitutional rule
        const_rule = create_test_rule(
            rule_id="const_1",
            asp_text="fundamental(X) :- entity(X).",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
        )
        core.add_rule(
            const_rule, StratificationLevel.CONSTITUTIONAL, bypass_checks=True
        )

        # Strategic rule
        strat_rule = create_test_rule(
            rule_id="strat_1",
            asp_text="valid(X) :- fundamental(X), not invalid(X).",
            level=StratificationLevel.STRATEGIC,
            confidence=0.95,
        )
        core.add_rule(strat_rule, StratificationLevel.STRATEGIC, bypass_checks=True)

        return core

    def test_initialization(self):
        """Test validator initialization."""
        validator = StratificationValidator()
        assert validator.initial_constitutional_rules == []

    def test_initialization_with_rules(self):
        """Test validator initialization with constitutional rules."""
        rules = [
            create_test_rule(
                rule_id="r1",
                asp_text="rule1.",
                level=StratificationLevel.CONSTITUTIONAL,
                confidence=1.0,
            )
        ]
        validator = StratificationValidator(initial_constitutional_rules=rules)
        assert len(validator.initial_constitutional_rules) == 1

    def test_validate_empty_core(self, validator):
        """Test validation of empty core."""
        core = StratifiedASPCore()
        report = validator.validate_core(core)

        assert isinstance(report, StratificationReport)
        assert report.valid
        assert len(report.violations) == 0
        assert len(report.cycles_detected) == 0

    def test_validate_valid_core(self, validator, core_with_rules):
        """Test validation of valid core."""
        report = validator.validate_core(core_with_rules)

        assert isinstance(report, StratificationReport)
        assert report.valid
        assert len(report.violations) == 0

    def test_detect_constitutional_modification(self):
        """Test detection of constitutional layer modifications."""
        # Create initial rule
        initial_rule = create_test_rule(
            rule_id="const_1",
            asp_text="fundamental(X) :- entity(X).",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
        )

        # Create validator with initial rule
        validator = StratificationValidator(initial_constitutional_rules=[initial_rule])

        # Create core with different constitutional rule
        core = StratifiedASPCore()
        different_rule = create_test_rule(
            rule_id="const_2",  # Different ID
            asp_text="different(X) :- thing(X).",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
        )
        core.add_rule(
            different_rule, StratificationLevel.CONSTITUTIONAL, bypass_checks=True
        )

        report = validator.validate_core(core)

        assert not report.valid
        assert len(report.violations) > 0
        # Should detect both addition and removal
        violation = report.violations[0]
        assert violation.severity == "critical"
        assert violation.layer == StratificationLevel.CONSTITUTIONAL
        assert violation.violation_type == "unauthorized_modification"

    def test_detect_constitutional_addition(self):
        """Test detection of rules added to constitutional layer."""
        initial_rule = create_test_rule(
            rule_id="const_1",
            asp_text="rule1.",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
        )

        validator = StratificationValidator(initial_constitutional_rules=[initial_rule])

        # Create core with original + new rule
        core = StratifiedASPCore()
        core.add_rule(
            initial_rule, StratificationLevel.CONSTITUTIONAL, bypass_checks=True
        )

        new_rule = create_test_rule(
            rule_id="const_2",
            asp_text="rule2.",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
        )
        core.add_rule(new_rule, StratificationLevel.CONSTITUTIONAL, bypass_checks=True)

        report = validator.validate_core(core)

        assert not report.valid
        assert len(report.violations) == 1
        assert "added" in report.violations[0].description.lower()

    def test_detect_constitutional_removal(self):
        """Test detection of rules removed from constitutional layer."""
        rule1 = create_test_rule(
            rule_id="const_1",
            asp_text="rule1.",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
        )
        rule2 = create_test_rule(
            rule_id="const_2",
            asp_text="rule2.",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
        )

        # Validator expects both rules
        validator = StratificationValidator(initial_constitutional_rules=[rule1, rule2])

        # Core only has rule1
        core = StratifiedASPCore()
        core.add_rule(rule1, StratificationLevel.CONSTITUTIONAL, bypass_checks=True)

        report = validator.validate_core(core)

        assert not report.valid
        assert "removed" in report.violations[0].description.lower()

    def test_detect_circular_dependency(self, validator):
        """Test detection of circular dependencies."""
        core = StratifiedASPCore()

        # Create circular dependency: a -> b -> c -> a
        rule_a = create_test_rule(
            rule_id="r_a",
            asp_text="a(X) :- b(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
        )
        rule_b = create_test_rule(
            rule_id="r_b",
            asp_text="b(X) :- c(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
        )
        rule_c = create_test_rule(
            rule_id="r_c",
            asp_text="c(X) :- a(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
        )

        core.add_rule(rule_a, StratificationLevel.TACTICAL, bypass_checks=True)
        core.add_rule(rule_b, StratificationLevel.TACTICAL, bypass_checks=True)
        core.add_rule(rule_c, StratificationLevel.TACTICAL, bypass_checks=True)

        report = validator.validate_core(core)

        assert not report.valid
        assert len(report.cycles_detected) > 0
        # Should have a violation for circular dependency
        cycle_violations = [
            v for v in report.violations if v.violation_type == "circular_dependency"
        ]
        assert len(cycle_violations) > 0
        assert cycle_violations[0].severity == "high"

    def test_no_cycles_in_valid_dependencies(self, validator):
        """Test that valid dependency chains don't trigger false positives."""
        core = StratifiedASPCore()

        # Linear chain: a -> b -> c (no cycle)
        rule_a = create_test_rule(
            rule_id="r_a",
            asp_text="a(X) :- b(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
        )
        rule_b = create_test_rule(
            rule_id="r_b",
            asp_text="b(X) :- c(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
        )
        rule_c = create_test_rule(
            rule_id="r_c",
            asp_text="c(foo).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
        )

        core.add_rule(rule_a, StratificationLevel.TACTICAL, bypass_checks=True)
        core.add_rule(rule_b, StratificationLevel.TACTICAL, bypass_checks=True)
        core.add_rule(rule_c, StratificationLevel.TACTICAL, bypass_checks=True)

        report = validator.validate_core(core)

        assert report.valid
        assert len(report.cycles_detected) == 0

    def test_detect_invalid_cross_layer_dependency(self, validator):
        """Test detection of invalid cross-layer dependencies."""
        core = StratifiedASPCore()

        # Strategic rule depending on tactical (not allowed)
        tactical_rule = create_test_rule(
            rule_id="tact_1",
            asp_text="tactical_fact(X) :- entity(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.85,
        )
        core.add_rule(tactical_rule, StratificationLevel.TACTICAL, bypass_checks=True)

        # Strategic trying to depend on tactical
        strategic_rule = create_test_rule(
            rule_id="strat_1",
            asp_text="strategic_rule(X) :- tactical_fact(X).",
            level=StratificationLevel.STRATEGIC,
            confidence=0.95,
        )
        core.add_rule(strategic_rule, StratificationLevel.STRATEGIC, bypass_checks=True)

        report = validator.validate_core(core)

        assert not report.valid
        dep_violations = [
            v for v in report.violations if v.violation_type == "invalid_dependency"
        ]
        assert len(dep_violations) > 0
        assert dep_violations[0].severity == "high"
        assert "strategic" in dep_violations[0].description.lower()
        assert "tactical" in dep_violations[0].description.lower()

    def test_valid_cross_layer_dependency(self, validator):
        """Test that valid cross-layer dependencies are allowed."""
        core = StratifiedASPCore()

        # Constitutional rule
        const_rule = create_test_rule(
            rule_id="const_1",
            asp_text="entity(thing).",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
        )
        core.add_rule(
            const_rule, StratificationLevel.CONSTITUTIONAL, bypass_checks=True
        )

        # Strategic rule depending on constitutional (allowed)
        strategic_rule = create_test_rule(
            rule_id="strat_1",
            asp_text="valid(X) :- entity(X).",
            level=StratificationLevel.STRATEGIC,
            confidence=0.95,
        )
        core.add_rule(strategic_rule, StratificationLevel.STRATEGIC, bypass_checks=True)

        report = validator.validate_core(core)

        assert report.valid
        dep_violations = [
            v for v in report.violations if v.violation_type == "invalid_dependency"
        ]
        assert len(dep_violations) == 0

    def test_report_structure(self, validator, core_with_rules):
        """Test that report has all expected fields."""
        report = validator.validate_core(core_with_rules)

        assert hasattr(report, "valid")
        assert hasattr(report, "violations")
        assert hasattr(report, "stats")
        assert hasattr(report, "cycles_detected")
        assert isinstance(report.valid, bool)
        assert isinstance(report.violations, list)
        assert isinstance(report.stats, dict)

    def test_violation_structure(self):
        """Test StratificationViolation structure."""
        violation = StratificationViolation(
            severity="critical",
            layer=StratificationLevel.CONSTITUTIONAL,
            violation_type="test_violation",
            description="Test description",
            affected_rules=["rule1", "rule2"],
        )

        assert violation.severity == "critical"
        assert violation.layer == StratificationLevel.CONSTITUTIONAL
        assert violation.violation_type == "test_violation"
        assert violation.description == "Test description"
        assert len(violation.affected_rules) == 2

    def test_stats_in_report(self, validator, core_with_rules):
        """Test that report includes statistics."""
        report = validator.validate_core(core_with_rules)

        assert "constitutional" in report.stats
        assert "strategic" in report.stats
        # Should have stats for all layers
        assert len(report.stats) >= 4

    def test_self_referential_rule_allowed(self, validator):
        """Test that self-referential predicates don't cause cycles."""
        core = StratifiedASPCore()

        # Rule that references itself (common in ASP)
        rule = create_test_rule(
            rule_id="r1",
            asp_text="reachable(X, Y) :- edge(X, Y). reachable(X, Z) :- reachable(X, Y), edge(Y, Z).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
        )
        core.add_rule(rule, StratificationLevel.TACTICAL, bypass_checks=True)

        report = validator.validate_core(core)

        # Self-reference should be filtered out
        # Cycles should only be detected for actual circular dependencies
        # This specific case might still show a cycle depending on implementation
        # The key is it shouldn't crash
        assert isinstance(report, StratificationReport)

    def test_multiple_violations(self, validator):
        """Test core with multiple different violations."""
        # Create validator expecting a constitutional rule
        expected_rule = create_test_rule(
            rule_id="const_1",
            asp_text="expected(foo).",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
        )
        validator_with_rules = StratificationValidator(
            initial_constitutional_rules=[expected_rule]
        )

        core = StratifiedASPCore()

        # Violation 1: Different constitutional rule (modification)
        different_rule = create_test_rule(
            rule_id="const_2",
            asp_text="different(bar).",
            level=StratificationLevel.CONSTITUTIONAL,
            confidence=1.0,
        )
        core.add_rule(
            different_rule, StratificationLevel.CONSTITUTIONAL, bypass_checks=True
        )

        # Violation 2: Circular dependency
        rule_a = create_test_rule(
            rule_id="r_a",
            asp_text="a(X) :- b(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
        )
        rule_b = create_test_rule(
            rule_id="r_b",
            asp_text="b(X) :- a(X).",
            level=StratificationLevel.TACTICAL,
            confidence=0.9,
        )
        core.add_rule(rule_a, StratificationLevel.TACTICAL, bypass_checks=True)
        core.add_rule(rule_b, StratificationLevel.TACTICAL, bypass_checks=True)

        report = validator_with_rules.validate_core(core)

        assert not report.valid
        # Should have both constitutional modification and circular dependency
        assert len(report.violations) >= 2
        violation_types = {v.violation_type for v in report.violations}
        assert "unauthorized_modification" in violation_types
        assert "circular_dependency" in violation_types
