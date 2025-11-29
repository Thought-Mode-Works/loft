"""Tests for rule evolution tracking."""

from datetime import datetime, timedelta

from loft.evolution.tracking import (
    RuleMetadata,
    ValidationResult,
    ABTestResult,
    DialecticalHistory,
    StratificationLayer,
    RuleStatus,
    RuleEvolutionTracker,
    generate_rule_id,
)


class TestRuleMetadata:
    """Tests for RuleMetadata dataclass."""

    def test_create_basic_metadata(self):
        """Test creating basic rule metadata."""
        metadata = RuleMetadata(
            rule_id="test_rule_001",
            rule_text="enforceable(C) :- has_consideration(C).",
            natural_language="A contract requires consideration",
            created_by="test",
        )

        assert metadata.rule_id == "test_rule_001"
        assert metadata.version == "1.0"
        assert metadata.status == RuleStatus.CANDIDATE
        assert metadata.current_layer == StratificationLayer.OPERATIONAL
        assert metadata.current_accuracy == 0.0

    def test_metadata_to_dict_roundtrip(self):
        """Test serialization and deserialization."""
        original = RuleMetadata(
            rule_id="test_rule_002",
            rule_text="valid(C) :- signed(C).",
            natural_language="A contract must be signed",
            created_by="test",
        )
        # Add a validation result to test accuracy
        original.validation_results.append(
            ValidationResult(
                timestamp=datetime.now(),
                test_cases_evaluated=100,
                passed=85,
                failed=15,
                accuracy=0.85,
            )
        )

        data = original.to_dict()
        restored = RuleMetadata.from_dict(data)

        assert restored.rule_id == original.rule_id
        assert restored.rule_text == original.rule_text
        assert restored.current_accuracy == original.current_accuracy

    def test_metadata_with_validation_results(self):
        """Test metadata with validation results."""
        metadata = RuleMetadata(
            rule_id="test_rule_003",
            rule_text="rule text",
            natural_language="description",
            created_by="test",
        )

        result = ValidationResult(
            timestamp=datetime.now(),
            test_cases_evaluated=100,
            passed=90,
            failed=10,
            accuracy=0.9,
        )
        metadata.validation_results.append(result)

        assert len(metadata.validation_results) == 1
        assert metadata.validation_results[0].accuracy == 0.9
        assert metadata.current_accuracy == 0.9

    def test_current_accuracy_property(self):
        """Test that current_accuracy returns latest validation accuracy."""
        metadata = RuleMetadata(
            rule_id="test_rule_004",
            rule_text="test",
            natural_language="test",
        )

        # No validations - should be 0
        assert metadata.current_accuracy == 0.0

        # Add first validation
        metadata.validation_results.append(
            ValidationResult(
                timestamp=datetime.now() - timedelta(hours=2),
                test_cases_evaluated=50,
                passed=35,
                failed=15,
                accuracy=0.7,
            )
        )
        assert metadata.current_accuracy == 0.7

        # Add second validation (latest)
        metadata.validation_results.append(
            ValidationResult(
                timestamp=datetime.now(),
                test_cases_evaluated=100,
                passed=85,
                failed=15,
                accuracy=0.85,
            )
        )
        assert metadata.current_accuracy == 0.85


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test validation result creation."""
        result = ValidationResult(
            timestamp=datetime.now(),
            test_cases_evaluated=50,
            passed=42,
            failed=8,
            accuracy=0.84,
        )

        assert result.test_cases_evaluated == 50
        assert result.passed == 42
        assert result.accuracy == 0.84

    def test_validation_result_serialization(self):
        """Test validation result serialization."""
        result = ValidationResult(
            timestamp=datetime.now(),
            test_cases_evaluated=100,
            passed=60,
            failed=40,
            accuracy=0.6,
            error_cases=["case_1", "case_2"],
            notes="Test notes",
        )

        data = result.to_dict()
        restored = ValidationResult.from_dict(data)

        assert restored.test_cases_evaluated == result.test_cases_evaluated
        assert restored.accuracy == result.accuracy
        assert restored.error_cases == result.error_cases
        assert restored.notes == result.notes


class TestABTestResult:
    """Tests for ABTestResult dataclass."""

    def test_ab_test_creation(self):
        """Test A/B test result creation."""
        result = ABTestResult(
            test_id="ab_test_001",
            started_at=datetime.now(),
            variant_a_id="rule_v1",
            variant_b_id="rule_v2",
            variant_a_accuracy=0.75,
            variant_b_accuracy=0.82,
            cases_evaluated=100,
        )

        assert result.test_id == "ab_test_001"
        assert result.winner is None
        assert result.completed_at is None

    def test_ab_test_with_winner(self):
        """Test A/B test with determined winner."""
        result = ABTestResult(
            test_id="ab_test_002",
            started_at=datetime.now() - timedelta(hours=1),
            completed_at=datetime.now(),
            variant_a_id="rule_v1",
            variant_b_id="rule_v2",
            variant_a_accuracy=0.75,
            variant_b_accuracy=0.88,
            cases_evaluated=200,
            p_value=0.01,
            winner="b",
        )

        assert result.winner == "b"
        assert result.p_value == 0.01
        assert result.completed_at is not None

    def test_ab_test_serialization(self):
        """Test A/B test serialization."""
        original = ABTestResult(
            test_id="ab_test_003",
            started_at=datetime.now(),
            variant_a_id="rule_v1",
            variant_b_id="rule_v2",
            variant_a_accuracy=0.8,
            variant_b_accuracy=0.8,
            cases_evaluated=50,
        )

        data = original.to_dict()
        restored = ABTestResult.from_dict(data)

        assert restored.test_id == original.test_id
        assert restored.variant_a_accuracy == original.variant_a_accuracy


class TestDialecticalHistory:
    """Tests for DialecticalHistory dataclass."""

    def test_dialectical_history_creation(self):
        """Test dialectical history creation."""
        history = DialecticalHistory(
            thesis_rule="original rule",
            antithesis_critiques=["critique 1", "critique 2"],
            synthesis_rule="improved rule",
            cycles_completed=1,
        )

        assert history.thesis_rule == "original rule"
        assert len(history.antithesis_critiques) == 2
        assert history.cycles_completed == 1

    def test_dialectical_history_serialization(self):
        """Test dialectical history serialization."""
        original = DialecticalHistory(
            thesis_rule="initial",
            antithesis_critiques=["issue"],
            synthesis_rule="refined",
            cycles_completed=2,
        )

        data = original.to_dict()
        restored = DialecticalHistory.from_dict(data)

        assert restored.thesis_rule == original.thesis_rule
        assert restored.cycles_completed == original.cycles_completed


class TestGenerateRuleId:
    """Tests for generate_rule_id function."""

    def test_generates_prefixed_id(self):
        """Test that generated IDs have rule_ prefix."""
        rule_id = generate_rule_id("test rule")
        assert rule_id.startswith("rule_")

    def test_different_text_different_id(self):
        """Test that different text produces different IDs."""
        id1 = generate_rule_id("rule one")
        id2 = generate_rule_id("rule two")
        assert id1 != id2


class TestRuleEvolutionTracker:
    """Tests for RuleEvolutionTracker."""

    @property
    def rules(self):
        """Expose internal rules for testing."""
        return self._rules

    @property
    def ab_tests(self):
        """Expose internal ab_tests for testing."""
        return self._ab_tests

    def test_create_rule(self):
        """Test creating a new rule."""
        tracker = RuleEvolutionTracker()

        metadata = tracker.create_rule(
            rule_text="enforceable(C) :- valid(C).",
            natural_language="Valid contracts are enforceable",
            created_by="test",
        )

        assert metadata.rule_id is not None
        assert metadata.version == "1.0"
        assert metadata.status == RuleStatus.CANDIDATE
        assert tracker.get_rule(metadata.rule_id) is not None

    def test_create_version(self):
        """Test creating a new version of a rule."""
        tracker = RuleEvolutionTracker()

        parent = tracker.create_rule(
            rule_text="rule v1",
            natural_language="version 1",
            created_by="test",
        )

        child = tracker.create_version(
            parent_id=parent.rule_id,
            new_rule_text="rule v2",
            change_reason="improvement",
        )

        assert child.version == "1.1"
        assert child.parent_rule_id == parent.rule_id
        assert tracker.get_rule(parent.rule_id) is not None

    def test_record_validation(self):
        """Test recording validation results."""
        tracker = RuleEvolutionTracker()

        metadata = tracker.create_rule(
            rule_text="test rule",
            natural_language="test",
            created_by="test",
        )

        tracker.record_validation(
            rule_id=metadata.rule_id,
            test_cases=100,
            passed=85,
            failed=15,
        )

        rule = tracker.get_rule(metadata.rule_id)
        assert len(rule.validation_results) == 1
        assert rule.current_accuracy == 0.85

    def test_promote_rule_layer(self):
        """Test promoting rule to higher layer."""
        tracker = RuleEvolutionTracker()

        metadata = tracker.create_rule(
            rule_text="test rule",
            natural_language="test",
            created_by="test",
        )

        tracker.promote_rule(
            metadata.rule_id,
            StratificationLayer.TACTICAL,
            "High accuracy achieved",
        )

        rule = tracker.get_rule(metadata.rule_id)
        assert rule.current_layer == StratificationLayer.TACTICAL
        assert len(rule.layer_history) >= 2

    def test_start_ab_test(self):
        """Test starting an A/B test."""
        tracker = RuleEvolutionTracker()

        rule_a = tracker.create_rule(
            rule_text="rule a",
            natural_language="version a",
            created_by="test",
        )

        rule_b = tracker.create_rule(
            rule_text="rule b",
            natural_language="version b",
            created_by="test",
        )

        test = tracker.start_ab_test(rule_a.rule_id, rule_b.rule_id)

        assert test.variant_a_id == rule_a.rule_id
        assert test.variant_b_id == rule_b.rule_id

        # Both rules should be in AB_TESTING status
        assert tracker.get_rule(rule_a.rule_id).status == RuleStatus.AB_TESTING
        assert tracker.get_rule(rule_b.rule_id).status == RuleStatus.AB_TESTING

    def test_record_dialectical_cycle(self):
        """Test recording dialectical refinement."""
        tracker = RuleEvolutionTracker()

        metadata = tracker.create_rule(
            rule_text="original rule",
            natural_language="original",
            created_by="test",
        )

        result = tracker.record_dialectical_cycle(
            rule_id=metadata.rule_id,
            thesis="original rule",
            thesis_reasoning="Initial proposal",
            critiques=["critique 1"],
            synthesis="improved rule",
            synthesis_reasoning="Addressed critique",
        )

        # If synthesis differs, a new rule is created
        assert result.dialectical.cycles_completed == 1
        assert result.dialectical.thesis_rule == "original rule"

    def test_deprecate_rule(self):
        """Test deprecating a rule with a replacement."""
        tracker = RuleEvolutionTracker()

        old_rule = tracker.create_rule(
            rule_text="old rule",
            natural_language="old",
            created_by="test",
        )

        new_rule = tracker.create_version(
            parent_id=old_rule.rule_id,
            new_rule_text="new rule",
            change_reason="improvement",
        )

        tracker.deprecate_rule(
            old_rule.rule_id,
            reason="Replaced with improved version",
            superseded_by=new_rule.rule_id,
        )

        old = tracker.get_rule(old_rule.rule_id)
        assert old.status == RuleStatus.SUPERSEDED
        assert old.superseded_by == new_rule.rule_id

    def test_get_version_history(self):
        """Test getting rule version history."""
        tracker = RuleEvolutionTracker()

        v1 = tracker.create_rule(
            rule_text="v1",
            natural_language="version 1",
            created_by="test",
        )

        v2 = tracker.create_version(
            parent_id=v1.rule_id,
            new_rule_text="v2",
            change_reason="update",
        )

        v3 = tracker.create_version(
            parent_id=v2.rule_id,
            new_rule_text="v3",
            change_reason="update",
        )

        history = tracker.get_version_history(v3.rule_id)

        assert len(history) == 3
        assert history[0].rule_id == v1.rule_id
        assert history[2].rule_id == v3.rule_id

    def test_get_rules_by_status(self):
        """Test filtering rules by status."""
        tracker = RuleEvolutionTracker()

        rule1 = tracker.create_rule(
            rule_text="rule 1",
            natural_language="test",
            created_by="test",
        )
        # Record high-accuracy validation to make it ACTIVE
        tracker.record_validation(
            rule_id=rule1.rule_id,
            test_cases=100,
            passed=90,
            failed=10,
        )

        tracker.create_rule(
            rule_text="rule 2",
            natural_language="test",
            created_by="test",
        )

        active_rules = tracker.get_rules_by_status(RuleStatus.ACTIVE)
        candidate_rules = tracker.get_rules_by_status(RuleStatus.CANDIDATE)

        assert len(active_rules) == 1
        assert len(candidate_rules) == 1

    def test_complete_ab_test(self):
        """Test completing an A/B test."""
        tracker = RuleEvolutionTracker()

        rule_a = tracker.create_rule(
            rule_text="rule a",
            natural_language="version a",
            created_by="test",
        )
        rule_b = tracker.create_rule(
            rule_text="rule b",
            natural_language="version b",
            created_by="test",
        )

        test = tracker.start_ab_test(rule_a.rule_id, rule_b.rule_id)

        # Complete test with B as winner
        result = tracker.complete_ab_test(
            test_id=test.test_id,
            winner="b",
            p_value=0.01,
        )

        assert result.winner == "b"
        assert result.completed_at is not None
        assert tracker.get_rule(rule_b.rule_id).status == RuleStatus.ACTIVE
        assert tracker.get_rule(rule_a.rule_id).status == RuleStatus.SUPERSEDED
