"""
Integration tests for rule incorporation with rollback mechanism.

Tests MVP criteria from Issue #42:
1. Successfully incorporates 10 new rules autonomously
2. Detects and rolls back harmful changes within 5 test cases
3. All modifications are traceable (snapshot IDs, timestamps)
"""

from loft.core.incorporation import (
    RuleIncorporationEngine,
    SimpleASPCore,
    SimpleTestSuite,
)
from loft.neural.rule_schemas import GeneratedRule
from loft.symbolic.stratification import StratificationLevel
from loft.validation.validation_schemas import ValidationReport


class MockRegressionTestSuite(SimpleTestSuite):
    """Test suite that simulates regression on specific rules."""

    def __init__(self, initial_accuracy: float = 0.85):
        super().__init__(initial_accuracy)
        self.regression_on_rule_numbers = set()  # Rule numbers that trigger regression

    def set_regression_rules(self, rule_numbers: set):
        """Configure which rule numbers cause regression."""
        self.regression_on_rule_numbers = rule_numbers

    def run_all(self):
        """
        Run tests with possible regression.

        Returns regression if current rule count matches a regression trigger.
        """
        self.test_count += 1

        # Check if this is a regression case
        if self.test_count in self.regression_on_rule_numbers:
            # Simulate regression: accuracy drops by 5%
            self.current_accuracy = max(0.0, self.current_accuracy - 0.05)
            return {
                "passed": False,
                "failures": [f"Regression detected on rule #{self.test_count}"],
                "accuracy": self.current_accuracy,
            }

        # Normal case: slight improvement
        import random

        random.seed(42 + self.test_count)
        self.current_accuracy = min(1.0, self.current_accuracy + random.uniform(0, 0.01))

        return {
            "passed": True,
            "failures": [],
            "accuracy": self.current_accuracy,
        }


class TestMVPCriteria:
    """Test MVP validation criteria from Issue #42."""

    def create_test_rule(self, rule_num: int, confidence: float = 0.85) -> GeneratedRule:
        """Create a numbered test rule."""
        return GeneratedRule(
            asp_rule=f"rule_{rule_num}(X) :- condition_{rule_num}(X).",
            confidence=confidence,
            reasoning=f"Test rule {rule_num}",
            source_type="gap_fill",
            source_text=f"Test source {rule_num}",
            predicates_used=[f"rule_{rule_num}", f"condition_{rule_num}"],
            new_predicates=[f"rule_{rule_num}"],
        )

    def create_test_validation_report(self) -> ValidationReport:
        """Create a test validation report."""
        report = ValidationReport(
            rule_asp="test_rule(X) :- condition(X).",
            rule_id="test",
            target_layer="tactical",
        )
        report.final_decision = "accept"
        return report

    def test_mvp_criterion_1_incorporate_10_rules(self):
        """
        MVP Criterion 1: Successfully incorporates 10 new rules autonomously.

        This demonstrates that the system can autonomously incorporate
        multiple rules following stratification policies.
        """
        engine = RuleIncorporationEngine(test_suite=MockRegressionTestSuite(initial_accuracy=0.80))
        report = self.create_test_validation_report()

        successful_incorporations = 0

        # Attempt to incorporate 10 rules
        for i in range(1, 11):
            rule = self.create_test_rule(i, confidence=0.85)

            result = engine.incorporate(
                rule=rule,
                target_layer=StratificationLevel.TACTICAL,
                validation_report=report,
                is_autonomous=True,
            )

            if result.is_success():
                successful_incorporations += 1

        # MVP criterion: Successfully incorporate 10 rules
        assert successful_incorporations == 10, (
            f"Expected 10 successful incorporations, got {successful_incorporations}"
        )

        # Verify all are tracked
        stats = engine.get_statistics()
        assert stats["total_modifications"] == 10
        assert len(engine.incorporation_history) == 10

        # Verify traceability: all have snapshot IDs and timestamps
        for history_item in engine.incorporation_history:
            assert "snapshot_id" in history_item
            assert "timestamp" in history_item
            assert history_item["snapshot_id"] is not None

        print(f"\n✓ MVP Criterion 1 PASSED: {successful_incorporations}/10 rules incorporated")
        print(f"  Total modifications: {stats['total_modifications']}")
        print(f"  History tracked: {len(engine.incorporation_history)} entries")

    def test_mvp_criterion_2_rollback_harmful_changes(self):
        """
        MVP Criterion 2: Detects and rolls back harmful changes within 5 test cases.

        This demonstrates that the regression detection system can identify
        and automatically rollback rules that cause accuracy drops.
        """
        # Create test suite that will regress on specific rule numbers
        test_suite = MockRegressionTestSuite(initial_accuracy=0.85)
        test_suite.set_regression_rules({3, 5})  # Rules 3 and 5 will cause regression

        engine = RuleIncorporationEngine(
            test_suite=test_suite,
            regression_threshold=0.02,  # 2% drop triggers rollback
        )
        report = self.create_test_validation_report()

        # Incorporate 5 rules, where some will regress
        results = []
        for i in range(1, 6):
            rule = self.create_test_rule(i, confidence=0.85)

            result = engine.incorporate(
                rule=rule,
                target_layer=StratificationLevel.TACTICAL,
                validation_report=report,
                is_autonomous=True,
            )

            results.append((i, result))

        # Count successful vs rolled back
        successful = sum(1 for _, r in results if r.is_success())
        rolled_back = len(engine.rollback_history)

        # MVP criterion: Detect and rollback harmful changes
        assert rolled_back == 2, f"Expected 2 rollbacks, got {rolled_back}"
        assert successful == 3, f"Expected 3 successful, got {successful}"

        # Verify rollback events are properly tracked
        rollback_events = engine.get_rollback_history()
        assert len(rollback_events) == 2

        for event in rollback_events:
            assert event.reason is not None
            assert event.snapshot_id is not None
            assert event.timestamp is not None
            assert event.failed_rule_text is not None
            assert "Regression" in event.reason or "regression" in event.reason

        stats = engine.get_statistics()

        print(f"\n✓ MVP Criterion 2 PASSED: Detected and rolled back {rolled_back} harmful changes")
        print(f"  Successful incorporations: {successful}/5")
        print(f"  Rollbacks: {rolled_back}/5")
        print(f"  Success rate: {stats['success_rate']:.1%}")

    def test_mvp_criterion_3_traceability(self):
        """
        MVP Criterion 3: All modifications are explainable and traceable.

        This demonstrates that every modification has:
        - Snapshot ID for rollback
        - Timestamp
        - Accuracy measurements
        - Reason for success/failure
        """
        engine = RuleIncorporationEngine(test_suite=MockRegressionTestSuite(initial_accuracy=0.85))
        report = self.create_test_validation_report()

        # Incorporate several rules
        for i in range(1, 6):
            rule = self.create_test_rule(i, confidence=0.85)

            result = engine.incorporate(
                rule=rule,
                target_layer=StratificationLevel.TACTICAL,
                validation_report=report,
                is_autonomous=True,
            )

            # Verify traceability for each result
            assert result.snapshot_id is not None, "Snapshot ID must be present"
            assert result.reason != "", "Reason must be provided"

            if result.is_success():
                assert result.accuracy_before >= 0.0
                assert result.accuracy_after >= 0.0
                assert result.modification_number > 0

        # Verify incorporation history traceability
        for history_item in engine.incorporation_history:
            assert "rule" in history_item
            assert "layer" in history_item
            assert "confidence" in history_item
            assert "snapshot_id" in history_item
            assert "accuracy_before" in history_item
            assert "accuracy_after" in history_item
            assert "timestamp" in history_item

        print("\n✓ MVP Criterion 3 PASSED: All modifications are traceable")
        print("  Every incorporation has:")
        print("    - Snapshot ID: ✓")
        print("    - Timestamp: ✓")
        print("    - Accuracy measurements: ✓")
        print("    - Layer and confidence: ✓")

    def test_regression_threshold_accuracy(self):
        """Test that regression threshold correctly triggers rollback."""
        # Create test suite with controllable accuracy
        test_suite = MockRegressionTestSuite(initial_accuracy=0.90)

        engine = RuleIncorporationEngine(
            test_suite=test_suite,
            regression_threshold=0.03,  # 3% threshold
        )
        report = self.create_test_validation_report()

        # Set rule 2 to cause 5% regression (above 3% threshold)
        test_suite.set_regression_rules({2})

        # Rule 1: Should succeed
        rule1 = self.create_test_rule(1, confidence=0.85)
        result1 = engine.incorporate(
            rule1, StratificationLevel.TACTICAL, report, is_autonomous=True
        )
        assert result1.is_success()

        # Rule 2: Should be rolled back due to regression
        rule2 = self.create_test_rule(2, confidence=0.85)
        result2 = engine.incorporate(
            rule2, StratificationLevel.TACTICAL, report, is_autonomous=True
        )
        assert not result2.is_success()
        assert result2.regression_detected
        assert "regression" in result2.reason.lower()

        # Verify rollback occurred
        assert len(engine.rollback_history) == 1
        assert len(engine.incorporation_history) == 1  # Only rule 1

        print("\n✓ Regression threshold test PASSED")
        print(f"  Threshold: {engine.regression_threshold:.1%}")
        print(f"  Rollbacks triggered: {len(engine.rollback_history)}")

    def test_complete_workflow(self):
        """
        Test complete incorporation workflow with mixed results.

        Demonstrates:
        - Multiple successful incorporations
        - Regression detection and rollback
        - Statistics tracking
        - History management
        """
        test_suite = MockRegressionTestSuite(initial_accuracy=0.80)
        test_suite.set_regression_rules({4, 7})  # Rules 4 and 7 will regress

        engine = RuleIncorporationEngine(
            test_suite=test_suite,
            regression_threshold=0.02,
        )
        report = self.create_test_validation_report()

        # Incorporate 10 rules with some regressions
        for i in range(1, 11):
            rule = self.create_test_rule(i, confidence=0.85)
            result = engine.incorporate(
                rule, StratificationLevel.TACTICAL, report, is_autonomous=True
            )

            # Print result
            if result.is_success():
                print(f"  Rule {i}: ✓ Incorporated")
            else:
                print(f"  Rule {i}: ✗ Rolled back ({result.reason})")

        # Check statistics
        stats = engine.get_statistics()

        assert stats["total_modifications"] == 8  # 10 - 2 rolled back
        assert stats["rollback_count"] == 2
        assert len(engine.incorporation_history) == 8
        assert len(engine.rollback_history) == 2

        print("\n✓ Complete workflow test PASSED")
        print(f"  Total modifications: {stats['total_modifications']}")
        print(f"  Rollbacks: {stats['rollback_count']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Final accuracy: {stats['current_accuracy']:.1%}")


class TestRollbackMechanism:
    """Test rollback mechanism details."""

    def test_rollback_restores_state(self):
        """Test that rollback fully restores previous state."""
        core = SimpleASPCore()

        # Initial state: 0 rules
        initial_state = core.get_state()
        assert len(initial_state["rules"]) == 0

        # Add a rule
        core.add_rule(
            "rule_1(X) :- cond(X).",
            StratificationLevel.TACTICAL,
            0.85,
            {},
        )
        assert len(core.rules) == 1

        # Rollback to initial state
        core.restore_state(initial_state)
        assert len(core.rules) == 0
        assert core.rule_count == 0

    def test_rollback_event_tracking(self):
        """Test that rollback events are properly tracked."""
        test_suite = MockRegressionTestSuite(initial_accuracy=0.85)
        test_suite.set_regression_rules({1})

        engine = RuleIncorporationEngine(test_suite=test_suite)
        report = ValidationReport(
            rule_asp="test(X) :- cond(X).",
            rule_id="test",
            target_layer="tactical",
        )
        report.final_decision = "accept"

        rule = GeneratedRule(
            asp_rule="bad_rule(X) :- harmful(X).",
            confidence=0.85,
            reasoning="Test",
            source_type="gap_fill",
            source_text="Test",
            predicates_used=["bad_rule", "harmful"],
            new_predicates=["bad_rule"],
        )

        # This should trigger rollback
        result = engine.incorporate(rule, StratificationLevel.TACTICAL, report, is_autonomous=True)

        assert not result.is_success()
        assert len(engine.rollback_history) == 1

        event = engine.rollback_history[0]
        assert event.reason is not None
        assert event.snapshot_id is not None
        assert event.failed_rule_text == "bad_rule(X) :- harmful(X)."
        assert event.regression_details is not None
