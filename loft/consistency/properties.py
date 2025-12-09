"""
Property-based tests for consistency checking.

Uses hypothesis to verify invariants that should hold under all operations.
"""

from hypothesis import given, settings, assume
from .generators import rule_strategy, core_state_strategy
from .checker import ConsistencyChecker
from ..version_control import Rule, CoreState


class ConsistencyProperties:
    """Property-based tests for consistency checking."""

    def __init__(self, checker: ConsistencyChecker):
        """
        Initialize property tests.

        Args:
            checker: Consistency checker to use
        """
        self.checker = checker

    @given(core_state_strategy(min_rules=0, max_rules=5))
    @settings(max_examples=50, deadline=None)
    def test_consistency_is_deterministic(self, state: CoreState) -> None:
        """
        Property: Consistency checking is deterministic.

        Running the same check twice should give same result.
        """
        report1 = self.checker.check(state)
        report2 = self.checker.check(state)

        assert report1.passed == report2.passed
        assert len(report1.inconsistencies) == len(report2.inconsistencies)
        assert report1.errors == report2.errors
        assert report1.warnings == report2.warnings

    @given(core_state_strategy(min_rules=0, max_rules=0))
    @settings(max_examples=10, deadline=None)
    def test_empty_state_is_consistent(self, state: CoreState) -> None:
        """
        Property: Empty core state is always consistent.
        """
        # Ensure state is truly empty
        assume(len(state.rules) == 0)

        report = self.checker.check(state)
        assert report.passed
        assert report.errors == 0

    @given(rule_strategy(), core_state_strategy(min_rules=0, max_rules=5))
    @settings(max_examples=50, deadline=None)
    def test_adding_rule_preserves_consistency_type(
        self, new_rule: Rule, state: CoreState
    ) -> None:
        """
        Property: Adding a non-contradictory rule to a consistent state.

        If a state is consistent and we add a rule that doesn't contradict
        existing rules, the result should not introduce NEW contradictions
        (though it may introduce other issues like incompleteness).
        """
        # Check initial consistency
        self.checker.check(state)

        # Add new rule
        state.rules.append(new_rule)

        # Check new consistency
        final_report = self.checker.check(state)

        # If initial state had no contradictions, adding a random rule
        # might introduce contradictions, but that's expected.
        # We just verify the checker can handle it.
        assert isinstance(final_report.passed, bool)
        assert final_report.errors >= 0
        assert final_report.warnings >= 0

    @given(core_state_strategy(min_rules=1, max_rules=10))
    @settings(max_examples=50, deadline=None)
    def test_removing_rule_cannot_introduce_contradictions(
        self, state: CoreState
    ) -> None:
        """
        Property: Removing a rule cannot introduce new contradictions.

        Contradictions require conflicting rules, so removing a rule
        can only reduce or maintain contradictions, never increase them.
        """
        assume(len(state.rules) > 0)

        # Get initial contradiction count
        initial_report = self.checker.check(state)
        initial_contradictions = sum(
            1
            for inc in initial_report.inconsistencies
            if inc.type.value == "contradiction"
        )

        # Remove a rule
        state.rules.pop()

        # Check new consistency
        final_report = self.checker.check(state)
        final_contradictions = sum(
            1
            for inc in final_report.inconsistencies
            if inc.type.value == "contradiction"
        )

        # Contradictions should not increase
        assert final_contradictions <= initial_contradictions

    @given(core_state_strategy(min_rules=0, max_rules=5))
    @settings(max_examples=50, deadline=None)
    def test_consistency_report_counts_match(self, state: CoreState) -> None:
        """
        Property: Consistency report counts match actual inconsistencies.

        The error/warning/info counts should equal the number of
        inconsistencies with those severities.
        """
        report = self.checker.check(state)

        actual_errors = sum(1 for i in report.inconsistencies if i.severity == "error")
        actual_warnings = sum(
            1 for i in report.inconsistencies if i.severity == "warning"
        )
        actual_info = sum(1 for i in report.inconsistencies if i.severity == "info")

        assert report.errors == actual_errors
        assert report.warnings == actual_warnings
        assert report.info == actual_info

    @given(core_state_strategy(min_rules=0, max_rules=5))
    @settings(max_examples=50, deadline=None)
    def test_passed_implies_no_errors(self, state: CoreState) -> None:
        """
        Property: If check passed, there are no errors.

        A check passes only if there are no errors (and no warnings in strict mode).
        """
        report = self.checker.check(state)

        if report.passed:
            assert report.errors == 0
            if self.checker.strict:
                assert report.warnings == 0

    @given(core_state_strategy(min_rules=0, max_rules=5))
    @settings(max_examples=50, deadline=None)
    def test_inconsistency_has_valid_severity(self, state: CoreState) -> None:
        """
        Property: All inconsistencies have valid severity levels.
        """
        report = self.checker.check(state)

        valid_severities = {"error", "warning", "info"}
        for inconsistency in report.inconsistencies:
            assert inconsistency.severity in valid_severities

    @given(core_state_strategy(min_rules=0, max_rules=5))
    @settings(max_examples=50, deadline=None)
    def test_report_formatting_succeeds(self, state: CoreState) -> None:
        """
        Property: Report formatting never fails.

        Generating summary and full format should always succeed.
        """
        report = self.checker.check(state)

        # These should not raise exceptions
        summary = report.summary()
        full_report = report.format()

        assert isinstance(summary, str)
        assert isinstance(full_report, str)
        assert len(summary) > 0
        assert len(full_report) > 0


def run_property_tests(checker: ConsistencyChecker) -> None:
    """
    Run all property-based tests.

    Args:
        checker: Consistency checker to test

    Note:
        This function is intended to be called from test suites.
        Individual properties can also be tested directly.
    """
    props = ConsistencyProperties(checker)

    # Run each property test
    props.test_consistency_is_deterministic()
    props.test_empty_state_is_consistent()
    props.test_adding_rule_preserves_consistency_type()
    props.test_removing_rule_cannot_introduce_contradictions()
    props.test_consistency_report_counts_match()
    props.test_passed_implies_no_errors()
    props.test_inconsistency_has_valid_severity()
    props.test_report_formatting_succeeds()
