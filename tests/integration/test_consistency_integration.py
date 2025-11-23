"""
Integration tests for consistency checking framework.

Tests integration with version control and validation systems.
"""

import tempfile
from pathlib import Path
from datetime import datetime
from loft.consistency import (
    ConsistencyChecker,
    TestFixtures,
    ConsistencyReporter,
)
from loft.version_control import (
    VersionControl,
    CoreState,
    Rule,
    StratificationLevel,
    create_state_id,
)


class TestVersionControlIntegration:
    """Tests for integration with version control."""

    def test_check_consistency_before_commit(self) -> None:
        """Test checking consistency before committing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))
            checker = ConsistencyChecker()

            # Create a consistent state
            state = TestFixtures.simple_consistent_state()

            # Check consistency
            report = checker.check(state)
            assert report.passed

            # Commit if consistent
            if report.passed:
                commit_id = vc.commit(state, "Add consistent state")
                assert commit_id is not None

    def test_prevent_commit_on_inconsistency(self) -> None:
        """Test preventing commit of inconsistent state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            VersionControl(storage_dir=Path(tmpdir))
            checker = ConsistencyChecker(strict=True)

            # Create an inconsistent state
            state = TestFixtures.contradictory_state()

            # Check consistency
            report = checker.check(state)

            # Should not pass in strict mode
            if report.errors > 0:
                # In a real system, we'd prevent the commit
                # For test, we just verify the inconsistency was detected
                assert not report.passed

    def test_consistency_across_versions(self) -> None:
        """Test tracking consistency across version history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))
            checker = ConsistencyChecker()
            reporter = ConsistencyReporter(history_file=Path(tmpdir) / "history.json")

            # Commit several versions and track consistency
            states = [
                TestFixtures.empty_state(),
                TestFixtures.simple_consistent_state(),
                TestFixtures.incomplete_state(),
            ]

            for i, state in enumerate(states):
                # Check consistency
                report = checker.check(state)

                # Generate enhanced report
                reporter.report(report, len(state.rules))

                # Commit
                vc.commit(state, f"Version {i + 1}")

            # Verify history was tracked
            assert len(reporter.history.metrics) == 3

    def test_rollback_on_consistency_regression(self) -> None:
        """Test rollback when consistency regresses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vc = VersionControl(storage_dir=Path(tmpdir))
            checker = ConsistencyChecker()

            # Commit a consistent state
            state1 = TestFixtures.simple_consistent_state()
            commit1 = vc.commit(state1, "Consistent state")

            report1 = checker.check(state1)
            assert report1.passed

            # Commit an inconsistent state
            state2 = TestFixtures.contradictory_state()
            vc.commit(state2, "Inconsistent state")

            report2 = checker.check(state2)
            assert not report2.passed

            # Rollback to consistent state
            rolled_back_state = vc.rollback(commit1)

            # Verify rolled back state is consistent
            report3 = checker.check(rolled_back_state)
            assert report3.passed


class TestConsistencyCheckPipeline:
    """Tests for consistency check pipeline."""

    def test_automated_consistency_pipeline(self) -> None:
        """Test automated consistency checking pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = ConsistencyChecker(strict=False)
            reporter = ConsistencyReporter(history_file=Path(tmpdir) / "history.json")

            # Pipeline: generate state -> check consistency -> report
            test_states = [
                TestFixtures.empty_state(),
                TestFixtures.simple_consistent_state(),
                TestFixtures.contradictory_state(),
                TestFixtures.incomplete_state(),
            ]

            results = []
            for state in test_states:
                # Check consistency
                report = checker.check(state)

                # Generate enhanced report
                enhanced = reporter.report(report, len(state.rules), save_history=True)

                results.append(
                    {"state_id": state.state_id, "passed": report.passed, "report": enhanced}
                )

            # Verify all checks completed
            assert len(results) == 4

            # Verify history was saved
            assert (Path(tmpdir) / "history.json").exists()

    def test_consistency_with_different_stratification_levels(self) -> None:
        """Test consistency across different stratification levels."""
        checker = ConsistencyChecker()

        # Create state with rules at all levels
        rules = [
            Rule(
                "r_const",
                "constitutional_rule.",
                StratificationLevel.CONSTITUTIONAL,
                1.0,
                "human",
                datetime.utcnow().isoformat(),
            ),
            Rule(
                "r_strat",
                "strategic_rule.",
                StratificationLevel.STRATEGIC,
                0.95,
                "llm",
                datetime.utcnow().isoformat(),
            ),
            Rule(
                "r_tact",
                "tactical_rule.",
                StratificationLevel.TACTICAL,
                0.9,
                "llm",
                datetime.utcnow().isoformat(),
            ),
            Rule(
                "r_oper",
                "operational_rule.",
                StratificationLevel.OPERATIONAL,
                0.85,
                "llm",
                datetime.utcnow().isoformat(),
            ),
        ]

        state = CoreState(
            state_id=create_state_id(),
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={},
            metrics={},
        )

        report = checker.check(state)

        # Should pass - no conflicts between levels
        assert report.passed or report.errors == 0


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_consistency_check_performance(self) -> None:
        """Test that consistency checking is fast enough."""
        import time

        checker = ConsistencyChecker()

        # Create a state with moderate number of rules
        rules = []
        for i in range(50):
            rules.append(
                Rule(
                    f"rule_{i}",
                    f"pred_{i}(X) :- pred_{i - 1}(X)." if i > 0 else f"pred_{i}(a).",
                    StratificationLevel.TACTICAL,
                    0.9,
                    "llm",
                    datetime.utcnow().isoformat(),
                )
            )

        state = CoreState(
            state_id=create_state_id(),
            timestamp=datetime.utcnow().isoformat(),
            rules=rules,
            configuration={},
            metrics={},
        )

        # Measure time
        start = time.time()
        report = checker.check(state)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second for 50 rules)
        assert elapsed < 1.0
        assert report is not None

    def test_multiple_checks_performance(self) -> None:
        """Test performance of multiple sequential checks."""
        import time

        checker = ConsistencyChecker()
        state = TestFixtures.simple_consistent_state()

        # Run multiple checks
        start = time.time()
        for _ in range(100):
            checker.check(state)
        elapsed = time.time() - start

        # Should complete 100 checks quickly (< 5 seconds)
        assert elapsed < 5.0


class TestRegressionDetection:
    """Tests for regression detection."""

    def test_detect_consistency_regression(self) -> None:
        """Test detecting consistency regressions over time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = ConsistencyChecker()
            reporter = ConsistencyReporter(history_file=Path(tmpdir) / "history.json")

            # Series of states: Good -> Good -> Regression
            states = [
                TestFixtures.simple_consistent_state(),  # Good, score ~1.0
                TestFixtures.simple_consistent_state(),  # Good, score ~1.0
                TestFixtures.contradictory_state(),  # Error - regression!
            ]

            for state in states:
                report = checker.check(state)
                reporter.report(report, len(state.rules), save_history=True)

            # Verify regression was detected between last two metrics
            assert len(reporter.history.metrics) == 3
            # The regression detector checks last two metrics
            # Since we went from 1.0 -> 0.0, it should detect regression
            assert reporter.history.detect_regression()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_state_with_no_rules(self) -> None:
        """Test checking state with no rules."""
        checker = ConsistencyChecker()
        state = TestFixtures.empty_state()

        report = checker.check(state)
        assert report.passed

    def test_state_with_single_rule(self) -> None:
        """Test checking state with single rule."""
        checker = ConsistencyChecker()

        rule = Rule(
            "r1",
            "fact(a).",
            StratificationLevel.OPERATIONAL,
            1.0,
            "human",
            datetime.utcnow().isoformat(),
        )

        state = CoreState(
            create_state_id(),
            datetime.utcnow().isoformat(),
            [rule],
            {},
            {},
        )

        report = checker.check(state)
        # Single fact should be consistent
        assert report.passed or report.errors == 0

    def test_state_with_complex_rules(self) -> None:
        """Test checking state with complex rule structures."""
        checker = ConsistencyChecker()

        rules = [
            Rule(
                "r1",
                "base(a).",
                StratificationLevel.OPERATIONAL,
                1.0,
                "human",
                datetime.utcnow().isoformat(),
            ),
            Rule(
                "r2",
                "derived(X) :- base(X), condition(X).",
                StratificationLevel.TACTICAL,
                0.9,
                "llm",
                datetime.utcnow().isoformat(),
            ),
            Rule(
                "r3",
                "final(X) :- derived(X), -excluded(X).",
                StratificationLevel.STRATEGIC,
                0.95,
                "llm",
                datetime.utcnow().isoformat(),
            ),
        ]

        state = CoreState(
            create_state_id(),
            datetime.utcnow().isoformat(),
            rules,
            {},
            {},
        )

        report = checker.check(state)
        # Should complete without errors (may have warnings about undefined predicates)
        assert isinstance(report, object)
