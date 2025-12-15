import pytest
import shutil
from pathlib import Path
from typing import Dict, List, Any # Import necessary types

from loft.persistence.asp_persistence import ASPPersistenceManager
from loft.persistence.metrics import PersistenceMetricsCollector  # Import the collector
from loft.symbolic.stratification import StratificationLevel
from loft.symbolic.asp_program import StratifiedASPCore
from loft.symbolic.asp_rule import ASPRule, RuleMetadata


# Import helper functions from test_asp_persistence_validation.py (or redefine/copy if necessary)
# For now, let's copy them to keep this test file self-contained.
# In a real project, these helpers would be in a common test_utils.py
def create_test_asp_core_with_rules(
    num_rules: int, start_idx: int = 0
) -> StratifiedASPCore:
    """Creates a StratifiedASPCore with a specified number of dummy rules."""
    core = StratifiedASPCore()
    for i in range(start_idx, start_idx + num_rules):
        confidence_val = 0.8 + (i % 10) / 100
        rule = ASPRule(
            rule_id=f"rule_{i}",
            asp_text=f"p({i}) :- q({i}).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=confidence_val,
            metadata=RuleMetadata(
                provenance="test_gen",
                timestamp=f"2025-01-01T{i%24:02d}:00:00Z",
                validation_score=confidence_val,  # Use confidence_val here
                tags=[f"case_{i%5}"],
                notes=f"Generated for test, author: TestRunner{i%2}",
            ),
        )
        core.add_rule(rule)
    return core


def add_rules_to_core(
    core: StratifiedASPCore, num_rules: int, start_idx: int = 0
) -> None:
    """Adds a specified number of dummy rules to an existing core."""
    for i in range(start_idx, start_idx + num_rules):
        confidence_val = 0.7 + (i % 10) / 100
        rule = ASPRule(
            rule_id=f"new_rule_{i}",
            asp_text=f"new_p({i}) :- new_q({i}).",
            stratification_level=StratificationLevel.OPERATIONAL,
            confidence=confidence_val,
            metadata=RuleMetadata(
                provenance="test_incremental",
                timestamp=f"2025-01-02T{i%24:02d}:00:00Z",
                validation_score=confidence_val,  # Use confidence_val here
                tags=[f"new_case_{i%3}"],
                notes=f"Incremental rule, author: TestRunner{i%2}",
            ),
        )
        core.add_rule(rule)


def count_rules(core: StratifiedASPCore) -> int:
    """Counts the total number of rules in a StratifiedASPCore."""
    return sum(len(core.get_program(level).rules) for level in StratificationLevel)


def assert_cores_equal(core1: StratifiedASPCore, core2: StratifiedASPCore) -> None:
    """Asserts that two StratifiedASPCore instances contain the same rules."""
    rules1 = {rule.rule_id: rule for rule in core1.get_all_rules()}
    rules2 = {rule.rule_id: rule for rule in core2.get_all_rules()}

    assert len(rules1) == len(
        rules2
    ), f"Rule counts differ: {len(rules1)} != {len(rules2)}"

    for rule_id, rule1 in rules1.items():
        assert rule_id in rules2, f"Rule {rule_id} missing from second core"
        rule2 = rules2[rule_id]
        assert rule1.asp_text == rule2.asp_text, f"ASP text for {rule_id} differs"
        assert (
            rule1.stratification_level == rule2.stratification_level
        ), f"Layer for {rule_id} differs"
        assert rule1.confidence == rule2.confidence, f"Confidence for {rule_id} differs"

        # Check metadata
        assert (
            rule1.metadata.provenance == rule2.metadata.provenance
        ), f"Metadata provenance for {rule_id} differs"
        assert (
            rule1.metadata.timestamp == rule2.metadata.timestamp
        ), f"Metadata timestamp for {rule_id} differs"
        assert (
            rule1.metadata.validation_score == rule2.metadata.validation_score
        ), f"Metadata validation_score for {rule_id} differs"
        assert (
            rule1.metadata.author == rule2.metadata.author
        ), f"Metadata author for {rule_id} differs"
        assert (
            rule1.metadata.tags == rule2.metadata.tags
        ), f"Metadata tags for {rule_id} differs"
        assert (
            rule1.metadata.notes == rule2.metadata.notes
        ), f"Metadata notes for {rule_id} differs"


# --- Long-Running Test Class ---


def create_or_modify_core(
    cycle: int, base_rules: int = 10, new_rules_per_cycle: int = 2
) -> StratifiedASPCore:
    """Creates a new core or modifies an existing one based on the cycle number."""
    # Start with a base set of rules
    core = create_test_asp_core_with_rules(base_rules, start_idx=0)

    # Add new rules based on the cycle number
    # This ensures the core grows over time in a long-running test
    if cycle > 0:
        add_rules_to_core(core, new_rules_per_cycle * cycle, start_idx=base_rules)

    return core


@pytest.mark.slow
class TestLongRunningPersistence:
    """Long-running tests for persistence stability."""

    @pytest.fixture
    def tmp_path(self, tmpdir_factory) -> Path:
        """Create a temporary directory for persistence operations."""
        return Path(tmpdir_factory.mktemp("asp_long_persistence_test"))

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path: Path) -> None:
        """Cleanup tmp_path before each test to ensure isolation."""
        # Ensure tmp_path is empty before test runs
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)

    def test_1000_save_cycles(self, tmp_path: Path) -> None:
        """Run 1000 save/load cycles to validate stability."""
        manager = ASPPersistenceManager(str(tmp_path), enable_git=False)
        metrics_collector = PersistenceMetricsCollector()

        num_cycles = 1000
        snapshot_interval = 100

        initial_rules_count = 10  # Base rules

        for i in range(num_cycles):
            # Create or modify rules - progressively larger core
            current_core = create_or_modify_core(i, base_rules=initial_rules_count)

            # Save with metrics
            rules_by_layer_current = {
                level: current_core.get_program(level).rules
                for level in StratificationLevel
            }
            _save_metrics = metrics_collector.measure_save_cycle( # Marked unused with _
                manager, rules_by_layer_current
            )

            # Periodic snapshot
            if i % snapshot_interval == 0:
                _snapshot_metrics = metrics_collector.measure_snapshot_cycle( # Marked unused with _
                    manager, i, f"cycle_{i}"
                )

            # Verify integrity by loading and asserting equality
            loaded_core = StratifiedASPCore()
            load_result = manager.load_all_rules()
            assert (
                not load_result.had_errors
            ), f"Errors during load in cycle {i}: {load_result.parsing_errors}"
            for level, rules in load_result.rules_by_layer.items():
                for rule in rules:
                    loaded_core.add_rule(rule)

            assert_cores_equal(current_core, loaded_core)

            # Measure load metrics
            _load_metrics = metrics_collector.measure_load_cycle( # Marked unused with _
                manager
            )  # This will create a new metric entry

        # Generate report (placeholder, actual report generation in script)
        # For now, just assert that metrics were collected and seem reasonable
        assert len(metrics_collector.collected_metrics) >= num_cycles
        # Ensure that no errors were recorded during the long run
        assert not metrics_collector.error_log

        baseline_report = metrics_collector.generate_baseline_report()
        # Assert that the number of metrics collected matches expectations
        expected_metrics_count = num_cycles * 2 + (num_cycles // snapshot_interval)
        assert baseline_report.total_cycles == expected_metrics_count
        assert baseline_report.avg_save_time_ms > 0
        assert baseline_report.avg_load_time_ms > 0

        # Save report (will be done by script)
        # with open(tmp_path / "persistence_baseline.md", "w") as f:
        #    f.write(baseline_report.to_markdown())