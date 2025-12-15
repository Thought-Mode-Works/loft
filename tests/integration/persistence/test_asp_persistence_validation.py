import pytest
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional # Add Optional

from loft.persistence.asp_persistence import ASPPersistenceManager, CorruptedFileError, LoadResult # Import LoadResult
from loft.symbolic.stratification import StratificationLevel
from loft.symbolic.asp_program import StratifiedASPCore
from loft.symbolic.asp_rule import ASPRule, RuleMetadata

# --- Helper Functions for Test Setup ---


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


# --- Test Class ---


class TestASPPersistenceValidation:
    """Comprehensive validation of ASP persistence."""

    # pytest fixture to create a temporary directory for each test
    @pytest.fixture
    def tmp_path(self, tmpdir_factory) -> Path:
        """Create a temporary directory for persistence operations."""
        return Path(tmpdir_factory.mktemp("asp_persistence_test"))

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path: Path) -> None: # Added type hint
        """Cleanup tmp_path before each test to ensure isolation."""
        # Ensure tmp_path is empty before test runs
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)

    def test_save_load_cycle_integrity(self, tmp_path: Path) -> None:
        """Verify rules survive save/load cycle with no data loss."""
        manager = ASPPersistenceManager(str(tmp_path), enable_git=False) # Convert Path to str
        core = create_test_asp_core_with_rules(50)
        core.add_rule(
            ASPRule(
                "c1",
                "c(1).",
                StratificationLevel.CONSTITUTIONAL,
                confidence=1.0,
                metadata=RuleMetadata(
                    provenance="test",
                    timestamp="2025-01-01T00:00:00Z",
                    validation_score=1.0,
                ),
            )
        )
        core.add_rule(
            ASPRule(
                "s1",
                "s(1).",
                StratificationLevel.STRATEGIC,
                confidence=0.9,
                metadata=RuleMetadata(
                    provenance="test",
                    timestamp="2025-01-01T00:00:00Z",
                    validation_score=0.9,
                ),
            )
        )
        core.add_rule(
            ASPRule(
                "o1",
                "o(1).",
                StratificationLevel.OPERATIONAL,
                confidence=0.8,
                metadata=RuleMetadata(
                    provenance="test",
                    timestamp="2025-01-01T00:00:00Z",
                    validation_score=0.8,
                ),
            )
        )

        # Save all rules
        rules_by_layer = {
            level: core.get_program(level).rules for level in StratificationLevel
        }
        manager.save_all_rules(rules_by_layer, overwrite=True)

        # Create new core and load
        new_core = StratifiedASPCore()
        load_result = manager.load_all_rules()  # Use new LoadResult
        assert not load_result.had_errors  # No errors expected here
        loaded_rules_by_layer = load_result.rules_by_layer

        # Add loaded rules to new_core (assuming add_rule is idempotent or handles this)
        for level, rules in loaded_rules_by_layer.items():
            for rule in rules:
                new_core.add_rule(rule)

        # Verify all rules present and identical
        assert_cores_equal(core, new_core)

    def test_snapshot_restore_cycle(self, tmp_path: Path) -> None:
        """Test snapshot creation and restoration."""
        manager = ASPPersistenceManager(str(tmp_path), enable_git=False) # Convert Path to str
        core = create_test_asp_core_with_rules(20)

        # Save initial state
        rules_by_layer_initial = {
            level: core.get_program(level).rules for level in StratificationLevel
        }
        manager.save_all_rules(rules_by_layer_initial, overwrite=True)
        _snapshot_path = manager.create_snapshot(1, "cycle_001") # Marked unused with _

        # Modify rules
        add_rules_to_core(core, 10, start_idx=20)
        rules_by_layer_modified = {
            level: core.get_program(level).rules for level in StratificationLevel
        }
        manager.save_all_rules(rules_by_layer_modified, overwrite=True)

        # Ensure modified rules are present before restore
        modified_core_check = StratifiedASPCore()
        load_result_modified = manager.load_all_rules()
        assert not load_result_modified.had_errors
        for level, rules in load_result_modified.rules_by_layer.items():
            for rule in rules:
                modified_core_check.add_rule(rule)
        assert count_rules(modified_core_check) == 30  # 20 initial + 10 new

        # Restore snapshot
        manager.restore_snapshot(1)

        # Load and verify original state restored
        restored_core = StratifiedASPCore()
        load_result_restored = manager.load_all_rules()
        assert not load_result_restored.had_errors
        for level, rules in load_result_restored.rules_by_layer.items():
            for rule in rules:
                restored_core.add_rule(rule)

        assert count_rules(restored_core) == 20
        # Assert that the restored core matches the original core (by content, not just rule count)
        original_core_for_comparison = create_test_asp_core_with_rules(20)
        assert_cores_equal(original_core_for_comparison, restored_core)

    def test_linked_asp_metadata_roundtrip(self, tmp_path: Path) -> None:
        """Verify LinkedASP metadata survives roundtrip."""
        manager = ASPPersistenceManager(str(tmp_path), enable_git=False) # Convert Path to str

        rule = ASPRule(
            rule_id="test_001",
            asp_text="valid(X) :- offer(X), acceptance(X).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.85,
            metadata=RuleMetadata(
                provenance="llm_generated",
                timestamp="2024-01-15T10:30:00Z",
                validation_score=0.85,  # Match confidence
                author="Claude",
                tags=["contract_001", "phase_3"],
                notes="Generated from principle: contract formation requires offer and acceptance.",
            ),
        )

        core = StratifiedASPCore()
        core.add_rule(rule)
        rules_by_layer = {
            level: core.get_program(level).rules for level in StratificationLevel
        }
        manager.save_all_rules(rules_by_layer, overwrite=True)

        # Load and verify metadata
        new_core = StratifiedASPCore()
        load_result = manager.load_all_rules()
        assert not load_result.had_errors
        loaded_rules_by_layer = load_result.rules_by_layer
        for level, rules in loaded_rules_by_layer.items():
            for loaded_rule in rules:
                new_core.add_rule(loaded_rule)

        loaded_rule_from_core = new_core.get_rule("test_001")
        assert loaded_rule_from_core is not None
        assert loaded_rule_from_core.metadata is not None

        # Verify specific metadata fields
        assert loaded_rule_from_core.metadata.provenance == "llm_generated"
        assert loaded_rule_from_core.metadata.timestamp == "2024-01-15T10:30:00Z"
        assert loaded_rule_from_core.metadata.validation_score == 0.85
        assert loaded_rule_from_core.metadata.author == "Claude"
        assert loaded_rule_from_core.metadata.tags == ["contract_001", "phase_3"]
        assert (
            loaded_rule_from_core.metadata.notes
            == "Generated from principle: contract formation requires offer and acceptance."
        )
        assert loaded_rule_from_core.confidence == 0.85

    def test_concurrent_save_operations(self, tmp_path: Path) -> None:
        """Test thread-safe concurrent saves."""
        manager = ASPPersistenceManager(str(tmp_path), enable_git=False) # Convert Path to str

        # Create multiple cores with some initial rules
        cores = []
        for i in range(5):
            core = create_test_asp_core_with_rules(5, start_idx=i * 5)
            core.add_rule(
                ASPRule(
                    f"core_{i}_initial",
                    f"initial({i}).",
                    StratificationLevel.CONSTITUTIONAL,
                    confidence=1.0,
                    metadata=RuleMetadata(
                        provenance="test",
                        timestamp="2025-01-01T00:00:00Z",
                        validation_score=1.0,
                    ),
                )
            )
            cores.append(core)

        # Function to save a core
        def save_core_task(core_to_save: StratifiedASPCore, manager: ASPPersistenceManager) -> bool: # Added type hints
            rules_by_layer = {
                level: core_to_save.get_program(level).rules
                for level in StratificationLevel
            }
            manager.save_all_rules(rules_by_layer, overwrite=True)
            return True

        # Run concurrent saves
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Modified to pass manager to save_core_task
            futures = [executor.submit(save_core_task, core, manager) for core in cores]
            results = [f.result() for f in futures]
            assert all(results)  # All saves should succeed

        final_core = StratifiedASPCore()
        load_result = manager.load_all_rules()
        assert not load_result.had_errors  # Assert no parsing errors occurred
        loaded_rules_by_layer = load_result.rules_by_layer
        for level, rules in loaded_rules_by_layer.items():
            for rule in rules:
                final_core.add_rule(rule)

        # Assert that the loaded core is valid and contains rules
        # With overwrite=True, only one of the concurrent writes will persist for each file.
        # We just need to ensure the final state is not corrupted and contains some rules.
        assert (
            count_rules(final_core) > 0
        ), "Final core should not be empty after concurrent saves"
        # Additional check: attempt to load again to ensure consistency (no hidden corruption)
        manager.load_all_rules()  # Should not raise

    def test_corruption_recovery(self, tmp_path: Path) -> None:
        """Test recovery from corrupted files."""
        manager = ASPPersistenceManager(str(tmp_path), enable_git=False) # Convert Path to str
        core = create_test_asp_core_with_rules(20)
        core.add_rule(
            ASPRule(
                "c1",
                "c(1).",
                StratificationLevel.CONSTITUTIONAL,
                confidence=1.0,
                metadata=RuleMetadata(
                    provenance="test",
                    timestamp="2025-01-01T00:00:00Z",
                    validation_score=1.0,
                ),
            )
        )
        core.add_rule(
            ASPRule(
                "s1",
                "s(1).",
                StratificationLevel.STRATEGIC,
                confidence=0.9,
                metadata=RuleMetadata(
                    provenance="test",
                    timestamp="2025-01-01T00:00:00Z",
                    validation_score=0.9,
                ),
            )
        )
        core.add_rule(
            ASPRule(
                "t1",
                "t(1).",
                StratificationLevel.TACTICAL,
                confidence=0.8,
                metadata=RuleMetadata(
                    provenance="test",
                    timestamp="2025-01-01T00:00:00Z",
                    validation_score=0.8,
                ),
            )
        )

        rules_by_layer = {
            level: core.get_program(level).rules for level in StratificationLevel
        }
        manager.save_all_rules(rules_by_layer, overwrite=True)

        # Corrupt a tactical layer file
        tactical_file = tmp_path / f"{StratificationLevel.TACTICAL.value}.lp"
        assert tactical_file.exists()
        with open(tactical_file, "w") as f:
            f.write("CORRUPTED DATA @#$%")  # Invalid ASP syntax

        # Attempt load without recovery (should raise CorruptedFileError)
        with pytest.raises(CorruptedFileError):
            manager.load_all_rules(recover_on_error=False)

        # Attempt load with recovery (should not raise, but report errors)
        load_result = manager.load_all_rules(recover_on_error=True)
        assert load_result.had_errors
        assert StratificationLevel.TACTICAL.value in load_result.recovered_layers
        assert len(load_result.parsing_errors) > 0
        # Rules from other layers should still be loaded
        assert StratificationLevel.CONSTITUTIONAL in load_result.rules_by_layer
        assert len(load_result.rules_by_layer[StratificationLevel.CONSTITUTIONAL]) == 1