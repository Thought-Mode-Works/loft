"""Tests for rule evolution storage."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from loft.evolution.tracking import (
    RuleMetadata,
    ValidationResult,
    ABTestResult,
    StratificationLayer,
    RuleStatus,
)
from loft.evolution.storage import (
    RuleEvolutionStorage,
    StorageConfig,
    CorpusSnapshot,
)


@pytest.fixture
def temp_storage():
    """Create a temporary storage instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(base_path=Path(tmpdir))
        yield RuleEvolutionStorage(config)


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = StorageConfig()

        assert config.base_path == Path("data/rule_evolution")
        assert config.metadata_dir == "metadata"
        assert config.versions_dir == "versions"

    def test_custom_config(self):
        """Test custom configuration."""
        config = StorageConfig(
            base_path=Path("/custom/path"),
            metadata_dir="meta",
        )

        assert config.base_path == Path("/custom/path")
        assert config.metadata_dir == "meta"

    def test_string_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = StorageConfig(base_path="/string/path")

        assert isinstance(config.base_path, Path)
        assert config.base_path == Path("/string/path")


class TestRuleEvolutionStorage:
    """Tests for RuleEvolutionStorage."""

    def test_save_and_load_rule(self, temp_storage):
        """Test saving and loading a rule."""
        metadata = RuleMetadata(
            rule_id="test_rule_001",
            rule_text="enforceable(C) :- valid(C).",
            natural_language="Valid contracts are enforceable",
            created_by="test",
        )
        # Add a validation result
        metadata.validation_results.append(
            ValidationResult(
                timestamp=datetime.now(),
                test_cases_evaluated=100,
                passed=85,
                failed=15,
                accuracy=0.85,
            )
        )

        temp_storage.save_rule(metadata)
        loaded = temp_storage.load_rule("test_rule_001")

        assert loaded is not None
        assert loaded.rule_id == metadata.rule_id
        assert loaded.rule_text == metadata.rule_text
        assert loaded.current_accuracy == 0.85

    def test_load_nonexistent_rule(self, temp_storage):
        """Test loading a rule that doesn't exist."""
        loaded = temp_storage.load_rule("nonexistent_rule")

        assert loaded is None

    def test_delete_rule(self, temp_storage):
        """Test deleting a rule."""
        metadata = RuleMetadata(
            rule_id="test_rule_002",
            rule_text="test rule",
            natural_language="test",
            created_by="test",
        )

        temp_storage.save_rule(metadata)
        assert temp_storage.load_rule("test_rule_002") is not None

        result = temp_storage.delete_rule("test_rule_002")

        assert result is True
        assert temp_storage.load_rule("test_rule_002") is None

    def test_delete_nonexistent_rule(self, temp_storage):
        """Test deleting a rule that doesn't exist."""
        result = temp_storage.delete_rule("nonexistent")

        assert result is False

    def test_list_rules(self, temp_storage):
        """Test listing all rules."""
        for i in range(3):
            metadata = RuleMetadata(
                rule_id=f"rule_{i}",
                rule_text=f"rule {i}",
                natural_language=f"rule {i}",
                created_by="test",
            )
            temp_storage.save_rule(metadata)

        rule_ids = temp_storage.list_rules()

        assert len(rule_ids) == 3
        assert "rule_0" in rule_ids
        assert "rule_1" in rule_ids
        assert "rule_2" in rule_ids

    def test_load_all_rules(self, temp_storage):
        """Test loading all rules."""
        for i in range(2):
            metadata = RuleMetadata(
                rule_id=f"rule_{i}",
                rule_text=f"rule {i}",
                natural_language=f"rule {i}",
                created_by="test",
            )
            temp_storage.save_rule(metadata)

        rules = temp_storage.load_all_rules()

        assert len(rules) == 2
        assert all(isinstance(r, RuleMetadata) for r in rules)

    def test_save_and_load_ab_test(self, temp_storage):
        """Test saving and loading an A/B test."""
        result = ABTestResult(
            test_id="ab_test_001",
            started_at=datetime.now(),
            variant_a_id="rule_v1",
            variant_b_id="rule_v2",
            variant_a_accuracy=0.75,
            variant_b_accuracy=0.82,
            cases_evaluated=100,
            p_value=0.02,
            winner="b",
        )

        temp_storage.save_ab_test(result)
        loaded = temp_storage.load_ab_test("ab_test_001")

        assert loaded is not None
        assert loaded.test_id == result.test_id
        assert loaded.winner == result.winner
        assert loaded.p_value == result.p_value

    def test_list_ab_tests(self, temp_storage):
        """Test listing A/B tests."""
        for i in range(2):
            result = ABTestResult(
                test_id=f"test_{i}",
                started_at=datetime.now(),
                variant_a_id=f"a_{i}",
                variant_b_id=f"b_{i}",
                variant_a_accuracy=0.7,
                variant_b_accuracy=0.8,
                cases_evaluated=50,
            )
            temp_storage.save_ab_test(result)

        test_ids = temp_storage.list_ab_tests()

        assert len(test_ids) == 2

    def test_save_and_load_genealogy(self, temp_storage):
        """Test saving and loading genealogy."""
        genealogy = {
            "root_1": ["child_1", "child_2"],
            "root_2": ["child_3"],
        }

        temp_storage.save_genealogy(genealogy)
        loaded = temp_storage.load_genealogy()

        assert loaded == genealogy

    def test_load_empty_genealogy(self, temp_storage):
        """Test loading genealogy when file doesn't exist."""
        loaded = temp_storage.load_genealogy()

        assert loaded == {}

    def test_get_rules_by_status(self, temp_storage):
        """Test getting rules by status."""
        active_rule = RuleMetadata(
            rule_id="active_rule",
            rule_text="active",
            natural_language="active",
            created_by="test",
            status=RuleStatus.ACTIVE,
        )
        candidate_rule = RuleMetadata(
            rule_id="candidate_rule",
            rule_text="candidate",
            natural_language="candidate",
            created_by="test",
            status=RuleStatus.CANDIDATE,
        )

        temp_storage.save_rule(active_rule)
        temp_storage.save_rule(candidate_rule)

        active_rules = temp_storage.get_rules_by_status(RuleStatus.ACTIVE)
        candidate_rules = temp_storage.get_rules_by_status(RuleStatus.CANDIDATE)

        assert len(active_rules) == 1
        assert active_rules[0].rule_id == "active_rule"
        assert len(candidate_rules) == 1

    def test_get_rules_by_layer(self, temp_storage):
        """Test getting rules by layer."""
        op_rule = RuleMetadata(
            rule_id="op_rule",
            rule_text="operational",
            natural_language="operational",
            created_by="test",
            current_layer=StratificationLayer.OPERATIONAL,
        )
        tac_rule = RuleMetadata(
            rule_id="tac_rule",
            rule_text="tactical",
            natural_language="tactical",
            created_by="test",
            current_layer=StratificationLayer.TACTICAL,
        )

        temp_storage.save_rule(op_rule)
        temp_storage.save_rule(tac_rule)

        op_rules = temp_storage.get_rules_by_layer(StratificationLayer.OPERATIONAL)
        tac_rules = temp_storage.get_rules_by_layer(StratificationLayer.TACTICAL)

        assert len(op_rules) == 1
        assert len(tac_rules) == 1

    def test_get_storage_stats(self, temp_storage):
        """Test getting storage statistics."""
        # Add some rules
        for i in range(3):
            rule = RuleMetadata(
                rule_id=f"rule_{i}",
                rule_text=f"rule {i}",
                natural_language=f"rule {i}",
                created_by="test",
                status=RuleStatus.ACTIVE if i < 2 else RuleStatus.CANDIDATE,
            )
            temp_storage.save_rule(rule)

        # Add an A/B test
        test = ABTestResult(
            test_id="test_1",
            started_at=datetime.now(),
            variant_a_id="a",
            variant_b_id="b",
            variant_a_accuracy=0.7,
            variant_b_accuracy=0.8,
            cases_evaluated=100,
        )
        temp_storage.save_ab_test(test)

        stats = temp_storage.get_storage_stats()

        assert stats["total_rules"] == 3
        assert stats["total_ab_tests"] == 1
        assert stats["active_ab_tests"] == 1
        assert stats["status_distribution"]["active"] == 2
        assert stats["status_distribution"]["candidate"] == 1

    def test_rule_version_file_created(self, temp_storage):
        """Test that rule version file is created."""
        metadata = RuleMetadata(
            rule_id="version_test",
            rule_text="rule text here",
            natural_language="description",
            created_by="test",
            version="1.0",
        )

        temp_storage.save_rule(metadata)

        version_path = (
            temp_storage.config.base_path
            / temp_storage.config.versions_dir
            / "version_test_1_0.asp"
        )

        assert version_path.exists()
        content = version_path.read_text()
        assert "rule text here" in content
        assert "% Rule: version_test" in content


class TestCorpusSnapshot:
    """Tests for CorpusSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a snapshot object."""
        snapshot = CorpusSnapshot(
            snapshot_id="snapshot_20241128_120000",
            name="test_snapshot",
            created_at=datetime.now(),
            description="Test description",
            rule_count=10,
            ab_test_count=2,
        )

        assert snapshot.name == "test_snapshot"
        assert snapshot.rule_count == 10
        assert snapshot.ab_test_count == 2

    def test_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        now = datetime.now()
        snapshot = CorpusSnapshot(
            snapshot_id="snapshot_001",
            name="my_snapshot",
            created_at=now,
            description="Test",
            rule_count=5,
            ab_test_count=1,
            metadata={"key": "value"},
        )

        data = snapshot.to_dict()

        assert data["snapshot_id"] == "snapshot_001"
        assert data["name"] == "my_snapshot"
        assert data["created_at"] == now.isoformat()
        assert data["description"] == "Test"
        assert data["rule_count"] == 5
        assert data["metadata"] == {"key": "value"}

    def test_snapshot_from_dict(self):
        """Test creating snapshot from dictionary."""
        now = datetime.now()
        data = {
            "snapshot_id": "snapshot_002",
            "name": "restored_snapshot",
            "created_at": now.isoformat(),
            "description": "Restored",
            "rule_count": 15,
            "ab_test_count": 3,
            "metadata": {"restored": True},
        }

        snapshot = CorpusSnapshot.from_dict(data)

        assert snapshot.snapshot_id == "snapshot_002"
        assert snapshot.name == "restored_snapshot"
        assert snapshot.rule_count == 15
        assert snapshot.metadata == {"restored": True}

    def test_snapshot_roundtrip(self):
        """Test snapshot serialization roundtrip."""
        original = CorpusSnapshot(
            snapshot_id="roundtrip_test",
            name="test",
            created_at=datetime.now(),
            description="Roundtrip test",
            rule_count=7,
            ab_test_count=0,
        )

        data = original.to_dict()
        restored = CorpusSnapshot.from_dict(data)

        assert restored.snapshot_id == original.snapshot_id
        assert restored.name == original.name
        assert restored.rule_count == original.rule_count


class TestSnapshotOperations:
    """Tests for snapshot operations in RuleEvolutionStorage."""

    def test_create_snapshot_empty(self, temp_storage):
        """Test creating a snapshot with no rules."""
        snapshot = temp_storage.create_snapshot(
            name="empty_snapshot",
            description="Testing empty snapshot",
        )

        assert snapshot.name == "empty_snapshot"
        assert snapshot.rule_count == 0
        assert snapshot.ab_test_count == 0
        assert snapshot.description == "Testing empty snapshot"

    def test_create_snapshot_with_rules(self, temp_storage):
        """Test creating a snapshot with existing rules."""
        # Add some rules
        for i in range(3):
            rule = RuleMetadata(
                rule_id=f"rule_{i}",
                rule_text=f"rule {i}",
                natural_language=f"rule {i}",
                created_by="test",
            )
            temp_storage.save_rule(rule)

        snapshot = temp_storage.create_snapshot(name="with_rules")

        assert snapshot.rule_count == 3

    def test_create_snapshot_duplicate_name_fails(self, temp_storage):
        """Test that creating a snapshot with duplicate name fails."""
        temp_storage.create_snapshot(name="unique_name")

        with pytest.raises(ValueError, match="already exists"):
            temp_storage.create_snapshot(name="unique_name")

    def test_get_snapshot(self, temp_storage):
        """Test getting snapshot details."""
        temp_storage.create_snapshot(name="get_test", description="Test getting")

        snapshot = temp_storage.get_snapshot("get_test")

        assert snapshot is not None
        assert snapshot.name == "get_test"
        assert snapshot.description == "Test getting"

    def test_get_nonexistent_snapshot(self, temp_storage):
        """Test getting a snapshot that doesn't exist."""
        snapshot = temp_storage.get_snapshot("nonexistent")

        assert snapshot is None

    def test_list_snapshots(self, temp_storage):
        """Test listing all snapshots."""
        temp_storage.create_snapshot(name="snap_a")
        temp_storage.create_snapshot(name="snap_b")
        temp_storage.create_snapshot(name="snap_c")

        names = temp_storage.list_snapshots()

        assert len(names) == 3
        assert "snap_a" in names
        assert "snap_b" in names
        assert "snap_c" in names

    def test_list_snapshots_empty(self, temp_storage):
        """Test listing snapshots when none exist."""
        names = temp_storage.list_snapshots()

        assert names == []

    def test_load_all_snapshots(self, temp_storage):
        """Test loading all snapshot objects."""
        temp_storage.create_snapshot(name="load_a")
        temp_storage.create_snapshot(name="load_b")

        snapshots = temp_storage.load_all_snapshots()

        assert len(snapshots) == 2
        assert all(isinstance(s, CorpusSnapshot) for s in snapshots)

    def test_delete_snapshot(self, temp_storage):
        """Test deleting a snapshot."""
        temp_storage.create_snapshot(name="to_delete")
        assert temp_storage.get_snapshot("to_delete") is not None

        result = temp_storage.delete_snapshot("to_delete")

        assert result is True
        assert temp_storage.get_snapshot("to_delete") is None

    def test_delete_nonexistent_snapshot(self, temp_storage):
        """Test deleting a snapshot that doesn't exist."""
        result = temp_storage.delete_snapshot("nonexistent")

        assert result is False

    def test_restore_snapshot(self, temp_storage):
        """Test restoring a snapshot."""
        # Add rules and create snapshot
        rule1 = RuleMetadata(
            rule_id="original_rule",
            rule_text="original",
            natural_language="original",
            created_by="test",
        )
        temp_storage.save_rule(rule1)
        temp_storage.create_snapshot(name="restore_test")

        # Add more rules after snapshot
        rule2 = RuleMetadata(
            rule_id="new_rule",
            rule_text="new",
            natural_language="new",
            created_by="test",
        )
        temp_storage.save_rule(rule2)
        assert len(temp_storage.list_rules()) == 2

        # Restore snapshot
        snapshot = temp_storage.restore_snapshot("restore_test")

        assert snapshot.rule_count == 1
        assert len(temp_storage.list_rules()) == 1
        assert temp_storage.load_rule("original_rule") is not None
        assert temp_storage.load_rule("new_rule") is None

    def test_restore_snapshot_nonexistent_fails(self, temp_storage):
        """Test that restoring nonexistent snapshot fails."""
        with pytest.raises(ValueError, match="does not exist"):
            temp_storage.restore_snapshot("nonexistent")

    def test_restore_snapshot_no_clear(self, temp_storage):
        """Test restoring snapshot without clearing existing rules."""
        # Add rule and create snapshot
        rule1 = RuleMetadata(
            rule_id="snap_rule",
            rule_text="snapshot rule",
            natural_language="snapshot",
            created_by="test",
        )
        temp_storage.save_rule(rule1)
        temp_storage.create_snapshot(name="merge_test")

        # Delete the rule
        temp_storage.delete_rule("snap_rule")

        # Add a different rule
        rule2 = RuleMetadata(
            rule_id="current_rule",
            rule_text="current",
            natural_language="current",
            created_by="test",
        )
        temp_storage.save_rule(rule2)

        # Restore without clearing
        temp_storage.restore_snapshot("merge_test", clear_existing=False)

        # Both rules should exist now
        assert temp_storage.load_rule("snap_rule") is not None
        assert temp_storage.load_rule("current_rule") is not None

    def test_compare_snapshots(self, temp_storage):
        """Test comparing two snapshots."""
        # Create first snapshot with 2 rules
        rule1 = RuleMetadata(
            rule_id="common_rule",
            rule_text="common",
            natural_language="common",
            created_by="test",
        )
        rule2 = RuleMetadata(
            rule_id="removed_rule",
            rule_text="removed",
            natural_language="removed",
            created_by="test",
        )
        temp_storage.save_rule(rule1)
        temp_storage.save_rule(rule2)
        temp_storage.create_snapshot(name="snap_1")

        # Modify and create second snapshot
        temp_storage.delete_rule("removed_rule")
        rule3 = RuleMetadata(
            rule_id="added_rule",
            rule_text="added",
            natural_language="added",
            created_by="test",
        )
        temp_storage.save_rule(rule3)
        temp_storage.create_snapshot(name="snap_2")

        # Compare
        comparison = temp_storage.compare_snapshots("snap_1", "snap_2")

        assert comparison["snapshot1"]["rule_count"] == 2
        assert comparison["snapshot2"]["rule_count"] == 2
        assert "removed_rule" in comparison["removed_rules"]
        assert "added_rule" in comparison["added_rules"]
        assert comparison["common_rules"] == 1

    def test_compare_snapshots_nonexistent_fails(self, temp_storage):
        """Test that comparing with nonexistent snapshot fails."""
        temp_storage.create_snapshot(name="exists")

        with pytest.raises(ValueError, match="does not exist"):
            temp_storage.compare_snapshots("exists", "nonexistent")

        with pytest.raises(ValueError, match="does not exist"):
            temp_storage.compare_snapshots("nonexistent", "exists")

    def test_storage_stats_includes_snapshots(self, temp_storage):
        """Test that storage stats includes snapshot count."""
        temp_storage.create_snapshot(name="stats_snap_1")
        temp_storage.create_snapshot(name="stats_snap_2")

        stats = temp_storage.get_storage_stats()

        assert stats["total_snapshots"] == 2

    def test_snapshot_preserves_ab_tests(self, temp_storage):
        """Test that snapshots preserve A/B tests."""
        # Add an A/B test
        ab_test = ABTestResult(
            test_id="preserved_test",
            started_at=datetime.now(),
            variant_a_id="a",
            variant_b_id="b",
            variant_a_accuracy=0.7,
            variant_b_accuracy=0.8,
            cases_evaluated=100,
        )
        temp_storage.save_ab_test(ab_test)
        temp_storage.create_snapshot(name="ab_test_snapshot")

        # Delete A/B test
        ab_test_path = temp_storage._ab_test_path("preserved_test")
        ab_test_path.unlink()
        assert temp_storage.load_ab_test("preserved_test") is None

        # Restore and verify
        temp_storage.restore_snapshot("ab_test_snapshot")
        restored_test = temp_storage.load_ab_test("preserved_test")

        assert restored_test is not None
        assert restored_test.test_id == "preserved_test"

    def test_snapshot_preserves_genealogy(self, temp_storage):
        """Test that snapshots preserve genealogy."""
        genealogy = {"root": ["child_1", "child_2"]}
        temp_storage.save_genealogy(genealogy)
        temp_storage.create_snapshot(name="genealogy_snapshot")

        # Clear genealogy
        temp_storage.save_genealogy({})
        assert temp_storage.load_genealogy() == {}

        # Restore and verify
        temp_storage.restore_snapshot("genealogy_snapshot")

        assert temp_storage.load_genealogy() == genealogy
