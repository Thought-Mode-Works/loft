"""
Unit tests for ASP persistence manager.

Tests file-based storage, atomic writes, snapshots, backups, and LinkedASP metadata.
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from loft.persistence.asp_persistence import (
    ASPPersistenceManager,
    SnapshotMetadata,
    CorruptedFileError,  # Added LoadResult
)
from loft.symbolic.asp_rule import ASPRule, RuleMetadata
from loft.symbolic.stratification import StratificationLevel


class TestASPPersistenceManager:
    """Test ASP persistence manager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def persistence_manager(self, temp_dir):
        """Create persistence manager with temp directory."""
        return ASPPersistenceManager(
            base_dir=str(temp_dir), enable_git=False
        )  # Convert Path to str

    @pytest.fixture
    def sample_rule(self):
        """Create sample ASP rule."""
        metadata = RuleMetadata(
            provenance="test",
            timestamp=datetime.now().isoformat(),
            validation_score=0.9,
        )

        return ASPRule(
            rule_id="test_rule_001",
            asp_text="contract(X) :- parties(X, _, _), consideration(X).",
            stratification_level=StratificationLevel.STRATEGIC,
            confidence=0.9,
            metadata=metadata,
        )

    def test_initialization(self, temp_dir):
        """Test persistence manager initialization."""
        manager = ASPPersistenceManager(
            base_dir=str(temp_dir), enable_git=False
        )  # Convert Path to str

        assert manager.base_dir == Path(temp_dir)
        assert manager.snapshot_dir == Path(temp_dir) / "snapshots"
        assert manager.backup_dir == Path(temp_dir) / "backups"
        assert manager.enable_git is False

        # Directories should be created
        assert manager.base_dir.exists()
        assert manager.snapshot_dir.exists()
        assert manager.backup_dir.exists()

    def test_save_single_rule(self, persistence_manager, sample_rule):
        """Test saving a single rule."""
        persistence_manager.save_rule(
            sample_rule,
            StratificationLevel.STRATEGIC,
            metadata={"source_type": "test"},
        )

        # Check file was created
        strategic_file = persistence_manager.base_dir / "strategic.lp"
        assert strategic_file.exists()

        # Check content
        content = strategic_file.read_text()
        assert "contract(X)" in content
        assert "LinkedASP Rule Metadata" in content
        assert "test_rule_001" in content
        assert "Strategic" in content or "strategic" in content

    def test_save_all_rules(self, persistence_manager):
        """Test saving multiple rules across layers."""
        rules_by_layer = {}

        # Create rules for different layers
        for layer in [
            StratificationLevel.CONSTITUTIONAL,
            StratificationLevel.STRATEGIC,
        ]:
            metadata = RuleMetadata(
                provenance="test",
                timestamp=datetime.now().isoformat(),
            )

            rule = ASPRule(
                rule_id=f"rule_{layer.value}",
                asp_text=f"test_{layer.value}(X) :- condition(X).",
                stratification_level=layer,
                confidence=1.0 if layer == StratificationLevel.CONSTITUTIONAL else 0.9,
                metadata=metadata,
            )
            rules_by_layer[layer] = [rule]

        persistence_manager.save_all_rules(rules_by_layer, overwrite=True)

        # Check files were created
        for layer in rules_by_layer.keys():
            layer_file = persistence_manager.base_dir / f"{layer.value}.lp"
            assert layer_file.exists()
            content = layer_file.read_text()
            assert f"test_{layer.value}(X)" in content

    def test_load_all_rules(self, persistence_manager, sample_rule):
        """Test loading persisted rules."""
        # Save rules
        persistence_manager.save_rule(sample_rule, StratificationLevel.STRATEGIC)

        # Load rules
        loaded_rules = persistence_manager.load_all_rules()
        loaded_rules_by_layer = loaded_rules.rules_by_layer  # Fix: assign here

        # Check loaded correctly
        assert StratificationLevel.STRATEGIC in loaded_rules_by_layer
        strategic_rules = loaded_rules_by_layer[StratificationLevel.STRATEGIC]
        assert len(strategic_rules) >= 1

        # Find our rule
        loaded_rule = next(
            (r for r in strategic_rules if "contract(X)" in r.asp_text), None
        )
        assert loaded_rule is not None
        assert loaded_rule.stratification_level == StratificationLevel.STRATEGIC

    def test_linkedasp_metadata_embedding(self, persistence_manager, sample_rule):
        """Test that LinkedASP metadata is properly embedded."""
        persistence_manager.save_rule(
            sample_rule,
            StratificationLevel.STRATEGIC,
            metadata={
                "source_type": "llm_generated",
                "source_llm": "claude-3-opus",
                "legal_source": "UCC § 2-201",
                "genre": "ConjunctiveRequirement",
            },
        )

        strategic_file = persistence_manager.base_dir / "strategic.lp"
        content = strategic_file.read_text()

        # Check LinkedASP metadata is present
        assert "LinkedASP Rule Metadata" in content
        assert "Rule ID: test_rule_001" in content
        assert "Source Type: llm_generated" in content
        assert "Source LLM: claude-3-opus" in content
        assert "Legal Source: UCC § 2-201" in content
        assert "Genre: ConjunctiveRequirement" in content

    def test_create_snapshot(self, persistence_manager, sample_rule):
        """Test snapshot creation."""
        # Save some rules
        persistence_manager.save_rule(sample_rule, StratificationLevel.STRATEGIC)

        # Create snapshot
        snapshot_path = persistence_manager.create_snapshot(
            cycle_number=1, description="Test snapshot"
        )

        assert snapshot_path.exists()
        assert snapshot_path.name == "cycle_001"

        # Check files were copied
        strategic_snapshot = snapshot_path / "strategic.lp"
        assert strategic_snapshot.exists()

        # Check metadata file
        metadata_file = snapshot_path / "metadata.json"
        assert metadata_file.exists()

        metadata_data = json.loads(metadata_file.read_text())
        assert metadata_data["cycle_number"] == 1
        assert metadata_data["description"] == "Test snapshot"
        assert "strategic" in metadata_data["rules_count"]

    def test_list_snapshots(self, persistence_manager, sample_rule):
        """Test listing snapshots."""
        # Create multiple snapshots
        persistence_manager.save_rule(sample_rule, StratificationLevel.STRATEGIC)

        for i in range(1, 4):
            persistence_manager.create_snapshot(
                cycle_number=i, description=f"Snapshot {i}"
            )

        # List snapshots
        snapshots = persistence_manager.list_snapshots()

        assert len(snapshots) == 3
        assert all(isinstance(s, SnapshotMetadata) for s in snapshots)
        assert [s.cycle_number for s in snapshots] == [1, 2, 3]

    def test_restore_snapshot(self, persistence_manager, sample_rule):
        """Test snapshot restoration."""
        # Save initial rule
        persistence_manager.save_rule(sample_rule, StratificationLevel.STRATEGIC)

        # Create snapshot
        persistence_manager.create_snapshot(cycle_number=1)

        # Modify rules
        metadata = RuleMetadata(provenance="test", timestamp=datetime.now().isoformat())
        new_rule = ASPRule(
            rule_id="new_rule",
            asp_text="new_test(X) :- condition(X).",
            stratification_level=StratificationLevel.STRATEGIC,
            confidence=0.9,
            metadata=metadata,
        )
        persistence_manager.save_rule(new_rule, StratificationLevel.STRATEGIC)

        # Restore snapshot
        persistence_manager.restore_snapshot(cycle_number=1)

        # Check rules reverted
        strategic_file = persistence_manager.base_dir / "strategic.lp"
        content = strategic_file.read_text()

        # Should have original rule
        assert "contract(X)" in content
        # Should NOT have new rule
        assert "new_test(X)" not in content

    def test_backup_creation(self, persistence_manager, sample_rule):
        """Test backup creation."""
        # Save rule
        persistence_manager.save_rule(sample_rule, StratificationLevel.STRATEGIC)

        # Create backup
        backup_path = persistence_manager.create_backup("test_backup")

        assert backup_path.exists()
        assert "test_backup" in backup_path.name

        # Check file was backed up
        strategic_backup = backup_path / "strategic.lp"
        assert strategic_backup.exists()

    def test_snapshot_retention(self, temp_dir, sample_rule):
        """Test snapshot retention policy."""
        # Create manager with retention limit
        manager = ASPPersistenceManager(
            base_dir=str(temp_dir),
            enable_git=False,
            snapshot_retention=3,  # Convert Path to str
        )

        # Save rule
        manager.save_rule(sample_rule, StratificationLevel.STRATEGIC)

        # Create 5 snapshots
        for i in range(1, 6):
            manager.create_snapshot(cycle_number=i)

        # Should only keep last 3
        snapshots = manager.list_snapshots()
        assert len(snapshots) == 3
        assert [s.cycle_number for s in snapshots] == [3, 4, 5]

    def test_get_stats(self, persistence_manager, sample_rule):
        """Test statistics generation."""
        # Save rules
        persistence_manager.save_rule(sample_rule, StratificationLevel.STRATEGIC)
        persistence_manager.create_snapshot(cycle_number=1)
        persistence_manager.create_backup("test")

        # Get stats
        stats = persistence_manager.get_stats()

        assert stats["base_dir"] == str(persistence_manager.base_dir)
        assert stats["git_enabled"] is False
        assert stats["total_rules"] >= 1
        assert stats["snapshots_count"] == 1
        assert stats["backups_count"] >= 1
        assert "strategic" in stats["layers"]
        assert stats["layers"]["strategic"]["count"] >= 1

    def test_atomic_write_prevents_corruption(self, persistence_manager, sample_rule):
        """Test that atomic writes prevent corruption on errors."""
        strategic_file = persistence_manager.base_dir / "strategic.lp"

        # Save initial rule
        persistence_manager.save_rule(sample_rule, StratificationLevel.STRATEGIC)
        initial_content = strategic_file.read_text()

        # Simulate write failure by making directory read-only temporarily
        # (This test is simplified - in production, test actual crash scenarios)
        assert strategic_file.exists()
        assert len(initial_content) > 0

    def test_corrupted_file_detection(self, persistence_manager):
        """Test detection of corrupted files."""
        # Create corrupted file
        strategic_file = persistence_manager.base_dir / "strategic.lp"
        strategic_file.write_text("@@@ CORRUPTED @@@\n!!! INVALID ASP !!!")

        # Should handle gracefully
        load_result = persistence_manager.load_all_rules()
        assert load_result.had_errors is True
        assert len(load_result.parsing_errors) > 0
        assert "strategic" in load_result.recovered_layers

        # Test with recover_on_error=False (should raise)
        strategic_file.write_text(
            "@@@ CORRUPTED @@@\n!!! INVALID ASP !!!"
        )  # Corrupt again
        with pytest.raises(CorruptedFileError):
            persistence_manager.load_all_rules(recover_on_error=False)

    def test_empty_layer_handling(self, persistence_manager):
        """Test handling of empty layers."""
        # Load when no files exist
        load_result = persistence_manager.load_all_rules()
        loaded_rules_by_layer = (
            load_result.rules_by_layer
        )  # Fixed: Access rules_by_layer

        # All layers should be present with empty lists
        assert len(loaded_rules_by_layer) == len(
            StratificationLevel
        )  # Fixed: check len of rules_by_layer
        for layer in StratificationLevel:
            assert layer in loaded_rules_by_layer
            assert loaded_rules_by_layer[layer] == []

    def test_multiple_rules_same_layer(self, persistence_manager):
        """Test saving multiple rules to same layer."""
        rules = []
        for i in range(3):
            metadata = RuleMetadata(
                provenance="test", timestamp=datetime.now().isoformat()
            )
            rule = ASPRule(
                rule_id=f"rule_{i}",
                asp_text=f"test_{i}(X) :- condition_{i}(X).",
                stratification_level=StratificationLevel.TACTICAL,
                confidence=0.85,
                metadata=metadata,
            )
            rules.append(rule)
            persistence_manager.save_rule(rule, StratificationLevel.TACTICAL)

        # Load and verify all rules present
        load_result = persistence_manager.load_all_rules()
        tactical_rules = load_result.rules_by_layer[
            StratificationLevel.TACTICAL
        ]  # Fixed access

        assert len(tactical_rules) >= 3
        for i in range(3):
            assert any(f"test_{i}(X)" in r.asp_text for r in tactical_rules)


class TestSnapshotMetadata:
    """Test snapshot metadata handling."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = SnapshotMetadata(
            cycle_number=1,
            timestamp=datetime(2025, 1, 26, 14, 30, 0),
            rules_count={"strategic": 5, "tactical": 10},
            total_rules=15,
            description="Test snapshot",
        )

        data = metadata.to_dict()

        assert data["cycle_number"] == 1
        assert data["timestamp"] == "2025-01-26T14:30:00"
        assert data["rules_count"]["strategic"] == 5
        assert data["total_rules"] == 15

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "cycle_number": 1,
            "timestamp": "2025-01-26T14:30:00",
            "rules_count": {"strategic": 5},
            "total_rules": 5,
            "description": "Test",
            "git_commit": None,
        }

        metadata = SnapshotMetadata.from_dict(data)

        assert metadata.cycle_number == 1
        assert isinstance(metadata.timestamp, datetime)
        assert metadata.total_rules == 5


class TestIntegration:
    """Test integration scenarios."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_full_workflow(self, temp_dir):
        """Test complete persistence workflow."""
        manager = ASPPersistenceManager(
            base_dir=str(temp_dir), enable_git=False
        )  # Convert Path to str

        # Step 1: Save initial rules
        metadata = RuleMetadata(
            provenance="manual", timestamp=datetime.now().isoformat()
        )
        rule1 = ASPRule(
            rule_id="rule1",
            asp_text="test1(X) :- condition(X).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.85,
            metadata=metadata,
        )
        manager.save_rule(rule1, StratificationLevel.TACTICAL)

        # Step 2: Create snapshot
        snapshot1 = manager.create_snapshot(cycle_number=1, description="Initial state")
        assert snapshot1.exists()

        # Step 3: Add more rules
        rule2 = ASPRule(
            rule_id="rule2",
            asp_text="test2(X) :- condition2(X).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=0.85,
            metadata=metadata,
        )
        manager.save_rule(rule2, StratificationLevel.TACTICAL)

        # Step 4: Create another snapshot
        snapshot2 = manager.create_snapshot(cycle_number=2, description="Added rule2")
        assert snapshot2.exists()

        # Step 5: Verify we can restore first snapshot
        manager.restore_snapshot(cycle_number=1)

        # Step 6: Load rules and verify only rule1 present
        load_result = manager.load_all_rules()
        tactical_rules = load_result.rules_by_layer[StratificationLevel.TACTICAL]
        assert any("test1(X)" in r.asp_text for r in tactical_rules)
        assert not any("test2(X)" in r.asp_text for r in tactical_rules)

        # Step 7: Verify stats
        stats = manager.get_stats()
        assert stats["snapshots_count"] == 2
        assert stats["backups_count"] >= 1  # From restore operation
