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
