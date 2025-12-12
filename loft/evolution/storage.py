"""
Persistence layer for rule evolution data.

Provides file-based storage for rule metadata, version history,
A/B test results, genealogy graphs, and corpus snapshots.
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .tracking import (
    RuleMetadata,
    ABTestResult,
    StratificationLayer,
    RuleStatus,
)


@dataclass
class StorageConfig:
    """Configuration for rule evolution storage."""

    base_path: Path = field(default_factory=lambda: Path("data/rule_evolution"))
    metadata_dir: str = "metadata"
    versions_dir: str = "versions"
    ab_tests_dir: str = "ab_tests"
    snapshots_dir: str = "snapshots"
    genealogy_file: str = "genealogy.json"

    def __post_init__(self):
        """Ensure base_path is a Path object."""
        if isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)


@dataclass
class CorpusSnapshot:
    """
    A timestamped snapshot of the rule corpus.

    Enables rollback to previous corpus states and A/B testing
    between corpus versions.
    """

    snapshot_id: str
    name: str
    created_at: datetime
    description: str = ""
    rule_count: int = 0
    ab_test_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "snapshot_id": self.snapshot_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "rule_count": self.rule_count,
            "ab_test_count": self.ab_test_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorpusSnapshot":
        """Create from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data.get("description", ""),
            rule_count=data.get("rule_count", 0),
            ab_test_count=data.get("ab_test_count", 0),
            metadata=data.get("metadata", {}),
        )


class RuleEvolutionStorage:
    """
    File-based storage for rule evolution data.

    Storage structure:
        data/rule_evolution/
        ├── metadata/
        │   ├── rule_abc123.json
        │   └── ...
        ├── versions/
        │   ├── abc123_v1.0.asp
        │   └── ...
        ├── ab_tests/
        │   ├── ab_test123.json
        │   └── ...
        ├── snapshots/
        │   ├── snapshot_20241128/
        │   │   ├── manifest.json
        │   │   ├── metadata/
        │   │   ├── versions/
        │   │   └── genealogy.json
        │   └── ...
        └── genealogy.json

    Example:
        storage = RuleEvolutionStorage()
        storage.save_rule(metadata)
        loaded = storage.load_rule("rule_abc123")

        # Create and restore snapshots
        snapshot = storage.create_snapshot("before_experiment")
        # ... make changes ...
        storage.restore_snapshot("before_experiment")
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize storage with configuration.

        Args:
            config: Storage configuration (uses defaults if not provided)
        """
        self.config = config or StorageConfig()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        base = self.config.base_path
        (base / self.config.metadata_dir).mkdir(parents=True, exist_ok=True)
        (base / self.config.versions_dir).mkdir(parents=True, exist_ok=True)
        (base / self.config.ab_tests_dir).mkdir(parents=True, exist_ok=True)
        (base / self.config.snapshots_dir).mkdir(parents=True, exist_ok=True)

    def _metadata_path(self, rule_id: str) -> Path:
        """Get path for rule metadata file."""
        return self.config.base_path / self.config.metadata_dir / f"{rule_id}.json"

    def _version_path(self, rule_id: str, version: str) -> Path:
        """Get path for rule version file."""
        safe_version = version.replace(".", "_")
        return (
            self.config.base_path
            / self.config.versions_dir
            / f"{rule_id}_{safe_version}.asp"
        )

    def _ab_test_path(self, test_id: str) -> Path:
        """Get path for A/B test file."""
        return self.config.base_path / self.config.ab_tests_dir / f"{test_id}.json"

    def _genealogy_path(self) -> Path:
        """Get path for genealogy file."""
        return self.config.base_path / self.config.genealogy_file

    def save_rule(self, metadata: RuleMetadata) -> None:
        """
        Save rule metadata to storage.

        Args:
            metadata: Rule metadata to save
        """
        # Save metadata JSON
        meta_path = self._metadata_path(metadata.rule_id)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Save rule version
        version_path = self._version_path(metadata.rule_id, metadata.version)
        with open(version_path, "w", encoding="utf-8") as f:
            f.write(f"% Rule: {metadata.rule_id}\n")
            f.write(f"% Version: {metadata.version}\n")
            f.write(f"% Description: {metadata.natural_language}\n")
            f.write(f"% Created: {metadata.created_at.isoformat()}\n\n")
            f.write(metadata.rule_text)

        logger.debug(f"Saved rule {metadata.rule_id} v{metadata.version}")

    def load_rule(self, rule_id: str) -> Optional[RuleMetadata]:
        """
        Load rule metadata from storage.

        Args:
            rule_id: ID of the rule to load

        Returns:
            RuleMetadata if found, None otherwise
        """
        meta_path = self._metadata_path(rule_id)
        if not meta_path.exists():
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return RuleMetadata.from_dict(data)

    def delete_rule(self, rule_id: str) -> bool:
        """
        Delete rule from storage.

        Args:
            rule_id: ID of the rule to delete

        Returns:
            True if deleted, False if not found
        """
        meta_path = self._metadata_path(rule_id)
        if not meta_path.exists():
            return False

        # Load to get version for cleanup
        metadata = self.load_rule(rule_id)
        if metadata:
            version_path = self._version_path(rule_id, metadata.version)
            if version_path.exists():
                version_path.unlink()

        meta_path.unlink()
        logger.debug(f"Deleted rule {rule_id}")
        return True

    def list_rules(self) -> List[str]:
        """
        List all rule IDs in storage.

        Returns:
            List of rule IDs
        """
        meta_dir = self.config.base_path / self.config.metadata_dir
        return [p.stem for p in meta_dir.glob("*.json")]

    def load_all_rules(self) -> List[RuleMetadata]:
        """
        Load all rules from storage.

        Returns:
            List of all RuleMetadata objects
        """
        rules = []
        for rule_id in self.list_rules():
            metadata = self.load_rule(rule_id)
            if metadata:
                rules.append(metadata)
        return rules

    def save_ab_test(self, result: ABTestResult) -> None:
        """
        Save A/B test result to storage.

        Args:
            result: A/B test result to save
        """
        path = self._ab_test_path(result.test_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.debug(f"Saved A/B test {result.test_id}")

    def load_ab_test(self, test_id: str) -> Optional[ABTestResult]:
        """
        Load A/B test result from storage.

        Args:
            test_id: ID of the A/B test to load

        Returns:
            ABTestResult if found, None otherwise
        """
        path = self._ab_test_path(test_id)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return ABTestResult.from_dict(data)

    def list_ab_tests(self) -> List[str]:
        """
        List all A/B test IDs in storage.

        Returns:
            List of test IDs
        """
        ab_dir = self.config.base_path / self.config.ab_tests_dir
        return [p.stem for p in ab_dir.glob("*.json")]

    def load_all_ab_tests(self) -> List[ABTestResult]:
        """
        Load all A/B tests from storage.

        Returns:
            List of all ABTestResult objects
        """
        tests = []
        for test_id in self.list_ab_tests():
            result = self.load_ab_test(test_id)
            if result:
                tests.append(result)
        return tests

    def save_genealogy(self, genealogy: Dict[str, List[str]]) -> None:
        """
        Save rule genealogy graph to storage.

        Args:
            genealogy: Dictionary mapping root rule ID to list of descendant IDs
        """
        path = self._genealogy_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(genealogy, f, indent=2)
        logger.debug("Saved genealogy graph")

    def load_genealogy(self) -> Dict[str, List[str]]:
        """
        Load rule genealogy graph from storage.

        Returns:
            Dictionary mapping root rule ID to list of descendant IDs
        """
        path = self._genealogy_path()
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_rules_by_status(self, status: RuleStatus) -> List[RuleMetadata]:
        """
        Get all rules with a given status.

        Args:
            status: Status to filter by

        Returns:
            List of matching RuleMetadata objects
        """
        return [meta for meta in self.load_all_rules() if meta.status == status]

    def get_rules_by_layer(self, layer: StratificationLayer) -> List[RuleMetadata]:
        """
        Get all rules in a given stratification layer.

        Args:
            layer: Layer to filter by

        Returns:
            List of matching RuleMetadata objects
        """
        return [meta for meta in self.load_all_rules() if meta.current_layer == layer]

    def get_storage_stats(self) -> Dict:
        """
        Get statistics about stored data.

        Returns:
            Dictionary with storage statistics
        """
        rules = self.load_all_rules()
        tests = self.load_all_ab_tests()

        status_counts = {}
        layer_counts = {}

        for rule in rules:
            status_counts[rule.status.value] = (
                status_counts.get(rule.status.value, 0) + 1
            )
            layer_counts[rule.current_layer.value] = (
                layer_counts.get(rule.current_layer.value, 0) + 1
            )

        return {
            "total_rules": len(rules),
            "total_ab_tests": len(tests),
            "active_ab_tests": sum(1 for t in tests if t.completed_at is None),
            "status_distribution": status_counts,
            "layer_distribution": layer_counts,
            "total_snapshots": len(self.list_snapshots()),
        }

    # ========== Snapshot Methods ==========

    def _snapshot_path(self, snapshot_name: str) -> Path:
        """Get path for snapshot directory."""
        return self.config.base_path / self.config.snapshots_dir / snapshot_name

    def _snapshot_manifest_path(self, snapshot_name: str) -> Path:
        """Get path for snapshot manifest file."""
        return self._snapshot_path(snapshot_name) / "manifest.json"

    def create_snapshot(
        self,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CorpusSnapshot:
        """
        Create a timestamped snapshot of the current rule corpus.

        Copies all rule metadata, versions, A/B tests, and genealogy
        to a snapshot directory for later restoration.

        Args:
            name: Human-readable name for the snapshot
            description: Optional description of the snapshot purpose
            metadata: Optional additional metadata to store

        Returns:
            CorpusSnapshot object with snapshot details

        Raises:
            ValueError: If a snapshot with this name already exists
        """
        timestamp = datetime.now()
        snapshot_id = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        snapshot_dir = self._snapshot_path(name)

        if snapshot_dir.exists():
            raise ValueError(f"Snapshot '{name}' already exists")

        snapshot_dir.mkdir(parents=True)

        # Copy metadata directory
        src_metadata = self.config.base_path / self.config.metadata_dir
        dst_metadata = snapshot_dir / self.config.metadata_dir
        if src_metadata.exists():
            shutil.copytree(src_metadata, dst_metadata)

        # Copy versions directory
        src_versions = self.config.base_path / self.config.versions_dir
        dst_versions = snapshot_dir / self.config.versions_dir
        if src_versions.exists():
            shutil.copytree(src_versions, dst_versions)

        # Copy A/B tests directory
        src_ab_tests = self.config.base_path / self.config.ab_tests_dir
        dst_ab_tests = snapshot_dir / self.config.ab_tests_dir
        if src_ab_tests.exists():
            shutil.copytree(src_ab_tests, dst_ab_tests)

        # Copy genealogy file
        src_genealogy = self._genealogy_path()
        if src_genealogy.exists():
            shutil.copy2(src_genealogy, snapshot_dir / self.config.genealogy_file)

        # Create snapshot object
        rules = self.list_rules()
        ab_tests = self.list_ab_tests()

        snapshot = CorpusSnapshot(
            snapshot_id=snapshot_id,
            name=name,
            created_at=timestamp,
            description=description,
            rule_count=len(rules),
            ab_test_count=len(ab_tests),
            metadata=metadata or {},
        )

        # Save manifest
        manifest_path = self._snapshot_manifest_path(name)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

        logger.info(
            f"Created snapshot '{name}' with {len(rules)} rules and {len(ab_tests)} A/B tests"
        )
        return snapshot

    def restore_snapshot(
        self, name: str, clear_existing: bool = True
    ) -> CorpusSnapshot:
        """
        Restore the rule corpus from a snapshot.

        Args:
            name: Name of the snapshot to restore
            clear_existing: If True, clear existing rules before restoring

        Returns:
            CorpusSnapshot object with restored snapshot details

        Raises:
            ValueError: If snapshot does not exist
        """
        snapshot_dir = self._snapshot_path(name)
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot '{name}' does not exist")

        # Load manifest
        manifest_path = self._snapshot_manifest_path(name)
        with open(manifest_path, "r", encoding="utf-8") as f:
            snapshot = CorpusSnapshot.from_dict(json.load(f))

        if clear_existing:
            # Clear existing data
            for rule_id in self.list_rules():
                self.delete_rule(rule_id)

            # Clear A/B tests
            ab_tests_dir = self.config.base_path / self.config.ab_tests_dir
            for test_file in ab_tests_dir.glob("*.json"):
                test_file.unlink()

            # Clear genealogy
            genealogy_path = self._genealogy_path()
            if genealogy_path.exists():
                genealogy_path.unlink()

        # Restore metadata
        src_metadata = snapshot_dir / self.config.metadata_dir
        dst_metadata = self.config.base_path / self.config.metadata_dir
        if src_metadata.exists():
            for src_file in src_metadata.glob("*.json"):
                shutil.copy2(src_file, dst_metadata / src_file.name)

        # Restore versions
        src_versions = snapshot_dir / self.config.versions_dir
        dst_versions = self.config.base_path / self.config.versions_dir
        if src_versions.exists():
            for src_file in src_versions.glob("*.asp"):
                shutil.copy2(src_file, dst_versions / src_file.name)

        # Restore A/B tests
        src_ab_tests = snapshot_dir / self.config.ab_tests_dir
        dst_ab_tests = self.config.base_path / self.config.ab_tests_dir
        if src_ab_tests.exists():
            for src_file in src_ab_tests.glob("*.json"):
                shutil.copy2(src_file, dst_ab_tests / src_file.name)

        # Restore genealogy
        src_genealogy = snapshot_dir / self.config.genealogy_file
        if src_genealogy.exists():
            shutil.copy2(src_genealogy, self._genealogy_path())

        logger.info(
            f"Restored snapshot '{name}' with {snapshot.rule_count} rules "
            f"and {snapshot.ab_test_count} A/B tests"
        )
        return snapshot

    def get_snapshot(self, name: str) -> Optional[CorpusSnapshot]:
        """
        Get snapshot details without restoring.

        Args:
            name: Name of the snapshot

        Returns:
            CorpusSnapshot if found, None otherwise
        """
        manifest_path = self._snapshot_manifest_path(name)
        if not manifest_path.exists():
            return None

        with open(manifest_path, "r", encoding="utf-8") as f:
            return CorpusSnapshot.from_dict(json.load(f))

    def list_snapshots(self) -> List[str]:
        """
        List all available snapshot names.

        Returns:
            List of snapshot names
        """
        snapshots_dir = self.config.base_path / self.config.snapshots_dir
        if not snapshots_dir.exists():
            return []

        return sorted(
            [
                p.name
                for p in snapshots_dir.iterdir()
                if p.is_dir() and (p / "manifest.json").exists()
            ]
        )

    def load_all_snapshots(self) -> List[CorpusSnapshot]:
        """
        Load all snapshot manifests.

        Returns:
            List of CorpusSnapshot objects
        """
        snapshots = []
        for name in self.list_snapshots():
            snapshot = self.get_snapshot(name)
            if snapshot:
                snapshots.append(snapshot)
        return snapshots

    def delete_snapshot(self, name: str) -> bool:
        """
        Delete a snapshot.

        Args:
            name: Name of the snapshot to delete

        Returns:
            True if deleted, False if not found
        """
        snapshot_dir = self._snapshot_path(name)
        if not snapshot_dir.exists():
            return False

        shutil.rmtree(snapshot_dir)
        logger.info(f"Deleted snapshot '{name}'")
        return True

    def compare_snapshots(self, name1: str, name2: str) -> Dict[str, Any]:
        """
        Compare two snapshots.

        Args:
            name1: First snapshot name
            name2: Second snapshot name

        Returns:
            Dictionary with comparison results

        Raises:
            ValueError: If either snapshot does not exist
        """
        snapshot1 = self.get_snapshot(name1)
        snapshot2 = self.get_snapshot(name2)

        if not snapshot1:
            raise ValueError(f"Snapshot '{name1}' does not exist")
        if not snapshot2:
            raise ValueError(f"Snapshot '{name2}' does not exist")

        # Get rule IDs from each snapshot
        snapshot1_dir = self._snapshot_path(name1)
        snapshot2_dir = self._snapshot_path(name2)

        rules1 = (
            set(
                p.stem
                for p in (snapshot1_dir / self.config.metadata_dir).glob("*.json")
            )
            if (snapshot1_dir / self.config.metadata_dir).exists()
            else set()
        )

        rules2 = (
            set(
                p.stem
                for p in (snapshot2_dir / self.config.metadata_dir).glob("*.json")
            )
            if (snapshot2_dir / self.config.metadata_dir).exists()
            else set()
        )

        added_rules = rules2 - rules1
        removed_rules = rules1 - rules2
        common_rules = rules1 & rules2

        return {
            "snapshot1": {
                "name": name1,
                "created_at": snapshot1.created_at.isoformat(),
                "rule_count": snapshot1.rule_count,
                "ab_test_count": snapshot1.ab_test_count,
            },
            "snapshot2": {
                "name": name2,
                "created_at": snapshot2.created_at.isoformat(),
                "rule_count": snapshot2.rule_count,
                "ab_test_count": snapshot2.ab_test_count,
            },
            "added_rules": list(added_rules),
            "removed_rules": list(removed_rules),
            "common_rules": len(common_rules),
            "rule_count_delta": snapshot2.rule_count - snapshot1.rule_count,
            "ab_test_count_delta": snapshot2.ab_test_count - snapshot1.ab_test_count,
        }
