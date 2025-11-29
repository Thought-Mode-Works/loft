"""
Persistence layer for rule evolution data.

Provides file-based storage for rule metadata, version history,
A/B test results, and genealogy graphs.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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

    base_path: Path = Path("data/rule_evolution")
    metadata_dir: str = "metadata"
    versions_dir: str = "versions"
    ab_tests_dir: str = "ab_tests"
    genealogy_file: str = "genealogy.json"

    def __post_init__(self):
        """Ensure base_path is a Path object."""
        if isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)


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
        └── genealogy.json

    Example:
        storage = RuleEvolutionStorage()
        storage.save_rule(metadata)
        loaded = storage.load_rule("rule_abc123")
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

    def _metadata_path(self, rule_id: str) -> Path:
        """Get path for rule metadata file."""
        return self.config.base_path / self.config.metadata_dir / f"{rule_id}.json"

    def _version_path(self, rule_id: str, version: str) -> Path:
        """Get path for rule version file."""
        safe_version = version.replace(".", "_")
        return self.config.base_path / self.config.versions_dir / f"{rule_id}_{safe_version}.asp"

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
            status_counts[rule.status.value] = status_counts.get(rule.status.value, 0) + 1
            layer_counts[rule.current_layer.value] = (
                layer_counts.get(rule.current_layer.value, 0) + 1
            )

        return {
            "total_rules": len(rules),
            "total_ab_tests": len(tests),
            "active_ab_tests": sum(1 for t in tests if t.completed_at is None),
            "status_distribution": status_counts,
            "layer_distribution": layer_counts,
        }
