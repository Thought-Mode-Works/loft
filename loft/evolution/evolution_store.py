"""
Persistent storage for rule evolution history (Phase 4.3).

Stores evolution data in JSON format compatible with existing demo structure.
Future: Can be migrated to RDF/Triple store for LinkedASP querying.
"""

import json
from pathlib import Path
from typing import List, Optional
from loguru import logger

from loft.evolution.evolution_schemas import RuleVersion, RuleLineage


class RuleEvolutionStore:
    """
    File-based storage for rule evolution history.

    Storage structure:
        evolution_data/
            <rule_family_id>/
                lineage.json          # Complete lineage
                versions/
                    <version_id>.json # Individual versions
                metadata.json         # Family metadata
    """

    def __init__(self, base_path: Path = None):
        """
        Initialize evolution store.

        Args:
            base_path: Root directory for evolution data.
                      Defaults to loft/evolution_data/
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent / "evolution_data"

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RuleEvolutionStore at {self.base_path}")

    def save_version(self, version: RuleVersion) -> None:
        """
        Save a rule version.

        Args:
            version: The rule version to save
        """
        family_path = self._get_family_path(version.rule_family_id)
        family_path.mkdir(parents=True, exist_ok=True)

        versions_path = family_path / "versions"
        versions_path.mkdir(exist_ok=True)

        # Save individual version
        version_file = versions_path / f"{version.rule_id}.json"
        with open(version_file, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

        logger.debug(f"Saved version {version.rule_id} to {version_file}")

    def get_version(self, rule_family_id: str, version_id: str) -> Optional[RuleVersion]:
        """
        Get a specific rule version.

        Args:
            rule_family_id: The rule family ID
            version_id: The specific version ID

        Returns:
            RuleVersion if found, None otherwise
        """
        version_file = self._get_family_path(rule_family_id) / "versions" / f"{version_id}.json"

        if not version_file.exists():
            logger.warning(f"Version {version_id} not found for family {rule_family_id}")
            return None

        with open(version_file, "r") as f:
            data = json.load(f)

        return RuleVersion.from_dict(data)

    def get_all_versions(self, rule_family_id: str) -> List[RuleVersion]:
        """
        Get all versions for a rule family.

        Args:
            rule_family_id: The rule family ID

        Returns:
            List of all versions
        """
        versions_path = self._get_family_path(rule_family_id) / "versions"

        if not versions_path.exists():
            logger.warning(f"No versions found for family {rule_family_id}")
            return []

        versions = []
        for version_file in versions_path.glob("*.json"):
            with open(version_file, "r") as f:
                data = json.load(f)
            versions.append(RuleVersion.from_dict(data))

        # Sort by creation time
        versions.sort(key=lambda v: v.created_at)

        logger.debug(f"Loaded {len(versions)} versions for family {rule_family_id}")
        return versions

    def save_lineage(self, lineage: RuleLineage) -> None:
        """
        Save complete lineage.

        Args:
            lineage: The rule lineage to save
        """
        family_path = self._get_family_path(lineage.rule_family_id)
        family_path.mkdir(parents=True, exist_ok=True)

        # Save lineage summary
        lineage_file = family_path / "lineage.json"
        with open(lineage_file, "w") as f:
            json.dump(lineage.to_dict(), f, indent=2)

        # Save all versions
        for version in lineage.all_versions:
            self.save_version(version)

        logger.info(
            f"Saved lineage for {lineage.rule_family_id} ({len(lineage.all_versions)} versions)"
        )

    def get_lineage(self, rule_family_id: str) -> Optional[RuleLineage]:
        """
        Get complete lineage for a rule family.

        Args:
            rule_family_id: The rule family ID

        Returns:
            RuleLineage if found, None otherwise
        """
        lineage_file = self._get_family_path(rule_family_id) / "lineage.json"

        if not lineage_file.exists():
            logger.warning(f"Lineage not found for family {rule_family_id}")
            return None

        with open(lineage_file, "r") as f:
            data = json.load(f)

        return RuleLineage.from_dict(data)

    def get_all_families(self) -> List[str]:
        """
        Get list of all rule family IDs.

        Returns:
            List of family IDs
        """
        families = []
        for path in self.base_path.iterdir():
            if path.is_dir():
                families.append(path.name)

        return sorted(families)

    def query_by_method(self, evolution_method: str) -> List[RuleVersion]:
        """
        Query versions by evolution method.

        Args:
            evolution_method: The method to filter by

        Returns:
            List of matching versions
        """
        matching = []

        for family_id in self.get_all_families():
            versions = self.get_all_versions(family_id)
            for version in versions:
                if (
                    version.evolution_context
                    and version.evolution_context.evolution_method.value == evolution_method
                ):
                    matching.append(version)

        logger.debug(f"Found {len(matching)} versions with method={evolution_method}")
        return matching

    def query_by_stratification(self, stratification_level: str) -> List[RuleVersion]:
        """
        Query versions by stratification level.

        Args:
            stratification_level: The level to filter by

        Returns:
            List of matching versions
        """
        matching = []

        for family_id in self.get_all_families():
            versions = self.get_all_versions(family_id)
            for version in versions:
                if version.stratification_level.value == stratification_level:
                    matching.append(version)

        logger.debug(f"Found {len(matching)} versions at level={stratification_level}")
        return matching

    def query_active_rules(self) -> List[RuleVersion]:
        """
        Get all currently active rules.

        Returns:
            List of active rule versions
        """
        active = []

        for family_id in self.get_all_families():
            lineage = self.get_lineage(family_id)
            if lineage and lineage.current_version.is_active:
                active.append(lineage.current_version)

        logger.debug(f"Found {len(active)} active rules")
        return active

    def export_asp_file(
        self, output_path: Path, stratification_level: Optional[str] = None
    ) -> None:
        """
        Export active rules to ASP file (demo format).

        Args:
            output_path: Where to write the .lp file
            stratification_level: Filter by level (optional)
        """
        active_rules = self.query_active_rules()

        if stratification_level:
            active_rules = [
                r for r in active_rules if r.stratification_level.value == stratification_level
            ]

        lines = []
        for rule in active_rules:
            # Add metadata comment (demo format)
            timestamp = rule.incorporated_at or rule.created_at
            confidence = rule.performance.confidence if rule.performance else 0.0

            comment = f"% Added: {timestamp.isoformat()}, Version: {rule.version}, Confidence: {confidence}"
            lines.append(comment)
            lines.append(rule.asp_rule)
            lines.append("")  # Blank line

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Exported {len(active_rules)} rules to {output_path}")

    def delete_family(self, rule_family_id: str) -> bool:
        """
        Delete a rule family and all its versions.

        Args:
            rule_family_id: The family to delete

        Returns:
            True if deleted, False if not found
        """
        family_path = self._get_family_path(rule_family_id)

        if not family_path.exists():
            return False

        import shutil

        shutil.rmtree(family_path)
        logger.info(f"Deleted family {rule_family_id}")
        return True

    def _get_family_path(self, rule_family_id: str) -> Path:
        """Get path for a rule family."""
        return self.base_path / rule_family_id
