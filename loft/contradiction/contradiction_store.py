"""
Persistent storage for contradiction management (Phase 4.4).

Stores contradictions, interpretations, and resolution history in JSON format.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict
from loguru import logger

from loft.contradiction.contradiction_schemas import (
    ContradictionReport,
    RuleInterpretation,
    ContradictionSeverity,
    ContradictionType,
)


class ContradictionStore:
    """
    File-based storage for contradiction management.

    Storage structure:
        contradiction_data/
            contradictions/
                <contradiction_id>.json
            interpretations/
                <interpretation_id>.json
            resolutions/
                <contradiction_id>_resolution.json
            indexes/
                by_severity.json
                by_type.json
                by_rule.json
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize contradiction store.

        Args:
            base_path: Root directory for contradiction data.
                      Defaults to contradiction_data/
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent / "contradiction_data"

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.base_path / "contradictions").mkdir(exist_ok=True)
        (self.base_path / "interpretations").mkdir(exist_ok=True)
        (self.base_path / "resolutions").mkdir(exist_ok=True)
        (self.base_path / "indexes").mkdir(exist_ok=True)

        logger.info(f"Initialized ContradictionStore at {self.base_path}")

    def save_contradiction(self, contradiction: ContradictionReport) -> None:
        """
        Save a contradiction report.

        Args:
            contradiction: The contradiction to save
        """
        file_path = self.base_path / "contradictions" / f"{contradiction.contradiction_id}.json"

        with open(file_path, "w") as f:
            json.dump(contradiction.to_dict(), f, indent=2)

        # Update indexes
        self._update_indexes(contradiction)

        logger.debug(f"Saved contradiction {contradiction.contradiction_id} to {file_path}")

    def get_contradiction(self, contradiction_id: str) -> Optional[ContradictionReport]:
        """
        Get a specific contradiction report.

        Args:
            contradiction_id: The contradiction ID

        Returns:
            ContradictionReport if found, None otherwise
        """
        file_path = self.base_path / "contradictions" / f"{contradiction_id}.json"

        if not file_path.exists():
            logger.warning(f"Contradiction {contradiction_id} not found")
            return None

        with open(file_path, "r") as f:
            data = json.load(f)

        return ContradictionReport.from_dict(data)

    def get_all_contradictions(self) -> List[ContradictionReport]:
        """
        Get all contradiction reports.

        Returns:
            List of all contradictions
        """
        contradictions_path = self.base_path / "contradictions"

        if not contradictions_path.exists():
            return []

        contradictions = []
        for file_path in contradictions_path.glob("*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
            contradictions.append(ContradictionReport.from_dict(data))

        # Sort by detected_at
        contradictions.sort(key=lambda c: c.detected_at, reverse=True)

        logger.debug(f"Loaded {len(contradictions)} contradictions")
        return contradictions

    def query_by_severity(self, severity: ContradictionSeverity) -> List[ContradictionReport]:
        """
        Query contradictions by severity.

        Args:
            severity: The severity level to filter by

        Returns:
            List of matching contradictions
        """
        all_contradictions = self.get_all_contradictions()
        return [c for c in all_contradictions if c.severity == severity]

    def query_by_type(self, contradiction_type: ContradictionType) -> List[ContradictionReport]:
        """
        Query contradictions by type.

        Args:
            contradiction_type: The type to filter by

        Returns:
            List of matching contradictions
        """
        all_contradictions = self.get_all_contradictions()
        return [c for c in all_contradictions if c.contradiction_type == contradiction_type]

    def query_by_rule(self, rule_id: str) -> List[ContradictionReport]:
        """
        Query contradictions involving a specific rule.

        Args:
            rule_id: The rule ID

        Returns:
            List of contradictions involving this rule
        """
        all_contradictions = self.get_all_contradictions()
        return [c for c in all_contradictions if c.rule_a_id == rule_id or c.rule_b_id == rule_id]

    def query_unresolved(self) -> List[ContradictionReport]:
        """
        Get all unresolved contradictions.

        Returns:
            List of unresolved contradictions
        """
        all_contradictions = self.get_all_contradictions()
        return [c for c in all_contradictions if not c.resolved]

    def query_critical(self) -> List[ContradictionReport]:
        """
        Get all critical contradictions.

        Returns:
            List of critical contradictions
        """
        return self.query_by_severity(ContradictionSeverity.CRITICAL)

    def save_interpretation(self, interpretation: RuleInterpretation) -> None:
        """
        Save a rule interpretation.

        Args:
            interpretation: The interpretation to save
        """
        file_path = self.base_path / "interpretations" / f"{interpretation.interpretation_id}.json"

        with open(file_path, "w") as f:
            json.dump(interpretation.to_dict(), f, indent=2)

        logger.debug(f"Saved interpretation {interpretation.interpretation_id} to {file_path}")

    def get_interpretation(self, interpretation_id: str) -> Optional[RuleInterpretation]:
        """
        Get a specific interpretation.

        Args:
            interpretation_id: The interpretation ID

        Returns:
            RuleInterpretation if found, None otherwise
        """
        file_path = self.base_path / "interpretations" / f"{interpretation_id}.json"

        if not file_path.exists():
            logger.warning(f"Interpretation {interpretation_id} not found")
            return None

        with open(file_path, "r") as f:
            data = json.load(f)

        return RuleInterpretation.from_dict(data)

    def get_interpretations_by_principle(self, principle: str) -> List[RuleInterpretation]:
        """
        Get all interpretations for a specific principle.

        Args:
            principle: The principle name

        Returns:
            List of interpretations for this principle
        """
        interpretations_path = self.base_path / "interpretations"

        if not interpretations_path.exists():
            return []

        interpretations = []
        for file_path in interpretations_path.glob("*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
            interpretation = RuleInterpretation.from_dict(data)
            if interpretation.principle == principle:
                interpretations.append(interpretation)

        logger.debug(f"Found {len(interpretations)} interpretations for principle '{principle}'")
        return interpretations

    def delete_contradiction(self, contradiction_id: str) -> bool:
        """
        Delete a contradiction.

        Args:
            contradiction_id: The contradiction ID

        Returns:
            True if deleted, False if not found
        """
        file_path = self.base_path / "contradictions" / f"{contradiction_id}.json"

        if not file_path.exists():
            return False

        file_path.unlink()
        logger.info(f"Deleted contradiction {contradiction_id}")
        return True

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about stored contradictions.

        Returns:
            Dictionary with statistics
        """
        all_contradictions = self.get_all_contradictions()

        stats = {
            "total": len(all_contradictions),
            "unresolved": len([c for c in all_contradictions if not c.resolved]),
            "resolved": len([c for c in all_contradictions if c.resolved]),
            "by_severity": {},
            "by_type": {},
            "critical_unresolved": len(
                [c for c in all_contradictions if not c.resolved and c.is_critical()]
            ),
        }

        # Count by severity
        for severity in ContradictionSeverity:
            count = len([c for c in all_contradictions if c.severity == severity])
            stats["by_severity"][severity.value] = count

        # Count by type
        for ctype in ContradictionType:
            count = len([c for c in all_contradictions if c.contradiction_type == ctype])
            stats["by_type"][ctype.value] = count

        return stats

    def _update_indexes(self, contradiction: ContradictionReport) -> None:
        """Update index files for faster queries."""
        # For now, indexes are built on-demand during queries
        # Future: maintain persistent indexes for large datasets
        pass
