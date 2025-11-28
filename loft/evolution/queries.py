"""
Query interface for rule evolution database.

Provides high-level queries for analyzing rule evolution patterns.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from loft.evolution.evolution_store import RuleEvolutionStore
from loft.evolution.evolution_schemas import (
    RuleLineage,
    EvolutionMethod,
    StratificationLevel,
)


class RuleEvolutionDB:
    """
    Query interface for rule evolution data.

    Provides semantic queries over rule evolution history.
    """

    def __init__(self, store: RuleEvolutionStore):
        """Initialize with evolution store."""
        self.store = store

    def get_active_rules(self) -> List[RuleLineage]:
        """Get all rules with active versions."""
        all_lineages = self.store.list_all()
        return [
            lineage
            for lineage in all_lineages
            if any(v.status == "active" for v in lineage.versions)
        ]

    def get_deprecated_rules(self) -> List[RuleLineage]:
        """Get all deprecated rules."""
        all_lineages = self.store.list_all()
        return [
            lineage
            for lineage in all_lineages
            if all(v.status == "deprecated" for v in lineage.versions)
        ]

    def get_rules_by_method(self, method: EvolutionMethod) -> List[RuleLineage]:
        """Get all rules created via specific method."""
        all_lineages = self.store.list_all()
        return [
            lineage
            for lineage in all_lineages
            if any(v.evolution_method == method for v in lineage.versions)
        ]

    def get_rules_by_layer(self, layer: StratificationLevel) -> List[RuleLineage]:
        """Get all rules in specific stratification layer."""
        all_lineages = self.store.list_all()
        return [lineage for lineage in all_lineages if lineage.current_layer == layer]

    def get_recent_rules(self, days: int = 7) -> List[RuleLineage]:
        """Get rules created in last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        all_lineages = self.store.list_all()

        return [
            lineage
            for lineage in all_lineages
            if any(v.created_at >= cutoff for v in lineage.versions)
        ]

    def get_high_performing_rules(self, min_accuracy: float = 0.8) -> List[RuleLineage]:
        """Get rules with accuracy above threshold."""
        all_lineages = self.store.list_all()
        high_performers = []

        for lineage in all_lineages:
            for version in lineage.versions:
                if version.performance_history:
                    latest_perf = version.performance_history[-1]
                    if latest_perf.accuracy >= min_accuracy:
                        high_performers.append(lineage)
                        break

        return high_performers

    def get_rule_history(
        self,
        rule_id: str,
        include_deprecated: bool = True,
    ) -> Optional[RuleLineage]:
        """
        Get complete history for a rule.

        Args:
            rule_id: Rule identifier
            include_deprecated: Whether to include deprecated versions

        Returns:
            Complete lineage or None if not found
        """
        lineage = self.store.load_lineage(rule_id)

        if not lineage:
            return None

        if not include_deprecated:
            # Filter out deprecated versions
            lineage.versions = [v for v in lineage.versions if v.status != "deprecated"]

        return lineage

    def search_by_source_gap(self, gap_id: str) -> List[RuleLineage]:
        """Find all rules generated from a specific gap."""
        all_lineages = self.store.list_all()
        return [lineage for lineage in all_lineages if lineage.source_gap_id == gap_id]

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics about rule evolution."""
        all_lineages = self.store.list_all()

        total_rules = len(all_lineages)
        total_versions = sum(len(lineage.versions) for lineage in all_lineages)

        # Count by method
        method_counts: Dict[str, int] = {}
        for lineage in all_lineages:
            for version in lineage.versions:
                method = version.evolution_method.value
                method_counts[method] = method_counts.get(method, 0) + 1

        # Count by layer
        layer_counts: Dict[str, int] = {}
        for lineage in all_lineages:
            layer = lineage.current_layer.value
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        # Count by status
        active_count = len(self.get_active_rules())
        deprecated_count = len(self.get_deprecated_rules())

        # Average versions per rule
        avg_versions = total_versions / total_rules if total_rules > 0 else 0

        return {
            "total_rules": total_rules,
            "total_versions": total_versions,
            "avg_versions_per_rule": avg_versions,
            "active_rules": active_count,
            "deprecated_rules": deprecated_count,
            "by_evolution_method": method_counts,
            "by_stratification_layer": layer_counts,
        }

    def find_superseded_rules(self) -> List[RuleLineage]:
        """Find rules that have been superseded by newer versions."""
        all_lineages = self.store.list_all()
        superseded = []

        for lineage in all_lineages:
            has_superseded = any(v.status == "superseded" for v in lineage.versions)
            if has_superseded:
                superseded.append(lineage)

        return superseded

    def get_learning_trajectory(self, rule_id: str) -> List[Dict[str, Any]]:
        """
        Get learning trajectory for a rule (accuracy over time).

        Args:
            rule_id: Rule identifier

        Returns:
            List of performance snapshots over time
        """
        lineage = self.store.load_lineage(rule_id)
        if not lineage:
            return []

        trajectory = []

        for version in lineage.versions:
            for snapshot in version.performance_history:
                trajectory.append(
                    {
                        "version": version.version_number,
                        "timestamp": snapshot.timestamp,
                        "accuracy": snapshot.accuracy,
                        "confidence": snapshot.confidence,
                        "tests_passed": snapshot.test_cases_passed,
                        "tests_total": snapshot.test_cases_total,
                    }
                )

        # Sort by timestamp
        trajectory.sort(key=lambda x: x["timestamp"])

        return trajectory

    def compare_evolution_methods(self) -> Dict[str, Dict[str, float]]:
        """
        Compare effectiveness of different evolution methods.

        Returns:
            Dict mapping method to performance metrics
        """
        all_lineages = self.store.list_all()

        method_performance: Dict[str, List[float]] = {}

        for lineage in all_lineages:
            for version in lineage.versions:
                method = version.evolution_method.value

                if version.performance_history:
                    latest_perf = version.performance_history[-1]
                    if method not in method_performance:
                        method_performance[method] = []
                    method_performance[method].append(latest_perf.accuracy)

        # Calculate averages
        results = {}
        for method, accuracies in method_performance.items():
            results[method] = {
                "avg_accuracy": sum(accuracies) / len(accuracies),
                "count": len(accuracies),
                "max_accuracy": max(accuracies),
                "min_accuracy": min(accuracies),
            }

        return results
