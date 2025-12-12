"""
Rule evolution tracking system (Phase 4.3).

Integrates with debate framework to automatically track rule evolution.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from loft.evolution.evolution_schemas import (
    RuleVersion,
    RuleLineage,
    EvolutionContext,
    EvolutionMethod,
    PerformanceSnapshot,
    StratificationLevel,
)
from loft.evolution.evolution_store import RuleEvolutionStore


class RuleEvolutionTracker:
    """
    Tracks rule evolution across iterations.

    Integrates with:
    - Phase 4.2: Debate Framework (tracks dialectical cycles)
    - Phase 3.4: A/B Testing (performance metrics)
    - Phase 3.5: Performance Monitoring (metrics over time)
    """

    def __init__(self, store: Optional[RuleEvolutionStore] = None):
        """
        Initialize evolution tracker.

        Args:
            store: Evolution store for persistence (creates default if None)
        """
        self.store = store or RuleEvolutionStore()
        logger.info("Initialized RuleEvolutionTracker")

    def track_new_version(
        self,
        asp_rule: str,
        evolution_context: EvolutionContext,
        performance: Optional[PerformanceSnapshot] = None,
        parent_id: Optional[str] = None,
        rule_family_id: Optional[str] = None,
        predicates_used: Optional[List[str]] = None,
        stratification_level: StratificationLevel = StratificationLevel.TACTICAL,
    ) -> RuleVersion:
        """
        Track a new rule version.

        Args:
            asp_rule: The ASP rule text
            evolution_context: How this version was created
            performance: Performance metrics (optional)
            parent_id: Parent version ID (None for root)
            rule_family_id: Family ID (generated if None)
            predicates_used: Predicates in this rule
            stratification_level: ASP layer

        Returns:
            The created RuleVersion
        """
        # Generate IDs
        rule_id = self._generate_rule_id()
        if rule_family_id is None:
            rule_family_id = rule_id  # First version uses same ID

        # Determine version number
        version = self._calculate_version(rule_family_id, parent_id)

        # Calculate improvement if parent exists
        improvement = 0.0
        if parent_id and performance:
            parent = self.store.get_version(rule_family_id, parent_id)
            if parent and parent.performance:
                improvement = performance.confidence - parent.performance.confidence

        # Create version
        new_version = RuleVersion(
            rule_id=rule_id,
            rule_family_id=rule_family_id,
            version=version,
            asp_rule=asp_rule,
            predicates_used=predicates_used or [],
            parent_version=parent_id,
            evolution_context=evolution_context,
            performance=performance,
            improvement_over_parent=improvement,
            stratification_level=stratification_level,
            created_at=datetime.now(),
        )

        # Save version
        self.store.save_version(new_version)

        # Update parent's children list
        if parent_id:
            parent = self.store.get_version(rule_family_id, parent_id)
            if parent:
                parent.children_versions.append(rule_id)
                self.store.save_version(parent)

        # Update or create lineage
        self._update_lineage(new_version)

        logger.info(
            f"Tracked new version {version} for family {rule_family_id} "
            f"(method: {evolution_context.evolution_method.value})"
        )

        return new_version

    def track_from_debate(
        self,
        asp_rule: str,
        debate_cycle_id: str,
        round_number: int,
        thesis_rule: str,
        critique_summary: str,
        confidence: float,
        parent_id: Optional[str] = None,
        rule_family_id: Optional[str] = None,
        predicates_used: Optional[List[str]] = None,
        source_llm: str = "anthropic/claude-3-5-haiku",
    ) -> RuleVersion:
        """
        Track a version created from dialectical debate.

        Args:
            asp_rule: The synthesized rule
            debate_cycle_id: ID of the debate cycle
            round_number: Which round produced this
            thesis_rule: The thesis being refined
            critique_summary: Summary of critique
            confidence: Confidence score
            parent_id: Parent version (if refinement)
            rule_family_id: Family ID
            predicates_used: Predicates in rule
            source_llm: LLM that generated this

        Returns:
            The created RuleVersion
        """
        evolution_context = EvolutionContext(
            evolution_method=EvolutionMethod.DIALECTICAL,
            dialectical_cycle_id=debate_cycle_id,
            debate_round=round_number,
            thesis_rule=thesis_rule,
            critique_summary=critique_summary,
            source_llm=source_llm,
            reasoning=f"Synthesized from dialectical debate round {round_number}",
        )

        performance = PerformanceSnapshot(
            timestamp=datetime.now(),
            accuracy=0.0,  # To be updated after validation
            confidence=confidence,
            test_cases_passed=0,
            test_cases_total=0,
            success_rate=0.0,
        )

        return self.track_new_version(
            asp_rule=asp_rule,
            evolution_context=evolution_context,
            performance=performance,
            parent_id=parent_id,
            rule_family_id=rule_family_id,
            predicates_used=predicates_used,
        )

    def update_performance(
        self, rule_family_id: str, version_id: str, performance: PerformanceSnapshot
    ) -> None:
        """
        Update performance metrics for a version.

        Args:
            rule_family_id: The rule family
            version_id: The specific version
            performance: Updated performance metrics
        """
        version = self.store.get_version(rule_family_id, version_id)
        if not version:
            logger.warning(f"Version {version_id} not found for performance update")
            return

        old_confidence = version.performance.confidence if version.performance else 0.0
        version.performance = performance

        # Recalculate improvement
        if version.parent_version:
            parent = self.store.get_version(rule_family_id, version.parent_version)
            if parent and parent.performance:
                version.improvement_over_parent = (
                    performance.confidence - parent.performance.confidence
                )

        self.store.save_version(version)

        logger.info(
            f"Updated performance for {version_id}: "
            f"confidence {old_confidence:.2f} → {performance.confidence:.2f}"
        )

    def deprecate_version(
        self,
        rule_family_id: str,
        version_id: str,
        replaced_by: str,
        reason: str,
    ) -> None:
        """
        Mark a version as deprecated.

        Args:
            rule_family_id: The rule family
            version_id: Version to deprecate
            replaced_by: ID of replacement version
            reason: Why it was deprecated
        """
        version = self.store.get_version(rule_family_id, version_id)
        if not version:
            logger.warning(f"Version {version_id} not found for deprecation")
            return

        version.is_deprecated = True
        version.is_active = False
        version.replaced_by = replaced_by
        version.deprecation_reason = reason

        self.store.save_version(version)

        logger.info(f"Deprecated version {version_id}: {reason}")

    def get_lineage(self, rule_family_id: str) -> Optional[RuleLineage]:
        """
        Get complete evolution lineage.

        Args:
            rule_family_id: The rule family

        Returns:
            RuleLineage if found
        """
        return self.store.get_lineage(rule_family_id)

    def get_evolution_path(
        self, rule_family_id: str, to_version_id: Optional[str] = None
    ) -> List[RuleVersion]:
        """
        Get evolution path from root to version.

        Args:
            rule_family_id: The rule family
            to_version_id: Target version (current if None)

        Returns:
            List of versions in evolution order
        """
        lineage = self.get_lineage(rule_family_id)
        if not lineage:
            return []

        return lineage.get_evolution_path(to_version_id)

    def analyze_evolution_patterns(self) -> Dict[str, any]:
        """
        Analyze evolution patterns across all rules.

        Returns:
            Dictionary with analysis results
        """
        families = self.store.get_all_families()

        # Collect statistics
        method_counts = {}
        method_improvements = {}
        total_versions = 0
        total_families = len(families)

        for family_id in families:
            lineage = self.get_lineage(family_id)
            if not lineage:
                continue

            total_versions += len(lineage.all_versions)

            # Track by method
            for version in lineage.all_versions:
                if not version.evolution_context:
                    continue

                method = version.evolution_context.evolution_method.value
                method_counts[method] = method_counts.get(method, 0) + 1

                if version.improvement_over_parent != 0:
                    if method not in method_improvements:
                        method_improvements[method] = []
                    method_improvements[method].append(version.improvement_over_parent)

        # Calculate averages
        method_avg_improvement = {}
        for method, improvements in method_improvements.items():
            method_avg_improvement[method] = sum(improvements) / len(improvements)

        analysis = {
            "total_families": total_families,
            "total_versions": total_versions,
            "avg_versions_per_family": (
                total_versions / total_families if total_families > 0 else 0
            ),
            "methods_used": method_counts,
            "avg_improvement_by_method": method_avg_improvement,
            "most_effective_method": (
                max(method_avg_improvement.items(), key=lambda x: x[1])[0]
                if method_avg_improvement
                else None
            ),
        }

        logger.info(f"Evolution analysis: {analysis}")
        return analysis

    def get_top_performers(self, n: int = 10) -> List[RuleVersion]:
        """
        Get top N performing active rules.

        Args:
            n: Number of top rules to return

        Returns:
            List of top performing versions
        """
        active_rules = self.store.query_active_rules()

        # Filter rules with performance data
        with_performance = [r for r in active_rules if r.performance]

        # Sort by confidence
        sorted_rules = sorted(
            with_performance,
            key=lambda r: r.performance.confidence,
            reverse=True,
        )

        return sorted_rules[:n]

    def get_most_iterated(self, n: int = 10) -> List[tuple]:
        """
        Get rule families with most iterations.

        Args:
            n: Number of families to return

        Returns:
            List of (family_id, iteration_count) tuples
        """
        families = self.store.get_all_families()
        iteration_counts = []

        for family_id in families:
            lineage = self.get_lineage(family_id)
            if lineage:
                iteration_counts.append((family_id, lineage.total_iterations))

        # Sort by iteration count
        iteration_counts.sort(key=lambda x: x[1], reverse=True)

        return iteration_counts[:n]

    def rollback_to_version(self, rule_family_id: str, target_version_id: str) -> RuleVersion:
        """
        Rollback to a previous version.

        Args:
            rule_family_id: The rule family
            target_version_id: Version to rollback to

        Returns:
            The activated version

        Raises:
            ValueError: If version not found
        """
        target = self.store.get_version(rule_family_id, target_version_id)
        if not target:
            raise ValueError(f"Version {target_version_id} not found in family {rule_family_id}")

        # Deactivate current version
        lineage = self.get_lineage(rule_family_id)
        if lineage:
            current = lineage.current_version
            current.is_active = False
            self.store.save_version(current)

        # Activate target version
        target.is_active = True
        self.store.save_version(target)

        # Update lineage
        if lineage:
            lineage.current_version = target
            self.store.save_lineage(lineage)

        logger.info(f"Rolled back {rule_family_id} to version {target.version}")
        return target

    def _generate_rule_id(self) -> str:
        """Generate unique rule ID."""
        return str(uuid.uuid4())

    def _calculate_version(self, rule_family_id: str, parent_id: Optional[str]) -> str:
        """
        Calculate semantic version number.

        Args:
            rule_family_id: The rule family
            parent_id: Parent version ID

        Returns:
            Version string (e.g., "1.2.3")
        """
        if not parent_id:
            # Root version
            return "1.0.0"

        # Get parent version
        parent = self.store.get_version(rule_family_id, parent_id)
        if not parent:
            return "1.0.0"

        # Parse parent version
        parts = parent.version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        # Increment patch version
        patch += 1

        return f"{major}.{minor}.{patch}"

    def _update_lineage(self, new_version: RuleVersion) -> None:
        """Update lineage with new version."""
        lineage = self.store.get_lineage(new_version.rule_family_id)

        if lineage:
            # Update existing lineage
            lineage.all_versions.append(new_version)
            lineage.current_version = new_version
            lineage.total_iterations += 1
            lineage.last_updated = datetime.now()

            # Recalculate overall improvement
            if lineage.root_version.performance and new_version.performance:
                lineage.overall_improvement = (
                    new_version.performance.confidence - lineage.root_version.performance.confidence
                )
        else:
            # Create new lineage
            lineage = RuleLineage(
                rule_family_id=new_version.rule_family_id,
                root_version=new_version,
                all_versions=[new_version],
                current_version=new_version,
                total_iterations=1,
                overall_improvement=0.0,
                created_at=datetime.now(),
                last_updated=datetime.now(),
            )

        self.store.save_lineage(lineage)
