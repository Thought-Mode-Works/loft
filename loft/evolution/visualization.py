"""
Text-based visualization for rule evolution genealogy.

Generates ASCII art trees showing rule lineage and evolution paths.
"""

from typing import List, Optional

from loft.evolution.evolution_schemas import RuleLineage, RuleVersion


class EvolutionVisualizer:
    """Generate text-based visualizations of rule evolution."""

    def __init__(self, lineage: RuleLineage):
        """Initialize visualizer with rule lineage."""
        self.lineage = lineage

    def generate_tree(self, max_depth: Optional[int] = None) -> str:
        """
        Generate ASCII art tree of rule evolution.

        Args:
            max_depth: Maximum depth to display (None = all)

        Returns:
            ASCII art string representation
        """
        lines = []
        lines.append("Rule Evolution Tree")
        lines.append("=" * 80)
        lines.append(f"Rule ID: {self.lineage.rule_id}")
        lines.append(f"Total Versions: {len(self.lineage.versions)}")
        lines.append("")

        # Build tree recursively
        if self.lineage.versions:
            root = self.lineage.versions[0]
            lines.extend(self._build_tree_lines(root, 0, max_depth))

        return "\n".join(lines)

    def _build_tree_lines(
        self,
        version: RuleVersion,
        depth: int,
        max_depth: Optional[int],
    ) -> List[str]:
        """Build tree lines recursively."""
        lines = []

        if max_depth is not None and depth >= max_depth:
            return lines

        # Indent based on depth
        indent = "  " * depth
        connector = "└─ " if depth > 0 else ""

        # Version info
        status_icon = "✓" if version.status == "active" else "✗"
        lines.append(
            f"{indent}{connector}{status_icon} v{version.version_number} "
            f"({version.created_at.strftime('%Y-%m-%d')})"
        )

        # Show method and confidence
        method_indent = "  " * (depth + 1)
        lines.append(
            f"{method_indent}Method: {version.evolution_method.value}, "
            f"Confidence: {version.confidence:.2f}"
        )

        # Show performance if available
        if version.performance_history:
            latest_perf = version.performance_history[-1]
            lines.append(f"{method_indent}Performance: {latest_perf.accuracy:.1%} accuracy")

        # Show children (future versions derived from this one)
        children = self._find_children(version)
        for child in children:
            lines.extend(self._build_tree_lines(child, depth + 1, max_depth))

        return lines

    def _find_children(self, version: RuleVersion) -> List[RuleVersion]:
        """Find versions that are derived from this version."""
        children = []
        for v in self.lineage.versions:
            if v.parent_version_id == version.version_id:
                children.append(v)
        return sorted(children, key=lambda x: x.version_number)

    def generate_timeline(self) -> str:
        """
        Generate timeline view of rule evolution.

        Returns:
            Timeline string representation
        """
        lines = []
        lines.append("Rule Evolution Timeline")
        lines.append("=" * 80)
        lines.append(f"Rule ID: {self.lineage.rule_id}")
        lines.append("")

        # Sort versions by creation date
        sorted_versions = sorted(self.lineage.versions, key=lambda v: v.created_at)

        for version in sorted_versions:
            lines.append(
                f"{version.created_at.strftime('%Y-%m-%d %H:%M:%S')} - "
                f"v{version.version_number} ({version.status})"
            )
            lines.append(f"  Method: {version.evolution_method.value}")
            lines.append(f"  Created by: {version.created_by}")

            if version.change_description:
                lines.append(f"  Changes: {version.change_description}")

            if version.performance_history:
                latest = version.performance_history[-1]
                lines.append(
                    f"  Performance: {latest.accuracy:.1%} accuracy, "
                    f"{latest.test_cases_passed}/{latest.test_cases_total} tests"
                )

            lines.append("")

        return "\n".join(lines)

    def generate_performance_graph(self, width: int = 60) -> str:
        """
        Generate ASCII art performance graph over time.

        Args:
            width: Width of graph in characters

        Returns:
            ASCII graph string
        """
        lines = []
        lines.append("Performance Over Time")
        lines.append("=" * 80)

        # Collect all performance snapshots chronologically
        all_snapshots = []
        for version in self.lineage.versions:
            for snapshot in version.performance_history:
                all_snapshots.append((version.version_number, snapshot))

        if not all_snapshots:
            lines.append("No performance data available")
            return "\n".join(lines)

        # Sort by timestamp
        all_snapshots.sort(key=lambda x: x[1].timestamp)

        # Create simple ASCII graph
        lines.append("Accuracy (0% to 100%)")
        lines.append("")

        for version_num, snapshot in all_snapshots:
            # Create bar
            bar_length = int(snapshot.accuracy * width)
            bar = "█" * bar_length + "░" * (width - bar_length)

            lines.append(
                f"v{version_num} {snapshot.timestamp.strftime('%m/%d')} | "
                f"{bar} {snapshot.accuracy:.1%}"
            )

        return "\n".join(lines)


def print_rule_genealogy(lineage: RuleLineage) -> None:
    """
    Print complete genealogy information for a rule.

    Args:
        lineage: Rule lineage to display
    """
    viz = EvolutionVisualizer(lineage)

    print(viz.generate_tree())
    print("\n")
    print(viz.generate_timeline())
    print("\n")
    print(viz.generate_performance_graph())


def compare_versions(version1: RuleVersion, version2: RuleVersion) -> str:
    """
    Generate comparison between two rule versions.

    Args:
        version1: First version
        version2: Second version

    Returns:
        Comparison string
    """
    lines = []
    lines.append("Version Comparison")
    lines.append("=" * 80)
    lines.append("")

    lines.append(f"Version {version1.version_number} vs Version {version2.version_number}")
    lines.append("")

    lines.append("ASP Code:")
    lines.append("-" * 80)
    lines.append(f"v{version1.version_number}:")
    lines.append(version1.asp_code)
    lines.append("")
    lines.append(f"v{version2.version_number}:")
    lines.append(version2.asp_code)
    lines.append("")

    # Performance comparison
    if version1.performance_history and version2.performance_history:
        perf1 = version1.performance_history[-1]
        perf2 = version2.performance_history[-1]

        lines.append("Performance:")
        lines.append("-" * 80)
        lines.append(
            f"v{version1.version_number}: {perf1.accuracy:.1%} accuracy, "
            f"{perf1.test_cases_passed}/{perf1.test_cases_total} tests"
        )
        lines.append(
            f"v{version2.version_number}: {perf2.accuracy:.1%} accuracy, "
            f"{perf2.test_cases_passed}/{perf2.test_cases_total} tests"
        )

        improvement = perf2.accuracy - perf1.accuracy
        lines.append(f"Change: {improvement:+.1%}")
        lines.append("")

    # Method comparison
    lines.append("Evolution Method:")
    lines.append("-" * 80)
    lines.append(f"v{version1.version_number}: {version1.evolution_method.value}")
    lines.append(f"v{version2.version_number}: {version2.evolution_method.value}")

    return "\n".join(lines)
