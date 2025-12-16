"""
Living Document Manager for iterative rule building.

Manages LIVING_DOCUMENT.md that tracks all rule adjustments,
coverage progression, and reasoning over time.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class RuleAdjustment:
    """Record of a rule being added, modified, or removed."""

    timestamp: str  # ISO format
    action: str  # "added", "modified", "removed", "rejected"
    rule_id: str
    rule_text: str
    layer: str
    coverage_change: float  # Change in predicate coverage
    reason: str  # Why this adjustment was made
    source: str  # Where rule came from (e.g., "llm", "gap_fill", "validation")
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CycleSummary:
    """Summary of a learning cycle."""

    cycle_number: int
    timestamp: str
    cases_processed: int
    rules_added: int
    rules_rejected: int
    coverage_start: float
    coverage_end: float
    highlights: List[str]


class LivingDocumentManager:
    """
    Manages the LIVING_DOCUMENT.md for rule evolution tracking.

    The living document provides a human-readable history of:
    - All rule adjustments over time
    - Coverage progression
    - Reasoning for decisions
    - Learning cycle summaries
    """

    def __init__(self, document_path: Path = Path("LIVING_DOCUMENT.md")):
        """
        Initialize living document manager.

        Args:
            document_path: Path to living document file
        """
        self.path = document_path

        # Create document if it doesn't exist
        if not self.path.exists():
            self._initialize_document()

    def _initialize_document(self) -> None:
        """Initialize document with header and structure."""
        header = f"""# LOFT Living Document

**Generated**: {datetime.utcnow().isoformat()}

This document tracks the evolution of the ASP rule base over time.
Each rule adjustment is documented with reasoning, coverage impact,
and metadata for full transparency and auditability.

---

## Rule Adjustments

"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(header)

    def append_adjustment(self, adjustment: RuleAdjustment) -> None:
        """
        Append a rule adjustment entry to the document.

        Args:
            adjustment: RuleAdjustment record to append
        """
        entry = f"""
### {adjustment.timestamp} - Rule {adjustment.action.title()}: `{adjustment.rule_id}`

**Layer**: {adjustment.layer}
**Action**: {adjustment.action}
**Coverage Change**: {adjustment.coverage_change:+.2%}
**Source**: {adjustment.source}
**Reason**: {adjustment.reason}

```asp
{adjustment.rule_text}
```

"""

        if adjustment.metadata:
            entry += "**Metadata**:\n"
            for key, value in adjustment.metadata.items():
                entry += f"- {key}: {value}\n"
            entry += "\n"

        entry += "---\n"

        # Append to file
        with open(self.path, "a") as f:
            f.write(entry)

    def append_cycle_summary(self, summary: CycleSummary) -> None:
        """
        Append a learning cycle summary.

        Args:
            summary: CycleSummary to append
        """
        entry = f"""
## Cycle {summary.cycle_number} Summary

**Timestamp**: {summary.timestamp}
**Cases Processed**: {summary.cases_processed}
**Rules Added**: {summary.rules_added}
**Rules Rejected**: {summary.rules_rejected}
**Coverage**: {summary.coverage_start:.1%} → {summary.coverage_end:.1%} ({summary.coverage_end - summary.coverage_start:+.1%})

"""

        if summary.highlights:
            entry += "**Highlights**:\n"
            for highlight in summary.highlights:
                entry += f"- {highlight}\n"
            entry += "\n"

        entry += "---\n"

        with open(self.path, "a") as f:
            f.write(entry)

    def append_section(self, title: str, content: str) -> None:
        """
        Append a custom section to the document.

        Args:
            title: Section title
            content: Section content (markdown)
        """
        entry = f"\n## {title}\n\n{content}\n\n---\n"

        with open(self.path, "a") as f:
            f.write(entry)

    def append_coverage_analysis(
        self,
        timestamp: str,
        coverage_metrics: Dict[str, Any],
        trend: str,
        monotonic: bool,
    ) -> None:
        """
        Append coverage analysis section.

        Args:
            timestamp: When analysis was performed
            coverage_metrics: Current coverage metrics
            trend: Coverage trend description
            monotonic: Whether coverage is monotonic
        """
        content = f"""
### Coverage Analysis - {timestamp}

**Predicate Coverage**: {coverage_metrics.get('predicate_coverage', 0.0):.1%}
**Case Coverage**: {coverage_metrics.get('case_coverage', 0.0):.1%}
**Scenario Coverage**: {coverage_metrics.get('scenario_coverage', 0.0):.1%}

**Trend**: {trend}
**Monotonicity**: {'✓ Maintained' if monotonic else '✗ Violated'}

**Total Rules**: {coverage_metrics.get('total_rules', 0)}

"""

        rules_by_layer = coverage_metrics.get("rules_by_layer", {})
        if rules_by_layer:
            content += "**Rules by Layer**:\n"
            for layer, count in sorted(rules_by_layer.items()):
                content += f"- {layer}: {count}\n"
            content += "\n"

        content += "---\n"

        with open(self.path, "a") as f:
            f.write(content)

    def get_recent_adjustments(self, count: int = 10) -> List[str]:
        """
        Get recent adjustment entries.

        Args:
            count: Number of recent entries to retrieve

        Returns:
            List of recent adjustment entry strings
        """
        if not self.path.exists():
            return []

        content = self.path.read_text()

        # Split by adjustment markers
        sections = content.split("### ")
        # Filter to adjustment sections (have "Rule" in them)
        adjustments = ["### " + s for s in sections[1:] if "Rule" in s.split("\n")[0]]

        return adjustments[-count:]

    def count_total_adjustments(self) -> int:
        """
        Count total number of adjustments in document.

        Returns:
            Total adjustment count
        """
        if not self.path.exists():
            return 0

        content = self.path.read_text()
        # Count "### " markers that are adjustments
        return content.count("### ") - content.count("### Coverage Analysis")

    def backup(self, backup_path: Optional[Path] = None) -> Path:
        """
        Create a backup of the living document.

        Args:
            backup_path: Optional path for backup, defaults to timestamped name

        Returns:
            Path to backup file
        """
        if backup_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = self.path.parent / f"LIVING_DOCUMENT_{timestamp}.md"

        # Copy file
        import shutil

        shutil.copy2(self.path, backup_path)

        return backup_path

    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics from document.

        Returns:
            Dictionary with summary statistics
        """
        if not self.path.exists():
            return {
                "total_adjustments": 0,
                "adjustments_by_action": {},
                "adjustments_by_layer": {},
            }

        content = self.path.read_text()

        # Count adjustments by action type
        actions = ["added", "modified", "removed", "rejected"]
        adjustments_by_action = {
            action: content.count(f"**Action**: {action}") for action in actions
        }

        # Count adjustments by layer
        layers = ["constitutional", "strategic", "tactical", "operational"]
        adjustments_by_layer = {
            layer: content.count(f"**Layer**: {layer}") for layer in layers
        }

        # Count cycle summaries
        cycle_count = content.count("## Cycle ")

        return {
            "total_adjustments": self.count_total_adjustments(),
            "adjustments_by_action": adjustments_by_action,
            "adjustments_by_layer": adjustments_by_layer,
            "cycle_count": cycle_count,
            "document_size_bytes": self.path.stat().st_size,
        }
