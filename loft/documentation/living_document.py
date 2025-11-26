"""
Living Document Generator for ASP Core Evolution.

Automatically generates and maintains human-readable documentation of the
evolving ASP core, showing rules, their history, and system performance over time.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any
from loguru import logger

from loft.symbolic.stratification import StratificationLevel
from loft.core.integration_schemas import ImprovementCycleResult


class LivingDocumentGenerator:
    """
    Generates living documentation for the self-modifying system.

    Features:
    - Automatic generation after each improvement cycle
    - Markdown format for easy viewing and version control
    - Organized by stratification layers
    - Includes rule statistics and metadata
    - Shows incorporation history with success/failure tracking
    - Displays improvement cycle metrics
    """

    def __init__(self, system: Optional[Any] = None):
        """
        Initialize living document generator.

        Args:
            system: SelfModifyingSystem instance
        """
        self.system = system
        self.cycle_history: List[ImprovementCycleResult] = []

    def generate(
        self,
        output_path: str = "./LIVING_DOCUMENT.md",
        include_metadata: bool = True,
    ) -> str:
        """
        Generate comprehensive living document.

        Args:
            output_path: Path to save the generated document
            include_metadata: Whether to include generation metadata

        Returns:
            Generated document content as string
        """
        logger.info(f"Generating living document: {output_path}")

        # Build document sections
        sections = []

        # Header
        sections.append(self._generate_header())

        # Table of contents
        sections.append(self._generate_table_of_contents())

        # Overview section
        sections.append(self._generate_overview())

        # Rules by layer
        sections.append(self._generate_rules_by_layer())

        # Incorporation history
        if self.system and hasattr(self.system, "incorporation_engine"):
            sections.append(self._generate_incorporation_history())

        # Improvement cycles
        if self.cycle_history:
            sections.append(self._generate_improvement_cycles())

        # Self-analysis
        if self.system and hasattr(self.system, "analyze_self"):
            sections.append(self._generate_self_analysis())

        # Rule evolution timeline
        sections.append(self._generate_evolution_timeline())

        # Footer with metadata
        if include_metadata:
            sections.append(self._generate_footer())

        # Combine all sections
        document = "\n\n".join(sections)

        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            f.write(document)

        logger.info(f"Living document generated: {output_path}")

        return document

    def update_cycle_history(self, cycle_result: ImprovementCycleResult) -> None:
        """
        Update cycle history with new cycle result.

        Args:
            cycle_result: Result from improvement cycle
        """
        self.cycle_history.append(cycle_result)

        # Keep only last 20 cycles for document generation
        if len(self.cycle_history) > 20:
            self.cycle_history = self.cycle_history[-20:]

    def _generate_header(self) -> str:
        """Generate document header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""# ASP Core Living Document

*Generated: {timestamp}*

---

This document provides a comprehensive view of the evolving ASP (Answer Set Programming)
core of the self-modifying system. It automatically updates after each improvement cycle
to show the current state of rules, their history, and system performance."""

    def _generate_table_of_contents(self) -> str:
        """Generate table of contents."""
        return """## Table of Contents

- [Overview](#overview)
- [Rules by Stratification Layer](#rules-by-stratification-layer)
  - [Constitutional Layer](#constitutional-layer)
  - [Strategic Layer](#strategic-layer)
  - [Tactical Layer](#tactical-layer)
  - [Operational Layer](#operational-layer)
- [Incorporation History](#incorporation-history)
- [Improvement Cycles](#improvement-cycles)
- [Self-Analysis](#self-analysis)
- [Rule Evolution Timeline](#rule-evolution-timeline)"""

    def _generate_overview(self) -> str:
        """Generate overview section."""
        lines = ["## Overview\n"]

        if self.system and hasattr(self.system, "asp_core"):
            asp_core = self.system.asp_core

            # Count rules by layer
            total_rules = 0
            rules_by_layer = {}

            for layer in StratificationLevel:
                rules = asp_core.get_rules_by_layer(layer)
                count = len(rules)
                rules_by_layer[layer.value] = count
                total_rules += count

            lines.append(f"**Total Rules**: {total_rules}\n")
            lines.append("**Rules by Layer**:")

            for layer, count in rules_by_layer.items():
                lines.append(f"- {layer.title()}: {count} rules")

            lines.append("")

            # Cycle count
            if self.cycle_history:
                lines.append(f"**Improvement Cycles Completed**: {len(self.cycle_history)}\n")

            # LLM status
            if hasattr(self.system, "rule_generator") and self.system.rule_generator:
                lines.append("**LLM Integration**: Enabled\n")
            else:
                lines.append("**LLM Integration**: Disabled\n")

            # System health
            if hasattr(self.system, "performance_monitor") and self.system.performance_monitor:
                try:
                    health = self.system.performance_monitor.get_system_health()
                    lines.append(f"**System Health**: {health.overall_health.title()}\n")
                    lines.append(f"**Success Rate**: {health.success_rate:.1%}\n")
                except Exception:
                    pass
        else:
            lines.append("*System not initialized*\n")

        return "\n".join(lines)

    def _generate_rules_by_layer(self) -> str:
        """Generate rules section organized by stratification layer."""
        lines = ["## Rules by Stratification Layer\n"]

        if not self.system or not hasattr(self.system, "asp_core"):
            lines.append("*No ASP core available*\n")
            return "\n".join(lines)

        asp_core = self.system.asp_core

        # Generate section for each layer
        for layer in StratificationLevel:
            lines.append(f"### {layer.value.title()} Layer\n")

            # Add layer description
            layer_descriptions = {
                StratificationLevel.CONSTITUTIONAL: (
                    "Constitutional rules are immutable and define core legal principles. "
                    "These rules cannot be modified by the system."
                ),
                StratificationLevel.STRATEGIC: (
                    "Strategic rules define high-level legal doctrines and principles. "
                    "Requires human review for modification."
                ),
                StratificationLevel.TACTICAL: (
                    "Tactical rules implement specific legal tests and procedures. "
                    "Can be autonomously modified with rollback protection."
                ),
                StratificationLevel.OPERATIONAL: (
                    "Operational rules handle procedural and administrative matters. "
                    "Freely modifiable for experimentation."
                ),
            }

            if layer in layer_descriptions:
                lines.append(f"*{layer_descriptions[layer]}*\n")

            # Get rules for this layer
            rules = asp_core.get_rules_by_layer(layer)
            lines.append(f"**Count**: {len(rules)} rules\n")

            if rules:
                lines.append("```prolog")
                for rule in rules:
                    lines.append(rule)
                lines.append("```\n")
            else:
                lines.append("*No rules in this layer*\n")

        return "\n".join(lines)

    def _generate_incorporation_history(self) -> str:
        """Generate incorporation history section."""
        lines = ["## Incorporation History\n"]

        if not hasattr(self.system, "incorporation_engine"):
            lines.append("*No incorporation engine available*\n")
            return "\n".join(lines)

        # Get recent incorporations from cycle history
        recent_incorporations = []
        for cycle in reversed(self.cycle_history[-10:]):  # Last 10 cycles
            recent_incorporations.extend(cycle.successful_incorporations)

        if not recent_incorporations:
            lines.append("*No incorporations yet*\n")
            return "\n".join(lines)

        lines.append("Recent successful incorporations:\n")
        lines.append("| Cycle | Modification # | Status | Accuracy Improvement |")
        lines.append("|-------|----------------|--------|---------------------|")

        # Track which cycle each incorporation came from
        for cycle in reversed(self.cycle_history[-10:]):  # Last 10 cycles
            for inc in cycle.successful_incorporations[:5]:  # Show first 5 from each cycle
                mod_num = inc.modification_number
                status = "✅" if inc.is_success() else "❌"
                improvement = inc.accuracy_after - inc.accuracy_before
                impact = f"+{improvement:.1%}" if improvement > 0 else f"{improvement:.1%}"

                lines.append(f"| #{cycle.cycle_number} | {mod_num} | {status} | {impact} |")

        lines.append("")

        # Incorporation statistics
        if self.cycle_history:
            total_incorporated = sum(c.rules_incorporated for c in self.cycle_history)
            total_attempts = sum(c.variants_generated for c in self.cycle_history)
            success_rate = (total_incorporated / total_attempts * 100) if total_attempts > 0 else 0

            lines.append(f"**Total Incorporations**: {total_incorporated}")
            lines.append(f"**Success Rate**: {success_rate:.1f}%")

        return "\n".join(lines)

    def _generate_improvement_cycles(self) -> str:
        """Generate improvement cycles section."""
        lines = ["## Improvement Cycles\n"]

        if not self.cycle_history:
            lines.append("*No improvement cycles completed yet*\n")
            return "\n".join(lines)

        lines.append("Historical cycle results and performance trends:\n")
        lines.append("| Cycle # | Timestamp | Gaps | Variants | Incorporated | Accuracy Δ |")
        lines.append("|---------|-----------|------|----------|--------------|-----------|")

        for cycle in reversed(self.cycle_history[-15:]):  # Last 15 cycles
            timestamp = cycle.timestamp.strftime("%Y-%m-%d %H:%M")
            improvement = (
                f"+{cycle.overall_improvement:.1%}"
                if cycle.overall_improvement > 0
                else f"{cycle.overall_improvement:.1%}"
            )

            lines.append(
                f"| #{cycle.cycle_number} | {timestamp} | {cycle.gaps_identified} | "
                f"{cycle.variants_generated} | {cycle.rules_incorporated} | {improvement} |"
            )

        lines.append("")

        # Cycle statistics
        avg_improvement = sum(c.overall_improvement for c in self.cycle_history) / len(
            self.cycle_history
        )
        total_gaps = sum(c.gaps_identified for c in self.cycle_history)
        total_incorporated = sum(c.rules_incorporated for c in self.cycle_history)

        lines.append("**Cycle Statistics**:")
        lines.append(f"- Total Cycles: {len(self.cycle_history)}")
        lines.append(f"- Average Improvement: {avg_improvement:.1%}")
        lines.append(f"- Total Gaps Identified: {total_gaps}")
        lines.append(f"- Total Rules Incorporated: {total_incorporated}")

        return "\n".join(lines)

    def _generate_self_analysis(self) -> str:
        """Generate self-analysis section."""
        lines = ["## Self-Analysis\n"]

        if not hasattr(self.system, "analyze_self"):
            lines.append("*Self-analysis not available*\n")
            return "\n".join(lines)

        try:
            analysis = self.system.analyze_self()

            lines.append(f"**Self-Confidence**: {analysis.self_confidence:.1%}\n")
            lines.append(f"**Overall Health**: {analysis.system_health.overall_health.title()}\n")

            # Strengths
            if analysis.identified_strengths:
                lines.append("**Strengths**:")
                for strength in analysis.identified_strengths[:5]:
                    lines.append(f"- {strength}")
                lines.append("")

            # Weaknesses
            if analysis.identified_weaknesses:
                lines.append("**Areas for Improvement**:")
                for weakness in analysis.identified_weaknesses[:5]:
                    lines.append(f"- {weakness}")
                lines.append("")

            # Recommendations
            if analysis.recommendations:
                lines.append("**Recommendations**:")
                for rec in analysis.recommendations[:5]:
                    lines.append(f"- {rec}")
                lines.append("")

        except Exception as e:
            lines.append(f"*Error generating self-analysis: {e}*\n")

        return "\n".join(lines)

    def _generate_evolution_timeline(self) -> str:
        """Generate rule evolution timeline."""
        lines = ["## Rule Evolution Timeline\n"]

        if not self.cycle_history:
            lines.append("*No evolution history available*\n")
            return "\n".join(lines)

        lines.append("Timeline showing how the system has evolved:\n")

        for cycle in self.cycle_history[-10:]:  # Last 10 cycles
            timestamp = cycle.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"### Cycle #{cycle.cycle_number} - {timestamp}\n")
            lines.append(f"- **Status**: {cycle.status}")
            lines.append(f"- **Gaps Identified**: {cycle.gaps_identified}")
            lines.append(f"- **Rules Incorporated**: {cycle.rules_incorporated}")
            lines.append(
                f"- **Accuracy**: {cycle.baseline_accuracy:.1%} → {cycle.final_accuracy:.1%}"
            )

            if cycle.successful_incorporations:
                lines.append("\n**New Rules**:")
                for inc in cycle.successful_incorporations[:3]:  # Show first 3
                    rule_preview = (
                        inc.rule.asp_rule[:60] + "..."
                        if len(inc.rule.asp_rule) > 60
                        else inc.rule.asp_rule
                    )
                    lines.append(f"- `{rule_preview}`")

            lines.append("")

        return "\n".join(lines)

    def _generate_footer(self) -> str:
        """Generate document footer with metadata."""
        timestamp = datetime.now().isoformat()

        return f"""---

**Document Metadata**

- Generated: {timestamp}
- Generator: Living Document Generator v1.0
- System: Self-Modifying ASP System

This document is automatically generated after each improvement cycle.
Changes to this document reflect the system's evolution over time."""
