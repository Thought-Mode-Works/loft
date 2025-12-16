"""
Translation pattern documentation and analysis.

Documents translation patterns across runs, identifies success/failure patterns,
and generates pattern guides for improving translation quality.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from statistics import mean
import re


@dataclass
class TranslationPattern:
    """A recorded translation pattern."""

    original: str
    translated: str
    back_translated: str
    fidelity: float
    detected_type: str
    predicates: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternAnalysis:
    """Analysis results for translation patterns."""

    total_translations: int
    avg_fidelity: float
    fidelity_by_type: Dict[str, float]
    common_failure_patterns: List[Dict[str, Any]]
    successful_patterns: List[Dict[str, Any]]
    edge_cases: List[Dict[str, Any]] = field(default_factory=list)


class TranslationPatternDocumenter:
    """
    Documents translation patterns across runs.

    Analyzes patterns, identifies success/failure modes,
    and generates comprehensive pattern guides.
    """

    def __init__(self, output_path: Optional[Path] = None):
        """
        Initialize pattern documenter.

        Args:
            output_path: Optional path for pattern guide output
        """
        self.output_path = output_path
        self.patterns: List[TranslationPattern] = []

    def record_translation(
        self,
        original: str,
        translated: str,
        back_translated: str,
        fidelity: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a translation for pattern analysis.

        Args:
            original: Original ASP text
            translated: Natural language translation
            back_translated: Back-translated ASP
            fidelity: Translation fidelity score
            metadata: Optional metadata
        """
        pattern = TranslationPattern(
            original=original,
            translated=translated,
            back_translated=back_translated,
            fidelity=fidelity,
            detected_type=self._detect_rule_type(original),
            predicates=self._extract_predicates(original),
            metadata=metadata or {},
        )
        self.patterns.append(pattern)

    def _detect_rule_type(self, asp_text: str) -> str:
        """
        Detect ASP rule type.

        Args:
            asp_text: ASP rule text

        Returns:
            Rule type string
        """
        asp_text = asp_text.strip()

        # Constraint
        if asp_text.startswith(":-"):
            return "constraint"

        # Choice rule
        if "|" in asp_text.split(":-")[0] if ":-" in asp_text else False:
            return "choice"

        # Fact (no :-)
        if ":-" not in asp_text:
            return "fact"

        # Check for special constructs
        if "#count" in asp_text or "#sum" in asp_text or "#max" in asp_text:
            return "aggregate"

        # Check for negation
        if "not " in asp_text:
            return "negation"

        # Check for disjunction in body
        if ";" in asp_text.split(":-")[1] if ":-" in asp_text else False:
            return "disjunction"

        # Default: standard rule
        return "rule"

    def _extract_predicates(self, asp_text: str) -> List[str]:
        """
        Extract predicate names from ASP text.

        Args:
            asp_text: ASP rule text

        Returns:
            List of predicate names
        """
        # Extract predicates (lowercase identifiers followed by '(')
        pattern = r"\b([a-z][a-z0-9_]*)(?:\()"
        matches = re.findall(pattern, asp_text)

        # Filter out ASP keywords
        keywords = {"not", "count", "sum", "max", "min"}
        predicates = [m for m in matches if m not in keywords]

        return list(set(predicates))

    def analyze_patterns(self) -> PatternAnalysis:
        """
        Analyze recorded patterns.

        Returns:
            PatternAnalysis object with analysis results
        """
        if not self.patterns:
            return PatternAnalysis(
                total_translations=0,
                avg_fidelity=0.0,
                fidelity_by_type={},
                common_failure_patterns=[],
                successful_patterns=[],
            )

        return PatternAnalysis(
            total_translations=len(self.patterns),
            avg_fidelity=mean([p.fidelity for p in self.patterns]),
            fidelity_by_type=self._group_by_type(),
            common_failure_patterns=self._identify_failure_patterns(),
            successful_patterns=self._identify_success_patterns(),
            edge_cases=self._identify_edge_cases(),
        )

    def _group_by_type(self) -> Dict[str, float]:
        """Group patterns by type and compute average fidelity."""
        by_type: Dict[str, List[float]] = defaultdict(list)

        for pattern in self.patterns:
            by_type[pattern.detected_type].append(pattern.fidelity)

        return {rtype: mean(fidelities) for rtype, fidelities in by_type.items()}

    def _identify_failure_patterns(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Identify common failure patterns.

        Args:
            threshold: Fidelity threshold below which to consider failure

        Returns:
            List of failure pattern dictionaries
        """
        failures = [p for p in self.patterns if p.fidelity < threshold]

        if not failures:
            return []

        # Group by rule type
        by_type = defaultdict(list)
        for p in failures:
            by_type[p.detected_type].append(p)

        failure_patterns = []
        for rtype, patterns in by_type.items():
            if len(patterns) < 2:  # Skip single occurrences
                continue

            avg_fidelity = mean([p.fidelity for p in patterns])
            examples = patterns[:3]  # First 3 examples

            failure_patterns.append(
                {
                    "rule_type": rtype,
                    "count": len(patterns),
                    "avg_fidelity": avg_fidelity,
                    "examples": [
                        {
                            "original": p.original,
                            "translated": p.translated,
                            "back_translated": p.back_translated,
                            "fidelity": p.fidelity,
                        }
                        for p in examples
                    ],
                }
            )

        # Sort by count (most common failures first)
        failure_patterns.sort(key=lambda x: x["count"], reverse=True)

        return failure_patterns

    def _identify_success_patterns(self, threshold: float = 0.95) -> List[Dict[str, Any]]:
        """
        Identify successful translation patterns.

        Args:
            threshold: Fidelity threshold above which to consider success

        Returns:
            List of success pattern dictionaries
        """
        successes = [p for p in self.patterns if p.fidelity >= threshold]

        if not successes:
            return []

        # Group by rule type
        by_type = defaultdict(list)
        for p in successes:
            by_type[p.detected_type].append(p)

        success_patterns = []
        for rtype, patterns in by_type.items():
            avg_fidelity = mean([p.fidelity for p in patterns])
            examples = patterns[:3]  # First 3 examples

            success_patterns.append(
                {
                    "rule_type": rtype,
                    "count": len(patterns),
                    "avg_fidelity": avg_fidelity,
                    "examples": [
                        {
                            "original": p.original,
                            "translated": p.translated,
                            "fidelity": p.fidelity,
                        }
                        for p in examples
                    ],
                }
            )

        # Sort by avg fidelity (best first)
        success_patterns.sort(key=lambda x: x["avg_fidelity"], reverse=True)

        return success_patterns

    def _identify_edge_cases(self) -> List[Dict[str, Any]]:
        """
        Identify edge cases (unusual patterns).

        Returns:
            List of edge case dictionaries
        """
        # Find patterns with unusual characteristics
        edge_cases = []

        # Edge case 1: Low fidelity despite simple structure
        simple_types = ["fact", "rule"]
        for p in self.patterns:
            if p.detected_type in simple_types and p.fidelity < 0.8:
                edge_cases.append(
                    {
                        "category": "simple_but_failed",
                        "type": p.detected_type,
                        "original": p.original,
                        "fidelity": p.fidelity,
                    }
                )

        # Edge case 2: High fidelity despite complex structure
        complex_types = ["aggregate", "choice", "disjunction"]
        for p in self.patterns:
            if p.detected_type in complex_types and p.fidelity > 0.95:
                edge_cases.append(
                    {
                        "category": "complex_but_succeeded",
                        "type": p.detected_type,
                        "original": p.original,
                        "fidelity": p.fidelity,
                    }
                )

        return edge_cases[:10]  # Limit to 10 edge cases

    def generate_pattern_guide(self) -> str:
        """
        Generate markdown pattern guide.

        Returns:
            Markdown-formatted pattern guide
        """
        analysis = self.analyze_patterns()

        guide = f"""# Translation Pattern Guide

## Overview

- **Total translations analyzed**: {analysis.total_translations}
- **Average fidelity**: {analysis.avg_fidelity:.2%}

## Fidelity by Rule Type

| Rule Type | Average Fidelity | Performance |
|-----------|------------------|-------------|
"""

        for rtype, fidelity in sorted(
            analysis.fidelity_by_type.items(), key=lambda x: x[1], reverse=True
        ):
            performance = "✓ Good" if fidelity >= 0.9 else "⚠ Needs Work"
            guide += f"| {rtype} | {fidelity:.2%} | {performance} |\n"

        # Common failure patterns
        if analysis.common_failure_patterns:
            guide += "\n## Common Failure Patterns\n\n"

            for idx, failure in enumerate(analysis.common_failure_patterns, 1):
                guide += f"### {idx}. {failure['rule_type'].title()} Rules\n\n"
                guide += f"- **Occurrences**: {failure['count']}\n"
                guide += f"- **Average Fidelity**: {failure['avg_fidelity']:.2%}\n\n"

                guide += "**Examples**:\n\n"
                for ex in failure["examples"][:2]:  # Show 2 examples
                    guide += f"```asp\n{ex['original']}\n```\n\n"
                    guide += f'*Translated to*: "{ex["translated"]}"\n\n'
                    guide += f"*Back-translated*: `{ex['back_translated']}`\n\n"
                    guide += f"*Fidelity*: {ex['fidelity']:.2%}\n\n"

        # Successful patterns
        if analysis.successful_patterns:
            guide += "\n## Successful Translation Patterns\n\n"

            for idx, success in enumerate(analysis.successful_patterns, 1):
                guide += f"### {idx}. {success['rule_type'].title()} Rules\n\n"
                guide += f"- **Occurrences**: {success['count']}\n"
                guide += f"- **Average Fidelity**: {success['avg_fidelity']:.2%}\n\n"

                guide += "**Examples**:\n\n"
                for ex in success["examples"][:2]:  # Show 2 examples
                    guide += f"```asp\n{ex['original']}\n```\n\n"
                    guide += f'*Translated to*: "{ex["translated"]}"\n\n'
                    guide += f"*Fidelity*: {ex['fidelity']:.2%}\n\n"

        # Edge cases
        if analysis.edge_cases:
            guide += "\n## Edge Cases\n\n"

            for edge in analysis.edge_cases[:5]:  # Show top 5
                guide += f"### {edge['category'].replace('_', ' ').title()}\n\n"
                guide += f"- **Type**: {edge['type']}\n"
                guide += f"- **Fidelity**: {edge['fidelity']:.2%}\n\n"
                guide += f"```asp\n{edge['original']}\n```\n\n"

        # Recommendations
        guide += self._generate_recommendations(analysis)

        return guide

    def _generate_recommendations(self, analysis: PatternAnalysis) -> str:
        """Generate recommendations based on analysis."""
        recommendations = "\n## Recommendations\n\n"

        # Find rule types that need improvement
        needs_work = [
            (rtype, fidelity)
            for rtype, fidelity in analysis.fidelity_by_type.items()
            if fidelity < 0.9
        ]

        if needs_work:
            recommendations += "### Priority Improvements\n\n"
            for rtype, fidelity in sorted(needs_work, key=lambda x: x[1]):
                recommendations += f"1. **{rtype.title()} rules** (current: {fidelity:.2%})\n"
                recommendations += f"   - Review translation templates for {rtype} patterns\n"
                recommendations += "   - Add more test cases for this rule type\n"
                recommendations += "   - Consider specialized handling\n\n"

        # General recommendations
        recommendations += "### General Best Practices\n\n"
        recommendations += "- Validate roundtrip translations regularly\n"
        recommendations += "- Maintain diverse test cases covering all rule types\n"
        recommendations += "- Monitor fidelity trends for regressions\n"
        recommendations += "- Document challenging edge cases as they're discovered\n\n"

        return recommendations

    def save_guide(self, filepath: Optional[str] = None) -> Path:
        """
        Save pattern guide to file.

        Args:
            filepath: Optional path to save guide (defaults to output_path)

        Returns:
            Path where guide was saved
        """
        if filepath is None:
            if self.output_path is None:
                filepath = "reports/translation_patterns.md"
            else:
                filepath = str(self.output_path)

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        guide = self.generate_pattern_guide()
        path.write_text(guide)

        return path
