"""
ASP program representation and composition.

Provides structures for managing collections of ASP rules and facts,
with support for composition and stratification.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from .asp_rule import ASPRule, StratificationLevel


@dataclass
class ASPProgram:
    """
    Collection of ASP rules and facts.

    Represents a complete or partial ASP program that can be
    composed with other programs and converted to Clingo format.
    """

    rules: List[ASPRule] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)  # Raw ASP facts
    name: str = "unnamed"
    description: str = ""

    def add_rule(self, rule: ASPRule) -> None:
        """Add a rule to the program."""
        self.rules.append(rule)

    def add_fact(self, fact: str) -> None:
        """Add a fact to the program."""
        if not fact.strip().endswith("."):
            fact = fact.strip() + "."
        self.facts.append(fact)

    def to_asp(self) -> str:
        """
        Convert program to ASP text format.

        Returns:
            Complete ASP program as string
        """
        lines = []

        # Add header comment
        if self.name != "unnamed":
            lines.append(f"% Program: {self.name}")
            if self.description:
                lines.append(f"% {self.description}")
            lines.append("")

        # Add facts
        if self.facts:
            lines.append("% === Facts ===")
            lines.extend(self.facts)
            lines.append("")

        # Add rules
        if self.rules:
            lines.append("% === Rules ===")
            for rule in self.rules:
                lines.append(rule.to_clingo())
                lines.append("")  # Blank line between rules

        return "\n".join(lines)

    def get_rules_by_level(self, level: StratificationLevel) -> List[ASPRule]:
        """Get all rules at a specific stratification level."""
        return [r for r in self.rules if r.stratification_level == level]

    def get_rule(self, rule_id: str) -> Optional[ASPRule]:
        """Get a specific rule by its ID."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def count_rules_by_level(self) -> Dict[str, int]:
        """Count rules at each stratification level."""
        counts: Dict[str, int] = {level.value: 0 for level in StratificationLevel}
        for rule in self.rules:
            counts[rule.stratification_level.value] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert program to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "rules": [r.to_dict() for r in self.rules],
            "facts": self.facts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ASPProgram":
        """Create program from dictionary."""
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            rules=[ASPRule.from_dict(r) for r in data.get("rules", [])],
            facts=data.get("facts", []),
        )

    def to_json(self) -> str:
        """Convert program to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ASPProgram":
        """Create program from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class StratifiedASPCore:
    """
    Four-layer stratified architecture for ASP rules.

    Layers:
    - Constitutional: Immutable core principles
    - Strategic: High-level planning and policy
    - Tactical: Mid-level operational rules
    - Operational: Low-level rapid adaptation
    """

    constitutional: ASPProgram = field(
        default_factory=lambda: ASPProgram(name="constitutional")
    )
    strategic: ASPProgram = field(default_factory=lambda: ASPProgram(name="strategic"))
    tactical: ASPProgram = field(default_factory=lambda: ASPProgram(name="tactical"))
    operational: ASPProgram = field(
        default_factory=lambda: ASPProgram(name="operational")
    )

    def get_program(self, level: StratificationLevel) -> ASPProgram:
        """Get the program for a specific stratification level."""
        level_map = {
            StratificationLevel.CONSTITUTIONAL: self.constitutional,
            StratificationLevel.STRATEGIC: self.strategic,
            StratificationLevel.TACTICAL: self.tactical,
            StratificationLevel.OPERATIONAL: self.operational,
        }
        return level_map[level]

    def add_rule(self, rule: ASPRule) -> None:
        """Add a rule to the appropriate stratification level."""
        program = self.get_program(rule.stratification_level)
        program.add_rule(rule)

    def get_rule(self, rule_id: str) -> Optional[ASPRule]:
        """Get a specific rule by its ID across all layers."""
        for rule in self.get_all_rules():
            if rule.rule_id == rule_id:
                return rule
        return None

    def get_full_program(self) -> str:
        """
        Compose all layers into a single ASP program.

        Returns:
            Complete ASP program with all layers
        """
        sections = [
            "% ============================================",
            "% STRATIFIED ASP CORE",
            "% ============================================",
            "",
            "% === CONSTITUTIONAL LAYER ===",
            "% Immutable core principles",
            self.constitutional.to_asp(),
            "",
            "% === STRATEGIC LAYER ===",
            "% High-level planning and policy",
            self.strategic.to_asp(),
            "",
            "% === TACTICAL LAYER ===",
            "% Mid-level operational rules",
            self.tactical.to_asp(),
            "",
            "% === OPERATIONAL LAYER ===",
            "% Low-level rapid adaptation",
            self.operational.to_asp(),
            "",
            "% ============================================",
        ]
        return "\n".join(sections)

    def get_all_rules(self) -> List[ASPRule]:
        """Get all rules from all layers."""
        return (
            self.constitutional.rules
            + self.strategic.rules
            + self.tactical.rules
            + self.operational.rules
        )

    def count_total_rules(self) -> int:
        """Count total number of rules across all layers."""
        return len(self.get_all_rules())

    def find_rules_mentioning(self, predicate: str) -> List[ASPRule]:
        """
        Find all rules that mention a specific predicate.

        Args:
            predicate: Predicate name to search for

        Returns:
            List of rules that contain the predicate
        """
        matching_rules = []
        for rule in self.get_all_rules():
            if predicate in rule.extract_predicates():
                matching_rules.append(rule)
        return matching_rules

    def to_dict(self) -> Dict[str, Any]:
        """Convert stratified core to dictionary."""
        return {
            "constitutional": self.constitutional.to_dict(),
            "strategic": self.strategic.to_dict(),
            "tactical": self.tactical.to_dict(),
            "operational": self.operational.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StratifiedASPCore":
        """Create stratified core from dictionary."""
        return cls(
            constitutional=ASPProgram.from_dict(data["constitutional"]),
            strategic=ASPProgram.from_dict(data["strategic"]),
            tactical=ASPProgram.from_dict(data["tactical"]),
            operational=ASPProgram.from_dict(data["operational"]),
        )


def compose_programs(prog1: ASPProgram, prog2: ASPProgram) -> ASPProgram:
    """
    Compose two ASP programs into one.

    Args:
        prog1: First program
        prog2: Second program

    Returns:
        New program containing rules and facts from both
    """
    return ASPProgram(
        name=f"{prog1.name}+{prog2.name}",
        description=f"Composition of {prog1.name} and {prog2.name}",
        rules=prog1.rules + prog2.rules,
        facts=prog1.facts + prog2.facts,
    )
