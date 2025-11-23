"""
ASP (Answer Set Programming) rule representation.

Provides Python wrappers for ASP rules with stratification levels,
confidence scores, and metadata tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import hashlib


class StratificationLevel(str, Enum):
    """
    Stratification levels for rules in the symbolic core.

    Four-layer architecture from constitutional (immutable) to operational (rapid adaptation).
    """

    CONSTITUTIONAL = "constitutional"  # Immutable, require code change
    STRATEGIC = "strategic"  # Slow change, high validation (>0.9 confidence)
    TACTICAL = "tactical"  # Frequent updates, moderate validation (>0.8 confidence)
    OPERATIONAL = "operational"  # Rapid adaptation, low validation (>0.6 confidence)


@dataclass
class RuleMetadata:
    """
    Metadata for ASP rules.

    Tracks provenance, timestamps, and validation information.
    """

    provenance: str  # Source: "llm", "human", "validation", "system"
    timestamp: str  # ISO format timestamp
    validation_score: float = 1.0  # Validation confidence (0.0-1.0)
    author: Optional[str] = None  # Author identifier
    tags: list[str] = field(default_factory=list)  # Categorization tags
    notes: str = ""  # Additional notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "provenance": self.provenance,
            "timestamp": self.timestamp,
            "validation_score": self.validation_score,
            "author": self.author,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleMetadata":
        """Create metadata from dictionary."""
        return cls(
            provenance=data["provenance"],
            timestamp=data["timestamp"],
            validation_score=data.get("validation_score", 1.0),
            author=data.get("author"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )


@dataclass
class ASPRule:
    """
    Python wrapper for ASP (Answer Set Programming) rules.

    Represents a single rule with stratification level, confidence,
    and metadata for tracking and validation.
    """

    rule_id: str  # Unique identifier
    asp_text: str  # The actual ASP rule text
    stratification_level: StratificationLevel  # Layer in the core
    confidence: float  # Confidence score (0.0-1.0)
    metadata: RuleMetadata  # Provenance and validation info

    def __post_init__(self) -> None:
        """Validate rule after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        # Validate confidence meets stratification requirements
        min_confidence = self.get_min_confidence_for_level(self.stratification_level)
        if self.confidence < min_confidence:
            raise ValueError(
                f"{self.stratification_level.value} level requires confidence >= {min_confidence}, "
                f"got {self.confidence}"
            )

    @staticmethod
    def get_min_confidence_for_level(level: StratificationLevel) -> float:
        """Get minimum confidence threshold for a stratification level."""
        thresholds = {
            StratificationLevel.CONSTITUTIONAL: 1.0,  # Perfect confidence required
            StratificationLevel.STRATEGIC: 0.9,
            StratificationLevel.TACTICAL: 0.8,
            StratificationLevel.OPERATIONAL: 0.6,
        }
        return thresholds[level]

    def to_clingo(self) -> str:
        """
        Convert to Clingo-compatible format with metadata as comments.

        Returns:
            Formatted ASP rule with metadata comments
        """
        lines = [
            f"% Rule ID: {self.rule_id}",
            f"% Level: {self.stratification_level.value}",
            f"% Confidence: {self.confidence}",
            f"% Provenance: {self.metadata.provenance}",
            self.asp_text,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary for serialization."""
        return {
            "rule_id": self.rule_id,
            "asp_text": self.asp_text,
            "stratification_level": self.stratification_level.value,
            "confidence": self.confidence,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ASPRule":
        """Create rule from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            asp_text=data["asp_text"],
            stratification_level=StratificationLevel(data["stratification_level"]),
            confidence=data["confidence"],
            metadata=RuleMetadata.from_dict(data["metadata"]),
        )

    def compute_hash(self) -> str:
        """Compute hash of rule for comparison and versioning."""
        content = f"{self.asp_text}:{self.stratification_level.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def is_fact(self) -> bool:
        """Check if this rule is a fact (no rule body)."""
        return ":-" not in self.asp_text

    def is_constraint(self) -> bool:
        """Check if this rule is a constraint (starts with :-)."""
        return self.asp_text.strip().startswith(":-")

    def is_choice_rule(self) -> bool:
        """Check if this rule is a choice rule (contains {})."""
        return "{" in self.asp_text and "}" in self.asp_text

    def extract_predicates(self) -> list[str]:
        """
        Extract predicate names from the rule.

        Returns:
            List of predicate names found in the rule
        """
        import re

        # Simple predicate extraction - matches predicate_name( or predicate_name.
        pattern = r"(\w+)(?:\(|\.)"
        matches = re.findall(pattern, self.asp_text)

        # Filter out keywords
        keywords = {"not", "if", "then", "else"}
        return [p for p in matches if p not in keywords]


def create_rule_id() -> str:
    """Generate a unique rule ID based on timestamp."""
    timestamp = datetime.utcnow().isoformat()
    return f"rule_{hashlib.sha256(timestamp.encode()).hexdigest()[:12]}"
