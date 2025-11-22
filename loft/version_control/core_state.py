"""
Core state representation for version control.

Defines the serializable state of the symbolic core including rules,
metadata, and configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import hashlib
import json


class StratificationLevel(str, Enum):
    """Stratification levels for rules in the symbolic core."""

    CONSTITUTIONAL = "constitutional"
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"


@dataclass
class Rule:
    """
    Represents a single rule in the symbolic core.

    Attributes:
        rule_id: Unique identifier for the rule
        content: The ASP rule content
        level: Stratification level
        confidence: Confidence score (0.0-1.0)
        provenance: Origin of the rule (e.g., "llm", "human", "validation")
        timestamp: When the rule was added
        metadata: Additional metadata
    """

    rule_id: str
    content: str
    level: StratificationLevel
    confidence: float
    provenance: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary for serialization."""
        return {
            "rule_id": self.rule_id,
            "content": self.content,
            "level": self.level.value,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        """Create rule from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            content=data["content"],
            level=StratificationLevel(data["level"]),
            confidence=data["confidence"],
            provenance=data["provenance"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )

    def __hash__(self) -> int:
        """Hash based on rule content and level."""
        return hash((self.content, self.level))


@dataclass
class CoreState:
    """
    Complete state of the symbolic core at a point in time.

    This is the unit of version control - each commit saves a CoreState.
    """

    state_id: str
    timestamp: str
    rules: List[Rule]
    configuration: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert core state to dictionary for serialization."""
        return {
            "state_id": self.state_id,
            "timestamp": self.timestamp,
            "rules": [rule.to_dict() for rule in self.rules],
            "configuration": self.configuration,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoreState":
        """Create core state from dictionary."""
        return cls(
            state_id=data["state_id"],
            timestamp=data["timestamp"],
            rules=[Rule.from_dict(r) for r in data["rules"]],
            configuration=data["configuration"],
            metrics=data["metrics"],
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Convert core state to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "CoreState":
        """Create core state from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def compute_hash(self) -> str:
        """Compute hash of core state for comparison."""
        # Create deterministic representation
        rule_hashes = sorted(
            [
                hashlib.sha256(f"{r.content}:{r.level.value}".encode()).hexdigest()
                for r in self.rules
            ]
        )
        config_str = json.dumps(self.configuration, sort_keys=True)

        combined = f"{''.join(rule_hashes)}:{config_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get_rules_by_level(self, level: StratificationLevel) -> List[Rule]:
        """Get all rules at a specific stratification level."""
        return [r for r in self.rules if r.level == level]

    def count_rules_by_level(self) -> Dict[str, int]:
        """Count rules at each stratification level."""
        counts: Dict[str, int] = {level.value: 0 for level in StratificationLevel}
        for rule in self.rules:
            counts[rule.level.value] += 1
        return counts


@dataclass
class Commit:
    """
    Represents a commit in the version history.

    A commit is a snapshot of the core state with metadata about why it was saved.
    """

    commit_id: str
    parent_id: Optional[str]
    state: CoreState
    message: str
    author: str
    timestamp: str
    branch: str = "main"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert commit to dictionary for serialization."""
        return {
            "commit_id": self.commit_id,
            "parent_id": self.parent_id,
            "state": self.state.to_dict(),
            "message": self.message,
            "author": self.author,
            "timestamp": self.timestamp,
            "branch": self.branch,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Commit":
        """Create commit from dictionary."""
        return cls(
            commit_id=data["commit_id"],
            parent_id=data.get("parent_id"),
            state=CoreState.from_dict(data["state"]),
            message=data["message"],
            author=data["author"],
            timestamp=data["timestamp"],
            branch=data.get("branch", "main"),
            tags=data.get("tags", []),
        )

    def to_json(self) -> str:
        """Convert commit to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Commit":
        """Create commit from JSON string."""
        return cls.from_dict(json.loads(json_str))


def create_state_id() -> str:
    """Generate a unique state ID based on timestamp."""
    timestamp = datetime.utcnow().isoformat()
    return hashlib.sha256(timestamp.encode()).hexdigest()[:16]


def create_commit_id() -> str:
    """Generate a unique commit ID based on timestamp."""
    timestamp = datetime.utcnow().isoformat()
    return hashlib.sha256(timestamp.encode()).hexdigest()[:16]
