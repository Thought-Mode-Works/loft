"""
ASP (Answer Set Programming) rule representation.

Provides Python wrappers for ASP rules with stratification levels,
confidence scores, and metadata tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import re

# Import StratificationLevel from central location
from loft.symbolic.stratification import StratificationLevel


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

    # Added for Issue #235: Party Symmetry Invariance Testing
    name: str = field(init=False)  # Name property for easier identification
    parties_in_rule: List[str] = field(
        default_factory=list, init=False
    )  # Parties directly referenced in the rule

    # Fields for stratification validation (auto-populated from asp_text)
    predicates_used: List[str] = field(default_factory=list)
    new_predicates: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate rule after initialization."""
        self.name = self.rule_id  # Default name to rule_id

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        # Validate confidence meets stratification requirements
        min_confidence = self.get_min_confidence_for_level(self.stratification_level)
        if self.confidence < min_confidence:
            raise ValueError(
                f"{self.stratification_level.value} level requires confidence >= {min_confidence}, "
                f"got {self.confidence}"
            )

        # Extract predicates from asp_text if not provided
        if not self.predicates_used or not self.new_predicates:
            self._extract_predicates()

        # Validate that predicates were extracted (basic syntax check)
        if not self.is_fact():  # Only check rules, facts can just be "p."
            if not (self.new_predicates or self.predicates_used):
                raise ValueError(
                    f"Invalid ASP rule text: no predicates found in '{self.asp_text}'"
                )

        # Extract parties from rule content
        self.parties_in_rule = self._extract_parties_from_rule_text(self.asp_text)

        # Basic ASP syntax validation
        if not self._is_valid_asp_syntax():
            raise ValueError(f"Invalid ASP rule syntax: '{self.asp_text}'")

    def add_annotation(self, annotation: str) -> None:
        """Add an annotation to the rule's metadata tags."""
        if annotation not in self.metadata.tags:
            self.metadata.tags.append(annotation)

    def evaluate(self, case: Dict[str, Any]) -> Any:
        """
        Evaluate the rule against a given case (placeholder for actual ASP evaluation).
        This method would typically interface with an ASP solver.
        For symmetry testing, it needs to return a consistent outcome.
        """
        # This is a placeholder. A real implementation would involve:
        # 1. Combining case facts with the rule's asp_text.
        # 2. Running an ASP solver (e.g., Clingo) on the combined program.
        # 3. Parsing the answer set to determine the outcome.
        # For the purpose of symmetry testing, we need a mockable/predictable outcome.
        # For now, we'll return a deterministic value based on the case, or True by default.
        # A more complex mock in tests would provide specific side_effects.
        return True  # Default to True for now, assume basic rule is 'satisfied'

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

    def _is_valid_asp_syntax(self) -> bool:
        """Perform basic validation of ASP syntax."""
        text = self.asp_text.strip()
        if not text:
            return False

        # Must end with a period
        if not text.endswith("."):
            return False

        # If it's a constraint (no head), check its format
        if self.is_constraint():
            # A basic constraint is ':- body.'
            # It must have a body (even if just a variable or complex term)
            # The regex ensures there's something after ':- ' and it's not just whitespace
            if not re.fullmatch(r":-\s*.+\.$", text):
                return False
        elif self.is_fact():
            # Check for simple predicate format: 'name.' or 'name(args).'
            # This regex allows basic predicates and facts, but prevents random strings
            # e.g., "p." or "p(a,b)." but not "CORRUPTED DATA @#$%" or just "p"
            if not re.fullmatch(r"^[a-z][a-z0-9_]*(\([a-zA-Z0-9_,\s]*\))?\.$", text):
                return False
        else:  # Regular rule with head and body
            # For rules, there must be a head and a body (simplified check)
            if ":-" not in text:
                # This case shouldn't be reached if is_fact and is_constraint are robust,
                # but as a fallback, it's an invalid rule if no ":-"
                return False

            parts = text.split(":-", 1)
            head = parts[0].strip()
            body = parts[1].strip() if len(parts) > 1 else ""

            # Both head and body should not be empty for a non-constraint rule
            if not head or not body:
                return False
            # Ensure predicates are found in either head or body for a meaningful rule
            if not self.new_predicates and not self.predicates_used:
                return False

        return True

    def extract_predicates(self) -> List[str]:
        """
        Get all predicates used in this rule.

        Returns:
            List of unique predicate names from both head and body
        """
        return list(set(self.new_predicates + self.predicates_used))

    def _extract_predicates(self) -> None:
        """
        Extract predicates from ASP text and populate predicate fields.

        Populates:
        - new_predicates: Predicates defined in the head
        - predicates_used: Predicates referenced in the body
        """
        # Split on :- to separate head and body
        if ":-" in self.asp_text:
            parts = self.asp_text.split(":-", 1)
            head = parts[0].strip()
            body = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Fact (no body)
            head = self.asp_text.strip()
            body = ""

        # Extract predicates from head (new_predicates)
        self.new_predicates = self._extract_predicate_names(head)

        # Extract predicates from body (predicates_used)
        if body:
            self.predicates_used = self._extract_predicate_names(body)
        else:
            self.predicates_used = []

    def _extract_predicate_names(self, text: str) -> List[str]:
        """
        Extract predicate names from ASP text.

        Args:
            text: ASP text to extract from

        Returns:
            List of unique predicate names
        """
        # Match predicate_name( or predicate_name (p(X), p(a)), or just predicate_name (p.)
        # The (?:...) makes the group non-capturing for the optional part.
        # It now matches words followed by an optional parenthesis group or followed by nothing.
        pattern = r"\b([a-z][a-z0-9_]*)(?:\(.*\))?\b"
        matches = re.findall(pattern, text)

        # Filter out ASP keywords
        keywords = {
            "not",
            "and",
            "or",
        }  # Removed 'then', 'else' as they are not ASP keywords
        predicates = [m for m in matches if m not in keywords]

        return list(set(predicates))  # Return unique predicates

    def _extract_parties_from_rule_text(self, text: str) -> List[str]:
        """
        Extract potential party variables (uppercase letters) from ASP text.
        This is a heuristic for symmetry testing.

        Args:
            text: ASP rule text.

        Returns:
            List of unique uppercase single-letter variables found.
        """
        # A simple heuristic: assume single uppercase letters are party variables
        # This can be refined later if a more robust party identification is needed.
        parties = set(re.findall(r"\b[A-Z]\b", text))
        return sorted(list(parties))


def create_rule_id() -> str:
    """Generate a unique rule ID based on timestamp."""
    timestamp = datetime.utcnow().isoformat()
    return f"rule_{hashlib.sha256(timestamp.encode()).hexdigest()[:12]}"
