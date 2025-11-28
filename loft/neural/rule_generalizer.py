"""
Rule generalization post-processor.

Replaces specific constants with variables to improve rule coverage
and generalization. Preserves intentional constants like yes/no/land.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from loguru import logger


@dataclass
class GeneralizationResult:
    """Result of rule generalization."""

    original_rule: str
    generalized_rule: str
    changes: List[str] = field(default_factory=list)
    is_valid: bool = True
    error: Optional[str] = None

    @property
    def was_modified(self) -> bool:
        """Whether the rule was modified."""
        return len(self.changes) > 0


@dataclass
class GeneralizationStats:
    """Statistics for rule generalization operations."""

    rules_processed: int = 0
    rules_modified: int = 0
    constants_replaced: int = 0
    validation_failures: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "rules_processed": self.rules_processed,
            "rules_modified": self.rules_modified,
            "constants_replaced": self.constants_replaced,
            "validation_failures": self.validation_failures,
            "modification_rate": (
                self.rules_modified / self.rules_processed if self.rules_processed > 0 else 0.0
            ),
        }


class RuleGeneralizer:
    """
    Post-process ASP rules to improve generalization.

    Identifies and replaces specific constants (like entity names) with
    anonymous variables, while preserving intentional constants.

    Example:
        generalizer = RuleGeneralizer(known_entities={'alice', 'bob', 'c1'})
        result = generalizer.generalize(
            "enforceable(X) :- claimant(X, alice), occupation_years(X, N), N >= 20."
        )
        # Result: "enforceable(X) :- claimant(X, _), occupation_years(X, N), N >= 20."
    """

    # Constants that should NOT be generalized
    PRESERVE_CONSTANTS = {
        # Boolean-like values
        "yes",
        "no",
        "true",
        "false",
        # Property types
        "land",
        "goods",
        "services",
        "real_property",
        "personal_property",
        # Document types
        "written",
        "oral",
        # Status values
        "valid",
        "invalid",
        "enforceable",
        "unenforceable",
        # Classification values
        "commercial",
        "residential",
        "agricultural",
    }

    # Predicate names that should never have their arguments generalized
    # (the arguments are intentional classifications)
    PRESERVE_PREDICATES = {
        "property_type",
        "contract_type",
        "document_type",
        "classification",
    }

    def __init__(
        self,
        known_entities: Optional[Set[str]] = None,
        additional_preserve: Optional[Set[str]] = None,
    ):
        """
        Initialize rule generalizer.

        Args:
            known_entities: Entity names from dataset facts that should be
                           generalized (e.g., {"alice", "bob", "c1", "claim1"})
            additional_preserve: Additional constants to preserve (not generalize)
        """
        self.known_entities = {e.lower() for e in (known_entities or set())}
        self.preserve_constants = self.PRESERVE_CONSTANTS.copy()
        if additional_preserve:
            self.preserve_constants.update(c.lower() for c in additional_preserve)

        self.stats = GeneralizationStats()

    def generalize(self, rule: str) -> GeneralizationResult:
        """
        Generalize a rule by replacing specific constants with variables.

        Args:
            rule: ASP rule string

        Returns:
            GeneralizationResult with original, generalized, and changes
        """
        self.stats.rules_processed += 1
        changes: List[str] = []

        # Don't modify comments or empty lines
        stripped = rule.strip()
        if not stripped or stripped.startswith("%"):
            return GeneralizationResult(original_rule=rule, generalized_rule=rule)

        # Pattern to match predicates with arguments
        # Matches: predicate(arg1, arg2, ...) including nested parentheses for comparisons
        pred_pattern = r"([a-z][a-z0-9_]*)\s*\(([^)]+)\)"

        def replace_constants(match: re.Match) -> str:
            pred_name = match.group(1)
            args_str = match.group(2)

            # Skip predicates that should preserve all arguments
            if pred_name.lower() in self.PRESERVE_PREDICATES:
                return match.group(0)

            # Split arguments carefully (handle nested structures)
            args = self._split_arguments(args_str)
            new_args = []

            for arg in args:
                arg = arg.strip()
                if self._should_generalize(arg, pred_name):
                    changes.append(f"Replaced '{arg}' with '_' in {pred_name}()")
                    new_args.append("_")
                    self.stats.constants_replaced += 1
                else:
                    new_args.append(arg)

            return f"{pred_name}({', '.join(new_args)})"

        generalized = re.sub(pred_pattern, replace_constants, rule)

        if changes:
            self.stats.rules_modified += 1

        return GeneralizationResult(
            original_rule=rule,
            generalized_rule=generalized,
            changes=changes,
        )

    def _split_arguments(self, args_str: str) -> List[str]:
        """
        Split predicate arguments, handling nested parentheses.

        Args:
            args_str: Arguments string like "X, alice, foo(Y)"

        Returns:
            List of individual arguments
        """
        args = []
        current = ""
        depth = 0

        for char in args_str:
            if char == "(":
                depth += 1
                current += char
            elif char == ")":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            args.append(current.strip())

        return args

    def _should_generalize(self, arg: str, pred_name: str) -> bool:
        """
        Determine if an argument should be replaced with a variable.

        Args:
            arg: The argument string
            pred_name: Name of the containing predicate

        Returns:
            True if the argument should be generalized
        """
        arg = arg.strip()

        # Empty or whitespace
        if not arg:
            return False

        # Already a variable (uppercase start) or anonymous variable
        if arg[0].isupper() or arg == "_":
            return False

        # Numeric values should be preserved (they're thresholds)
        if arg.lstrip("-").isdigit():
            return False

        # Comparison expressions should be preserved
        if any(op in arg for op in [">=", "<=", ">", "<", "==", "!="]):
            return False

        # Preserved constants (yes, no, land, etc.)
        if arg.lower() in self.preserve_constants:
            return False

        # Known entity from dataset (alice, bob, c1, etc.)
        if arg.lower() in self.known_entities:
            return True

        # Heuristic: Short lowercase identifiers that look like entity names
        # and aren't keywords
        if self._looks_like_entity_name(arg):
            return True

        return False

    def _looks_like_entity_name(self, arg: str) -> bool:
        """
        Heuristic to detect entity-like names.

        Args:
            arg: The argument to check

        Returns:
            True if it looks like a specific entity name
        """
        # Must be lowercase
        if not arg.islower():
            return False

        # Skip if it's a preserved constant
        if arg in self.preserve_constants:
            return False

        # Skip common ASP patterns
        if arg.startswith("_"):
            return False

        # Pattern: lowercase letters/numbers like "alice", "bob", "c1", "claim1"
        # These are typically entity identifiers
        if re.match(r"^[a-z][a-z0-9_]*$", arg):
            # Skip very long names (likely predicate-like)
            if len(arg) > 15:
                return False

            # Check if it looks like a name (common patterns)
            # Names ending in numbers often are entities (c1, claim1, contract2)
            if re.search(r"\d$", arg):
                return True

            # Short names (2-6 chars) that aren't keywords
            if 2 <= len(arg) <= 6:
                # Skip obvious legal terms
                legal_terms = {
                    "claim",
                    "rule",
                    "fact",
                    "type",
                    "case",
                    "law",
                    "act",
                    "tort",
                    "duty",
                }
                if arg not in legal_terms:
                    return True

        return False

    def generalize_rules(self, rules: List[str]) -> List[GeneralizationResult]:
        """
        Generalize multiple rules.

        Args:
            rules: List of ASP rules

        Returns:
            List of GeneralizationResults
        """
        return [self.generalize(rule) for rule in rules]

    def get_stats(self) -> GeneralizationStats:
        """Get generalization statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = GeneralizationStats()


def extract_entities_from_facts(facts: str) -> Set[str]:
    """
    Extract entity constants from ASP facts.

    Identifies entity names that appear as arguments in facts,
    excluding known value constants like 'yes', 'no'.

    Args:
        facts: ASP facts string

    Returns:
        Set of entity names found
    """
    entities: Set[str] = set()

    # Value constants that aren't entity names
    value_constants = {"yes", "no", "true", "false", "land", "goods", "services"}

    # Pattern to match predicate arguments
    # Captures lowercase identifiers in predicate arguments
    pattern = r"[a-z_]+\s*\(([^)]+)\)"

    for match in re.finditer(pattern, facts):
        args_str = match.group(1)

        # Split and process each argument
        for arg in args_str.split(","):
            arg = arg.strip()

            # Only consider lowercase identifiers
            if re.match(r"^[a-z][a-z0-9_]*$", arg):
                if arg not in value_constants:
                    entities.add(arg)

    return entities


def generalize_rule_with_validation(
    rule: str,
    sample_facts: str,
    known_entities: Optional[Set[str]] = None,
) -> GeneralizationResult:
    """
    Generalize a rule and validate it still parses correctly.

    Args:
        rule: ASP rule to generalize
        sample_facts: Sample facts for validation context
        known_entities: Known entity names to generalize

    Returns:
        GeneralizationResult with validation status
    """
    # Extract entities from facts if not provided
    if known_entities is None:
        known_entities = extract_entities_from_facts(sample_facts)

    generalizer = RuleGeneralizer(known_entities=known_entities)
    result = generalizer.generalize(rule)

    if not result.was_modified:
        return result

    # Validate the generalized rule still parses
    try:
        import clingo

        ctl = clingo.Control()
        # Try to parse the generalized rule with facts
        program = f"{result.generalized_rule}\n{sample_facts}"
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        result.is_valid = True
    except Exception as e:
        logger.warning(f"Generalized rule validation failed: {e}")
        result.is_valid = False
        result.error = str(e)
        # Return original rule if generalization broke it
        result.generalized_rule = result.original_rule
        result.changes = []

    return result
