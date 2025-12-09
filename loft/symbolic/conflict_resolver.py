"""
Conflict resolution for contradictory ASP predictions.

Implements strategies for resolving cases where both enforceable(X) and
unenforceable(X) are derived for the same entity.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger


class ResolutionStrategy(Enum):
    """Available conflict resolution strategies."""

    SPECIFICITY = "specificity"  # More conditions wins
    CONFIDENCE = "confidence"  # Higher confidence wins
    SPECIFICITY_THEN_CONFIDENCE = "specificity_then_confidence"  # Combined
    LEGAL_DEFAULT = "legal_default"  # Use legal default (typically unenforceable)
    RECENCY = "recency"  # Most recently added rule wins


@dataclass
class ConflictInfo:
    """Information about a detected conflict."""

    entity: str
    positive_atoms: Set[str]  # e.g., enforceable(c1)
    negative_atoms: Set[str]  # e.g., unenforceable(c1)
    positive_rules: List[Tuple[str, float]]  # (rule, confidence)
    negative_rules: List[Tuple[str, float]]  # (rule, confidence)


@dataclass
class ResolutionResult:
    """Result of conflict resolution."""

    prediction: str  # "enforceable", "unenforceable", or "unknown"
    confidence: float  # 0.0 to 1.0
    method: str  # How the conflict was resolved
    conflict_detected: bool  # Was there a conflict to resolve?
    resolution_details: Dict = field(default_factory=dict)  # Additional info

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "method": self.method,
            "conflict_detected": self.conflict_detected,
            "resolution_details": self.resolution_details,
        }


@dataclass
class ConflictStats:
    """Statistics for conflict resolution operations."""

    total_resolutions: int = 0
    conflicts_detected: int = 0
    resolved_by_specificity: int = 0
    resolved_by_confidence: int = 0
    resolved_by_legal_default: int = 0
    unresolved_conflicts: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_resolutions": self.total_resolutions,
            "conflicts_detected": self.conflicts_detected,
            "resolved_by_specificity": self.resolved_by_specificity,
            "resolved_by_confidence": self.resolved_by_confidence,
            "resolved_by_legal_default": self.resolved_by_legal_default,
            "unresolved_conflicts": self.unresolved_conflicts,
            "conflict_rate": (
                self.conflicts_detected / self.total_resolutions
                if self.total_resolutions > 0
                else 0.0
            ),
        }


class ConflictResolver:
    """
    Resolve conflicting predictions from ASP reasoning.

    When both enforceable(X) and unenforceable(X) are derived, this class
    applies various strategies to determine which prediction should prevail.

    Example:
        resolver = ConflictResolver(strategy=ResolutionStrategy.SPECIFICITY_THEN_CONFIDENCE)
        result = resolver.resolve(
            derived_atoms={'enforceable(c1)', 'unenforceable(c1)', 'claim(c1)'},
            fired_rules=[
                ('enforceable(X) :- a(X), b(X), c(X).', 0.9),
                ('unenforceable(X) :- a(X).', 0.8)
            ],
            entity='c1'
        )
        # Result: enforceable wins (3 conditions vs 1)
    """

    # Default predicates for positive/negative outcomes
    POSITIVE_PREDICATES = {"enforceable", "valid", "entitled", "acquired"}
    NEGATIVE_PREDICATES = {"unenforceable", "invalid", "not_entitled", "not_acquired"}

    def __init__(
        self,
        strategy: ResolutionStrategy = ResolutionStrategy.SPECIFICITY_THEN_CONFIDENCE,
        legal_default: str = "unenforceable",
        positive_predicates: Optional[Set[str]] = None,
        negative_predicates: Optional[Set[str]] = None,
    ):
        """
        Initialize conflict resolver.

        Args:
            strategy: Resolution strategy to use
            legal_default: Default prediction when resolution is tied
                          ("unenforceable" favors caution in legal contexts)
            positive_predicates: Predicates indicating positive outcomes
            negative_predicates: Predicates indicating negative outcomes
        """
        self.strategy = strategy
        self.legal_default = legal_default
        self.positive_predicates = positive_predicates or self.POSITIVE_PREDICATES
        self.negative_predicates = negative_predicates or self.NEGATIVE_PREDICATES
        self.stats = ConflictStats()

    def resolve(
        self,
        derived_atoms: Set[str],
        fired_rules: List[Tuple[str, float]],
        entity: str,
    ) -> ResolutionResult:
        """
        Resolve conflict for an entity.

        Args:
            derived_atoms: Set of atoms derived by ASP reasoning
            fired_rules: List of (rule, confidence) tuples for rules that fired
            entity: The entity identifier (e.g., 'c1')

        Returns:
            ResolutionResult with prediction, confidence, and resolution method
        """
        self.stats.total_resolutions += 1

        # Check what was derived for this entity
        positive_atoms = self._find_atoms_for_entity(
            derived_atoms, self.positive_predicates, entity
        )
        negative_atoms = self._find_atoms_for_entity(
            derived_atoms, self.negative_predicates, entity
        )

        has_positive = len(positive_atoms) > 0
        has_negative = len(negative_atoms) > 0

        # No conflict cases
        if has_positive and not has_negative:
            return ResolutionResult(
                prediction="enforceable",
                confidence=0.9,
                method="no_conflict",
                conflict_detected=False,
            )

        if has_negative and not has_positive:
            return ResolutionResult(
                prediction="unenforceable",
                confidence=0.9,
                method="no_conflict",
                conflict_detected=False,
            )

        if not has_positive and not has_negative:
            return ResolutionResult(
                prediction="unknown",
                confidence=0.0,
                method="no_derivation",
                conflict_detected=False,
            )

        # Conflict exists
        self.stats.conflicts_detected += 1
        logger.info(
            f"Conflict detected for entity {entity}: {positive_atoms} vs {negative_atoms}"
        )

        # Classify fired rules
        positive_rules = self._classify_rules(fired_rules, self.positive_predicates)
        negative_rules = self._classify_rules(fired_rules, self.negative_predicates)

        conflict_info = ConflictInfo(
            entity=entity,
            positive_atoms=positive_atoms,
            negative_atoms=negative_atoms,
            positive_rules=positive_rules,
            negative_rules=negative_rules,
        )

        # Apply resolution strategy
        if self.strategy == ResolutionStrategy.SPECIFICITY:
            return self._resolve_by_specificity(conflict_info)
        elif self.strategy == ResolutionStrategy.CONFIDENCE:
            return self._resolve_by_confidence(conflict_info)
        elif self.strategy == ResolutionStrategy.SPECIFICITY_THEN_CONFIDENCE:
            return self._resolve_specificity_then_confidence(conflict_info)
        elif self.strategy == ResolutionStrategy.LEGAL_DEFAULT:
            return self._resolve_by_legal_default(conflict_info)
        else:
            self.stats.unresolved_conflicts += 1
            return ResolutionResult(
                prediction="unknown",
                confidence=0.0,
                method="unsupported_strategy",
                conflict_detected=True,
            )

    def _find_atoms_for_entity(
        self, derived_atoms: Set[str], predicates: Set[str], entity: str
    ) -> Set[str]:
        """Find atoms matching predicates for a specific entity."""
        matching = set()
        for atom in derived_atoms:
            pred_name = atom.split("(")[0] if "(" in atom else atom
            if pred_name in predicates:
                # Check if atom refers to this entity
                if f"({entity})" in atom or f"({entity}," in atom:
                    matching.add(atom)
        return matching

    def _classify_rules(
        self, fired_rules: List[Tuple[str, float]], predicates: Set[str]
    ) -> List[Tuple[str, float]]:
        """Classify rules by whether their head matches given predicates."""
        matching = []
        for rule, confidence in fired_rules:
            if ":-" in rule:
                head = rule.split(":-")[0].strip()
            else:
                head = rule.strip().rstrip(".")

            head_pred = head.split("(")[0] if "(" in head else head
            if head_pred in predicates:
                matching.append((rule, confidence))

        return matching

    def _count_conditions(self, rule: str) -> int:
        """Count number of conditions in a rule body."""
        if ":-" not in rule:
            return 0

        body = rule.split(":-")[1]
        # Remove trailing period
        body = body.rstrip(".")

        # Count comma-separated predicates, handling nested parentheses
        count = 0
        depth = 0
        current = ""

        for char in body:
            if char == "(":
                depth += 1
                current += char
            elif char == ")":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                if current.strip():
                    count += 1
                current = ""
            else:
                current += char

        # Count the last condition
        if current.strip():
            count += 1

        return count

    def _resolve_by_specificity(self, conflict: ConflictInfo) -> ResolutionResult:
        """Resolve conflict by preferring rules with more conditions."""
        positive_spec = 0
        negative_spec = 0

        if conflict.positive_rules:
            positive_spec = max(
                self._count_conditions(rule) for rule, _ in conflict.positive_rules
            )
        if conflict.negative_rules:
            negative_spec = max(
                self._count_conditions(rule) for rule, _ in conflict.negative_rules
            )

        details = {
            "positive_specificity": positive_spec,
            "negative_specificity": negative_spec,
        }

        if positive_spec > negative_spec:
            self.stats.resolved_by_specificity += 1
            confidence = 0.7 + 0.1 * min(positive_spec - negative_spec, 3) / 3
            return ResolutionResult(
                prediction="enforceable",
                confidence=confidence,
                method="specificity",
                conflict_detected=True,
                resolution_details=details,
            )
        elif negative_spec > positive_spec:
            self.stats.resolved_by_specificity += 1
            confidence = 0.7 + 0.1 * min(negative_spec - positive_spec, 3) / 3
            return ResolutionResult(
                prediction="unenforceable",
                confidence=confidence,
                method="specificity",
                conflict_detected=True,
                resolution_details=details,
            )
        else:
            # Tie - use legal default
            return self._resolve_by_legal_default(
                conflict, tie_reason="specificity_tie"
            )

    def _resolve_by_confidence(self, conflict: ConflictInfo) -> ResolutionResult:
        """Resolve conflict by preferring rules with higher confidence."""
        positive_conf = 0.0
        negative_conf = 0.0

        if conflict.positive_rules:
            positive_conf = max(conf for _, conf in conflict.positive_rules)
        if conflict.negative_rules:
            negative_conf = max(conf for _, conf in conflict.negative_rules)

        details = {
            "positive_confidence": positive_conf,
            "negative_confidence": negative_conf,
        }

        if positive_conf > negative_conf:
            self.stats.resolved_by_confidence += 1
            return ResolutionResult(
                prediction="enforceable",
                confidence=positive_conf,
                method="confidence",
                conflict_detected=True,
                resolution_details=details,
            )
        elif negative_conf > positive_conf:
            self.stats.resolved_by_confidence += 1
            return ResolutionResult(
                prediction="unenforceable",
                confidence=negative_conf,
                method="confidence",
                conflict_detected=True,
                resolution_details=details,
            )
        else:
            # Tie - use legal default
            return self._resolve_by_legal_default(conflict, tie_reason="confidence_tie")

    def _resolve_specificity_then_confidence(
        self, conflict: ConflictInfo
    ) -> ResolutionResult:
        """Resolve by specificity first, then confidence if tied."""
        positive_spec = 0
        negative_spec = 0

        if conflict.positive_rules:
            positive_spec = max(
                self._count_conditions(rule) for rule, _ in conflict.positive_rules
            )
        if conflict.negative_rules:
            negative_spec = max(
                self._count_conditions(rule) for rule, _ in conflict.negative_rules
            )

        if positive_spec > negative_spec:
            self.stats.resolved_by_specificity += 1
            confidence = 0.7 + 0.1 * min(positive_spec - negative_spec, 3) / 3
            return ResolutionResult(
                prediction="enforceable",
                confidence=confidence,
                method="specificity",
                conflict_detected=True,
                resolution_details={
                    "positive_specificity": positive_spec,
                    "negative_specificity": negative_spec,
                },
            )
        elif negative_spec > positive_spec:
            self.stats.resolved_by_specificity += 1
            confidence = 0.7 + 0.1 * min(negative_spec - positive_spec, 3) / 3
            return ResolutionResult(
                prediction="unenforceable",
                confidence=confidence,
                method="specificity",
                conflict_detected=True,
                resolution_details={
                    "positive_specificity": positive_spec,
                    "negative_specificity": negative_spec,
                },
            )
        else:
            # Specificity tied, use confidence
            return self._resolve_by_confidence(conflict)

    def _resolve_by_legal_default(
        self, conflict: ConflictInfo, tie_reason: str = "legal_default"
    ) -> ResolutionResult:
        """Resolve by using the legal default (typically unenforceable)."""
        self.stats.resolved_by_legal_default += 1
        return ResolutionResult(
            prediction=self.legal_default,
            confidence=0.5,
            method=tie_reason,
            conflict_detected=True,
            resolution_details={"default_used": self.legal_default},
        )

    def detect_conflicts(self, derived_atoms: Set[str]) -> List[str]:
        """
        Detect all entities with conflicting predictions.

        Args:
            derived_atoms: Set of atoms derived by ASP reasoning

        Returns:
            List of entity identifiers with conflicts
        """
        # Extract entities from positive predictions
        positive_entities = set()
        for atom in derived_atoms:
            pred_name = atom.split("(")[0] if "(" in atom else atom
            if pred_name in self.positive_predicates:
                # Extract entity from atom
                match = re.search(r"\(([^,)]+)", atom)
                if match:
                    positive_entities.add(match.group(1))

        # Extract entities from negative predictions
        negative_entities = set()
        for atom in derived_atoms:
            pred_name = atom.split("(")[0] if "(" in atom else atom
            if pred_name in self.negative_predicates:
                match = re.search(r"\(([^,)]+)", atom)
                if match:
                    negative_entities.add(match.group(1))

        # Find intersection (conflicts)
        return list(positive_entities & negative_entities)

    def resolve_all_conflicts(
        self,
        derived_atoms: Set[str],
        fired_rules: List[Tuple[str, float]],
    ) -> Dict[str, ResolutionResult]:
        """
        Resolve all conflicts in a set of derived atoms.

        Args:
            derived_atoms: Set of atoms derived by ASP reasoning
            fired_rules: List of (rule, confidence) tuples

        Returns:
            Dictionary mapping entity to resolution result
        """
        conflicting_entities = self.detect_conflicts(derived_atoms)
        results = {}

        for entity in conflicting_entities:
            results[entity] = self.resolve(derived_atoms, fired_rules, entity)

        return results

    def get_stats(self) -> ConflictStats:
        """Get conflict resolution statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = ConflictStats()

    def __repr__(self) -> str:
        return (
            f"ConflictResolver(strategy={self.strategy.value}, "
            f"legal_default='{self.legal_default}')"
        )
