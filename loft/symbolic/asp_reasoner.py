"""
ASP-based reasoning for legal scenario predictions.

This module provides the ASPReasoner class that uses the Clingo solver
to make predictions based on ASP rules and scenario facts.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import clingo
from loguru import logger

if TYPE_CHECKING:
    from .conflict_resolver import ConflictResolver


@dataclass
class ReasoningResult:
    """Result from ASP reasoning on a scenario."""

    prediction: str  # "enforceable", "unenforceable", or "unknown"
    confidence: float  # Confidence in the prediction (0.0-1.0)
    satisfiable: bool  # Whether the program was satisfiable
    answer_set_count: int  # Number of answer sets found
    derived_atoms: Set[str]  # Atoms derived in the answer set
    rules_fired: List[str]  # Rules that contributed to the result
    grounding_time_ms: float  # Time spent grounding
    solving_time_ms: float  # Time spent solving
    error: Optional[str] = None  # Error message if reasoning failed
    conflict_resolved: bool = False  # Whether a conflict was resolved
    resolution_method: Optional[str] = None  # How conflict was resolved
    resolution_details: Dict = field(default_factory=dict)  # Additional conflict info

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/reporting."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "satisfiable": self.satisfiable,
            "answer_set_count": self.answer_set_count,
            "derived_atoms": list(self.derived_atoms),
            "rules_fired": self.rules_fired,
            "grounding_time_ms": self.grounding_time_ms,
            "solving_time_ms": self.solving_time_ms,
            "error": self.error,
            "conflict_resolved": self.conflict_resolved,
            "resolution_method": self.resolution_method,
            "resolution_details": self.resolution_details,
        }


@dataclass
class ReasoningStats:
    """Statistics from a batch of reasoning operations."""

    total_scenarios: int = 0
    correct_predictions: int = 0
    incorrect_predictions: int = 0
    unknown_predictions: int = 0
    reasoning_errors: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    total_grounding_time_ms: float = 0.0
    total_solving_time_ms: float = 0.0
    scenarios_with_answer_sets: int = 0
    coverage: float = 0.0  # % of scenarios with definitive predictions

    def update(self, result: ReasoningResult, correct: Optional[bool] = None) -> None:
        """Update stats with a new result."""
        self.total_scenarios += 1
        self.total_grounding_time_ms += result.grounding_time_ms
        self.total_solving_time_ms += result.solving_time_ms

        if result.conflict_resolved:
            self.conflicts_detected += 1
            self.conflicts_resolved += 1

        if result.error:
            self.reasoning_errors += 1
        elif result.prediction == "unknown":
            self.unknown_predictions += 1
        else:
            self.scenarios_with_answer_sets += 1
            if correct is True:
                self.correct_predictions += 1
            elif correct is False:
                self.incorrect_predictions += 1

        # Update coverage
        definitive = (
            self.total_scenarios - self.unknown_predictions - self.reasoning_errors
        )
        self.coverage = (
            definitive / self.total_scenarios if self.total_scenarios > 0 else 0.0
        )

    def get_accuracy(self) -> float:
        """Get accuracy over scenarios with definitive predictions."""
        definitive = self.correct_predictions + self.incorrect_predictions
        return self.correct_predictions / definitive if definitive > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_scenarios": self.total_scenarios,
            "correct_predictions": self.correct_predictions,
            "incorrect_predictions": self.incorrect_predictions,
            "unknown_predictions": self.unknown_predictions,
            "reasoning_errors": self.reasoning_errors,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "accuracy": self.get_accuracy(),
            "coverage": self.coverage,
            "avg_grounding_time_ms": (
                self.total_grounding_time_ms / self.total_scenarios
                if self.total_scenarios > 0
                else 0.0
            ),
            "avg_solving_time_ms": (
                self.total_solving_time_ms / self.total_scenarios
                if self.total_scenarios > 0
                else 0.0
            ),
        }


class ASPReasoner:
    """
    ASP-based reasoner for legal scenario predictions.

    Uses Clingo to solve ASP programs combining knowledge base rules
    with scenario-specific facts to derive predictions.

    Supports optional conflict resolution for cases where contradictory
    predictions (both enforceable and unenforceable) are derived.
    """

    # Target predicates for prediction extraction
    POSITIVE_PREDICATES = {"enforceable", "valid", "entitled", "acquired"}
    NEGATIVE_PREDICATES = {"unenforceable", "invalid", "not_entitled", "not_acquired"}

    def __init__(
        self,
        positive_predicates: Optional[Set[str]] = None,
        negative_predicates: Optional[Set[str]] = None,
        conflict_resolver: Optional["ConflictResolver"] = None,
        rules_with_confidence: Optional[List[Tuple[str, float]]] = None,
    ):
        """
        Initialize ASP reasoner.

        Args:
            positive_predicates: Predicates indicating positive outcome
            negative_predicates: Predicates indicating negative outcome
            conflict_resolver: Optional resolver for contradictory predictions
            rules_with_confidence: Optional list of (rule, confidence) tuples
                                   for conflict resolution weighting
        """
        self.positive_predicates = positive_predicates or self.POSITIVE_PREDICATES
        self.negative_predicates = negative_predicates or self.NEGATIVE_PREDICATES
        self.conflict_resolver = conflict_resolver
        self.rules_with_confidence = rules_with_confidence or []
        self.stats = ReasoningStats()

    def reason(
        self,
        knowledge_base: List[str],
        scenario_facts: str,
        max_models: int = 1,
    ) -> ReasoningResult:
        """
        Perform ASP reasoning on a scenario.

        Args:
            knowledge_base: List of ASP rules in the knowledge base
            scenario_facts: ASP facts describing the scenario
            max_models: Maximum number of models to compute (0 = all)

        Returns:
            ReasoningResult with prediction and diagnostic information
        """
        import time

        # Combine knowledge base and scenario facts
        program = self._build_program(knowledge_base, scenario_facts)

        derived_atoms: Set[str] = set()
        rules_fired: List[str] = []
        answer_set_count = 0
        grounding_time_ms = 0.0
        solving_time_ms = 0.0

        try:
            # Create Clingo control
            ctl = clingo.Control(["0"] if max_models == 0 else [str(max_models)])

            # Add program
            ctl.add("base", [], program)

            # Ground
            start_ground = time.perf_counter()
            ctl.ground([("base", [])])
            grounding_time_ms = (time.perf_counter() - start_ground) * 1000

            # Solve
            start_solve = time.perf_counter()

            def on_model(model: clingo.Model) -> None:
                nonlocal answer_set_count, derived_atoms
                answer_set_count += 1
                # Get all shown atoms
                for symbol in model.symbols(shown=True):
                    derived_atoms.add(str(symbol))

            with ctl.solve(on_model=on_model, yield_=True) as handle:
                for _ in handle:
                    pass
                solve_result = handle.get()

            solving_time_ms = (time.perf_counter() - start_solve) * 1000

            # Identify which rules fired (contributed to the answer set)
            rules_fired = self._identify_fired_rules(knowledge_base, derived_atoms)

            # Extract prediction from derived atoms (with conflict resolution)
            (
                prediction,
                confidence,
                conflict_resolved,
                resolution_method,
                resolution_details,
            ) = self._extract_prediction(derived_atoms, rules_fired)

            return ReasoningResult(
                prediction=prediction,
                confidence=confidence,
                satisfiable=bool(solve_result.satisfiable),
                answer_set_count=answer_set_count,
                derived_atoms=derived_atoms,
                rules_fired=rules_fired,
                grounding_time_ms=grounding_time_ms,
                solving_time_ms=solving_time_ms,
                conflict_resolved=conflict_resolved,
                resolution_method=resolution_method,
                resolution_details=resolution_details,
            )

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"ASP reasoning error: {error_msg}")
            return ReasoningResult(
                prediction="unknown",
                confidence=0.0,
                satisfiable=False,
                answer_set_count=0,
                derived_atoms=set(),
                rules_fired=[],
                grounding_time_ms=grounding_time_ms,
                solving_time_ms=solving_time_ms,
                error=error_msg,
            )

    def _build_program(self, knowledge_base: List[str], scenario_facts: str) -> str:
        """Build complete ASP program from knowledge base and facts."""
        program_parts = []

        # Add knowledge base rules
        if knowledge_base:
            program_parts.append("% === Knowledge Base Rules ===")
            for rule in knowledge_base:
                # Clean and add rule
                rule = rule.strip()
                if rule and not rule.startswith("%"):
                    program_parts.append(rule)

        # Add scenario facts
        if scenario_facts:
            program_parts.append("\n% === Scenario Facts ===")
            program_parts.append(scenario_facts.strip())

        return "\n".join(program_parts)

    def _extract_prediction(
        self,
        derived_atoms: Set[str],
        fired_rules: Optional[List[str]] = None,
    ) -> Tuple[str, float, bool, Optional[str], Dict]:
        """
        Extract prediction from derived atoms.

        Args:
            derived_atoms: Atoms derived by ASP reasoning
            fired_rules: Rules that contributed to the derivation

        Returns:
            Tuple of (prediction, confidence, conflict_resolved, resolution_method, details)
        """
        positive_found = False
        negative_found = False
        positive_entities: Set[str] = set()
        negative_entities: Set[str] = set()

        for atom in derived_atoms:
            # Extract predicate name (before the first parenthesis)
            pred_name = atom.split("(")[0] if "(" in atom else atom

            if pred_name in self.positive_predicates:
                positive_found = True
                # Extract entity
                match = re.search(r"\(([^,)]+)", atom)
                if match:
                    positive_entities.add(match.group(1))
            elif pred_name in self.negative_predicates:
                negative_found = True
                match = re.search(r"\(([^,)]+)", atom)
                if match:
                    negative_entities.add(match.group(1))

        # Determine prediction
        if positive_found and not negative_found:
            return "enforceable", 0.9, False, None, {}
        elif negative_found and not positive_found:
            return "unenforceable", 0.9, False, None, {}
        elif positive_found and negative_found:
            # Conflicting atoms
            logger.warning(f"Conflicting predictions found: {derived_atoms}")

            # Try to resolve with conflict resolver if available
            if self.conflict_resolver:
                # Find conflicting entities
                conflicting = positive_entities & negative_entities

                if conflicting:
                    entity = list(conflicting)[0]  # Resolve first conflict

                    # Get rules with confidence
                    rules_conf = self.rules_with_confidence
                    if not rules_conf and fired_rules:
                        # Use fired rules with default confidence
                        rules_conf = [(r, 0.8) for r in fired_rules]

                    resolution = self.conflict_resolver.resolve(
                        derived_atoms, rules_conf, entity
                    )

                    logger.info(
                        f"Conflict resolved for {entity}: "
                        f"{resolution.prediction} via {resolution.method}"
                    )

                    return (
                        resolution.prediction,
                        resolution.confidence,
                        resolution.conflict_detected,
                        resolution.method,
                        resolution.resolution_details,
                    )

            # No resolver or couldn't resolve
            return "unknown", 0.3, True, "unresolved", {}
        else:
            # No relevant atoms derived
            return "unknown", 0.0, False, None, {}

    def _identify_fired_rules(
        self, knowledge_base: List[str], derived_atoms: Set[str]
    ) -> List[str]:
        """
        Identify which rules from the knowledge base fired.

        A rule is considered "fired" if its head appears in the derived atoms.
        """
        fired = []
        for rule in knowledge_base:
            if ":-" in rule:
                # Extract head
                head = rule.split(":-")[0].strip()
                # Check if head matches any derived atom (ignoring arguments)
                head_pred = head.split("(")[0] if "(" in head else head
                for atom in derived_atoms:
                    atom_pred = atom.split("(")[0] if "(" in atom else atom
                    if head_pred == atom_pred:
                        fired.append(rule)
                        break
        return fired

    def make_prediction(
        self,
        knowledge_base: List[str],
        scenario_facts: str,
        expected_outcome: Optional[str] = None,
    ) -> tuple[str, float]:
        """
        Make a prediction for a scenario and update stats.

        Args:
            knowledge_base: List of ASP rules
            scenario_facts: ASP facts for the scenario
            expected_outcome: Optional expected outcome for stats tracking

        Returns:
            Tuple of (prediction, confidence)
        """
        result = self.reason(knowledge_base, scenario_facts)

        # Update stats
        correct = None
        if expected_outcome and result.prediction != "unknown":
            correct = result.prediction == expected_outcome
        self.stats.update(result, correct)

        return result.prediction, result.confidence

    def reset_stats(self) -> None:
        """Reset reasoning statistics."""
        self.stats = ReasoningStats()

    def get_stats(self) -> ReasoningStats:
        """Get current reasoning statistics."""
        return self.stats
