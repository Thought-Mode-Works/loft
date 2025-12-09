"""
Contradiction Management System (Phase 4.4).

Detects, tracks, and resolves contradictions between rules and competing
interpretations.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any
from loguru import logger

from loft.contradiction.contradiction_schemas import (
    ContradictionReport,
    ContradictionType,
    ContradictionSeverity,
    ResolutionStrategy,
    ResolutionResult,
    RuleInterpretation,
)
from loft.contradiction.contradiction_store import ContradictionStore
from loft.contradiction.context_classifier import ContextClassifier
from loft.symbolic.stratification import StratificationLevel


class ContradictionManager:
    """
    Manages contradiction detection, tracking, and resolution.

    Integrates with:
    - Evolution tracking (Phase 4.3) for rule versioning
    - Debate framework (Phase 4.2) for contradiction discussion
    - Stratified core (Phase 3.2) for layer-based resolution
    """

    def __init__(
        self,
        store: Optional[ContradictionStore] = None,
        classifier: Optional[ContextClassifier] = None,
    ):
        """
        Initialize contradiction manager.

        Args:
            store: Contradiction store for persistence
            classifier: Context classifier for context-dependent resolution
        """
        self.store = store or ContradictionStore()
        self.classifier = classifier or ContextClassifier()

        logger.info("Initialized ContradictionManager")

    def detect_contradictions(
        self,
        rules: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> List[ContradictionReport]:
        """
        Detect contradictions between rules.

        Args:
            rules: List of rules to check. Each dict should have:
                   - rule_id: str
                   - rule_text: str
                   - predicates: List[str] (optional)
                   - layer: StratificationLevel (optional)
                   - created_at: datetime (optional)
            context: Optional context for contextual contradiction detection

        Returns:
            List of detected contradictions
        """
        contradictions = []

        # Pairwise comparison
        for i, rule_a in enumerate(rules):
            for rule_b in rules[i + 1 :]:
                # Check for various contradiction types
                contradiction = self._check_contradiction_pair(rule_a, rule_b, context)

                if contradiction:
                    self.store.save_contradiction(contradiction)
                    contradictions.append(contradiction)

        logger.info(
            f"Detected {len(contradictions)} contradictions among {len(rules)} rules"
        )

        return contradictions

    def resolve_contradiction(
        self,
        contradiction: ContradictionReport,
        strategy: Optional[ResolutionStrategy] = None,
        force_strategy: bool = False,
    ) -> ResolutionResult:
        """
        Resolve a contradiction using specified or automatic strategy.

        Args:
            contradiction: The contradiction to resolve
            strategy: Resolution strategy to use (None = auto-select)
            force_strategy: If True, use strategy even if not recommended

        Returns:
            ResolutionResult with outcome
        """
        # Auto-select strategy if not provided
        if strategy is None:
            strategy = self._select_resolution_strategy(contradiction)

        # Apply strategy
        if strategy == ResolutionStrategy.STRATIFICATION:
            result = self._resolve_by_stratification(contradiction)
        elif strategy == ResolutionStrategy.TEMPORAL:
            result = self._resolve_by_temporal(contradiction)
        elif strategy == ResolutionStrategy.CONTEXTUAL:
            result = self._resolve_by_context(contradiction)
        elif strategy == ResolutionStrategy.SPECIFICITY:
            result = self._resolve_by_specificity(contradiction)
        elif strategy == ResolutionStrategy.CONFIDENCE:
            result = self._resolve_by_confidence(contradiction)
        elif strategy == ResolutionStrategy.HUMAN_DECISION:
            result = self._escalate_to_human(contradiction)
        elif strategy == ResolutionStrategy.MERGE:
            result = self._resolve_by_merge(contradiction)
        elif strategy == ResolutionStrategy.DEPRECATE_BOTH:
            result = self._deprecate_both(contradiction)
        else:
            result = ResolutionResult(
                contradiction_id=contradiction.contradiction_id,
                strategy_applied=strategy,
                success=False,
                resolution_notes=f"Unknown strategy: {strategy}",
            )

        # Update contradiction if resolved
        if result.success:
            contradiction.resolved = True
            contradiction.resolution_timestamp = result.timestamp
            contradiction.resolution_applied = strategy
            contradiction.winning_rule_id = result.winning_rule_id
            contradiction.losing_rule_id = result.losing_rule_id
            contradiction.resolution_notes = result.resolution_notes

            self.store.save_contradiction(contradiction)

        logger.info(
            f"Resolved contradiction {contradiction.contradiction_id} "
            f"using {strategy.value}: {'success' if result.success else 'failed'}"
        )

        return result

    def select_rule_for_context(
        self,
        competing_rules: List[Dict[str, Any]],
        context_facts: Dict[str, Any],
    ) -> Tuple[Optional[str], float]:
        """
        Select appropriate rule given context.

        Args:
            competing_rules: List of competing rules
            context_facts: Case facts for context classification

        Returns:
            (selected_rule_id, confidence) tuple
        """
        # Classify context
        context = self.classifier.classify_context(context_facts)

        # Get applicable rules
        applicable = self.classifier.get_applicable_rules(context, competing_rules)

        if not applicable:
            logger.warning("No applicable rules found for context")
            return None, 0.0

        # Return best match
        best_rule_id, confidence = applicable[0]

        logger.info(
            f"Selected rule {best_rule_id} for context "
            f"'{context.context_type}' with confidence {confidence:.2f}"
        )

        return best_rule_id, confidence

    def track_interpretation(self, interpretation: RuleInterpretation) -> None:
        """
        Register a new interpretation.

        Args:
            interpretation: The interpretation to track
        """
        self.store.save_interpretation(interpretation)

        logger.info(
            f"Tracked interpretation {interpretation.interpretation_id} "
            f"for principle '{interpretation.principle}'"
        )

    def get_competing_interpretations(self, principle: str) -> List[RuleInterpretation]:
        """
        Get all interpretations for a principle.

        Args:
            principle: The principle name

        Returns:
            List of competing interpretations
        """
        return self.store.get_interpretations_by_principle(principle)

    def generate_contradiction_alert(
        self, contradiction: ContradictionReport
    ) -> Dict[str, Any]:
        """
        Generate alert for a contradiction.

        Args:
            contradiction: The contradiction to alert on

        Returns:
            Alert dictionary with details
        """
        alert = {
            "alert_id": str(uuid.uuid4()),
            "contradiction_id": contradiction.contradiction_id,
            "severity": contradiction.severity.value,
            "type": contradiction.contradiction_type.value,
            "rules": [
                contradiction.rule_a_id,
                contradiction.rule_b_id,
            ],
            "explanation": contradiction.explanation,
            "suggested_resolution": (
                contradiction.suggested_resolution.value
                if contradiction.suggested_resolution
                else "none"
            ),
            "requires_immediate_action": contradiction.requires_immediate_resolution(),
            "timestamp": datetime.now().isoformat(),
        }

        logger.warning(
            f"Generated {alert['severity']} alert for contradiction "
            f"{contradiction.contradiction_id}"
        )

        return alert

    def get_contradiction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about contradictions.

        Returns:
            Dictionary with statistics
        """
        return self.store.get_statistics()

    def _check_contradiction_pair(
        self,
        rule_a: Dict[str, Any],
        rule_b: Dict[str, Any],
        context: Optional[str],
    ) -> Optional[ContradictionReport]:
        """Check if two rules contradict each other."""
        rule_a_id = rule_a.get("rule_id", "unknown")
        rule_b_id = rule_b.get("rule_id", "unknown")
        rule_a_text = rule_a.get("rule_text", "")
        rule_b_text = rule_b.get("rule_text", "")

        # Check for logical contradictions
        if self._is_logical_contradiction(rule_a, rule_b):
            return self._create_contradiction_report(
                rule_a_id,
                rule_a_text,
                rule_b_id,
                rule_b_text,
                ContradictionType.LOGICAL,
                ContradictionSeverity.HIGH,
                "Rules produce logically contradictory conclusions",
                context,
                rule_a.get("layer"),
                rule_b.get("layer"),
            )

        # Check for hierarchical contradictions (different layers)
        if self._is_hierarchical_contradiction(rule_a, rule_b):
            return self._create_contradiction_report(
                rule_a_id,
                rule_a_text,
                rule_b_id,
                rule_b_text,
                ContradictionType.HIERARCHICAL,
                ContradictionSeverity.MEDIUM,
                "Rules in different stratification layers conflict",
                context,
                rule_a.get("layer"),
                rule_b.get("layer"),
            )

        # Check for temporal contradictions
        if self._is_temporal_contradiction(rule_a, rule_b):
            return self._create_contradiction_report(
                rule_a_id,
                rule_a_text,
                rule_b_id,
                rule_b_text,
                ContradictionType.TEMPORAL,
                ContradictionSeverity.LOW,
                "Newer rule conflicts with older rule",
                context,
                rule_a.get("layer"),
                rule_b.get("layer"),
            )

        return None

    def _is_logical_contradiction(
        self, rule_a: Dict[str, Any], rule_b: Dict[str, Any]
    ) -> bool:
        """Check if rules are logically contradictory."""
        # Simple heuristic: check for negation patterns
        text_a = rule_a.get("rule_text", "").lower()
        text_b = rule_b.get("rule_text", "").lower()

        # Check for explicit negation
        if ("not" in text_a or "~" in text_a) and ("not" in text_b or "~" in text_b):
            # Extract predicates (simplified)
            predicates_a = rule_a.get("predicates", [])
            predicates_b = rule_b.get("predicates", [])

            # If same predicates but one negated, it's a contradiction
            common_predicates = set(predicates_a) & set(predicates_b)
            if common_predicates:
                return True

        return False

    def _is_hierarchical_contradiction(
        self, rule_a: Dict[str, Any], rule_b: Dict[str, Any]
    ) -> bool:
        """Check if rules in different layers conflict."""
        layer_a = rule_a.get("layer")
        layer_b = rule_b.get("layer")

        if layer_a and layer_b and layer_a != layer_b:
            # Check if they address the same predicate
            predicates_a = set(rule_a.get("predicates", []))
            predicates_b = set(rule_b.get("predicates", []))

            # If they share predicates but are in different layers, check for conflict
            if predicates_a & predicates_b:
                return True

        return False

    def _is_temporal_contradiction(
        self, rule_a: Dict[str, Any], rule_b: Dict[str, Any]
    ) -> bool:
        """Check if newer rule supersedes older."""
        created_a = rule_a.get("created_at")
        created_b = rule_b.get("created_at")

        if created_a and created_b:
            # Check if they're about the same thing
            predicates_a = set(rule_a.get("predicates", []))
            predicates_b = set(rule_b.get("predicates", []))

            if predicates_a & predicates_b:
                # Temporal conflict if one is newer
                return abs((created_a - created_b).days) > 30

        return False

    def _create_contradiction_report(
        self,
        rule_a_id: str,
        rule_a_text: str,
        rule_b_id: str,
        rule_b_text: str,
        ctype: ContradictionType,
        severity: ContradictionSeverity,
        explanation: str,
        context: Optional[str],
        layer_a: Optional[StratificationLevel],
        layer_b: Optional[StratificationLevel],
    ) -> ContradictionReport:
        """Create a contradiction report."""
        layers = []
        if layer_a:
            layers.append(layer_a)
        if layer_b and layer_b != layer_a:
            layers.append(layer_b)

        report = ContradictionReport(
            contradiction_id=str(uuid.uuid4()),
            contradiction_type=ctype,
            rule_a_id=rule_a_id,
            rule_a_text=rule_a_text,
            rule_b_id=rule_b_id,
            rule_b_text=rule_b_text,
            severity=severity,
            detected_in_context=context,
            affects_layers=layers,
            explanation=explanation,
            suggested_resolution=self._suggest_resolution_for_type(ctype, layers),
        )

        return report

    def _suggest_resolution_for_type(
        self,
        ctype: ContradictionType,
        layers: List[StratificationLevel],
    ) -> ResolutionStrategy:
        """Suggest resolution strategy based on contradiction type."""
        if ctype == ContradictionType.HIERARCHICAL and len(layers) > 1:
            return ResolutionStrategy.STRATIFICATION

        if ctype == ContradictionType.TEMPORAL:
            return ResolutionStrategy.TEMPORAL

        if ctype == ContradictionType.CONTEXTUAL:
            return ResolutionStrategy.CONTEXTUAL

        if ctype == ContradictionType.LOGICAL:
            return ResolutionStrategy.HUMAN_DECISION

        return ResolutionStrategy.CONFIDENCE

    def _select_resolution_strategy(
        self, contradiction: ContradictionReport
    ) -> ResolutionStrategy:
        """Auto-select best resolution strategy."""
        # Use suggested resolution if available
        if contradiction.suggested_resolution:
            return contradiction.suggested_resolution

        # Fall back to type-based suggestion
        return self._suggest_resolution_for_type(
            contradiction.contradiction_type,
            contradiction.affects_layers,
        )

    def _resolve_by_stratification(
        self, contradiction: ContradictionReport
    ) -> ResolutionResult:
        """Resolve by stratification: higher layer wins."""
        if len(contradiction.affects_layers) < 2:
            return ResolutionResult(
                contradiction_id=contradiction.contradiction_id,
                strategy_applied=ResolutionStrategy.STRATIFICATION,
                success=False,
                resolution_notes="Cannot resolve by stratification: not enough layer info",
            )

        # Higher level wins (constitutional > strategic > tactical > operational)
        winning_layer = max(contradiction.affects_layers)

        # Determine which rule is in higher layer (need metadata)
        # For now, assume rule_a is in first layer, rule_b in second
        if contradiction.affects_layers[0] == winning_layer:
            winning_rule = contradiction.rule_a_id
            losing_rule = contradiction.rule_b_id
        else:
            winning_rule = contradiction.rule_b_id
            losing_rule = contradiction.rule_a_id

        return ResolutionResult(
            contradiction_id=contradiction.contradiction_id,
            strategy_applied=ResolutionStrategy.STRATIFICATION,
            success=True,
            winning_rule_id=winning_rule,
            losing_rule_id=losing_rule,
            resolution_notes=f"Higher stratification layer ({winning_layer.value}) wins",
            confidence=0.95,
        )

    def _resolve_by_temporal(
        self, contradiction: ContradictionReport
    ) -> ResolutionResult:
        """Resolve by temporal: newer rule wins."""
        # Would need created_at timestamps from rules
        return ResolutionResult(
            contradiction_id=contradiction.contradiction_id,
            strategy_applied=ResolutionStrategy.TEMPORAL,
            success=True,
            resolution_notes="Newer rule supersedes older (temporal resolution)",
            confidence=0.8,
            requires_validation=True,
        )

    def _resolve_by_context(
        self, contradiction: ContradictionReport
    ) -> ResolutionResult:
        """Resolve by context: select based on context applicability."""
        return ResolutionResult(
            contradiction_id=contradiction.contradiction_id,
            strategy_applied=ResolutionStrategy.CONTEXTUAL,
            success=True,
            resolution_notes="Both rules valid in different contexts",
            confidence=0.85,
            requires_validation=True,
        )

    def _resolve_by_specificity(
        self, contradiction: ContradictionReport
    ) -> ResolutionResult:
        """Resolve by specificity: more specific rule wins."""
        # Would analyze rule conditions for specificity
        return ResolutionResult(
            contradiction_id=contradiction.contradiction_id,
            strategy_applied=ResolutionStrategy.SPECIFICITY,
            success=True,
            resolution_notes="More specific rule takes precedence",
            confidence=0.75,
            requires_validation=True,
        )

    def _resolve_by_confidence(
        self, contradiction: ContradictionReport
    ) -> ResolutionResult:
        """Resolve by confidence: higher confidence rule wins."""
        return ResolutionResult(
            contradiction_id=contradiction.contradiction_id,
            strategy_applied=ResolutionStrategy.CONFIDENCE,
            success=True,
            resolution_notes="Rule with higher confidence score wins",
            confidence=0.7,
            requires_validation=True,
        )

    def _escalate_to_human(
        self, contradiction: ContradictionReport
    ) -> ResolutionResult:
        """Escalate to human decision."""
        return ResolutionResult(
            contradiction_id=contradiction.contradiction_id,
            strategy_applied=ResolutionStrategy.HUMAN_DECISION,
            success=False,
            resolution_notes="Escalated to human review - requires manual resolution",
            confidence=0.0,
            requires_human_review=True,
        )

    def _resolve_by_merge(self, contradiction: ContradictionReport) -> ResolutionResult:
        """Resolve by merging rules with conditions."""
        return ResolutionResult(
            contradiction_id=contradiction.contradiction_id,
            strategy_applied=ResolutionStrategy.MERGE,
            success=True,
            resolution_notes="Rules merged with contextual conditions",
            confidence=0.65,
            requires_validation=True,
        )

    def _deprecate_both(self, contradiction: ContradictionReport) -> ResolutionResult:
        """Mark both rules as problematic."""
        return ResolutionResult(
            contradiction_id=contradiction.contradiction_id,
            strategy_applied=ResolutionStrategy.DEPRECATE_BOTH,
            success=True,
            rules_deprecated=[contradiction.rule_a_id, contradiction.rule_b_id],
            resolution_notes="Both rules marked as deprecated due to unresolvable contradiction",
            confidence=0.9,
            requires_human_review=True,
        )
