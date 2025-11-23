"""
Confidence gate for decision making based on confidence thresholds.

Gates rule acceptance based on stratified confidence thresholds,
accounting for rule impact and target layer.
"""

from typing import Literal, Optional
from loguru import logger

from loft.validation.confidence_schemas import AggregatedConfidence, GateDecision


class ConfidenceGate:
    """
    Gate decisions based on confidence thresholds.

    Uses stratified thresholds where higher-impact layers require
    higher confidence for automatic acceptance.
    """

    # Default thresholds from CLAUDE.md
    THRESHOLDS = {
        "operational": 0.6,  # Frequent updates, lower risk
        "tactical": 0.8,  # Moderate impact
        "strategic": 0.9,  # High impact, requires high confidence
        "constitutional": 1.0,  # Requires human approval
    }

    # Impact multipliers (adjust threshold based on rule impact)
    IMPACT_ADJUSTMENTS = {
        "low": -0.1,  # Lower threshold for low-impact rules
        "medium": 0.0,  # No adjustment
        "high": 0.1,  # Higher threshold for high-impact rules
    }

    def __init__(
        self,
        thresholds: Optional[dict[str, float]] = None,
        flag_margin: float = 0.05,
    ):
        """
        Initialize confidence gate.

        Args:
            thresholds: Custom thresholds by layer (default uses THRESHOLDS)
            flag_margin: Confidence margin below threshold to flag for review
        """
        self.thresholds = thresholds or self.THRESHOLDS.copy()
        self.flag_margin = flag_margin

        logger.debug(
            f"Initialized ConfidenceGate with thresholds: {self.thresholds}, "
            f"flag_margin={flag_margin}"
        )

    def should_accept(
        self,
        confidence: AggregatedConfidence,
        target_layer: str,
        rule_impact: str = "medium",
        use_calibrated: bool = True,
    ) -> GateDecision:
        """
        Determine if rule should be accepted based on confidence.

        Args:
            confidence: Aggregated confidence for the rule
            target_layer: Target stratification layer
            rule_impact: Estimated impact (low/medium/high)
            use_calibrated: Use calibrated score (if available) vs raw score

        Returns:
            GateDecision with action and reasoning

        Example:
            >>> gate = ConfidenceGate()
            >>> confidence = AggregatedConfidence(
            ...     score=0.85,
            ...     components={"generation": 0.85, "syntax": 1.0},
            ...     weights={"generation": 0.5, "syntax": 0.5},
            ...     variance=0.05,
            ...     is_reliable=True
            ... )
            >>> decision = gate.should_accept(
            ...     confidence, target_layer="tactical", rule_impact="medium"
            ... )
            >>> assert decision.action == "accept"
        """
        # Get base threshold for layer
        if target_layer not in self.thresholds:
            logger.warning(f"Unknown layer '{target_layer}', defaulting to 'tactical' threshold")
            target_layer = "tactical"

        base_threshold = self.thresholds[target_layer]

        # Adjust threshold based on rule impact
        impact_adjustment = self.IMPACT_ADJUSTMENTS.get(rule_impact, 0.0)
        threshold = base_threshold + impact_adjustment

        # Ensure threshold stays in valid range
        threshold = max(0.0, min(1.0, threshold))

        # Use raw or calibrated confidence
        # (In this implementation, we only have aggregated confidence)
        # In practice, this would be the calibrated score if calibrator was used
        confidence_score = confidence.score
        raw_confidence = confidence.score  # Same for now, would differ with calibrator

        # Decision logic
        action: Literal["accept", "reject", "flag_for_review"]
        if confidence_score >= threshold:
            # High confidence - accept
            action = "accept"
            reasoning = (
                f"Confidence {confidence_score:.2f} meets threshold {threshold:.2f} "
                f"for {target_layer} layer with {rule_impact} impact"
            )

            if not confidence.is_reliable:
                reasoning += " (Warning: high variance among confidence sources)"

        elif confidence_score >= threshold - self.flag_margin:
            # Close to threshold - flag for review
            action = "flag_for_review"
            gap = threshold - confidence_score
            reasoning = (
                f"Confidence {confidence_score:.2f} is {gap:.2f} below threshold {threshold:.2f} "
                f"for {target_layer} layer - flagging for human review"
            )

            if not confidence.is_reliable:
                reasoning += " (High variance among sources suggests uncertainty)"

        else:
            # Low confidence - reject
            action = "reject"
            gap = threshold - confidence_score
            reasoning = (
                f"Confidence {confidence_score:.2f} is {gap:.2f} below threshold {threshold:.2f} "
                f"for {target_layer} layer with {rule_impact} impact"
            )

            if not confidence.is_reliable:
                reasoning += " (High variance indicates disagreement among validation sources)"

        # Special case: constitutional layer always requires review
        if target_layer == "constitutional":
            action = "flag_for_review"
            reasoning = "Constitutional layer requires human approval regardless of confidence"

        logger.info(
            f"Gate decision: {action} (confidence={confidence_score:.2f}, "
            f"threshold={threshold:.2f}, layer={target_layer})"
        )

        return GateDecision(
            action=action,
            reasoning=reasoning,
            raw_confidence=raw_confidence,
            calibrated_confidence=confidence_score,
            threshold=threshold,
            target_layer=target_layer,
            rule_impact=rule_impact,
        )

    def batch_evaluate(
        self,
        confidences: list[AggregatedConfidence],
        target_layer: str,
        rule_impacts: Optional[list[str]] = None,
    ) -> list[GateDecision]:
        """
        Evaluate multiple confidence scores in batch.

        Args:
            confidences: List of confidence scores
            target_layer: Target layer for all rules
            rule_impacts: List of impact levels (default: all "medium")

        Returns:
            List of GateDecision objects

        Example:
            >>> gate = ConfidenceGate()
            >>> decisions = gate.batch_evaluate(
            ...     [confidence1, confidence2],
            ...     target_layer="tactical"
            ... )
        """
        if rule_impacts is None:
            rule_impacts = ["medium"] * len(confidences)

        if len(rule_impacts) != len(confidences):
            raise ValueError(
                f"Length mismatch: {len(confidences)} confidences but {len(rule_impacts)} impacts"
            )

        decisions = []
        for conf, impact in zip(confidences, rule_impacts):
            decision = self.should_accept(conf, target_layer, impact)
            decisions.append(decision)

        return decisions

    def get_statistics(self, decisions: list[GateDecision]) -> dict:
        """
        Get statistics on gate decisions.

        Args:
            decisions: List of gate decisions

        Returns:
            Dict with decision statistics

        Example:
            >>> gate = ConfidenceGate()
            >>> # ... make decisions ...
            >>> stats = gate.get_statistics(decisions)
            >>> assert "acceptance_rate" in stats
        """
        if not decisions:
            return {
                "total": 0,
                "accepted": 0,
                "rejected": 0,
                "flagged": 0,
                "acceptance_rate": 0.0,
                "rejection_rate": 0.0,
                "flag_rate": 0.0,
            }

        total = len(decisions)
        accepted = sum(1 for d in decisions if d.action == "accept")
        rejected = sum(1 for d in decisions if d.action == "reject")
        flagged = sum(1 for d in decisions if d.action == "flag_for_review")

        return {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "flagged": flagged,
            "acceptance_rate": accepted / total,
            "rejection_rate": rejected / total,
            "flag_rate": flagged / total,
            "mean_confidence": sum(d.calibrated_confidence for d in decisions) / total,
            "mean_threshold": sum(d.threshold for d in decisions) / total,
        }

    def update_threshold(self, layer: str, new_threshold: float) -> None:
        """
        Update threshold for a specific layer.

        Args:
            layer: Layer to update
            new_threshold: New threshold value (0.0-1.0)
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0.0, 1.0], got {new_threshold}")

        old_threshold = self.thresholds.get(layer)
        self.thresholds[layer] = new_threshold

        logger.info(f"Updated threshold for {layer} layer: {old_threshold} -> {new_threshold}")

    def get_thresholds(self) -> dict[str, float]:
        """Get current thresholds for all layers."""
        return self.thresholds.copy()
