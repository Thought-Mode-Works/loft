"""
Working example demonstrating confidence gating.

This example shows how to make accept/reject/flag decisions based on
confidence scores and stratification layers.
"""

from loft.validation.confidence_schemas import AggregatedConfidence
from loft.validation.confidence_gate import ConfidenceGate

# Create gate with default thresholds
gate = ConfidenceGate()

print("Default thresholds:")
for layer, threshold in gate.get_thresholds().items():
    print(f"  {layer}: {threshold:.2f}")

# Example confidence scores
high_confidence = AggregatedConfidence(
    score=0.88,
    components={"generation": 0.85, "empirical": 0.90},
    weights={"generation": 0.4, "empirical": 0.6},
    variance=0.02,
    is_reliable=True,
)

medium_confidence = AggregatedConfidence(
    score=0.77,
    components={"generation": 0.75, "empirical": 0.79},
    weights={"generation": 0.4, "empirical": 0.6},
    variance=0.02,
    is_reliable=True,
)

low_confidence = AggregatedConfidence(
    score=0.55,
    components={"generation": 0.50, "empirical": 0.60},
    weights={"generation": 0.4, "empirical": 0.6},
    variance=0.05,
    is_reliable=True,
)

unreliable_confidence = AggregatedConfidence(
    score=0.82,
    components={"generation": 0.95, "empirical": 0.65},
    weights={"generation": 0.4, "empirical": 0.6},
    variance=0.25,
    is_reliable=False,  # High variance
)

print("\n" + "=" * 70)
print("Example 1: High confidence rule for tactical layer")
print("=" * 70)

decision1 = gate.should_accept(
    confidence=high_confidence, target_layer="tactical", rule_impact="medium"
)

print(f"Decision: {decision1.action.upper()}")
print(f"Reasoning: {decision1.reasoning}")
print(f"Confidence: {decision1.calibrated_confidence:.2f} vs threshold {decision1.threshold:.2f}")

print("\n" + "=" * 70)
print("Example 2: Medium confidence (borderline) for tactical layer")
print("=" * 70)

decision2 = gate.should_accept(
    confidence=medium_confidence, target_layer="tactical", rule_impact="medium"
)

print(f"Decision: {decision2.action.upper()}")
print(f"Reasoning: {decision2.reasoning}")
print(f"Confidence: {decision2.calibrated_confidence:.2f} vs threshold {decision2.threshold:.2f}")

print("\n" + "=" * 70)
print("Example 3: Low confidence for tactical layer")
print("=" * 70)

decision3 = gate.should_accept(
    confidence=low_confidence, target_layer="tactical", rule_impact="medium"
)

print(f"Decision: {decision3.action.upper()}")
print(f"Reasoning: {decision3.reasoning}")
print(f"Confidence: {decision3.calibrated_confidence:.2f} vs threshold {decision3.threshold:.2f}")

print("\n" + "=" * 70)
print("Example 4: Impact-based threshold adjustment")
print("=" * 70)

# Same confidence, but different impacts
decision_low_impact = gate.should_accept(
    confidence=medium_confidence,
    target_layer="tactical",
    rule_impact="low",  # Lowers threshold by 0.1
)

decision_high_impact = gate.should_accept(
    confidence=medium_confidence,
    target_layer="tactical",
    rule_impact="high",  # Raises threshold by 0.1
)

print(
    f"Low impact:  {decision_low_impact.action:20} (threshold: {decision_low_impact.threshold:.2f})"
)
print(
    f"High impact: {decision_high_impact.action:20} (threshold: {decision_high_impact.threshold:.2f})"
)

print("\n" + "=" * 70)
print("Example 5: Different layers with same confidence")
print("=" * 70)

for layer in ["operational", "tactical", "strategic"]:
    decision = gate.should_accept(
        confidence=high_confidence, target_layer=layer, rule_impact="medium"
    )
    print(f"{layer:15} (threshold={decision.threshold:.2f}): {decision.action}")

print("\n" + "=" * 70)
print("Example 6: Unreliable confidence (high variance)")
print("=" * 70)

decision6 = gate.should_accept(
    confidence=unreliable_confidence, target_layer="tactical", rule_impact="medium"
)

print(f"Decision: {decision6.action.upper()}")
print(f"Reasoning: {decision6.reasoning}")
print(f"Note: High variance ({unreliable_confidence.variance:.2f}) triggered warning")

print("\n" + "=" * 70)
print("Example 7: Batch evaluation")
print("=" * 70)

confidences = [high_confidence, medium_confidence, low_confidence]
decisions = gate.batch_evaluate(confidences, target_layer="tactical")

for i, decision in enumerate(decisions, 1):
    print(f"{i}. {decision.action:20} (confidence: {decision.calibrated_confidence:.2f})")

# Get statistics
stats = gate.get_statistics(decisions)
print("\nStatistics:")
print(f"  Acceptance rate: {stats['acceptance_rate']:.1%}")
print(f"  Rejection rate:  {stats['rejection_rate']:.1%}")
print(f"  Flag rate:       {stats['flag_rate']:.1%}")
