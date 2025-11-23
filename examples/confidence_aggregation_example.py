"""
Working example demonstrating confidence aggregation.

This example shows how to aggregate confidence scores from multiple
validation sources into a single weighted score.
"""

from loft.validation.confidence_aggregator import ConfidenceAggregator

# Create aggregator with default weights
aggregator = ConfidenceAggregator()

print("Default weights:")
for source, weight in aggregator.get_weights().items():
    print(f"  {source}: {weight:.2f}")

print("\n" + "=" * 50)
print("Example 1: High-quality rule with all validation sources")
print("=" * 50)

confidence1 = aggregator.aggregate(
    generation_confidence=0.87,  # LLM was confident
    syntax_valid=True,  # Passes syntax check
    semantic_valid=True,  # Passes semantic check
    empirical_accuracy=0.82,  # 82% accuracy on test cases
    consensus_strength=0.90,  # High agreement among voters
)

print(confidence1.explanation())

print("\n" + "=" * 50)
print("Example 2: Lower-quality rule with missing empirical data")
print("=" * 50)

confidence2 = aggregator.aggregate(
    generation_confidence=0.65,  # LLM was less confident
    syntax_valid=True,  # Passes syntax
    semantic_valid=False,  # Fails semantic validation
    # No empirical data available
    # No consensus data available
)

print(confidence2.explanation())

print("\n" + "=" * 50)
print("Example 3: High variance (unreliable) case")
print("=" * 50)

confidence3 = aggregator.aggregate(
    generation_confidence=0.95,  # LLM very confident
    syntax_valid=True,
    semantic_valid=True,
    empirical_accuracy=0.55,  # But actual performance is poor!
    consensus_strength=0.60,  # And low agreement
)

print(confidence3.explanation())

# Custom weights example
print("\n" + "=" * 50)
print("Example 4: Custom weights (prioritize empirical)")
print("=" * 50)

custom_aggregator = ConfidenceAggregator(
    weights={
        "generation": 0.05,
        "syntax": 0.05,
        "semantic": 0.10,
        "empirical": 0.70,  # Much higher weight on actual performance
        "consensus": 0.10,
    }
)

confidence4 = custom_aggregator.aggregate(
    generation_confidence=0.90,
    syntax_valid=True,
    semantic_valid=True,
    empirical_accuracy=0.75,
    consensus_strength=0.85,
)

print(f"With custom weights (empirical=70%):")
print(f"  Overall confidence: {confidence4.score:.2f}")
print(f"  Empirical contribution: {0.70 * 0.75:.2f}")
