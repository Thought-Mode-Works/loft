"""
Working example demonstrating confidence tracking.

This example shows how to track confidence scores over time and
analyze trends for different rule categories.
"""

from loft.validation.confidence_schemas import AggregatedConfidence
from loft.validation.confidence_tracker import ConfidenceTracker

# Create tracker
tracker = ConfidenceTracker(history_limit=1000)

print("=" * 70)
print("Recording confidence scores for different rule categories")
print("=" * 70)

# Simulate recording confidence scores for contract rules (high quality)
print("\nContract rules (high quality):")
for i in range(10):
    confidence = AggregatedConfidence(
        score=0.85 + (i % 3) * 0.03,  # 0.85-0.91
        components={"generation": 0.85, "empirical": 0.88},
        weights={"generation": 0.4, "empirical": 0.6},
        variance=0.02,
        is_reliable=True,
    )
    tracker.record(confidence, category="contract_rules")
    print(f"  Rule {i + 1}: {confidence.score:.2f}")

# Simulate recording confidence scores for experimental rules (lower quality)
print("\nExperimental rules (lower quality):")
for i in range(10):
    confidence = AggregatedConfidence(
        score=0.60 + (i % 4) * 0.05,  # 0.60-0.75
        components={"generation": 0.65, "empirical": 0.58},
        weights={"generation": 0.4, "empirical": 0.6},
        variance=0.08,
        is_reliable=False,
    )
    tracker.record(confidence, category="experimental_rules")
    print(f"  Rule {i + 1}: {confidence.score:.2f}")

# Simulate some mixed quality rules
print("\nMixed quality rules:")
for i in range(5):
    score = 0.70 if i % 2 == 0 else 0.85
    confidence = AggregatedConfidence(
        score=score,
        components={"generation": score, "empirical": score},
        weights={"generation": 0.5, "empirical": 0.5},
        variance=0.05,
        is_reliable=True,
    )
    tracker.record(confidence, category="mixed_rules")
    print(f"  Rule {i + 1}: {confidence.score:.2f}")

print("\n" + "=" * 70)
print("Overall Trends")
print("=" * 70)

trends = tracker.get_trends()
print(trends.summary())

print("\n" + "=" * 70)
print("Trends by Category")
print("=" * 70)

comparison = tracker.get_category_comparison()
for category, stats in comparison.items():
    print(f"\n{category}:")
    print(f"  Count: {stats['count']}")
    print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Reliability rate: {stats['reliability_rate']:.1%}")

print("\n" + "=" * 70)
print("Low Confidence Cases (threshold < 0.7)")
print("=" * 70)

low_conf_cases = tracker.get_low_confidence_cases(threshold=0.7)
print(f"\nFound {len(low_conf_cases)} low-confidence cases:")
for i, conf in enumerate(low_conf_cases[:5], 1):  # Show first 5
    print(f"  {i}. Score: {conf.score:.2f}, Variance: {conf.variance:.3f}")

print("\n" + "=" * 70)
print("High Variance Cases (variance > 0.05)")
print("=" * 70)

high_var_cases = tracker.get_high_variance_cases(variance_threshold=0.05)
print(f"\nFound {len(high_var_cases)} high-variance cases:")
for i, conf in enumerate(high_var_cases[:5], 1):  # Show first 5
    print(
        f"  {i}. Score: {conf.score:.2f}, Variance: {conf.variance:.3f}, Reliable: {conf.is_reliable}"
    )

print("\n" + "=" * 70)
print("Recent Scores (last 5)")
print("=" * 70)

recent = tracker.get_recent_scores(count=5)
for i, conf in enumerate(recent, 1):
    print(f"  {i}. Score: {conf.score:.2f}, Timestamp: {conf.timestamp}")

print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)

summary = tracker.get_summary()
for key, value in summary.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")
