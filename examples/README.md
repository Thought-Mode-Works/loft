# Confidence System Examples

This directory contains working examples demonstrating the confidence scoring and calibration system.

## Running the Examples

All examples can be run directly from the project root:

```bash
# Set PYTHONPATH to find the loft module
export PYTHONPATH=/path/to/loft:$PYTHONPATH

# Or run with PYTHONPATH inline
PYTHONPATH=.:$PYTHONPATH python3 examples/confidence_calibration_example.py
```

## Available Examples

### 1. Confidence Calibration (`confidence_calibration_example.py`)

Demonstrates how to calibrate confidence scores using historical data to improve accuracy.

**Features:**
- Recording calibration points (predicted vs actual)
- Isotonic regression calibration
- Before/after metrics (MSE, ECE)
- Applying calibration to new predictions

**Run:**
```bash
python3 examples/confidence_calibration_example.py
```

**Expected Output:**
```
Calibration Report (isotonic)
  Data Points: 15
  Before: MSE=0.0011, ECE=0.0313
  After:  MSE=0.0000, ECE=0.0000
  Improvement: 100.0%
```

### 2. Confidence Aggregation (`confidence_aggregation_example.py`)

Shows how to combine confidence from multiple validation sources into a weighted score.

**Features:**
- Default weight configuration
- Multi-source aggregation (generation, syntax, semantic, empirical, consensus)
- Variance-based reliability detection
- Custom weight configuration

**Run:**
```bash
python3 examples/confidence_aggregation_example.py
```

**Expected Output:**
```
Overall Confidence: 0.89
Breakdown:
  Generation: 0.87 (weight: 0.15, contribution: 0.13)
  Empirical: 0.82 (weight: 0.40, contribution: 0.33)
  ...
Variance: 0.005 (reliable)
```

### 3. Confidence Gating (`confidence_gating_example.py`)

Demonstrates decision-making based on stratified confidence thresholds.

**Features:**
- Layer-specific thresholds (operational: 0.6, tactical: 0.8, strategic: 0.9)
- Impact-based threshold adjustments
- Accept/reject/flag decisions
- Batch evaluation
- Decision statistics

**Run:**
```bash
python3 examples/confidence_gating_example.py
```

**Expected Output:**
```
Decision: ACCEPT
Reasoning: Confidence 0.88 meets threshold 0.80 for tactical layer
```

### 4. Confidence Tracking (`confidence_tracking_example.py`)

Shows how to track confidence scores over time and analyze trends.

**Features:**
- Historical tracking
- Category-based analysis
- Trend identification (mean, median, std)
- Low-confidence case detection
- High-variance case detection

**Run:**
```bash
python3 examples/confidence_tracking_example.py
```

**Expected Output:**
```
Confidence Trends
  Mean: 0.77 ± 0.11
  Median: 0.75
  Reliability: 60.0% (15/25)
```

## Quick Start

Try all examples at once:

```bash
# From project root
export PYTHONPATH=.:$PYTHONPATH

for example in examples/confidence_*.py; do
    echo "Running $example..."
    python3 "$example" 2>&1 | grep -v "^2025-"  # Filter debug logs
    echo ""
done
```

## Integration Example

Here's how these components work together in a real validation pipeline:

```python
from loft.validation.validation_pipeline import ValidationPipeline
from loft.validation.confidence_aggregator import ConfidenceAggregator
from loft.validation.confidence_calibrator import ConfidenceCalibrator
from loft.validation.confidence_gate import ConfidenceGate
from loft.validation.confidence_tracker import ConfidenceTracker

# Setup
pipeline = ValidationPipeline(stages=["syntactic", "semantic", "empirical"])
aggregator = ConfidenceAggregator()
calibrator = ConfidenceCalibrator(min_calibration_points=10)
gate = ConfidenceGate()
tracker = ConfidenceTracker()

# Validate rule
report = pipeline.validate(rule_text, test_cases)

# Aggregate confidence
confidence = aggregator.aggregate_from_report(report)

# Calibrate (if calibrator is trained)
if calibrator.is_calibrated():
    calibrated_score = calibrator.calibrate_score(confidence.score)
    # Update confidence with calibrated score

# Make gating decision
decision = gate.should_accept(
    confidence=confidence,
    target_layer="tactical",
    rule_impact="medium"
)

# Track for analysis
tracker.record(confidence, category="contract_rules")

# Act on decision
if decision.action == "accept":
    # Deploy rule
    pass
elif decision.action == "flag_for_review":
    # Queue for human review
    pass
else:  # reject
    # Discard rule
    pass
```

## Dependencies

These examples require:
- Python 3.9+
- scikit-learn (for isotonic regression in calibrator)
- pydantic
- loguru

Install with:
```bash
pip install scikit-learn pydantic loguru
```
