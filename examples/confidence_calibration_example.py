"""
Working example demonstrating confidence calibration.

This example shows how to use the ConfidenceCalibrator to improve
confidence score accuracy by calibrating against historical data.
"""

from loft.validation.confidence_calibrator import ConfidenceCalibrator

# Create calibrator
calibrator = ConfidenceCalibrator(min_calibration_points=10)

# Example training data: rule_id -> (predicted_confidence, actual_accuracy)
# In practice, this would come from historical validation runs
training_data = {
    "rule_1": (0.90, 0.85),  # Model was slightly overconfident
    "rule_2": (0.85, 0.82),
    "rule_3": (0.88, 0.84),
    "rule_4": (0.75, 0.72),
    "rule_5": (0.70, 0.68),
    "rule_6": (0.92, 0.88),
    "rule_7": (0.65, 0.63),
    "rule_8": (0.80, 0.78),
    "rule_9": (0.95, 0.90),
    "rule_10": (0.78, 0.75),
    "rule_11": (0.83, 0.80),
    "rule_12": (0.72, 0.70),
    "rule_13": (0.87, 0.84),
    "rule_14": (0.68, 0.66),
    "rule_15": (0.91, 0.87),
}

# Record historical predictions vs actual accuracy
print("Recording calibration data...")
for rule_id, (predicted, actual) in training_data.items():
    calibrator.record(predicted, actual, rule_id)
    print(f"  {rule_id}: predicted={predicted:.2f}, actual={actual:.2f}")

print(f"\nRecorded {len(training_data)} calibration points\n")

# Calibrate using isotonic regression
print("Calibrating with isotonic regression...")
report = calibrator.calibrate(method="isotonic")

print("\n" + report.summary())

# Apply calibration to new predictions
print("\n" + "=" * 50)
print("Applying calibration to new predictions:")
print("=" * 50)

new_predictions = [0.88, 0.75, 0.92, 0.65]

for pred in new_predictions:
    calibrated = calibrator.calibrate_score(pred)
    print(f"Raw: {pred:.2f} -> Calibrated: {calibrated:.2f}")

# Show calibration improvement
print("\n" + "=" * 50)
print("Calibration Improvement:")
print("=" * 50)
print(f"MSE improved by: {(report.before_mse - report.after_mse):.4f}")
print(f"ECE improved by: {(report.before_ece - report.after_ece):.4f}")
print(f"Overall improvement: {report.improvement:.1%}")
