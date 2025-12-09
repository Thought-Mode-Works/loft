"""
Confidence calibrator for adjusting confidence scores to match empirical accuracy.

Uses isotonic regression or other methods to calibrate LLM confidence scores.
"""

from typing import List, Optional, Callable
import statistics
from loguru import logger

from loft.validation.confidence_schemas import CalibrationPoint, CalibrationReport


class ConfidenceCalibrator:
    """
    Calibrate confidence scores to match empirical accuracy.

    Learns from historical (predicted, actual) pairs to adjust future predictions.
    """

    def __init__(self, min_calibration_points: int = 10):
        """
        Initialize calibrator.

        Args:
            min_calibration_points: Minimum points needed for reliable calibration
        """
        self.calibration_data: List[CalibrationPoint] = []
        self.calibration_function: Optional[Callable] = None
        self.min_calibration_points = min_calibration_points
        self.last_calibration_report: Optional[CalibrationReport] = None

        logger.debug(
            f"Initialized ConfidenceCalibrator (min_points={min_calibration_points})"
        )

    def record(
        self, predicted_confidence: float, actual_accuracy: float, rule_id: str
    ) -> None:
        """
        Record a calibration data point.

        Args:
            predicted_confidence: Model's predicted confidence
            actual_accuracy: Actual accuracy achieved
            rule_id: Identifier for the rule

        Example:
            >>> calibrator = ConfidenceCalibrator()
            >>> calibrator.record(0.9, 0.85, "rule_1")
            >>> calibrator.record(0.6, 0.65, "rule_2")
        """
        point = CalibrationPoint(
            predicted=predicted_confidence, actual=actual_accuracy, rule_id=rule_id
        )
        self.calibration_data.append(point)

        logger.debug(
            f"Recorded calibration point: predicted={predicted_confidence:.2f}, "
            f"actual={actual_accuracy:.2f} for {rule_id}"
        )

    def calibrate(self, method: str = "isotonic") -> CalibrationReport:
        """
        Learn calibration function from historical data.

        Args:
            method: Calibration method ("isotonic", "linear", or "identity")

        Returns:
            CalibrationReport with before/after metrics

        Raises:
            ValueError: If insufficient calibration data

        Example:
            >>> calibrator = ConfidenceCalibrator()
            >>> # Record many points...
            >>> report = calibrator.calibrate(method="isotonic")
            >>> assert report.after_ece < report.before_ece
        """
        if len(self.calibration_data) < self.min_calibration_points:
            raise ValueError(
                f"Need at least {self.min_calibration_points} calibration points, "
                f"have {len(self.calibration_data)}"
            )

        logger.info(
            f"Calibrating with {len(self.calibration_data)} points using {method}"
        )

        predicted = [p.predicted for p in self.calibration_data]
        actual = [p.actual for p in self.calibration_data]

        # Metrics before calibration
        before_mse = self._mse(predicted, actual)
        before_ece = self._expected_calibration_error(predicted, actual)

        # Apply calibration method
        if method == "isotonic":
            self.calibration_function = self._isotonic_calibration(predicted, actual)
        elif method == "linear":
            self.calibration_function = self._linear_calibration(predicted, actual)
        elif method == "identity":
            self.calibration_function = lambda x: x  # No calibration
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        # Metrics after calibration
        calibrated = [self.calibrate_score(p) for p in predicted]
        after_mse = self._mse(calibrated, actual)
        after_ece = self._expected_calibration_error(calibrated, actual)

        report = CalibrationReport(
            method=method,
            num_points=len(self.calibration_data),
            before_mse=before_mse,
            after_mse=after_mse,
            before_ece=before_ece,
            after_ece=after_ece,
        )

        self.last_calibration_report = report

        logger.info(f"Calibration complete: {report.summary()}")

        return report

    def calibrate_score(self, raw_confidence: float) -> float:
        """
        Apply calibration to raw confidence.

        Args:
            raw_confidence: Uncalibrated confidence score

        Returns:
            Calibrated confidence score

        Example:
            >>> calibrator = ConfidenceCalibrator()
            >>> # ... calibrate ...
            >>> calibrated = calibrator.calibrate_score(0.85)
        """
        if self.calibration_function is None:
            logger.warning("No calibration function, returning raw score")
            return raw_confidence

        # Ensure input is in valid range
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        calibrated = float(self.calibration_function(raw_confidence))

        # Ensure output is in valid range
        calibrated = max(0.0, min(1.0, calibrated))

        return calibrated

    def _isotonic_calibration(
        self, predicted: List[float], actual: List[float]
    ) -> Callable:
        """
        Isotonic regression calibration.

        Fits a monotonic function that minimizes squared error.
        This is the recommended method from sklearn.

        Args:
            predicted: Predicted confidences
            actual: Actual accuracies

        Returns:
            Calibration function
        """
        try:
            from sklearn.isotonic import IsotonicRegression

            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(predicted, actual)

            def calibrate_fn(x: float) -> float:
                if isinstance(x, (list, tuple)):
                    return calibrator.predict(x)  # type: ignore
                return float(calibrator.predict([x])[0])

            return calibrate_fn

        except ImportError:
            logger.warning("sklearn not available, falling back to linear calibration")
            return self._linear_calibration(predicted, actual)

    def _linear_calibration(
        self, predicted: List[float], actual: List[float]
    ) -> Callable:
        """
        Linear calibration (Platt scaling).

        Fits a linear function: calibrated = a * predicted + b

        Args:
            predicted: Predicted confidences
            actual: Actual accuracies

        Returns:
            Calibration function
        """
        n = len(predicted)
        if n == 0:
            return lambda x: x

        # Simple linear regression
        mean_pred = statistics.mean(predicted)
        mean_actual = statistics.mean(actual)

        numerator = sum(
            (p - mean_pred) * (a - mean_actual) for p, a in zip(predicted, actual)
        )
        denominator = sum((p - mean_pred) ** 2 for p in predicted)

        if denominator > 0:
            slope = numerator / denominator
            intercept = mean_actual - slope * mean_pred
        else:
            slope = 1.0
            intercept = 0.0

        logger.debug(f"Linear calibration: y = {slope:.3f}x + {intercept:.3f}")

        def calibrate_fn(x: float) -> float:
            if isinstance(x, (list, tuple)):
                return [slope * xi + intercept for xi in x]  # type: ignore
            return slope * x + intercept

        return calibrate_fn

    def _mse(self, predicted: List[float], actual: List[float]) -> float:
        """
        Mean squared error.

        Args:
            predicted: Predicted values
            actual: Actual values

        Returns:
            MSE
        """
        if not predicted:
            return 0.0
        return sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(predicted)

    def _expected_calibration_error(
        self, predicted: List[float], actual: List[float], num_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE).

        Bins predictions, compares average confidence to average accuracy in each bin.

        Args:
            predicted: Predicted confidences
            actual: Actual accuracies
            num_bins: Number of bins for discretization

        Returns:
            ECE score (lower is better)
        """
        if not predicted:
            return 0.0

        bins: List[List[tuple[float, float]]] = [[] for _ in range(num_bins)]

        # Assign points to bins
        for p, a in zip(predicted, actual):
            bin_idx = min(int(p * num_bins), num_bins - 1)
            bins[bin_idx].append((p, a))

        ece = 0.0
        total = len(predicted)

        # Compute weighted error for each bin
        for bin_points in bins:
            if not bin_points:
                continue

            bin_confidence = sum(p for p, _ in bin_points) / len(bin_points)
            bin_accuracy = sum(a for _, a in bin_points) / len(bin_points)
            bin_weight = len(bin_points) / total

            ece += bin_weight * abs(bin_confidence - bin_accuracy)

        return ece

    def get_calibration_data(self) -> List[CalibrationPoint]:
        """Get all calibration data points."""
        return self.calibration_data.copy()

    def clear_calibration_data(self) -> None:
        """Clear all calibration data (useful for testing or reset)."""
        self.calibration_data.clear()
        self.calibration_function = None
        self.last_calibration_report = None
        logger.info("Cleared calibration data")

    def is_calibrated(self) -> bool:
        """Check if calibrator has been calibrated."""
        return self.calibration_function is not None
