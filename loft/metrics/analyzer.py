"""
Metrics analyzer for batch processing.

Provides trend analysis, anomaly detection, and reporting
capabilities for understanding system behavior at scale.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


@dataclass
class TrendPoint:
    """A single point in a trend analysis."""

    timestamp: datetime
    rules_count: int
    accuracy: float
    cases_processed: int
    avg_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "rules_count": self.rules_count,
            "accuracy": self.accuracy,
            "cases_processed": self.cases_processed,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class TrendAnalysis:
    """Analysis of metric trends over time."""

    metric_name: str
    data_points: List[TrendPoint]

    # Trend direction
    trend_direction: str  # "improving", "degrading", "stable"
    trend_slope: float

    # Key observations
    accuracy_trend: str
    latency_trend: str
    rules_growth_rate: float

    # Correlation analysis
    accuracy_vs_rules_correlation: float
    latency_vs_rules_correlation: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "data_points": [p.to_dict() for p in self.data_points],
            "trend_direction": self.trend_direction,
            "trend_slope": self.trend_slope,
            "accuracy_trend": self.accuracy_trend,
            "latency_trend": self.latency_trend,
            "rules_growth_rate": self.rules_growth_rate,
            "accuracy_vs_rules_correlation": self.accuracy_vs_rules_correlation,
            "latency_vs_rules_correlation": self.latency_vs_rules_correlation,
        }


@dataclass
class Anomaly:
    """A detected anomaly in metrics."""

    timestamp: datetime
    metric_name: str
    expected_value: float
    actual_value: float
    deviation_factor: float
    severity: str  # "low", "medium", "high"
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "deviation_factor": self.deviation_factor,
            "severity": self.severity,
            "description": self.description,
        }


@dataclass
class AnomalyReport:
    """Report of detected anomalies."""

    analyzed_at: datetime
    total_anomalies: int
    anomalies: List[Anomaly]
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analyzed_at": self.analyzed_at.isoformat(),
            "total_anomalies": self.total_anomalies,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "high_severity_count": self.high_severity_count,
            "medium_severity_count": self.medium_severity_count,
            "low_severity_count": self.low_severity_count,
            "recommendations": self.recommendations,
        }


@dataclass
class ScaleReport:
    """Comprehensive report on system behavior at scale."""

    batch_id: str
    generated_at: datetime

    # Volume summary
    total_cases: int
    total_rules: int
    total_runtime_seconds: float

    # Performance summary
    avg_case_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    peak_memory_mb: float

    # Quality summary
    initial_accuracy: float
    final_accuracy: float
    accuracy_change: float
    consistency_score: float

    # Trend analysis
    trends: TrendAnalysis

    # Anomalies
    anomaly_report: AnomalyReport

    # Bottlenecks
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "generated_at": self.generated_at.isoformat(),
            "total_cases": self.total_cases,
            "total_rules": self.total_rules,
            "total_runtime_seconds": self.total_runtime_seconds,
            "avg_case_latency_ms": self.avg_case_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "initial_accuracy": self.initial_accuracy,
            "final_accuracy": self.final_accuracy,
            "accuracy_change": self.accuracy_change,
            "consistency_score": self.consistency_score,
            "trends": self.trends.to_dict(),
            "anomaly_report": self.anomaly_report.to_dict(),
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
        }

    def save(self, filepath: Path) -> None:
        """Save report to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class MetricsAnalyzer:
    """Analyzer for batch processing metrics."""

    def __init__(
        self,
        anomaly_threshold_std: float = 2.0,
        min_data_points: int = 5,
    ):
        """Initialize analyzer.

        Args:
            anomaly_threshold_std: Standard deviations for anomaly detection
            min_data_points: Minimum points for statistical analysis
        """
        self.anomaly_threshold_std = anomaly_threshold_std
        self.min_data_points = min_data_points

    def analyze_trends(
        self,
        milestones: List[Dict[str, Any]],
    ) -> TrendAnalysis:
        """Analyze trends from milestone data.

        Args:
            milestones: List of milestone records

        Returns:
            Trend analysis results
        """
        if not milestones:
            return self._empty_trend_analysis()

        # Extract data points
        data_points = []
        for m in milestones:
            # Get average latency from timing aggregates
            timing_agg = m.get("timing_aggregates", {})
            avg_latency = 0.0
            if timing_agg:
                latencies = [t.get("mean_ms", 0) for t in timing_agg.values()]
                if latencies:
                    avg_latency = mean(latencies)

            point = TrendPoint(
                timestamp=datetime.fromisoformat(m["timestamp"]),
                rules_count=m.get("rules_count", 0),
                accuracy=m.get("accuracy", 0.0),
                cases_processed=m.get("cases_processed", 0),
                avg_latency_ms=avg_latency,
            )
            data_points.append(point)

        # Calculate trends
        accuracies = [p.accuracy for p in data_points]
        latencies = [p.avg_latency_ms for p in data_points]
        rules_counts = [p.rules_count for p in data_points]

        # Accuracy trend
        if len(accuracies) >= 2:
            accuracy_change = accuracies[-1] - accuracies[0]
            if accuracy_change > 0.05:
                accuracy_trend = "improving"
            elif accuracy_change < -0.05:
                accuracy_trend = "degrading"
            else:
                accuracy_trend = "stable"
        else:
            accuracy_trend = "insufficient_data"

        # Latency trend
        if len(latencies) >= 2:
            latency_change = latencies[-1] - latencies[0]
            if latency_change > latencies[0] * 0.2:  # >20% increase
                latency_trend = "degrading"
            elif latency_change < -latencies[0] * 0.2:  # >20% decrease
                latency_trend = "improving"
            else:
                latency_trend = "stable"
        else:
            latency_trend = "insufficient_data"

        # Rules growth rate
        if len(rules_counts) >= 2 and data_points[0].rules_count > 0:
            total_time = (
                data_points[-1].timestamp - data_points[0].timestamp
            ).total_seconds()
            if total_time > 0:
                rules_growth_rate = (rules_counts[-1] - rules_counts[0]) / (
                    total_time / 60
                )
            else:
                rules_growth_rate = 0.0
        else:
            rules_growth_rate = 0.0

        # Calculate correlations
        accuracy_vs_rules = self._calculate_correlation(rules_counts, accuracies)
        latency_vs_rules = self._calculate_correlation(rules_counts, latencies)

        # Overall trend direction
        if accuracy_trend == "improving" and latency_trend != "degrading":
            trend_direction = "improving"
        elif accuracy_trend == "degrading" or latency_trend == "degrading":
            trend_direction = "degrading"
        else:
            trend_direction = "stable"

        # Calculate trend slope (accuracy per 100 rules)
        if len(data_points) >= 2 and rules_counts[-1] != rules_counts[0]:
            trend_slope = (accuracies[-1] - accuracies[0]) / (
                (rules_counts[-1] - rules_counts[0]) / 100
            )
        else:
            trend_slope = 0.0

        return TrendAnalysis(
            metric_name="batch_performance",
            data_points=data_points,
            trend_direction=trend_direction,
            trend_slope=trend_slope,
            accuracy_trend=accuracy_trend,
            latency_trend=latency_trend,
            rules_growth_rate=rules_growth_rate,
            accuracy_vs_rules_correlation=accuracy_vs_rules,
            latency_vs_rules_correlation=latency_vs_rules,
        )

    def detect_anomalies(
        self,
        timing_stats: Dict[str, Dict[str, Any]],
        milestones: List[Dict[str, Any]],
    ) -> AnomalyReport:
        """Detect anomalies in metrics.

        Args:
            timing_stats: Timing statistics
            milestones: Milestone data

        Returns:
            Anomaly report
        """
        anomalies: List[Anomaly] = []

        # Analyze timing anomalies
        for metric_name, stats in timing_stats.items():
            mean_val = stats.get("mean", 0)
            std_val = stats.get("std_dev", 0)
            max_val = stats.get("max", 0)

            if std_val > 0 and mean_val > 0:
                deviation = (max_val - mean_val) / std_val
                if deviation > self.anomaly_threshold_std:
                    severity = self._classify_severity(deviation)
                    anomalies.append(
                        Anomaly(
                            timestamp=datetime.now(),
                            metric_name=metric_name,
                            expected_value=mean_val,
                            actual_value=max_val,
                            deviation_factor=deviation,
                            severity=severity,
                            description=f"{metric_name} max value ({max_val:.2f}ms) is {deviation:.1f} std devs above mean ({mean_val:.2f}ms)",
                        )
                    )

        # Analyze accuracy drops in milestones
        if len(milestones) >= 2:
            accuracies = [m.get("accuracy", 0) for m in milestones]
            for i in range(1, len(accuracies)):
                drop = accuracies[i - 1] - accuracies[i]
                if drop > 0.1:  # >10% accuracy drop
                    anomalies.append(
                        Anomaly(
                            timestamp=datetime.fromisoformat(
                                milestones[i]["timestamp"]
                            ),
                            metric_name="accuracy",
                            expected_value=accuracies[i - 1],
                            actual_value=accuracies[i],
                            deviation_factor=drop / 0.05,  # Normalized to 5% baseline
                            severity="high" if drop > 0.2 else "medium",
                            description=f"Accuracy dropped by {drop * 100:.1f}% between milestones",
                        )
                    )

        # Generate recommendations
        recommendations = self._generate_anomaly_recommendations(anomalies)

        return AnomalyReport(
            analyzed_at=datetime.now(),
            total_anomalies=len(anomalies),
            anomalies=anomalies,
            high_severity_count=sum(1 for a in anomalies if a.severity == "high"),
            medium_severity_count=sum(1 for a in anomalies if a.severity == "medium"),
            low_severity_count=sum(1 for a in anomalies if a.severity == "low"),
            recommendations=recommendations,
        )

    def generate_scale_report(
        self,
        batch_id: str,
        metrics_data: Dict[str, Any],
        profiler_data: Optional[Dict[str, Any]] = None,
    ) -> ScaleReport:
        """Generate comprehensive scale report.

        Args:
            batch_id: Batch identifier
            metrics_data: Metrics collector data
            profiler_data: Optional profiler data

        Returns:
            Scale report
        """
        milestones = metrics_data.get("milestones", [])
        timing_stats = metrics_data.get("timing_stats", {})
        counters = metrics_data.get("counters", {})

        # Analyze trends
        trends = self.analyze_trends(milestones)

        # Detect anomalies
        anomaly_report = self.detect_anomalies(timing_stats, milestones)

        # Extract performance metrics
        latencies = []
        for stat in timing_stats.values():
            if "mean" in stat:
                latencies.append(stat["mean"])

        if latencies:
            avg_latency = mean(latencies)
            sorted_latencies = sorted(latencies)
            p50 = self._percentile(sorted_latencies, 50)
            p95 = self._percentile(sorted_latencies, 95)
            p99 = self._percentile(sorted_latencies, 99)
        else:
            avg_latency = p50 = p95 = p99 = 0.0

        # Extract accuracy metrics
        if milestones:
            initial_accuracy = milestones[0].get("accuracy", 0.0)
            final_accuracy = milestones[-1].get("accuracy", 0.0)
        else:
            initial_accuracy = final_accuracy = 0.0

        # Extract from profiler
        peak_memory = 0.0
        bottlenecks = []
        if profiler_data:
            peak_memory = profiler_data.get("summary", {}).get("peak_memory_mb", 0.0)

            # Get bottlenecks from slow operations
            by_operation = profiler_data.get("summary", {}).get("by_operation", {})
            for op_name, op_stats in by_operation.items():
                if op_stats.get("avg_time_ms", 0) > 1000:  # >1s average
                    bottlenecks.append(
                        {
                            "operation": op_name,
                            "avg_time_ms": op_stats["avg_time_ms"],
                            "count": op_stats["count"],
                            "total_time_ms": op_stats["total_time_ms"],
                        }
                    )

        # Generate recommendations
        recommendations = self._generate_scale_recommendations(
            trends, anomaly_report, avg_latency, peak_memory
        )

        return ScaleReport(
            batch_id=batch_id,
            generated_at=datetime.now(),
            total_cases=counters.get("cases_processed", 0),
            total_rules=counters.get("rules_accepted", 0),
            total_runtime_seconds=metrics_data.get("elapsed_seconds", 0.0),
            avg_case_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            peak_memory_mb=peak_memory,
            initial_accuracy=initial_accuracy,
            final_accuracy=final_accuracy,
            accuracy_change=final_accuracy - initial_accuracy,
            consistency_score=1.0,  # Placeholder
            trends=trends,
            anomaly_report=anomaly_report,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

    def _empty_trend_analysis(self) -> TrendAnalysis:
        """Create empty trend analysis."""
        return TrendAnalysis(
            metric_name="batch_performance",
            data_points=[],
            trend_direction="insufficient_data",
            trend_slope=0.0,
            accuracy_trend="insufficient_data",
            latency_trend="insufficient_data",
            rules_growth_rate=0.0,
            accuracy_vs_rules_correlation=0.0,
            latency_vs_rules_correlation=0.0,
        )

    def _calculate_correlation(
        self, x_values: List[float], y_values: List[float]
    ) -> float:
        """Calculate Pearson correlation coefficient.

        Args:
            x_values: X values
            y_values: Y values

        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(x_values) < self.min_data_points or len(x_values) != len(y_values):
            return 0.0

        mean_x = mean(x_values)
        mean_y = mean(y_values)

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        denominator_x = sum((x - mean_x) ** 2 for x in x_values) ** 0.5
        denominator_y = sum((y - mean_y) ** 2 for y in y_values) ** 0.5

        if denominator_x == 0 or denominator_y == 0:
            return 0.0

        return numerator / (denominator_x * denominator_y)

    def _classify_severity(self, deviation: float) -> str:
        """Classify anomaly severity based on deviation.

        Args:
            deviation: Standard deviations from mean

        Returns:
            Severity level
        """
        if deviation > 4:
            return "high"
        elif deviation > 3:
            return "medium"
        else:
            return "low"

    def _percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile value.

        Args:
            sorted_values: Sorted list of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not sorted_values:
            return 0.0

        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = f + 1 if f < len(sorted_values) - 1 else f

        if f == c:
            return sorted_values[f]

        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    def _generate_anomaly_recommendations(self, anomalies: List[Anomaly]) -> List[str]:
        """Generate recommendations based on anomalies.

        Args:
            anomalies: Detected anomalies

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for latency anomalies
        latency_anomalies = [a for a in anomalies if "latency" in a.metric_name.lower()]
        if latency_anomalies:
            recommendations.append(
                "High latency variance detected. Consider profiling slow operations "
                "and implementing caching for repeated computations."
            )

        # Check for accuracy drops
        accuracy_anomalies = [a for a in anomalies if a.metric_name == "accuracy"]
        if accuracy_anomalies:
            recommendations.append(
                "Accuracy drops detected. Review recently added rules for "
                "conflicts or overly specific conditions."
            )

        # High severity anomalies
        high_severity = [a for a in anomalies if a.severity == "high"]
        if len(high_severity) > 3:
            recommendations.append(
                "Multiple high-severity anomalies detected. System may be "
                "experiencing instability. Consider pausing batch processing."
            )

        return recommendations

    def _generate_scale_recommendations(
        self,
        trends: TrendAnalysis,
        anomaly_report: AnomalyReport,
        avg_latency: float,
        peak_memory: float,
    ) -> List[str]:
        """Generate scale-specific recommendations.

        Args:
            trends: Trend analysis
            anomaly_report: Anomaly report
            avg_latency: Average latency
            peak_memory: Peak memory usage

        Returns:
            List of recommendations
        """
        recommendations = []

        # Trend-based recommendations
        if trends.accuracy_trend == "degrading":
            recommendations.append(
                "Accuracy is degrading as rules accumulate. Consider implementing "
                "rule pruning or conflict resolution strategies."
            )

        if trends.latency_trend == "degrading":
            recommendations.append(
                "Latency increasing with scale. Consider optimizing ASP solver "
                "queries or implementing rule indexing."
            )

        if trends.latency_vs_rules_correlation > 0.8:
            recommendations.append(
                "Strong correlation between rule count and latency. "
                "Performance may degrade significantly at larger scales."
            )

        # Performance-based recommendations
        if avg_latency > 5000:  # >5s average
            recommendations.append(
                "Average case latency exceeds 5 seconds. Consider parallel "
                "processing or batching optimizations."
            )

        if peak_memory > 2048:  # >2GB
            recommendations.append(
                "Peak memory usage exceeds 2GB. Consider implementing "
                "memory-efficient data structures or garbage collection tuning."
            )

        # Add anomaly recommendations
        recommendations.extend(anomaly_report.recommendations)

        return recommendations


def load_and_analyze_metrics(metrics_path: Path) -> ScaleReport:
    """Load metrics file and generate analysis report.

    Args:
        metrics_path: Path to metrics JSON file

    Returns:
        Scale report
    """
    with open(metrics_path) as f:
        metrics_data = json.load(f)

    analyzer = MetricsAnalyzer()
    return analyzer.generate_scale_report(
        batch_id=metrics_data.get("batch_id", "unknown"),
        metrics_data=metrics_data,
    )
