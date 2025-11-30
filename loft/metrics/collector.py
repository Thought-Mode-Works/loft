"""
Metrics collector for batch processing.

Provides real-time metrics collection, aggregation, and persistence
for monitoring batch learning performance and quality.
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Callable, Dict, Iterator, List, Optional


@dataclass
class MetricsSample:
    """A single metrics measurement point."""

    timestamp: datetime
    metric_name: str
    value: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsSample":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metric_name=data["metric_name"],
            value=data["value"],
            unit=data.get("unit", ""),
            tags=data.get("tags", {}),
        )


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, name: str, collector: Optional["MetricsCollector"] = None):
        """Initialize timer.

        Args:
            name: Name of the operation being timed
            collector: Optional MetricsCollector to record the timing
        """
        self.name = name
        self.collector = collector
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "PerformanceTimer":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record."""
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000

        if self.collector:
            self.collector.record_timing(self.name, self.elapsed_ms)


@dataclass
class AggregatedMetrics:
    """Aggregated statistics for a metric."""

    metric_name: str
    count: int
    total: float
    mean: float
    median: float
    min_value: float
    max_value: float
    std_dev: float
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "count": self.count,
            "total": self.total,
            "mean": self.mean,
            "median": self.median,
            "min": self.min_value,
            "max": self.max_value,
            "std_dev": self.std_dev,
            "unit": self.unit,
        }


class MetricsCollector:
    """Collects and aggregates metrics during batch processing."""

    def __init__(
        self,
        batch_id: str,
        output_dir: Optional[Path] = None,
        flush_interval: int = 100,
    ):
        """Initialize metrics collector.

        Args:
            batch_id: Identifier for this batch run
            output_dir: Directory for persisting metrics
            flush_interval: Number of samples before auto-flush
        """
        self.batch_id = batch_id
        self.output_dir = output_dir
        self.flush_interval = flush_interval

        self.started_at = datetime.now()
        self.samples: List[MetricsSample] = []
        self.timings: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}

        # Milestone tracking for trend analysis
        self.milestones: List[Dict[str, Any]] = []

        # Callbacks for real-time monitoring
        self._on_sample: Optional[Callable[[MetricsSample], None]] = None
        self._on_milestone: Optional[Callable[[Dict[str, Any]], None]] = None

        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_callbacks(
        self,
        on_sample: Optional[Callable[[MetricsSample], None]] = None,
        on_milestone: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Set callbacks for real-time monitoring.

        Args:
            on_sample: Called when a new sample is recorded
            on_milestone: Called when a milestone is recorded
        """
        self._on_sample = on_sample
        self._on_milestone = on_milestone

    def record_sample(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> MetricsSample:
        """Record a single metrics sample.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            tags: Optional tags for filtering

        Returns:
            The recorded sample
        """
        sample = MetricsSample(
            timestamp=datetime.now(),
            metric_name=name,
            value=value,
            unit=unit,
            tags=tags or {},
        )
        self.samples.append(sample)

        if self._on_sample:
            self._on_sample(sample)

        # Auto-flush if buffer is full
        if len(self.samples) >= self.flush_interval and self.output_dir:
            self._flush_samples()

        return sample

    def record_timing(self, name: str, elapsed_ms: float) -> None:
        """Record a timing measurement.

        Args:
            name: Name of the timed operation
            elapsed_ms: Elapsed time in milliseconds
        """
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed_ms)

        self.record_sample(name, elapsed_ms, unit="ms", tags={"type": "timing"})

    def increment_counter(self, name: str, amount: int = 1) -> int:
        """Increment a counter.

        Args:
            name: Counter name
            amount: Amount to increment

        Returns:
            New counter value
        """
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += amount
        return self.counters[name]

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value.

        Args:
            name: Gauge name
            value: Current value
        """
        self.gauges[name] = value
        self.record_sample(name, value, tags={"type": "gauge"})

    @contextmanager
    def time_operation(self, name: str) -> Iterator[PerformanceTimer]:
        """Context manager for timing operations.

        Args:
            name: Name of the operation

        Yields:
            PerformanceTimer instance
        """
        timer = PerformanceTimer(name, self)
        with timer:
            yield timer

    def record_milestone(
        self,
        milestone_name: str,
        rules_count: int,
        cases_processed: int,
        accuracy: float,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a milestone for trend analysis.

        Args:
            milestone_name: Name of the milestone (e.g., "10_rules", "50_rules")
            rules_count: Number of rules at this milestone
            cases_processed: Number of cases processed
            accuracy: Current accuracy
            additional_metrics: Optional additional metrics

        Returns:
            The milestone record
        """
        milestone = {
            "milestone_name": milestone_name,
            "timestamp": datetime.now().isoformat(),
            "rules_count": rules_count,
            "cases_processed": cases_processed,
            "accuracy": accuracy,
            "elapsed_seconds": (datetime.now() - self.started_at).total_seconds(),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            **(additional_metrics or {}),
        }

        # Add timing aggregates
        milestone["timing_aggregates"] = {}
        for timing_name, values in self.timings.items():
            if values:
                milestone["timing_aggregates"][timing_name] = {
                    "count": len(values),
                    "mean_ms": mean(values),
                    "median_ms": median(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                }

        self.milestones.append(milestone)

        if self._on_milestone:
            self._on_milestone(milestone)

        return milestone

    def get_timing_stats(self, name: str) -> Optional[AggregatedMetrics]:
        """Get aggregated statistics for a timing metric.

        Args:
            name: Timing name

        Returns:
            Aggregated statistics or None if no data
        """
        values = self.timings.get(name)
        if not values:
            return None

        return AggregatedMetrics(
            metric_name=name,
            count=len(values),
            total=sum(values),
            mean=mean(values),
            median=median(values),
            min_value=min(values),
            max_value=max(values),
            std_dev=stdev(values) if len(values) > 1 else 0.0,
            unit="ms",
        )

    def get_all_timing_stats(self) -> Dict[str, AggregatedMetrics]:
        """Get aggregated statistics for all timing metrics.

        Returns:
            Dictionary of timing name to aggregated stats
        """
        return {
            name: stats
            for name in self.timings
            if (stats := self.get_timing_stats(name)) is not None
        }

    def _flush_samples(self) -> None:
        """Flush samples to disk."""
        if not self.output_dir or not self.samples:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        samples_file = self.output_dir / f"samples_{timestamp}.json"

        with open(samples_file, "w") as f:
            json.dump([s.to_dict() for s in self.samples], f, indent=2)

        self.samples = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert collector state to dictionary.

        Returns:
            Complete collector state
        """
        timing_stats = {
            name: stats.to_dict() for name, stats in self.get_all_timing_stats().items()
        }

        return {
            "batch_id": self.batch_id,
            "started_at": self.started_at.isoformat(),
            "elapsed_seconds": (datetime.now() - self.started_at).total_seconds(),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "timing_stats": timing_stats,
            "milestones": self.milestones,
            "total_samples": len(self.samples),
        }

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save complete metrics to file.

        Args:
            filepath: Optional path, defaults to output_dir/metrics.json

        Returns:
            Path where metrics were saved
        """
        if filepath is None:
            if self.output_dir is None:
                raise ValueError("No output directory configured")
            filepath = self.output_dir / "metrics.json"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Flush any remaining samples
        self._flush_samples()

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "MetricsCollector":
        """Load metrics from file.

        Args:
            filepath: Path to metrics file

        Returns:
            MetricsCollector with loaded data
        """
        with open(filepath) as f:
            data = json.load(f)

        collector = cls(
            batch_id=data["batch_id"],
            output_dir=filepath.parent,
        )
        collector.started_at = datetime.fromisoformat(data["started_at"])
        collector.counters = data.get("counters", {})
        collector.gauges = data.get("gauges", {})
        collector.milestones = data.get("milestones", [])

        return collector
