"""
Persistence metrics collection and baseline validation.

Provides metrics collection for ASP rule persistence operations:
- Save/load performance timing
- Storage metrics (sizes, counts)
- Metadata integrity verification
- Error tracking and recovery monitoring

Issue #254: Phase 8 baseline validation infrastructure.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from loft.symbolic.stratification import (
    StratificationLevel,
)  # Added StratificationLevel import


@dataclass
class PersistenceMetrics:
    """Metrics for a single persistence operation."""

    # Timing metrics
    save_time_ms: float = 0.0
    load_time_ms: float = 0.0
    snapshot_time_ms: float = 0.0

    # Rule counts
    total_rules: int = 0
    rules_by_layer: Dict[str, int] = field(default_factory=dict)

    # Storage metrics
    file_sizes_bytes: Dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0

    # Snapshot metrics
    snapshot_count: int = 0
    snapshot_total_size_bytes: int = 0

    # Metadata integrity
    rules_with_metadata: int = 0
    metadata_fields_preserved: float = 1.0  # 0.0 - 1.0

    # Error tracking
    save_errors: int = 0
    load_errors: int = 0
    recovery_attempts: int = 0
    recovery_successes: int = 0

    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "save_time_ms": self.save_time_ms,
            "load_time_ms": self.load_time_ms,
            "snapshot_time_ms": self.snapshot_time_ms,
            "total_rules": self.total_rules,
            "rules_by_layer": self.rules_by_layer,
            "file_sizes_bytes": self.file_sizes_bytes,
            "total_size_bytes": self.total_size_bytes,
            "snapshot_count": self.snapshot_count,
            "snapshot_total_size_bytes": self.snapshot_total_size_bytes,
            "rules_with_metadata": self.rules_with_metadata,
            "metadata_fields_preserved": self.metadata_fields_preserved,
            "save_errors": self.save_errors,
            "load_errors": self.load_errors,
            "recovery_attempts": self.recovery_attempts,
            "recovery_successes": self.recovery_successes,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersistenceMetrics":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BaselineReport:
    """Baseline persistence metrics report."""

    # Summary statistics
    total_cycles: int = 0
    total_rules_processed: int = 0

    # Performance averages
    avg_save_time_ms: float = 0.0
    avg_load_time_ms: float = 0.0
    avg_snapshot_time_ms: float = 0.0

    # Storage averages
    avg_rule_size_bytes: float = 0.0
    metadata_overhead_percent: float = 0.0

    # Reliability
    save_error_rate: float = 0.0
    load_error_rate: float = 0.0
    recovery_success_rate: float = 0.0
    metadata_preservation_rate: float = 0.0

    # Scalability (rules -> time)
    scalability_data: List[Dict[str, float]] = field(default_factory=list)

    # Raw metrics
    individual_metrics: List[PersistenceMetrics] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# ASP Persistence Baseline Metrics",
            "",
            f"Generated: {datetime.now().isoformat()}",
            f"Total Cycles: {self.total_cycles}",
            f"Total Rules Processed: {self.total_rules_processed}",
            "",
            "## Performance",
            f"- Average save time: {self.avg_save_time_ms:.2f} ms per cycle",
            f"- Average load time: {self.avg_load_time_ms:.2f} ms per cycle",
            f"- Average snapshot time: {self.avg_snapshot_time_ms:.2f} ms",
            "",
            "## Storage",
            f"- Average rule size: {self.avg_rule_size_bytes:.1f} bytes",
            f"- Metadata overhead: {self.metadata_overhead_percent:.1f}%",
            "",
            "## Reliability",
            f"- Save error rate: {self.save_error_rate * 100:.2f}%",
            f"- Load error rate: {self.load_error_rate * 100:.2f}%",
            f"- Recovery success rate: {self.recovery_success_rate * 100:.1f}%",
            f"- Metadata preservation: {self.metadata_preservation_rate * 100:.1f}%",
            "",
        ]

        if self.scalability_data:
            lines.extend(
                [
                    "## Scalability",
                    "| Rules | Save (ms) | Load (ms) |",
                    "|-------|-----------|-----------|",
                ]
            )
            for entry in self.scalability_data:
                lines.append(
                    f"| {entry.get('rules', 0)} | "
                    f"{entry.get('save_ms', 0):.1f} | "
                    f"{entry.get('load_ms', 0):.1f} |"
                )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_cycles": self.total_cycles,
            "total_rules_processed": self.total_rules_processed,
            "avg_save_time_ms": self.avg_save_time_ms,
            "avg_load_time_ms": self.avg_load_time_ms,
            "avg_snapshot_time_ms": self.avg_snapshot_time_ms,
            "avg_rule_size_bytes": self.avg_rule_size_bytes,
            "metadata_overhead_percent": self.metadata_overhead_percent,
            "save_error_rate": self.save_error_rate,
            "load_error_rate": self.load_error_rate,
            "recovery_success_rate": self.recovery_success_rate,
            "metadata_preservation_rate": self.metadata_preservation_rate,
            "scalability_data": self.scalability_data,
            "individual_metrics": [m.to_dict() for m in self.individual_metrics],
        }


class PersistenceMetricsCollector:
    """
    Collects persistence operation metrics.

    Tracks save/load cycles, measures performance, and validates
    metadata integrity for baseline establishment.

    Example:
        >>> collector = PersistenceMetricsCollector()
        >>> metrics = collector.measure_save_cycle(manager, rules_by_layer)
        >>> print(f"Save took {metrics.save_time_ms:.2f}ms")
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.collected_metrics: List[PersistenceMetrics] = []
        self.error_log: List[Dict[str, Any]] = []  # Explicitly type error_log

    def measure_save_cycle(
        self,
        manager: Any,  # ASPPersistenceManager
        rules_by_layer: Dict[
            StratificationLevel, List[Any]
        ],  # Changed to StratificationLevel
    ) -> PersistenceMetrics:
        """
        Measure a complete save cycle.

        Args:
            manager: ASPPersistenceManager instance
            rules_by_layer: Rules organized by stratification layer

        Returns:
            PersistenceMetrics with save operation data
        """
        metrics = PersistenceMetrics()

        # Count rules
        for layer_name, rules in rules_by_layer.items():
            layer_key = (
                layer_name.value if hasattr(layer_name, "value") else str(layer_name)
            )
            metrics.rules_by_layer[layer_key] = len(rules)
            metrics.total_rules += len(rules)

            # Count rules with metadata
            for rule in rules:
                if hasattr(rule, "metadata") and rule.metadata:
                    metrics.rules_with_metadata += 1

        # Measure save time
        start_time = time.perf_counter()
        try:
            manager.save_all_rules(rules_by_layer, overwrite=True)
        except Exception as e:
            metrics.save_errors += 1
            self.error_log.append(
                {
                    "type": "save_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            logger.warning(f"Save error: {e}")

        metrics.save_time_ms = (time.perf_counter() - start_time) * 1000

        # Measure file sizes
        base_dir = Path(manager.base_dir)
        for layer_file in base_dir.glob("*.lp"):
            if layer_file.exists():
                metrics.file_sizes_bytes[layer_file.name] = layer_file.stat().st_size
                metrics.total_size_bytes += layer_file.stat().st_size

        # Get snapshot stats
        try:
            stats = manager.get_stats()
            metrics.snapshot_count = stats.get("snapshots_count", 0)
        except Exception:
            pass

        self.collected_metrics.append(metrics)
        return metrics

    def measure_load_cycle(
        self,
        manager: Any,  # ASPPersistenceManager
    ) -> PersistenceMetrics:
        """
        Measure a complete load cycle.

        Args:
            manager: ASPPersistenceManager instance

        Returns:
            PersistenceMetrics with load operation data
        """
        metrics = PersistenceMetrics()

        # Measure load time
        start_time = time.perf_counter()
        load_result = None
        try:
            load_result = manager.load_all_rules()
            metrics.load_errors += len(load_result.parsing_errors)
            if load_result.had_errors:
                metrics.recovery_attempts += 1
                if load_result.recovered_layers:
                    metrics.recovery_successes += len(load_result.recovered_layers)

            rules_by_layer = load_result.rules_by_layer

            # Count loaded rules
            for layer, rules in rules_by_layer.items():
                layer_key = layer.value if hasattr(layer, "value") else str(layer)
                metrics.rules_by_layer[layer_key] = len(rules)
                metrics.total_rules += len(rules)

                # Check metadata preservation
                for rule in rules:
                    if hasattr(rule, "metadata") and rule.metadata:
                        metrics.rules_with_metadata += 1

        except Exception as e:
            from loft.persistence.asp_persistence import (
                LoadResult,
            )  # Import here to avoid circular dependency

            metrics.load_errors += 1
            load_result = LoadResult(
                rules_by_layer={},
                parsing_errors=[str(e)],
                had_errors=True,
                recovered_layers=[],
            )
            self.error_log.append(
                {
                    "type": "load_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            logger.warning(f"Load error: {e}")

        metrics.load_time_ms = (time.perf_counter() - start_time) * 1000

        # Calculate metadata preservation rate
        if metrics.total_rules > 0:
            metrics.metadata_fields_preserved = (
                metrics.rules_with_metadata / metrics.total_rules
            )

        self.collected_metrics.append(metrics)
        return metrics

    def measure_snapshot_cycle(
        self,
        manager: Any,  # ASPPersistenceManager
        cycle_number: int,
        description: Optional[str] = None,
    ) -> PersistenceMetrics:
        """
        Measure snapshot creation.

        Args:
            manager: ASPPersistenceManager instance
            cycle_number: Cycle number for snapshot
            description: Optional snapshot description

        Returns:
            PersistenceMetrics with snapshot timing
        """
        metrics = PersistenceMetrics()

        start_time = time.perf_counter()
        try:
            manager.create_snapshot(cycle_number, description)
        except Exception as e:
            metrics.save_errors += 1
            self.error_log.append(
                {
                    "type": "snapshot_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            logger.warning(f"Snapshot error: {e}")

        metrics.snapshot_time_ms = (time.perf_counter() - start_time) * 1000

        # Get updated snapshot stats
        try:
            stats = manager.get_stats()
            metrics.snapshot_count = stats.get("snapshots_count", 0)
        except Exception:
            pass

        self.collected_metrics.append(metrics)
        return metrics

    def verify_roundtrip_integrity(
        self,
        original_rules: Dict[str, List[Any]],
        loaded_rules: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """
        Verify integrity after save/load roundtrip.

        Args:
            original_rules: Rules before save
            loaded_rules: Rules after load

        Returns:
            Dictionary with integrity check results
        """
        results: Dict[str, Any] = {
            "rules_match": True,
            "metadata_preserved": True,
            "missing_rules": [],
            "modified_rules": [],
            "metadata_issues": [],
        }

        for layer_name, original in original_rules.items():
            layer_key = (
                layer_name.value if hasattr(layer_name, "value") else str(layer_name)
            )

            # Find matching loaded layer
            loaded = None
            for loaded_layer, loaded_list in loaded_rules.items():
                loaded_key = (
                    loaded_layer.value
                    if hasattr(loaded_layer, "value")
                    else str(loaded_layer)
                )
                if loaded_key == layer_key:
                    loaded = loaded_list
                    break

            if loaded is None:
                results["rules_match"] = False
                results["missing_rules"].append(f"Layer {layer_key} missing")
                continue

            # Check rule counts
            if len(original) != len(loaded):
                results["rules_match"] = False
                results["missing_rules"].append(
                    f"Layer {layer_key}: {len(original)} -> {len(loaded)}"
                )

            # Check individual rules (by content)
            original_texts = {r.asp_text for r in original if hasattr(r, "asp_text")}
            loaded_texts = {r.asp_text for r in loaded if hasattr(r, "asp_text")}

            missing = original_texts - loaded_texts
            if missing:
                results["rules_match"] = False
                results["missing_rules"].extend(list(missing)[:5])

        return results

    def generate_baseline_report(
        self,
        metrics: Optional[List[PersistenceMetrics]] = None,
    ) -> BaselineReport:
        """
        Generate baseline report from collected metrics.

        Args:
            metrics: Optional list of metrics (uses collected if None)

        Returns:
            BaselineReport with aggregate statistics
        """
        metrics_list = metrics or self.collected_metrics

        if not metrics_list:
            return BaselineReport()

        report = BaselineReport()
        report.total_cycles = len(metrics_list)
        report.individual_metrics = metrics_list

        # Calculate totals
        total_save_time = 0.0
        total_load_time = 0.0
        total_snapshot_time = 0.0
        total_save_errors = 0
        total_load_errors = 0
        total_recovery_attempts = 0
        total_recovery_successes = 0
        total_rules = 0
        total_size = 0
        metadata_preserved_sum = 0.0
        metadata_count = 0

        for m in metrics_list:
            total_save_time += m.save_time_ms
            total_load_time += m.load_time_ms
            total_snapshot_time += m.snapshot_time_ms
            total_save_errors += m.save_errors
            total_load_errors += m.load_errors
            total_recovery_attempts += m.recovery_attempts
            total_recovery_successes += m.recovery_successes
            total_rules += m.total_rules
            total_size += m.total_size_bytes

            if m.total_rules > 0:
                metadata_preserved_sum += m.metadata_fields_preserved
                metadata_count += 1

        report.total_rules_processed = total_rules

        # Calculate averages
        if report.total_cycles > 0:
            report.avg_save_time_ms = total_save_time / report.total_cycles
            report.avg_load_time_ms = total_load_time / report.total_cycles
            report.avg_snapshot_time_ms = total_snapshot_time / report.total_cycles
            report.save_error_rate = total_save_errors / report.total_cycles
            report.load_error_rate = total_load_errors / report.total_cycles

        if total_rules > 0:
            report.avg_rule_size_bytes = total_size / total_rules

        if total_recovery_attempts > 0:
            report.recovery_success_rate = (
                total_recovery_successes / total_recovery_attempts
            )

        if metadata_count > 0:
            report.metadata_preservation_rate = metadata_preserved_sum / metadata_count

        # Build scalability data
        scalability_points: Dict[int, Dict[str, List[float]]] = {}
        for m in metrics_list:
            rule_count = m.total_rules
            if rule_count > 0:
                if rule_count not in scalability_points:
                    scalability_points[rule_count] = {
                        "save_times": [],
                        "load_times": [],
                    }
                scalability_points[rule_count]["save_times"].append(m.save_time_ms)
                scalability_points[rule_count]["load_times"].append(m.load_time_ms)

        for rule_count in sorted(scalability_points.keys()):
            data = scalability_points[rule_count]
            report.scalability_data.append(
                {
                    "rules": rule_count,
                    "save_ms": sum(data["save_times"]) / len(data["save_times"]),
                    "load_ms": sum(data["load_times"]) / len(data["load_times"]),
                }
            )

        return report

    def reset(self) -> None:
        """Reset collected metrics."""
        self.collected_metrics = []
        self.error_log = []


def create_metrics_collector() -> PersistenceMetricsCollector:
    """Factory function to create a metrics collector."""
    return PersistenceMetricsCollector()
