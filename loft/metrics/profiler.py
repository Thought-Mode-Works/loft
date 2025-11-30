"""
Performance profiler for batch processing.

Provides memory monitoring, CPU profiling, and performance
tracking capabilities for identifying bottlenecks.
"""

import gc
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    timestamp: datetime
    current_mb: float
    peak_mb: float
    allocated_blocks: int
    traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "current_mb": self.current_mb,
            "peak_mb": self.peak_mb,
            "allocated_blocks": self.allocated_blocks,
            "traceback": self.traceback,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemorySnapshot":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            current_mb=data["current_mb"],
            peak_mb=data["peak_mb"],
            allocated_blocks=data["allocated_blocks"],
            traceback=data.get("traceback"),
        )


@dataclass
class ProfileResult:
    """Result of profiling an operation."""

    operation_name: str
    started_at: datetime
    completed_at: datetime
    elapsed_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    peak_memory_mb: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_name": self.operation_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "elapsed_ms": self.elapsed_ms,
            "memory_before_mb": self.memory_before_mb,
            "memory_after_mb": self.memory_after_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfileResult":
        """Create from dictionary."""
        return cls(
            operation_name=data["operation_name"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            elapsed_ms=data["elapsed_ms"],
            memory_before_mb=data["memory_before_mb"],
            memory_after_mb=data["memory_after_mb"],
            memory_delta_mb=data["memory_delta_mb"],
            peak_memory_mb=data["peak_memory_mb"],
            success=data["success"],
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


class PerformanceProfiler:
    """Profiler for monitoring performance and memory during batch processing."""

    def __init__(
        self,
        enable_memory_tracking: bool = True,
        enable_gc_stats: bool = True,
        memory_sample_interval: int = 10,
    ):
        """Initialize profiler.

        Args:
            enable_memory_tracking: Whether to track memory usage
            enable_gc_stats: Whether to collect garbage collection stats
            memory_sample_interval: Cases between memory samples
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_gc_stats = enable_gc_stats
        self.memory_sample_interval = memory_sample_interval

        self.profile_results: List[ProfileResult] = []
        self.memory_snapshots: List[MemorySnapshot] = []
        self.gc_stats: List[Dict[str, Any]] = []

        self._tracking_started = False
        self._peak_memory_mb = 0.0
        self._operation_count = 0

    def start_tracking(self) -> None:
        """Start memory tracking."""
        if self.enable_memory_tracking and not self._tracking_started:
            tracemalloc.start()
            self._tracking_started = True

    def stop_tracking(self) -> None:
        """Stop memory tracking."""
        if self._tracking_started:
            tracemalloc.stop()
            self._tracking_started = False

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Current memory usage in megabytes
        """
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback to tracemalloc if psutil not available
            if self._tracking_started:
                current, _ = tracemalloc.get_traced_memory()
                return current / (1024 * 1024)
            return 0.0

    def take_memory_snapshot(self, include_traceback: bool = False) -> MemorySnapshot:
        """Take a memory snapshot.

        Args:
            include_traceback: Whether to include allocation traceback

        Returns:
            Memory snapshot
        """
        current_mb = self.get_memory_usage_mb()

        if self._tracking_started:
            _, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)
        else:
            peak_mb = current_mb

        self._peak_memory_mb = max(self._peak_memory_mb, current_mb)

        traceback_str = None
        if include_traceback and self._tracking_started:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")[:10]
            traceback_str = "\n".join(str(stat) for stat in top_stats)

        # Get allocated blocks count
        if self._tracking_started:
            snapshot = tracemalloc.take_snapshot()
            allocated_blocks = len(snapshot.statistics("lineno"))
        else:
            allocated_blocks = 0

        mem_snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            current_mb=current_mb,
            peak_mb=peak_mb,
            allocated_blocks=allocated_blocks,
            traceback=traceback_str,
        )

        self.memory_snapshots.append(mem_snapshot)
        return mem_snapshot

    def collect_gc_stats(self) -> Dict[str, Any]:
        """Collect garbage collection statistics.

        Returns:
            GC statistics dictionary
        """
        if not self.enable_gc_stats:
            return {}

        stats = {
            "timestamp": datetime.now().isoformat(),
            "gc_counts": gc.get_count(),
            "gc_threshold": gc.get_threshold(),
            "gc_objects": len(gc.get_objects()),
        }

        self.gc_stats.append(stats)
        return stats

    def profile_operation(
        self,
        operation_name: str,
        operation: Callable[[], T],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[T, ProfileResult]:
        """Profile a single operation.

        Args:
            operation_name: Name of the operation
            operation: Callable to profile
            metadata: Optional metadata to include

        Returns:
            Tuple of (operation result, profile result)
        """
        self._operation_count += 1

        # Take memory snapshot before
        memory_before = self.get_memory_usage_mb()

        # Reset peak tracking for this operation
        if self._tracking_started:
            tracemalloc.reset_peak()

        started_at = datetime.now()
        start_time = time.perf_counter()

        success = True
        error_message = None
        result = None

        try:
            result = operation()
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            completed_at = datetime.now()
            elapsed_ms = (end_time - start_time) * 1000

            memory_after = self.get_memory_usage_mb()

            if self._tracking_started:
                _, peak = tracemalloc.get_traced_memory()
                peak_memory_mb = peak / (1024 * 1024)
            else:
                peak_memory_mb = max(memory_before, memory_after)

            profile_result = ProfileResult(
                operation_name=operation_name,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_ms=elapsed_ms,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                memory_delta_mb=memory_after - memory_before,
                peak_memory_mb=peak_memory_mb,
                success=success,
                error_message=error_message,
                metadata=metadata or {},
            )

            self.profile_results.append(profile_result)
            self._peak_memory_mb = max(self._peak_memory_mb, peak_memory_mb)

            # Take periodic memory snapshots
            if self._operation_count % self.memory_sample_interval == 0:
                self.take_memory_snapshot()

            # Collect GC stats periodically
            if self.enable_gc_stats and self._operation_count % self.memory_sample_interval == 0:
                self.collect_gc_stats()

        return result, profile_result

    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary.

        Returns:
            Summary statistics
        """
        if not self.profile_results:
            return {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
            }

        successful = [r for r in self.profile_results if r.success]
        failed = [r for r in self.profile_results if not r.success]

        timing_values = [r.elapsed_ms for r in self.profile_results]
        memory_deltas = [r.memory_delta_mb for r in self.profile_results]

        # Group by operation name
        by_operation: Dict[str, List[ProfileResult]] = {}
        for result in self.profile_results:
            if result.operation_name not in by_operation:
                by_operation[result.operation_name] = []
            by_operation[result.operation_name].append(result)

        operation_summaries = {}
        for op_name, results in by_operation.items():
            times = [r.elapsed_ms for r in results]
            operation_summaries[op_name] = {
                "count": len(results),
                "total_time_ms": sum(times),
                "avg_time_ms": sum(times) / len(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "success_rate": sum(1 for r in results if r.success) / len(results),
            }

        return {
            "total_operations": len(self.profile_results),
            "successful_operations": len(successful),
            "failed_operations": len(failed),
            "total_time_ms": sum(timing_values),
            "avg_time_ms": sum(timing_values) / len(timing_values),
            "min_time_ms": min(timing_values),
            "max_time_ms": max(timing_values),
            "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
            "peak_memory_mb": self._peak_memory_mb,
            "memory_snapshots_count": len(self.memory_snapshots),
            "gc_collections_count": len(self.gc_stats),
            "by_operation": operation_summaries,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert profiler state to dictionary.

        Returns:
            Complete profiler state
        """
        return {
            "summary": self.get_summary(),
            "profile_results": [r.to_dict() for r in self.profile_results],
            "memory_snapshots": [s.to_dict() for s in self.memory_snapshots],
            "gc_stats": self.gc_stats,
        }

    def identify_bottlenecks(
        self, threshold_ms: float = 1000.0, top_n: int = 10
    ) -> List[ProfileResult]:
        """Identify slow operations.

        Args:
            threshold_ms: Minimum time to be considered slow
            top_n: Number of slowest operations to return

        Returns:
            List of slowest operations
        """
        slow_operations = [r for r in self.profile_results if r.elapsed_ms >= threshold_ms]
        slow_operations.sort(key=lambda x: x.elapsed_ms, reverse=True)
        return slow_operations[:top_n]

    def identify_memory_leaks(self, threshold_mb: float = 10.0) -> List[ProfileResult]:
        """Identify operations with high memory growth.

        Args:
            threshold_mb: Minimum memory growth to flag

        Returns:
            List of operations with high memory growth
        """
        high_memory = [r for r in self.profile_results if r.memory_delta_mb >= threshold_mb]
        high_memory.sort(key=lambda x: x.memory_delta_mb, reverse=True)
        return high_memory
