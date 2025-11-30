"""Tests for PerformanceProfiler."""

import time
from datetime import datetime

from loft.metrics.profiler import (
    MemorySnapshot,
    PerformanceProfiler,
    ProfileResult,
)


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a memory snapshot."""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            current_mb=512.5,
            peak_mb=600.0,
            allocated_blocks=1000,
        )

        assert snapshot.current_mb == 512.5
        assert snapshot.peak_mb == 600.0
        assert snapshot.allocated_blocks == 1000

    def test_snapshot_roundtrip(self):
        """Test serialization roundtrip."""
        original = MemorySnapshot(
            timestamp=datetime.now(),
            current_mb=256.0,
            peak_mb=300.0,
            allocated_blocks=500,
            traceback="test trace",
        )

        data = original.to_dict()
        restored = MemorySnapshot.from_dict(data)

        assert restored.current_mb == original.current_mb
        assert restored.peak_mb == original.peak_mb
        assert restored.traceback == original.traceback


class TestProfileResult:
    """Tests for ProfileResult dataclass."""

    def test_create_result(self):
        """Test creating a profile result."""
        now = datetime.now()
        result = ProfileResult(
            operation_name="test_op",
            started_at=now,
            completed_at=now,
            elapsed_ms=150.5,
            memory_before_mb=100.0,
            memory_after_mb=120.0,
            memory_delta_mb=20.0,
            peak_memory_mb=130.0,
            success=True,
        )

        assert result.operation_name == "test_op"
        assert result.elapsed_ms == 150.5
        assert result.success is True

    def test_result_roundtrip(self):
        """Test serialization roundtrip."""
        now = datetime.now()
        original = ProfileResult(
            operation_name="roundtrip_test",
            started_at=now,
            completed_at=now,
            elapsed_ms=200.0,
            memory_before_mb=50.0,
            memory_after_mb=75.0,
            memory_delta_mb=25.0,
            peak_memory_mb=80.0,
            success=False,
            error_message="Test error",
            metadata={"case_id": "123"},
        )

        data = original.to_dict()
        restored = ProfileResult.from_dict(data)

        assert restored.operation_name == original.operation_name
        assert restored.elapsed_ms == original.elapsed_ms
        assert restored.success == original.success
        assert restored.error_message == original.error_message


class TestPerformanceProfiler:
    """Tests for PerformanceProfiler."""

    def test_create_profiler(self):
        """Test creating a profiler."""
        profiler = PerformanceProfiler()

        assert profiler.enable_memory_tracking is True
        assert profiler.enable_gc_stats is True
        assert len(profiler.profile_results) == 0

    def test_profile_operation_success(self):
        """Test profiling a successful operation."""
        profiler = PerformanceProfiler(enable_memory_tracking=False)

        def test_operation():
            time.sleep(0.01)
            return "result"

        result, profile = profiler.profile_operation("test_op", test_operation)

        assert result == "result"
        assert profile.success is True
        assert profile.elapsed_ms >= 10
        assert profile.operation_name == "test_op"

    def test_profile_operation_failure(self):
        """Test profiling a failed operation."""
        profiler = PerformanceProfiler(enable_memory_tracking=False)

        def failing_operation():
            raise ValueError("Test error")

        try:
            profiler.profile_operation("failing_op", failing_operation)
        except ValueError:
            pass

        assert len(profiler.profile_results) == 1
        profile = profiler.profile_results[0]
        assert profile.success is False
        assert "Test error" in profile.error_message

    def test_profile_with_metadata(self):
        """Test profiling with metadata."""
        profiler = PerformanceProfiler(enable_memory_tracking=False)

        result, profile = profiler.profile_operation(
            "test_op",
            lambda: 42,
            metadata={"case_id": "test_123"},
        )

        assert result == 42
        assert profile.metadata["case_id"] == "test_123"

    def test_get_summary(self):
        """Test getting profiler summary."""
        profiler = PerformanceProfiler(enable_memory_tracking=False)

        # Profile several operations
        for i in range(5):
            profiler.profile_operation(f"op_{i}", lambda: time.sleep(0.001))

        summary = profiler.get_summary()

        assert summary["total_operations"] == 5
        assert summary["successful_operations"] == 5
        assert summary["failed_operations"] == 0
        assert "by_operation" in summary

    def test_identify_bottlenecks(self):
        """Test bottleneck identification."""
        profiler = PerformanceProfiler(enable_memory_tracking=False)

        # Create some fast and slow operations
        profiler.profile_operation("fast", lambda: None)
        profiler.profile_operation("slow", lambda: time.sleep(0.05))

        bottlenecks = profiler.identify_bottlenecks(threshold_ms=10)

        # Should identify the slow operation
        assert len(bottlenecks) >= 1
        assert bottlenecks[0].operation_name == "slow"

    def test_to_dict(self):
        """Test converting profiler to dictionary."""
        profiler = PerformanceProfiler(enable_memory_tracking=False)
        profiler.profile_operation("test", lambda: None)

        data = profiler.to_dict()

        assert "summary" in data
        assert "profile_results" in data
        assert "memory_snapshots" in data

    def test_memory_tracking(self):
        """Test memory tracking functionality."""
        profiler = PerformanceProfiler(
            enable_memory_tracking=True,
            memory_sample_interval=1,
        )

        profiler.start_tracking()

        # Profile an operation
        profiler.profile_operation("memory_test", lambda: [0] * 1000)

        # Take a snapshot
        snapshot = profiler.take_memory_snapshot()

        assert snapshot.current_mb >= 0
        assert len(profiler.memory_snapshots) >= 1

        profiler.stop_tracking()

    def test_get_memory_usage(self):
        """Test getting memory usage."""
        profiler = PerformanceProfiler(enable_memory_tracking=False)

        usage = profiler.get_memory_usage_mb()

        # Should return something (may be 0 if psutil not available)
        assert usage >= 0

    def test_identify_memory_leaks(self):
        """Test memory leak identification."""
        profiler = PerformanceProfiler(enable_memory_tracking=False)

        # Simulate a profile result with high memory growth
        now = datetime.now()
        profiler.profile_results.append(
            ProfileResult(
                operation_name="leaky_op",
                started_at=now,
                completed_at=now,
                elapsed_ms=100.0,
                memory_before_mb=100.0,
                memory_after_mb=200.0,
                memory_delta_mb=100.0,
                peak_memory_mb=200.0,
                success=True,
            )
        )

        leaks = profiler.identify_memory_leaks(threshold_mb=50.0)

        assert len(leaks) == 1
        assert leaks[0].operation_name == "leaky_op"

    def test_gc_stats_collection(self):
        """Test garbage collection stats collection."""
        profiler = PerformanceProfiler(enable_gc_stats=True)

        stats = profiler.collect_gc_stats()

        assert "gc_counts" in stats
        assert "gc_threshold" in stats
        assert "timestamp" in stats

    def test_empty_summary(self):
        """Test summary with no operations."""
        profiler = PerformanceProfiler()

        summary = profiler.get_summary()

        assert summary["total_operations"] == 0
        assert summary["successful_operations"] == 0
