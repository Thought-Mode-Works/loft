# LOFT Testing Guide

This guide covers testing strategies for LOFT, with emphasis on scale testing introduced in Phase 4.75.

## Quick Start

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run tests for a specific module
python -m pytest tests/unit/batch/ -v
python -m pytest tests/unit/metrics/ -v

# Run with coverage
python -m pytest tests/unit/ --cov=loft --cov-report=html
```

## Test Categories

### Unit Tests

Located in `tests/unit/`, organized by module:

| Module | Path | Tests | Purpose |
|--------|------|-------|---------|
| `batch` | `tests/unit/batch/` | 45 | Batch processing harness |
| `metrics` | `tests/unit/metrics/` | 53 | Metrics collection and analysis |
| `corpus` | `tests/unit/corpus/` | 29 | Dataset loading and domains |
| `evolution` | `tests/unit/evolution/` | 150+ | Rule evolution tracking |
| `validation` | `tests/unit/validation/` | 100+ | Validation pipeline |
| `translation` | `tests/unit/translation/` | 200+ | NL↔ASP translation |
| `dialectical` | `tests/unit/dialectical/` | 100+ | Dialectical reasoning |

### Integration Tests

Located in `experiments/`:

```bash
# LLM rule integration test
python experiments/llm_rule_integration_test.py

# Transfer study
python experiments/casework/transfer_study.py
```

### Scale Tests

See [Scale Testing](#scale-testing) section below.

---

## Scale Testing

Phase 4.75 introduced comprehensive infrastructure for testing at scale.

### Batch Processing

Use the `BatchLearningHarness` for processing large datasets:

```python
from loft.batch import BatchLearningHarness, BatchConfig

# Configure for scale
config = BatchConfig(
    max_cases=500,
    checkpoint_interval=25,
    max_total_rules=200,
    continue_on_error=True,
)

harness = BatchLearningHarness(config=config, output_dir="results/")

# Define your case processor
def process_case(case, accumulated_rules):
    # Your processing logic here
    ...

# Run batch
result = harness.run_batch(test_cases, process_case)
```

### Command-Line Batch Processing

```bash
# Process a single dataset
python scripts/run_batch_learning.py --dataset datasets/contracts/

# Process multiple datasets
python scripts/run_batch_learning.py \
    --datasets datasets/contracts/ datasets/torts/ datasets/procedural/

# Resume from checkpoint
python scripts/run_batch_learning.py \
    --resume results/batch_001/checkpoints/checkpoint_50.json

# List previous batch runs
python scripts/run_batch_learning.py --list-batches

# Show details of a specific batch
python scripts/run_batch_learning.py --show-batch batch_001
```

### Metrics Collection

Use `MetricsCollector` for comprehensive metrics:

```python
from loft.metrics import MetricsCollector

collector = MetricsCollector(
    batch_id="scale_test_001",
    output_dir="reports/",
)

# Time operations
with collector.time_operation("rule_generation"):
    # Your operation here
    ...

# Track counters
collector.increment_counter("cases_processed")
collector.increment_counter("rules_generated", 3)

# Set gauges
collector.set_gauge("memory_mb", get_memory_usage())

# Record milestones
collector.record_milestone(
    milestone_name="50_rules",
    rules_count=50,
    cases_processed=100,
    accuracy=0.85,
)

# Save metrics
collector.save()
```

### Performance Profiling

Use `PerformanceProfiler` for memory and CPU profiling:

```python
from loft.metrics import PerformanceProfiler

profiler = PerformanceProfiler(
    enable_memory_tracking=True,
    memory_sample_interval=10,
)

profiler.start_tracking()

# Profile specific operations
result, profile = profiler.profile_operation(
    "expensive_operation",
    lambda: your_function(),
)

print(f"Time: {profile.elapsed_ms:.2f}ms")
print(f"Memory delta: {profile.memory_delta_mb:.2f}MB")

# Identify bottlenecks
bottlenecks = profiler.identify_bottlenecks(threshold_ms=1000)
for b in bottlenecks:
    print(f"Slow: {b.operation_name} - {b.elapsed_ms:.2f}ms")

# Check for memory leaks
leaks = profiler.identify_memory_leaks(threshold_mb=50)

profiler.stop_tracking()
```

### Metrics Analysis

Analyze batch metrics with `MetricsAnalyzer`:

```python
from loft.metrics import MetricsAnalyzer, load_and_analyze_metrics
from pathlib import Path

# Load and analyze
report = load_and_analyze_metrics(Path("reports/batch_001/metrics.json"))

# Examine trends
print(f"Accuracy trend: {report.trends.accuracy_trend}")
print(f"Latency trend: {report.trends.latency_trend}")

# Check anomalies
print(f"Anomalies detected: {report.anomaly_report.total_anomalies}")

# Get recommendations
for rec in report.recommendations:
    print(f"- {rec}")
```

Command-line analysis:

```bash
# Text report
python scripts/analyze_metrics.py reports/batch_001/metrics.json

# JSON output
python scripts/analyze_metrics.py reports/batch_001/metrics.json \
    --format json --output analysis.json

# Compare batches
python scripts/analyze_metrics.py \
    reports/batch_001/metrics.json \
    reports/batch_002/metrics.json \
    --compare
```

---

## Dataset Validation

Validate datasets before batch processing:

```bash
# Validate a single domain
python scripts/validate_dataset.py datasets/contracts/

# Validate with specific checks
python scripts/validate_dataset.py datasets/contracts/ \
    --check-predicates \
    --check-balance

# Validate all datasets
for domain in contracts torts procedural; do
    python scripts/validate_dataset.py datasets/$domain/
done
```

Dataset requirements:
- JSON format with `predicates`, `ground_truth`, `reasoning`
- Domain-specific common predicates present
- Balanced positive/negative outcomes (recommended)

---

## Checkpoint and Resume

For long-running tests, use checkpointing:

```python
from loft.batch import BatchLearningHarness, BatchConfig

config = BatchConfig(
    checkpoint_interval=25,  # Checkpoint every 25 cases
)

harness = BatchLearningHarness(config=config, output_dir="results/")

# If interrupted, resume from checkpoint
result = harness.resume_from_checkpoint(
    checkpoint_path="results/batch_001/checkpoints/checkpoint_50.json",
    test_cases=remaining_cases,
    process_case_fn=process_case,
)
```

Checkpoints include:
- Processed case IDs
- Pending case IDs
- Case results
- Accumulated rule IDs
- Progress snapshot

---

## Corpus Snapshots

Save and restore corpus state:

```bash
# Create snapshot
python scripts/corpus_snapshot.py --create --output snapshots/

# List snapshots
python scripts/corpus_snapshot.py --list

# Restore from snapshot
python scripts/corpus_snapshot.py --restore snapshots/snapshot_20251130.json
```

---

## CI/CD Integration

### GitHub Actions

Tests run automatically on every push and PR:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: python -m pytest tests/unit/ -v

- name: Check coverage
  run: python -m pytest tests/unit/ --cov=loft --cov-fail-under=80
```

### Pre-commit Hooks

Recommended pre-commit configuration:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: python -m pytest tests/unit/ -x -q
        language: system
        pass_filenames: false
```

---

## Best Practices

### Writing Scale Tests

1. **Use configurable limits**: Set `max_cases`, `max_total_rules` appropriately
2. **Enable checkpointing**: Use `checkpoint_interval` for long tests
3. **Collect metrics**: Always use `MetricsCollector` for analysis
4. **Profile memory**: Enable `PerformanceProfiler` for large batches
5. **Handle errors gracefully**: Set `continue_on_error=True`

### Monitoring Scale Tests

1. **Set up callbacks** for real-time monitoring:
   ```python
   def on_progress(progress):
       print(f"Progress: {progress.completion_percentage:.1f}%")

   harness.set_callbacks(on_progress=on_progress)
   ```

2. **Record milestones** at key points (10, 50, 100 rules)

3. **Analyze trends** after completion:
   ```bash
   python scripts/analyze_metrics.py results/metrics.json
   ```

### Debugging Scale Failures

1. **Check anomaly report** for unusual patterns
2. **Review bottlenecks** in profiler output
3. **Examine checkpoint** before failure
4. **Analyze per-domain metrics** for domain-specific issues

---

## Troubleshooting

### Out of Memory

```python
config = BatchConfig(
    max_total_rules=100,  # Reduce rule limit
)
profiler = PerformanceProfiler(
    memory_sample_interval=5,  # More frequent snapshots
)
```

### Slow Processing

```python
# Identify bottlenecks
bottlenecks = profiler.identify_bottlenecks(threshold_ms=500)

# Check timing stats
stats = collector.get_all_timing_stats()
for name, stat in stats.items():
    print(f"{name}: avg={stat.mean:.2f}ms, max={stat.max_value:.2f}ms")
```

### Checkpoint Resume Fails

```bash
# Verify checkpoint integrity
python -c "import json; json.load(open('checkpoint.json'))"

# Check pending cases exist
# Ensure case IDs in pending_case_ids match available cases
```

---

## Reference

### Modules

- `loft.batch`: Batch processing harness
- `loft.metrics`: Metrics collection and analysis
- `loft.corpus`: Dataset loading and domains
- `loft.evolution`: Rule evolution tracking

### Scripts

- `scripts/run_batch_learning.py`: Batch processing CLI
- `scripts/analyze_metrics.py`: Metrics analysis
- `scripts/corpus_snapshot.py`: Corpus state management
- `scripts/validate_dataset.py`: Dataset validation

### Configuration

See `BatchConfig` for all configurable options:

```python
@dataclass
class BatchConfig:
    max_cases: Optional[int] = None
    max_rules_per_case: int = 3
    max_total_rules: int = 200
    validation_threshold: float = 0.8
    checkpoint_interval: int = 10
    continue_on_error: bool = True
    max_consecutive_errors: int = 10
    # ... and more
```
