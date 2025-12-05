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

---

## Autonomous Long-Running Tests

Phase 5 introduces the `loft.autonomous` module for running long-duration (4+ hour) autonomous test experiments with meta-reasoning integration.

### Quick Start

```bash
# Start a 4-hour autonomous run
loft-autonomous start --dataset datasets/contracts/ --duration 4h

# Resume from checkpoint
loft-autonomous resume --checkpoint data/autonomous_runs/run_001/checkpoints/latest.json

# Check run status
loft-autonomous status --run-id run_001

# Generate report
loft-autonomous report --run-id run_001 --format markdown
```

### Python API

```python
from loft.autonomous import AutonomousTestRunner, AutonomousRunConfig

# Configure run
config = AutonomousRunConfig(
    max_duration_hours=4.0,
    max_cases=0,  # Unlimited
    checkpoint_interval_minutes=15,
    llm_model="claude-3-5-haiku-20241022",
)

# Create runner
runner = AutonomousTestRunner(config, output_dir="data/autonomous_runs")

# Set callbacks for monitoring
runner.set_callbacks(
    on_progress=lambda p: print(f"Progress: {p.completion_percentage:.1f}%"),
    on_case_complete=lambda c: print(f"Case {c['case_id']}: {c['correct']}"),
    on_checkpoint=lambda cp: print(f"Checkpoint {cp.checkpoint_number} created"),
)

# Start run
result = runner.start(dataset_paths=["datasets/contracts/"])

print(f"Final accuracy: {result.final_metrics.overall_accuracy:.2%}")
print(f"Improvement cycles: {result.final_metrics.improvement_cycles_completed}")
```

### Docker Deployment

For remote or containerized runs:

```bash
# Build Docker image
docker build -f Dockerfile.autonomous -t loft-autonomous .

# Run with docker-compose
ANTHROPIC_API_KEY=xxx docker-compose -f docker-compose.autonomous.yml up

# Check health
curl http://localhost:8080/health
```

### Configuration

Create `config/my_run.yaml`:

```yaml
max_duration_hours: 4.0
max_cases: 0
checkpoint_interval_minutes: 15
llm_model: "claude-3-5-haiku-20241022"

meta_reasoning:
  enable_autonomous_improvement: true
  improvement_cycle_interval_cases: 50
  enable_prompt_optimization: true
  enable_failure_analysis: true

notification:
  notify_on_start: true
  notify_on_completion: true
  notify_on_error: true
  milestone_interval_cases: 100

health:
  enabled: true
  port: 8080
```

Use with:

```bash
loft-autonomous start --config config/my_run.yaml --dataset datasets/contracts/
```

### Slack Notifications

Enable Slack notifications by setting the webhook URL:

```bash
# Via environment variable
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/xxx"
loft-autonomous start --dataset datasets/contracts/

# Via command line
loft-autonomous start --dataset datasets/contracts/ \
    --slack-webhook "https://hooks.slack.com/services/xxx"
```

Notifications are sent for:
- Run started
- Processing milestones (every N cases)
- Improvement cycle completion
- Errors
- Run completion

### Run Output Structure

```
data/autonomous_runs/{run_id}/
├── config.json              # Run configuration
├── state.json               # Current state (for monitoring)
├── checkpoints/
│   ├── checkpoint_0001.json
│   ├── checkpoint_0002.json
│   └── latest.json -> checkpoint_0002.json
├── metrics/
│   ├── run_metrics.json     # Final metrics
│   └── timeline.jsonl       # Event timeline
├── rules/
│   └── evolved_rules.json   # Accumulated rules
└── reports/
    ├── final_report.json    # Structured results
    └── final_report.md      # Human-readable summary
```

### Running Tests

```bash
# Run autonomous module unit tests
python -m pytest tests/unit/autonomous/ -v

# Run integration tests
python -m pytest tests/integration/autonomous/ -v

# Run all autonomous tests
python -m pytest tests/unit/autonomous/ tests/integration/autonomous/ -v
```

### Example: Short Test Run

```python
from loft.autonomous import AutonomousTestRunner, AutonomousRunConfig

# Short test configuration (5 minutes)
config = AutonomousRunConfig(
    max_duration_hours=0.08,  # 5 minutes
    max_cases=50,
    checkpoint_interval_minutes=1,
)

runner = AutonomousTestRunner(config, output_dir="/tmp/test_runs")
result = runner.start(dataset_paths=["datasets/contracts/"])

assert result.status.value == "completed"
assert result.final_metrics.overall_accuracy >= 0.5
```

### Monitoring with Health Endpoint

The health endpoint provides Docker HEALTHCHECK support:

```bash
# Check health
curl http://localhost:8080/health

# Response:
{
  "healthy": true,
  "status": "running",
  "run_id": "run_20240101_120000_abc123",
  "progress": {
    "cases_processed": 150,
    "total_cases": 500,
    "current_accuracy": 0.85
  }
}

# Check readiness
curl http://localhost:8080/ready
```

### Troubleshooting

**Run exits early:**
- Check `max_duration_hours` and `max_cases` limits
- Review `state.json` for shutdown_requested flag
- Check for errors in timeline.jsonl

**Checkpoints not created:**
- Verify `checkpoint_interval_minutes` setting
- Check disk space in output directory
- Review logs for checkpoint errors

**Slack notifications not sending:**
- Verify webhook URL format (must start with `https://hooks.slack.com/`)
- Check rate limiting (30-second minimum between notifications)
- Review logs for notification errors
