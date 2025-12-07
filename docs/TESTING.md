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

---

## Autonomous ASP Generation Testing

This section documents requirements and known issues for running autonomous ASP rule generation tests.

### Prerequisites

1. **Environment Setup**
   ```bash
   # Ensure ANTHROPIC_API_KEY is set
   export ANTHROPIC_API_KEY="your-api-key"

   # Verify datasets exist
   ls datasets/contracts/ datasets/torts/ datasets/property_law/
   ```

2. **Dataset Format**
   Each case JSON file must contain:
   ```json
   {
     "id": "case_001",
     "domain": "contracts",
     "asp_facts": "contract(c1). parties(c1, seller, buyer).",
     "question": "Is there a valid contract?",
     "ground_truth": "enforceable"
   }
   ```

### Running ASP Generation Tests

> **Note:** As of PR #186, you must use the `--enable-llm` flag to activate LLM-powered case processing. Without this flag, the runner uses stub processing (for testing infrastructure without API calls).

**Quick 5-minute test:**
```bash
python3 -m loft.autonomous.cli start \
  --dataset datasets/contracts/ \
  --duration 5m \
  --max-cases 10 \
  --enable-llm \
  --model claude-3-5-haiku-20241022 \
  --output /tmp/asp_quick_test \
  --log-level DEBUG
```

**30-minute comprehensive test:**
```bash
python3 -m loft.autonomous.cli start \
  --dataset datasets/contracts/ \
  --dataset datasets/torts/ \
  --dataset datasets/property_law/ \
  --dataset datasets/statute_of_frauds/ \
  --dataset datasets/adverse_possession/ \
  --duration 30m \
  --max-cases 50 \
  --checkpoint-interval 2 \
  --enable-llm \
  --model claude-3-5-haiku-20241022 \
  --output /tmp/autonomous_30min_test \
  --log-level DEBUG 2>&1 | tee /tmp/autonomous_30min_test/run.log
```

**Multi-hour production test:**
```bash
python3 -m loft.autonomous.cli start \
  --dataset datasets/contracts/ \
  --dataset datasets/torts/ \
  --dataset datasets/property_law/ \
  --duration 4h \
  --max-cases 0 \
  --checkpoint-interval 5 \
  --enable-llm \
  --model claude-3-5-haiku-20241022 \
  --output /tmp/autonomous_4hr_test \
  --log-level INFO
```

### CLI Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Path to dataset directory (can specify multiple) | Required |
| `--duration` | Run duration (e.g., `5m`, `30m`, `2h`, `4h`) | `4h` |
| `--max-cases` | Maximum cases to process (0 = unlimited) | `0` |
| `--checkpoint-interval` | Minutes between checkpoints | `15` |
| `--enable-llm` | **Required for LLM processing** - Enable LLM-powered case processing | `false` |
| `--skip-api-check` | Skip ANTHROPIC_API_KEY validation (for testing) | `false` |
| `--model` | LLM model to use (only effective with `--enable-llm`) | `claude-3-5-haiku-20241022` |
| `--output` | Output directory for results | `data/autonomous_runs` |
| `--run-id` | Custom run identifier | Auto-generated |
| `--log-level` | Logging level (DEBUG, INFO, WARNING) | `INFO` |
| `--no-health` | Disable health endpoint | `false` |
| `--source` | Data source (`local`, `courtlistener`) | `local` |

### Verifying Test Results

**Check final report:**
```bash
# View final report
cat /tmp/autonomous_30min_test/*/reports/final_report.json | python3 -m json.tool

# Key metrics to check:
# - cases_processed > 0 (should match loaded cases)
# - rules_generated_total > 0 (if LLM processing is working)
# - llm_calls_total > 0 (confirms LLM is being called)
# - overall_accuracy (should be > 0.5 for good performance)
```

**Check logs for issues:**
```bash
# Look for errors
grep -i "error\|warning\|failed" /tmp/autonomous_30min_test/run.log

# Check case processing
grep "Processing case\|Case complete" /tmp/autonomous_30min_test/run.log

# Verify LLM calls
grep "LLM call\|API request" /tmp/autonomous_30min_test/run.log
```

### Known Issues and Solutions

#### Issue: Runner Completes Immediately (0 cases processed) - RESOLVED in PR #186

**Symptoms:**
- Run completes in < 1 second
- `cases_processed: 0` in final report
- `llm_calls_total: 0`

**Root Cause:**
The `AutonomousTestRunner._process_case()` method returns stub results when no `BatchLearningHarness` is configured. Prior to PR #186, the CLI did not set up this harness.

**Solution (PR #186):**
Use the `--enable-llm` flag to activate LLM-powered case processing:
```bash
python3 -m loft.autonomous.cli start \
  --dataset datasets/contracts/ \
  --enable-llm \
  --model claude-3-5-haiku-20241022 \
  --duration 30m
```

This flag activates the `LLMCaseProcessorAdapter` which:
1. Creates and initializes an `LLMCaseProcessor` with the specified model
2. Bridges the interface gap between the runner and processor
3. Accumulates generated rules across cases
4. Tracks failure patterns for meta-reasoning analysis

**Pre-flight Validation:**
When `--enable-llm` is used, the CLI validates that `ANTHROPIC_API_KEY` is set:
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Use `--skip-api-check` to bypass this validation (e.g., for unit testing with mocks).

**Legacy Workaround (before PR #186):**
If using an older version, manually set up the harness:
```python
from loft.batch import BatchLearningHarness, BatchConfig
from loft.autonomous import AutonomousTestRunner, AutonomousRunConfig

# Create batch harness
batch_config = BatchConfig(max_cases=50, checkpoint_interval=10)
harness = BatchLearningHarness(batch_config, output_dir="results/")

# Create autonomous runner
run_config = AutonomousRunConfig(max_duration_hours=0.5)
runner = AutonomousTestRunner(run_config, output_dir="results/")

# Connect harness to runner
runner.set_batch_harness(harness)

# Now run will process cases properly
result = runner.start(dataset_paths=["datasets/contracts/"])
```

**Related:** GitHub Issue #185, PR #186

#### Issue: Meta-Reasoning Cycles Don't Trigger

**Symptoms:**
- `improvement_cycles_completed: 1` (only initial cycle)
- No rules generated despite processing cases

**Root Cause:**
The `should_run_cycle()` method in `MetaReasoningOrchestrator` requires:
1. `enable_autonomous_improvement: true` in config
2. At least `min_cases_for_analysis` cases processed (default: 10)
3. Cases since last cycle >= `improvement_cycle_interval_cases` (default: 50)

**Solution:**
Adjust config for shorter test runs:
```yaml
meta_reasoning:
  enable_autonomous_improvement: true
  min_cases_for_analysis: 5
  improvement_cycle_interval_cases: 10
```

### Meta-Reasoning Architecture

The autonomous test harness includes a meta-reasoning layer for self-improvement:

```
┌─────────────────────────────────────────────────────────────┐
│                  MetaReasoningOrchestrator                  │
├─────────────────────────────────────────────────────────────┤
│  Components:                                                 │
│  - AutonomousImprover: Executes improvement cycles          │
│  - PromptOptimizer: Optimizes LLM prompt templates          │
│  - FailureAnalyzer: Analyzes prediction failures            │
│  - StrategySelector: Selects optimal strategies             │
├─────────────────────────────────────────────────────────────┤
│  Improvement Cycle Flow:                                     │
│  1. Analyze failure patterns                                 │
│  2. Generate improvement suggestions                         │
│  3. Apply improvements (prompt changes, strategy updates)    │
│  4. Validate improvements against safety thresholds          │
│  5. Rollback if accuracy drops > max_accuracy_drop           │
└─────────────────────────────────────────────────────────────┘
```

**Key Configuration:**
```python
MetaReasoningConfig(
    enable_autonomous_improvement=True,
    enable_prompt_optimization=True,
    enable_failure_analysis=True,
    enable_strategy_selection=True,
    improvement_cycle_interval_cases=50,
    min_cases_for_analysis=10,
    max_improvement_actions_per_cycle=5,
    confidence_threshold=0.7,
)
```

### Failure Pattern Categories

The system tracks these failure categories for meta-reasoning:

| Category | Description | Recommended Action |
|----------|-------------|-------------------|
| `unsafe_variable` | Variables only in negative literals | Add safety constraints to prompts |
| `embedded_period` | Periods in fact atoms | Sanitize periods before ASP processing |
| `syntax_error` | Invalid ASP syntax | Add syntax examples to prompts |
| `invalid_arithmetic` | Bad arithmetic expressions | Restrict arithmetic in templates |
| `grounding_error` | Cannot ground rules | Add domain bounding examples |
| `json_parse_error` | LLM output not parseable | Enforce JSON schema in prompts |
| `validation_error` | Rules fail validation | Improve quality guidance |

### Monitoring Long-Running Tests

**Real-time log monitoring:**
```bash
# Follow logs
tail -f /tmp/autonomous_test/run.log

# Monitor specific patterns
tail -f /tmp/autonomous_test/run.log | grep --line-buffered "cycle\|accuracy\|error"
```

**Check state file:**
```bash
# View current state
cat /tmp/autonomous_test/*/state.json | python3 -m json.tool

# Watch for updates
watch -n 5 'cat /tmp/autonomous_test/*/state.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Cases: {d.get(\"progress\",{}).get(\"cases_processed\",0)}, Accuracy: {d.get(\"progress\",{}).get(\"current_accuracy\",0):.2%}\")"'
```

**Health endpoint (if enabled):**
```bash
curl http://localhost:8080/health
```

### Validation Checklist

Before considering an autonomous test successful, verify:

- [ ] `cases_processed` matches expected count
- [ ] `llm_calls_total > 0` (LLM was used)
- [ ] `overall_accuracy > 0.5` (reasonable performance)
- [ ] `rules_generated_total > 0` (rules were created)
- [ ] No critical errors in logs
- [ ] Checkpoints were created at expected intervals
- [ ] Final report was generated successfully
- [ ] Run duration matches expected duration (within 10%)

### Suggested Improvements for Validation

Based on analysis of the autonomous testing infrastructure, the following improvements would enhance ASP generation validation:

#### 1. CLI-Harness Integration - IMPLEMENTED (PR #186)

**Status:** ✅ Resolved

PR #186 introduced `LLMCaseProcessorAdapter` class that bridges the interface between `AutonomousTestRunner` and `LLMCaseProcessor`. The adapter is activated via the `--enable-llm` CLI flag.

**Implementation:**
```python
# loft/autonomous/cli.py
class LLMCaseProcessorAdapter:
    """Adapter that wraps LLMCaseProcessor for AutonomousTestRunner."""

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        self._processor: Optional[Any] = None
        self._accumulated_rules: list = []
        self._model = model

    def initialize(self) -> None:
        if self._processor is not None:
            return
        from loft.autonomous.llm_processor import LLMCaseProcessor
        self._processor = LLMCaseProcessor(model=self._model)
        self._processor.initialize()

    def process_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        self.initialize()
        result = self._processor.process_case(case, self._accumulated_rules)
        # Convert CaseResult to Dict and track accumulated rules
        ...
```

**Usage:**
```bash
python3 -m loft.autonomous.cli start \
  --dataset datasets/contracts/ \
  --enable-llm \
  --model claude-3-5-haiku-20241022 \
  --duration 30m
```

#### 2. Pre-Flight Validation - PARTIALLY IMPLEMENTED (PR #186)

**Status:** ✅ API key validation implemented

PR #186 added pre-flight validation for the ANTHROPIC_API_KEY when `--enable-llm` is used:

```python
# In cli.py start command
if enable_llm and not skip_api_check:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        click.echo(
            "Error: ANTHROPIC_API_KEY environment variable not set. "
            "Required when using --enable-llm. "
            "Set it with: export ANTHROPIC_API_KEY='your-api-key'",
            err=True,
        )
        raise click.Abort()
```

**Remaining improvements (future work):**
```python
def validate_run_prerequisites(config, dataset_paths):
    """Validate prerequisites before starting autonomous run."""
    errors = []

    # Check datasets exist and are non-empty
    for path in dataset_paths:
        cases = load_cases_from_path(path)
        if not cases:
            errors.append(f"No cases found in {path}")

    return errors
```

#### 3. Early Exit Detection

Detect and warn about runs that complete too quickly:

```python
MIN_EXPECTED_DURATION_SECONDS = 60  # At least 1 minute for real processing

if result.total_duration_seconds < MIN_EXPECTED_DURATION_SECONDS:
    logger.warning(
        f"Run completed in {result.total_duration_seconds:.2f}s - "
        f"this is suspiciously fast. Check if cases were processed."
    )
```

#### 4. Metrics Validation

Add automated validation of metrics in the final report:

```python
def validate_run_metrics(result: RunResult) -> List[str]:
    """Validate that run metrics indicate successful processing."""
    warnings = []

    if result.final_metrics.cases_processed == 0:
        warnings.append("No cases were processed")

    if result.final_metrics.llm_calls_total == 0 and result.config.llm_model:
        warnings.append("No LLM calls were made despite LLM being configured")

    if result.final_metrics.overall_accuracy == 1.0:
        warnings.append("Perfect accuracy may indicate stub results")

    return warnings
```

#### 5. Integration Test Coverage

Add integration tests that verify end-to-end processing:

```python
# tests/integration/autonomous/test_end_to_end.py

def test_autonomous_processes_cases():
    """Verify that autonomous runner actually processes cases."""
    config = AutonomousRunConfig(max_duration_hours=0.01, max_cases=5)
    runner = AutonomousTestRunner(config, output_dir="/tmp/test")

    # Set up harness (the missing piece!)
    harness = MockBatchHarness()
    runner.set_batch_harness(harness)

    result = runner.start(dataset_paths=["datasets/contracts/"])

    assert result.final_metrics.cases_processed > 0
    assert harness.process_case_called_count > 0
```

#### 6. Meta-Reasoning Trigger Configuration

For shorter test runs, lower the thresholds for meta-reasoning cycles:

```yaml
# config/test_config.yaml
meta_reasoning:
  enable_autonomous_improvement: true
  min_cases_for_analysis: 3      # Lower for testing
  improvement_cycle_interval_cases: 5  # Trigger more frequently
  max_improvement_actions_per_cycle: 2
```

#### 7. Logging Improvements

Add structured logging for easier debugging:

```python
logger.info(
    "Case processing summary",
    extra={
        "cases_loaded": len(cases),
        "cases_processed": progress.cases_processed,
        "llm_calls": metrics.llm_calls_total,
        "harness_configured": runner._batch_harness is not None,
    }
)
```

### Recommended Test Sequence

For validating autonomous ASP generation, follow this sequence:

1. **Unit tests first:**
   ```bash
   python3 -m pytest tests/unit/autonomous/ -v
   ```

2. **Quick smoke test (5 cases):**
   ```bash
   python3 -m loft.autonomous.cli start \
     --dataset datasets/contracts/ \
     --duration 2m \
     --max-cases 5 \
     --enable-llm \
     --model claude-3-5-haiku-20241022 \
     --log-level DEBUG
   ```

3. **Short integration test (10-20 cases):**
   ```bash
   python3 -m loft.autonomous.cli start \
     --dataset datasets/contracts/ \
     --dataset datasets/torts/ \
     --duration 10m \
     --max-cases 20 \
     --checkpoint-interval 2 \
     --enable-llm \
     --model claude-3-5-haiku-20241022
   ```

4. **Full test (30+ minutes):**
   ```bash
   python3 -m loft.autonomous.cli start \
     --dataset datasets/contracts/ \
     --dataset datasets/torts/ \
     --dataset datasets/property_law/ \
     --duration 30m \
     --max-cases 50 \
     --checkpoint-interval 5 \
     --enable-llm \
     --model claude-3-5-haiku-20241022
   ```

5. **Validate results:**
   ```bash
   # Check final report
   cat /tmp/*/reports/final_report.json | python3 -m json.tool

   # Verify key metrics
   python3 -c "
   import json, sys
   data = json.load(open(sys.argv[1]))
   m = data['final_metrics']
   print(f'Cases: {m.get(\"cases_processed\", 0)}')
   print(f'LLM Calls: {m.get(\"llm_calls_total\", 0)}')
   print(f'Rules: {m.get(\"rules_generated_total\", 0)}')
   print(f'Accuracy: {m.get(\"overall_accuracy\", 0):.2%}')
   " /tmp/*/reports/final_report.json
   ```
