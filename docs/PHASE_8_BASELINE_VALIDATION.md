
# Phase 8: Baseline Infrastructure Validation

**Status**: Complete
**Phase**: 8 of 9
**Dependencies**: Phase 7 (Geometric Constraints) ✓

## Overview

Phase 8 establishes baseline validation of the LOFT infrastructure through long-running batch testing, iterative ASP rule building, and meta-reasoning integration. This phase bridges the gap between individual component functionality and end-to-end autonomous learning.

**Goal**: Validate infrastructure through long-running experiments with persistent, iterative ASP rule building.

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Experiment Runner                          │
│  (Long-running experiments with state management)            │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────┴─────────────┐
    │                          │
    v                          v
┌───────────────────┐  ┌──────────────────────┐
│  BatchLearning    │  │  MetaAware          │
│  Harness          │  │  BatchProcessor     │
│  (Framework)      │  │  (Strategy          │
│                   │  │   Selection)        │
└─────────┬─────────┘  └──────────┬───────────┘
          │                       │
          v                       v
┌──────────────────────────────────────────┐
│      FullPipelineProcessor               │
│  (Gap → Generate → Validate →            │
│   Incorporate → Persist)                 │
└────┬──────┬──────┬──────┬──────┬─────────┘
     │      │      │      │      │
     v      v      v      v      v
┌────────┐ ┌────┐ ┌────┐ ┌────┐ ┌─────────┐
│Gap ID  │ │LLM │ │Val │ │Inc │ │Persist  │
│        │ │Gen │ │    │ │    │ │         │
└────────┘ └────┘ └────┘ └────┘ └─────────┘
```

## Components

### 1. Full Pipeline Processor

**Location**: `loft/batch/full_pipeline.py`
**Issue**: #253

Connects all pipeline components for autonomous rule generation:

1. **Gap Identification**: Analyzes case to find missing knowledge
2. **Rule Generation**: Calls LLM with predicate alignment
3. **Validation**: Multi-stage pipeline (syntax, semantic, empirical, consensus)
4. **Incorporation**: Adds valid rules to stratified ASP core
5. **Persistence**: Saves rules to disk with snapshotting

**Key Features**:
- Predicate extraction from case facts for alignment
- LLM token usage tracking
- Per-stage timing metrics
- Configurable target stratification layer
- Automatic snapshot creation

**Example Usage**:

```python
from loft.batch.full_pipeline import create_full_pipeline_processor

# Create processor with all dependencies
processor = create_full_pipeline_processor(
    model="claude-3-5-haiku-20241022",
    persistence_dir="./asp_rules",
    enable_persistence=True,
    target_layer=StratificationLevel.TACTICAL,
)

# Process a case
result = processor.process_case(
    case={
        "id": "case_001",
        "asp_facts": "offer(c1). acceptance(c1).",
        "ground_truth": "valid",
        "legal_principle": "Contract formation principles",
    },
    accumulated_rules=[],
)

print(f"Generated {result.rules_generated} rules")
print(f"Accepted {result.rules_accepted} rules")
```

### 2. Meta-Aware Batch Processor

**Location**: `loft/batch/meta_aware_processor.py`
**Issue**: #255

Wraps pipeline processor with meta-reasoning capabilities:

1. **Strategy Selection**: Chooses reasoning strategy based on case type
2. **Failure Analysis**: Detects and categorizes failure patterns
3. **Adaptive Weights**: Adjusts strategy weights based on performance
4. **Performance Tracking**: Monitors success rates per strategy

**Key Features**:
- Lazy-loaded meta-reasoning components
- Configurable adaptation thresholds
- Failure pattern detection with root cause analysis
- Strategy performance tracking
- Automatic adaptation triggers

**Example Usage**:

```python
from loft.batch.meta_aware_processor import (
    MetaAwareBatchConfig,
    MetaAwareBatchProcessor,
)

# Create config
config = MetaAwareBatchConfig(
    enable_strategy_selection=True,
    enable_failure_analysis=True,
    min_failures_for_adaptation=5,
    failure_pattern_threshold=0.5,
)

# Wrap pipeline with meta-awareness
meta_processor = MetaAwareBatchProcessor(
    pipeline_processor=pipeline,
    config=config,
)

# Process with meta-reasoning
result = meta_processor.process_case_with_meta(case, accumulated_rules=[])

print(f"Strategy used: {result.strategy_used}")
print(f"Adaptation triggered: {result.adaptation_triggered}")
if result.failure_pattern_detected:
    print(f"Pattern: {result.failure_pattern_detected.failure_type}")
```

### 3. Experiment Runner

**Location**: `loft/experiments/experiment_runner.py`
**Issue**: #256

Orchestrates long-running experiments with state management:

1. **Cycle Management**: Runs multiple improvement cycles
2. **State Persistence**: Saves state for resumption
3. **Checkpoint Creation**: Snapshots at cycle boundaries
4. **Report Generation**: Creates cycle and final reports

**Key Features**:
- Automatic state save/load
- Resumable experiments
- Cool-down periods between cycles
- Cumulative metrics tracking
- Markdown report generation

**Example Usage**:

```python
from loft.experiments import ExperimentConfig, ExperimentRunner

# Configure experiment
config = ExperimentConfig(
    experiment_id="batch_learning_001",
    description="Baseline validation experiment",
    dataset_path=Path("datasets/contracts/"),
    state_path=Path("state/"),
    reports_path=Path("reports/"),
    rules_path=Path("asp_rules/"),
    max_cycles=10,
    cases_per_cycle=20,
    cool_down_seconds=60,
)

# Create and run experiment
runner = ExperimentRunner(
    config=config,
    processor=processor,
    persistence=persistence_manager,
)

report = runner.run()

print(f"Completed {report.cycles_completed} cycles")
print(f"Processed {report.cases_processed} cases")
print(f"Generated {report.rules_generated} rules")
```

### 4. ASP Persistence Manager

**Location**: `loft/persistence/asp_persistence.py`
**Issue**: #254

Manages ASP rule persistence with versioning:

1. **Stratified Save/Load**: Separate files per layer
2. **Git Versioning**: Optional git-based snapshots
3. **Snapshot Management**: Named snapshots with metadata
4. **Rollback Support**: Revert to previous states

**Example Usage**:

```python
from loft.persistence.asp_persistence import ASPPersistenceManager

# Create persistence manager
persistence = ASPPersistenceManager(
    base_dir="./asp_rules",
    enable_git=True,
)

# Save after processing
persistence.save_all(asp_core)

# Create snapshot
persistence.create_snapshot(
    cycle_number=1,
    description="After cycle 1: 15 rules",
)

# Rollback if needed
persistence.rollback_to_snapshot("cycle_001")
```

### 5. Batch Learning Harness

**Location**: `loft/batch/harness.py`

Framework for batch processing with checkpointing:

1. **Sequential Processing**: Process cases in order
2. **Rule Accumulation**: Track generated rules
3. **Checkpoint Support**: Save/restore batch state
4. **Metrics Collection**: Aggregate batch statistics

**Example Usage**:

```python
from loft.batch import BatchLearningHarness, BatchConfig

# Configure batch
config = BatchConfig(
    batch_id="batch_001",
    max_rules_per_case=3,
    checkpoint_interval=10,
)

# Create harness
harness = BatchLearningHarness(
    config=config,
    checkpoint_dir=Path("checkpoints/"),
)

# Run batch
report = harness.run_batch(
    cases=test_cases,
    process_case=processor.process_case,
)

print(f"Processed {report.cases_processed} cases")
print(f"Success rate: {report.success_rate:.2%}")
```

## Running Experiments

### Quick Validation (5 cases, no LLM)

```bash
# Run integration tests with mocked components
python -m pytest tests/integration/phase8/test_phase8_integration.py::TestPhase8EndToEnd::test_full_pipeline_5_cases -v
```

### Medium Validation (10 cases, mocked meta-reasoning)

```bash
# Test meta-aware processing
python -m pytest tests/integration/phase8/test_phase8_integration.py::TestPhase8EndToEnd::test_meta_aware_batch_10_cases -v
```

### Long Experiment (30 cases)

```bash
# Run longer experiment (marked as slow)
python -m pytest tests/integration/phase8/test_phase8_integration.py::TestLongRunningExperiments::test_30_case_experiment -v -s -m slow
```

### Dialogue Tests (Interactive Output)

```bash
# View dialogue test output showing component interactions
python -m pytest tests/integration/phase8/test_dialogue.py::TestFullPipelineDialogue::test_full_pipeline_dialogue_flow -v -s
```

### Component Analysis

```bash
# Generate component analysis table
python -m pytest tests/integration/phase8/test_dialogue.py::TestComponentAnalysisTable::test_generate_component_analysis_table -v -s
```

## Baseline Metrics

### Current Baseline (Mocked LLM)

These metrics are from integration tests with mocked components:

| Metric | Value | Target | Notes |
|--------|-------|--------|-------|
| Pipeline Success Rate | 100% | >95% | With mocked validation |
| Rules Generated per Case | 1-2 | 1-3 | Depends on gaps identified |
| Rules Accepted per Case | 0.8 | >0.7 | 80% validation pass rate |
| Meta-Reasoning Engagement | ~30% | Variable | Adapts based on failures |
| State Persistence | 100% | 100% | Save/load cycles |
| Experiment Resumption | 100% | 100% | Interrupted experiments |

### Metrics to Establish with Real LLM

When running with actual LLM (not mocked):

- [ ] ASP rule generation rate (rules/hour)
- [ ] LLM token usage per rule
- [ ] Validation pass rate by stage
- [ ] Meta-reasoning improvement detection rate
- [ ] Batch processing throughput
- [ ] Coverage expansion over iterations

## Integration Points

### With Phase 7 (Geometric Constraints)

Phase 8 builds on Phase 7's validated components:

- **Stratified ASP Core**: Used by FullPipelineProcessor for rule incorporation
- **Validation Pipeline**: 7-stage validation including geometric constraints
- **Confidence System**: Used for rule acceptance decisions
- **Persistence**: Extended with snapshot and rollback capabilities

### Preparing for Phase 9 (Production Hardening)

Phase 8 establishes the baseline for Phase 9:

- **Performance Baselines**: Documented metrics for regression detection
- **Long-Running Stability**: Validated multi-cycle experiments
- **State Management**: Robust save/load/resume capabilities
- **Monitoring Hooks**: Metrics collection points for observability

## Validation Criteria

Phase 8 is validated when:

- [x] Full pipeline processes cases end-to-end
- [x] Meta-aware processing selects strategies adaptively
- [x] Experiments run multiple cycles with state persistence
- [x] Rules persist and accumulate across cycles
- [x] Integration tests cover all components
- [x] Dialogue tests document expected behavior

## Testing Strategy

### Unit Tests

Individual component tests in `tests/unit/`:
- `test_full_pipeline.py` - Pipeline processor logic
- `test_meta_aware.py` - Meta-reasoning components
- `test_experiment_runner.py` - Experiment orchestration

### Integration Tests

Component integration in `tests/integration/phase8/`:
- `test_phase8_integration.py` - End-to-end flows
- `test_dialogue.py` - Interaction documentation

### Dialogue Tests

Per CLAUDE.md requirements, dialogue tests show:
- Component interaction flows
- Expected system behavior
- What's functional vs placeholder

Example output:

```
=============================================================
Test: Full Pipeline Dialogue
=============================================================

[User]: Process contract case with missing validity rule
[System]: Identifying knowledge gaps...
[System]: Identified 1 gap(s)

[System]: Generating rule for identified gap...
[System]: Calling LLM with gap description...
[System]: Generated rule: valid_contract(X) :- offer(X), acceptance(X), consideration(X).
[System]: Tokens used: 1234

[System]: Validating generated rule...
[System]: Syntax check: PASSED
[System]: Semantic check: PASSED
[System]: Empirical check: PASSED (accuracy: 0.85)
[System]: Consensus check: PASSED (votes: 3/3)

[System]: Incorporating validated rule...
[System]: Rule incorporated at TACTICAL level
[System]: New total rules: 15

[System]: Persisting rules to disk...
[System]: Saved to /path/to/asp_rules/tactical.lp
[System]: Snapshot created: cycle_001

=============================================================
Dialogue Test Complete
=============================================================
```

## Known Limitations

### Current Limitations

1. **LLM Dependency**: Pipeline requires external LLM API
2. **Predicate Alignment**: Depends on accurate extraction from case facts
3. **Meta-Reasoning**: Components lazy-loaded, may not always be available
4. **Coverage Tracking**: Not yet integrated into batch processing

### Future Enhancements

1. **Automated Gap Identification**: More sophisticated gap detection
2. **Coverage-Guided Processing**: Prioritize cases that expand coverage
3. **Multi-LLM Consensus**: Use multiple LLMs for rule generation
4. **Adaptive Validation**: Adjust thresholds based on performance

## Troubleshooting

### Common Issues

#### Issue: No rules generated for cases

**Cause**: Gap identification not finding gaps, or LLM not generating rules

**Solution**:
1. Check case has `ground_truth` and `prediction` fields
2. Ensure `prediction != ground_truth` to trigger gap
3. Verify LLM API credentials are set
4. Check logs for generation errors

#### Issue: All rules rejected by validation

**Cause**: Validation thresholds too strict, or rules have low quality

**Solution**:
1. Review validation pipeline configuration
2. Check `min_confidence` threshold
3. Examine rejection reasons in validation reports
4. Adjust LLM prompts for better rule quality

#### Issue: Experiment not resuming from state

**Cause**: State file missing or corrupted

**Solution**:
1. Check `state_path` directory exists
2. Verify `{experiment_id}_state.json` file present
3. Check file permissions
4. Review state file JSON for corruption

## References

- **Issue #252**: Phase 8 Tracking Issue
- **Issue #253**: Full Pipeline Processor
- **Issue #254**: ASP Persistence Validation
- **Issue #255**: Meta-Reasoning Batch Integration
- **Issue #256**: Long-Running Experiment Runner
- **Issue #260**: Phase 8 Integration Tests and Documentation

- **ROADMAP.md**: Overall project roadmap
- **CLAUDE.md**: Development guidelines
- **TESTING.md**: Testing standards
