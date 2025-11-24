# LOFT Experiments

This directory contains experimental scripts for validating and testing the LOFT system before implementing full infrastructure.

## Experiments

### LLM Rule Integration Test (Phase 3.0 Early Validation)

**File**: `llm_rule_integration_test.py`

**Purpose**: Validate the complete workflow of LLM-generated rule integration before implementing full Phase 3 safe self-modification infrastructure.

**What it does**:
1. Identifies knowledge gaps using statute of frauds test suite
2. Generates candidate rules (mock implementation for now)
3. Validates candidates using simplified validation
4. Measures accuracy impact of accepted rules
5. Documents findings and lessons learned

**Usage**:

```bash
# Run with defaults (5 gaps, 3 candidates per gap)
python experiments/llm_rule_integration_test.py

# Custom configuration
python experiments/llm_rule_integration_test.py --max-gaps 10 --candidates-per-gap 5

# Specify output file
python experiments/llm_rule_integration_test.py --output results/my_experiment.json
```

**Expected Output**:

The script will:
- Print progress for each step (gap identification, generation, validation, measurement)
- Display a summary of results
- Export detailed JSON results to `experiments/results/`

**Example Summary**:

```
================================================================================
 Experiment Summary
================================================================================

Gaps Identified: 5
Candidates Generated: 15
Accepted: 4 (26.7%)
Rejected: 11

Baseline Accuracy: 87.5%
Best Improvement: +8.3%

Experiment Success: ✅ YES

Best Performing Rule:
merchant_confirmation_satisfies(C) :-
    goods_sale_contract(C),
    between_merchants(C),
    confirmation_sent(C, P1, P2),
    not objection_within_10_days(P2).
```

## Results Directory

The `results/` subdirectory contains exported experiment results in JSON format.

**File naming convention**: `llm_integration_test_YYYYMMDD_HHMMSS.json`

**Result structure**:

```json
{
  "config": {
    "max_gaps": 5,
    "candidates_per_gap": 3,
    "validation_threshold": 0.75,
    "experiment_date": "2025-11-23T19:30:00"
  },
  "gaps": [
    {
      "test_case_id": "tc1",
      "description": "...",
      "query": "enforceable",
      "expected": true,
      "actual": false,
      "missing_reasoning": "..."
    }
  ],
  "candidates": [...],
  "validation_results": [...],
  "performance_results": [...],
  "summary": {
    "gaps_identified": 5,
    "candidates_generated": 15,
    "accepted_count": 4,
    "rejected_count": 11,
    "acceptance_rate": 0.267,
    "baseline_accuracy": 0.875,
    "best_improvement": 0.083,
    "best_rule": "...",
    "success": true
  }
}
```

## Implementation Notes

### Current Limitations

**Mock Implementation Areas** (to be replaced with real implementations):

1. **Candidate Generation**: Currently generates mock rules based on patterns. In full implementation:
   - Use `RuleGenerator` with actual LLM calls
   - Generate semantically correct rules from gap analysis
   - Use proper prompting for different rule types

2. **Validation**: Simplified validation checking syntax and confidence. In full implementation:
   - Use `ValidationPipeline` with full multi-stage validation
   - Run empirical tests against test suite
   - Check logical consistency with ASP core
   - Use multi-LLM consensus voting

3. **Performance Measurement**: Simplified simulation. In full implementation:
   - Actually add rules to ASP core temporarily
   - Run complete test suite
   - Measure actual accuracy changes
   - Track which specific test cases are fixed

### Why Mock for Now?

This experiment focuses on **validating the workflow and infrastructure** rather than testing actual LLM rule generation. The mock implementation allows us to:

- Test the experiment framework itself
- Validate data flow between components
- Ensure result collection and reporting works
- Identify infrastructure needs for Phase 3

Once the workflow is validated, we can replace mock components with real implementations incrementally.

## Success Criteria

An experiment is considered successful if:

✅ At least 1 candidate rule is accepted (confidence ≥ threshold)
✅ Accepted rule(s) improve accuracy by >2%
✅ No logical inconsistencies introduced
✅ Process identifies areas needing infrastructure improvements

Even if experiments reveal issues (validation too strict/lenient, rules don't generalize, etc.), this is **valuable feedback** for refining Phase 3 plans.

## Next Steps After Running Experiments

### If Successful:
1. Review accepted rules for incorporation into main `statute_of_frauds.lp`
2. Update Phase 3 plans based on findings
3. Replace mock implementations with real ones
4. Proceed with full Phase 3 infrastructure

### If Issues Found:
1. Analyze failure modes (validation, generation, gap detection)
2. Fix identified issues
3. Re-run experiment
4. Update Phase 3 requirements based on lessons learned

## Integration with Phase 3

This experiment validates the **end-to-end workflow** that Phase 3 will automate:

- **Phase 3.1**: Safe self-modification infrastructure → Automates the manual rule incorporation tested here
- **Phase 3.2**: Stratified modification authority → Ensures only validated rules modify the core
- **Phase 3.3**: A/B testing framework → Systematizes the performance measurement done here
- **Phase 3.4**: Performance monitoring → Tracks the accuracy improvements measured here

By validating the workflow early, we ensure Phase 3 infrastructure is built on proven foundations.
