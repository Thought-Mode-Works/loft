# Codex Implementation Guide

This file captures how to work on LOFT when operating as Codex. It summarizes the essential guardrails from `CLAUDE.md` and adds a pull-request scaffold so changes stay aligned with the self-reflexive symbolic core.

## Core Guardrails (from CLAUDE.md)
- Preserve the self-modifying symbolic core: changes must keep the system able to explain, question, and revise its own reasoning (traceable, reversible, justifiable).
- Validate continuously across layers: syntactic, semantic, empirical, and meta-validation must all remain operational, with clear pass/fail criteria and logged results.
- Protect the ontological bridge: NL ↔ ASP roundtrips should preserve meaning, surface ambiguity explicitly, and track provenance/confidence.
- Respect stratified authority and safety: constitutional constraints are immutable; higher-impact changes need higher validation thresholds and rollback plans.
- Practice epistemic humility: propagate confidence/uncertainty, avoid silent failures, and request human input when outside competence.

## Working Steps
- **Before coding:** Read `ROADMAP.md` for phase targets, review recent validation results, and note core state or predicate vocabulary to avoid ontology drift.
- **During coding:** Design small, composable functions; log key decisions and validation data; version prompts; and ensure translation and validation functions can explain failures.
- **Tests to run:** Prefer targeted suites that exercise the bridge and reflexive loop. Helpful commands:
  ```bash
  # Ontological bridge validation (fidelity, edge cases, adequacy)
  pytest tests/integration/ontological_bridge/ -v

  # Integration smoke (no LLM cost)
  python experiments/llm_rule_integration_test.py --dry-run
  ```
- **Out-of-scope issues:** Record them as GitHub issues with repro steps, expected vs actual behavior, and validation impact.

## Pull Request Template
Use this template when proposing changes.

```
## Summary
- [brief description of changes]

## Phase & Validation Criteria
- Phase: [0-9] (see ROADMAP.md)
- Relevant MVP criteria: [list the specific items]

## Validation Results
- [ ] All existing tests pass (commands + outcomes)
- [ ] New tests added for new functionality
- Validation metrics (before → after):
  - Accuracy:
  - Consistency score:
  - Translation fidelity:
- [ ] Self-reflexivity preserved (explain how)
- [ ] Ontological bridge integrity (tests/metrics)

## Tested Examples
- [working, tested code paths or CLI invocations]

## Impact on Symbolic Core
- Rules added/modified: [count + description]
- Meta-reasoning changes: [summary]
- Stability impact: [analysis]

## Risks & Mitigations
- [issues + mitigations]

## Rollback Plan
- [how to revert if problems occur]
```

## Validation Checklist (quick)
- Self-reflexivity intact
- Ontological bridge fidelity validated
- Validation framework operational
- Safety mechanisms (rollback/thresholds) active
- Explainability maintained
- Performance/consistency stable
- Tests updated and passing

## Notes for Codex
- Default to `claude-haiku-3-5-20241022` for budgeted LLM runs; prefer dry-run modes when possible.
- Keep comments focused on rationale or invariants; avoid restating code.
- If translation or predicate alignment is touched, add/extend roundtrip and alignment tests to prevent ontology mismatch.
