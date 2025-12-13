# LOFT (Logical Ontological Framework Translator)

## Project Overview

LOFT is a research project focusing on **Reflexive Neuro-Symbolic AI**. It bridges the gap between:
*   **Symbolic Core**: Explicit, logical reasoning using Answer Set Programming (ASP).
*   **Neural Interface**: Flexible pattern recognition and rule generation using Large Language Models (LLMs).

The system features a **self-reflexive core** that can question its logic, use LLMs to generate new rules, and validly incorporate them through a rigorous multi-stage pipeline.

## Tech Stack & Dependencies

*   **Language**: Python 3.11+
*   **Symbolic Reasoning**: `clingo` (ASP solver)
*   **LLM Interface**: `anthropic`, `openai`, `instructor` (for structured output)
*   **Data Validation**: `pydantic`
*   **Testing**: `pytest`, `hypothesis`
*   **Tooling**: `ruff` (linting), `black` (formatting), `mypy` (types)

## Setup & Configuration

1.  **Installation**:
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt # For dev tools
    ```

2.  **Environment Variables**:
    *   Copy `.env.example` to `.env`.
    *   Configure API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

---

## Pre-Commit Checklist (MANDATORY)

**Before every commit, run these checks in order:**

```bash
# 1. Format code with Black (CI uses Black, not ruff format)
black loft/ tests/

# 2. Check linting with ruff
ruff check loft/ tests/

# 3. Run tests for affected modules
python -m pytest tests/unit/<affected_module>/ -v --tb=short

# 4. Verify formatting passes
black --check --diff loft/ tests/
```

**CI runs both `ruff check` AND `black --check --diff`. Both must pass.**

---

## Common CI Failures & Fixes

### 1. Missing Trailing Newline (Black)
**Symptom**: Black diff shows `]` vs `]\n` at end of file
```
-]
\ No newline at end of file
+]
```
**Fix**: Run `black <file>` or ensure file ends with newline

### 2. Unused Imports (F401)
**Symptom**: `F401 'module.X' imported but unused`
**Fix**: Remove the unused import from the import statement

### 3. Redefined Import (F811)
**Symptom**: `F811 redefinition of unused 'X' from line N`
**Fix**: Remove duplicate import (often local imports inside functions when same import exists at module level)

### 4. Unused Variable (F841)
**Symptom**: `F841 local variable 'x' is assigned to but never used`
**Fix**: Either use the variable in an assertion or remove it

### 5. Test Data Edge Cases
**Symptom**: Test passes locally but fails in CI due to edge case transformations
**Example**: Temporal tests with dates 1 day apart fail when scaled by 0.5 (truncates to same day)
**Fix**: Use test data with sufficient margin (e.g., dates 30+ days apart for scale tests)

---

## PR Workflow

### Creating a New PR

```bash
# 1. Always start from fresh main
git fetch origin main
git checkout -b feature/issue-XXX-description origin/main

# 2. Make changes...

# 3. Run pre-commit checks (see above)
black loft/ tests/
ruff check loft/ tests/
python -m pytest tests/unit/<module>/ -v

# 4. Commit with conventional format
git add <files>
git commit -m "feat(module): description (#XXX)"

# 5. Push and create PR
git push -u origin feature/issue-XXX-description
gh pr create --title "feat(module): description (#XXX)" --body "..."
```

### Fixing CI Failures

```bash
# 1. Checkout the PR branch
git fetch origin <branch>
git checkout <branch>

# 2. Get CI logs to identify failure
gh run view <run_id> --repo Thought-Mode-Works/loft --log-failed

# 3. Fix issues locally
black loft/ tests/  # For formatting
ruff check --fix loft/ tests/  # For auto-fixable lint issues

# 4. Run failing tests
python -m pytest <test_path> -v --tb=short

# 5. Commit and push fix
git add <files>
git commit -m "fix: resolve CI failures"
git push
```

---

## Code Review Abstractions

### Review Checklist

Before approving any PR, verify:

1. **Formatting**: `black --check --diff loft/ tests/` passes
2. **Linting**: `ruff check loft/ tests/` passes
3. **Tests**: All new code has corresponding tests
4. **Edge Cases**: Test data handles boundary conditions (see temporal test example)
5. **Imports**: No unused imports, no duplicate imports
6. **File Endings**: All files end with newline

### Common Patterns to Watch

| Pattern | Issue | Fix |
|---------|-------|-----|
| `__init__.py` exports | Missing trailing newline | Run `black` |
| New module imports | Unused imports from copy-paste | Remove unused |
| Test fixtures | Edge case data too close to boundaries | Increase margins |
| Mock objects | Missing method on spec | Add method to source class |

### Test Design Guidelines

1. **Temporal Tests**: Use dates 30+ days apart to survive 0.5x scale factors
2. **Boundary Tests**: Test both sides of thresholds, not just at threshold
3. **Mock Specs**: When using `spec=ClassName`, ensure all called methods exist on class
4. **Assertions**: Always use variables you assign (or remove them)

---

## Development Workflow

### Code Quality
Adhere to the project's strict validation and type safety standards.

*   **Linting**: `ruff check loft/ tests/`
*   **Formatting**: `black loft/ tests/` (NOT `ruff format`)
*   **Type Checking**: `mypy loft/` (Strict typing required)

### Testing
Testing is paramount given the self-modifying nature of the system.

*   **Run All Tests**: `pytest`
*   **Run with Coverage**: `pytest --cov=loft --cov-report=html`
*   **Specific Test**: `pytest tests/unit/constraints/test_temporal.py -v`
*   **Failed Tests Only**: `pytest --lf` (last failed)

### Commit Standards
*   Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
*   Reference issue numbers: `feat(constraints): add temporal testing (#236)`
*   Focus on *why* a change was made
*   Ensure all tests pass before committing

---

## Project Structure

*   `loft/`: Main package source code.
    *   `autonomous/`: Self-reflexive loops and agents.
    *   `constraints/`: Phase 7 geometric constraints (equivariance, temporal, measure theory).
    *   `core/`: Core definitions and abstractions.
    *   `legal/`: Domain-specific legal logic modules.
    *   `neural/`: LLM interaction and prompt management.
    *   `symbolic/`: ASP rule management and solver integration.
    *   `translation/`: NL <-> ASP translation layer.
    *   `validation/`: Multi-stage validation logic.
*   `asp_rules/`: Raw Answer Set Programming (.lp) files.
*   `datasets/`: Validation datasets (e.g., `adverse_possession`, `contracts`).
*   `examples/`: Usage demonstrations (Start here to understand the system).
*   `tests/`: Comprehensive test suite (Unit, Integration, E2E).
*   `CLAUDE.md`: Detailed architectural guidelines and principles.
*   `ROADMAP.md`: Project phases and MVP criteria.

---

## Key Architectural Concepts

1.  **Self-Reflexivity**: The system must be able to reason about its own reasoning. Code changes should preserve this capability.
2.  **Ontological Bridge**: Translation between Symbolic (ASP) and Natural Language must maintain semantic fidelity.
3.  **Validation**: Every generated rule undergoes syntactic, semantic, and empirical validation before being added to the core.
4.  **Stratification**: Logic is layered (Constitutional, Strategic, Tactical, Operational) to ensure safety and stability.

---

## Quick Reference Commands

```bash
# Format and lint (run before every commit)
black loft/ tests/ && ruff check loft/ tests/

# Run specific test module
python -m pytest tests/unit/constraints/ -v

# Check PR CI status
gh pr checks <pr_number> --repo Thought-Mode-Works/loft

# Get failed CI logs
gh run view <run_id> --repo Thought-Mode-Works/loft --log-failed

# Watch CI progress
gh pr checks <pr_number> --repo Thought-Mode-Works/loft --watch --interval 30
```
