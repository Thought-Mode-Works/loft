# Contributing to LOFT

<!-- Last verified: 2024-12-07 -->
> **Document Status:** Last updated 2024-12-07 | Closes #21
>
> **Requirements:** Python 3.11+ | pip 21.0+ | Git 2.20+

Welcome to LOFT (Logical Ontological Framework Translator)! This guide will help you get up to speed quickly and start contributing effectively.

**Time estimates:**
- Quick Start: ~10 minutes
- Full Onboarding: ~45 minutes
- First Contribution: ~2 hours

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Start](#2-quick-start)
3. [Codebase Tour](#3-codebase-tour)
4. [Development Workflow](#4-development-workflow)
5. [Testing Philosophy](#5-testing-philosophy)
6. [Common Tasks](#6-common-tasks)
7. [Validation and Review Process](#7-validation-and-review-process)
8. [Getting Help](#8-getting-help)
9. [Project Principles](#9-project-principles)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Project Overview

**Reading time: ~5 minutes**

### What is LOFT?

LOFT is a research project building a **self-reflexive neuro-symbolic AI system**. The core innovation is a symbolic reasoning engine that can:

- **Question its own logic** - identify gaps and inconsistencies
- **Use LLMs for pattern recognition** - generate candidate logical rules from natural language
- **Validate and incorporate new rules** - rigorous multi-stage verification
- **Reason about its own reasoning** - meta-level self-improvement

### The Ontological Bridge

The key concept is the **ontological bridge** - bidirectional translation between:
- **Symbolic representations** (Answer Set Programming rules)
- **Neural representations** (LLM natural language understanding)

This bridge allows the strengths of each paradigm to compensate for the other's weaknesses.

### Current Domain: Legal Reasoning

LOFT is initially targeting legal document analysis, specifically contract law (Statute of Frauds). Legal reasoning is an ideal domain because:
- Rules are structured but application requires pattern recognition
- Meta-reasoning is natural (precedent, analogies, distinctions)
- Court decisions provide ground truth for validation

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│         Reflexive Meta-Reasoner                 │  ← Reasons about reasoning
├─────────────────────────────────────────────────┤
│         Symbolic Core (Self-Modifying)          │  ← ASP rules, compositional
│  - Constitutional Layer (immutable)             │
│  - Strategic/Tactical/Operational Layers        │
├─────────────────────────────────────────────────┤
│      Translation Layer (Ontological Bridge)     │  ← Symbolic ↔ Natural Language
├─────────────────────────────────────────────────┤
│         Neural Interface & LLM Ensemble         │  ← Multiple specialized LLMs
├─────────────────────────────────────────────────┤
│      Validation & Verification Framework        │  ← Multi-stage validation
└─────────────────────────────────────────────────┘
```

---

## 2. Quick Start

**Time: ~10 minutes**

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Required |
| pip | 21.0+ | For dependency management |
| Git | 2.20+ | For version control |
| clingo | 5.4+ | Optional, auto-installed via pip |
| Docker | 24.0+ | Optional, for containerized development |

**API Keys (optional for most development):**
- `ANTHROPIC_API_KEY` - For Anthropic Claude models
- `OPENAI_API_KEY` - For OpenAI models

### Installation

```bash
# Clone the repository
git clone https://github.com/justin4957/loft.git
cd loft

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools

# Set up environment (for LLM features)
cp .env.example .env
# Edit .env to add your API keys if needed
```

### Verify Installation

```bash
# Run the test suite
python -m pytest tests/unit/ -v --tb=short

# Expected output: All tests passing (3000+ tests)
```

### Run a Quick Example

```bash
# Legal reasoning example
python examples/statute_of_frauds_demo.py

# Translation example
python examples/nl_to_asp_examples.py
```

---

## 3. Codebase Tour

**Reading time: ~15 minutes**

### Directory Structure

```
loft/                        # Repository root
│
├── loft/                    # Main source code (Python package)
│   │
│   ├── autonomous/          # Long-running autonomous test infrastructure
│   │   ├── cli.py           # Command-line interface
│   │   ├── runner.py        # Test runner orchestration
│   │   └── checkpoints.py   # State persistence
│   │
│   ├── neural/              # LLM interface and rule generation
│   │   ├── providers/       # LLM provider implementations
│   │   │   ├── anthropic.py # Claude integration
│   │   │   └── openai.py    # GPT integration
│   │   └── rule_gen.py      # Rule generation from NL
│   │
│   ├── symbolic/            # ASP symbolic core
│   │   ├── asp_core.py      # Core ASP engine
│   │   └── rules.py         # Rule representations
│   │
│   ├── translation/         # NL ↔ ASP translation layer
│   │   ├── nl_to_asp.py     # Natural language to ASP
│   │   └── asp_to_nl.py     # ASP to natural language
│   │
│   ├── validation/          # Multi-stage validation pipeline
│   │   ├── asp_validators.py      # Syntax validation
│   │   └── semantic_validators.py # Semantic validation
│   │
│   ├── meta/                # Meta-reasoning orchestration
│   ├── legal/               # Legal domain-specific logic
│   └── ...                  # Other modules
│
├── tests/                   # Test suites (3000+ tests)
│   ├── unit/                # Unit tests by module
│   │   ├── test_asp_validators.py   # 110+ validation tests
│   │   ├── neural/                  # LLM interface tests
│   │   └── translation/             # Translation tests
│   ├── integration/         # Component interaction tests
│   └── e2e/                 # Full pipeline tests
│
├── datasets/                # Test case datasets
│   ├── contracts/           # Contract law (Statute of Frauds)
│   ├── torts/               # Tort law cases
│   └── property_law/        # Property law cases
│
├── experiments/             # Experiment scripts
├── scripts/                 # Utility scripts
├── docs/                    # Documentation
│   ├── CONTRIBUTING.md      # This guide
│   └── TESTING.md           # Testing documentation
│
├── examples/                # Example usage scripts
├── ROADMAP.md              # Phased development plan
├── CLAUDE.md               # AI development guidelines
└── README.md               # Project overview
```

### Module Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input                               │
│              (Natural language legal facts)                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  loft/translation/nl_to_asp.py                  │
│              (Translate NL facts to ASP predicates)             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   loft/symbolic/asp_core.py                     │
│              (Execute ASP rules, find answer sets)              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 loft/validation/asp_validators.py               │
│              (Validate generated rules)                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Legal Conclusion                          │
│              (Is the contract enforceable?)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Key Modules

#### `loft/symbolic/` - The ASP Core

The symbolic reasoning engine using Answer Set Programming:

```python
from loft.symbolic import ASPCore, ASPRule

core = ASPCore()
core.add_rule(ASPRule(
    head="enforceable(C)",
    body="contract(C), not void(C)",
    confidence=1.0
))
result = core.query("enforceable")
```

#### `loft/neural/` - LLM Interface

Interfaces with multiple LLM providers:

```python
from loft.neural import LLMInterface, AnthropicProvider

provider = AnthropicProvider(api_key="...", model="claude-3-5-haiku-20241022")
llm = LLMInterface(provider)

response = llm.query("What makes a contract enforceable?")
```

#### `loft/translation/` - Ontological Bridge

Translates between symbolic and natural language:

```python
from loft.translation import NLToASPTranslator

translator = NLToASPTranslator()
facts = translator.translate_to_facts("Alice signed a contract with Bob.")
```

#### `loft/validation/` - Verification Pipeline

Multi-stage validation for rule acceptance:

```python
from loft.validation import ASPSyntaxValidator

validator = ASPSyntaxValidator()
result = validator.validate_generated_rule("enforceable(C) :- contract(C).")
print(f"Valid: {result.is_valid}")
```

#### `loft/meta/` - Meta-Reasoning

Orchestrates self-improvement cycles:

```python
from loft.meta import MetaReasoningOrchestrator

orchestrator = MetaReasoningOrchestrator()
suggestions = orchestrator.suggest_prompt_improvements()
```

### Configuration

Key configuration is in `loft/config.py` and environment variables:

```python
# .env file
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
CLINGO_PATH=/path/to/clingo  # Optional, auto-detected
```

---

## 4. Development Workflow

### Branch Strategy

We use a simplified branching model. All branches should be created from and merged into `main`.

**Branch naming conventions:**

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/issue-XX-description` | `feature/issue-21-contributing-guide` |
| Bug fix | `bugfix/issue-XX-description` | `bugfix/issue-42-validation-error` |
| Hotfix | `hotfix/issue-XX-description` | `hotfix/issue-99-critical-bug` |
| Refactor | `refactor/description` | `refactor/simplify-translation-layer` |

```bash
# Always start from latest main
git fetch origin main
git checkout -b feature/issue-XX-short-description origin/main

# Examples:
git checkout -b feature/issue-177-context-aware-validators origin/main
git checkout -b bugfix/issue-42-embedded-period-detection origin/main
git checkout -b refactor/simplify-meta-reasoning origin/main
```

### Making Changes

1. **Read the issue thoroughly** - check all comments
2. **Check ROADMAP.md** - understand the current phase
3. **Review CLAUDE.md** - follow architectural principles
4. **Write tests first** - TDD is encouraged
5. **Make incremental commits** - logical, reviewable chunks

### Running Quality Checks

```bash
# Linting
ruff check loft/
ruff format loft/

# Type checking
mypy loft/

# Run tests
python -m pytest tests/unit/ -v

# Run with coverage
python -m pytest tests/unit/ --cov=loft --cov-report=html
```

### Pre-Commit Checklist

Before creating a PR:

```bash
# 1. Run linter
ruff check loft/ && ruff format --check loft/

# 2. Run type checker
mypy loft/

# 3. Run full test suite
python -m pytest tests/unit/ -v

# 4. Check coverage hasn't dropped
python -m pytest tests/unit/ --cov=loft --cov-fail-under=70
```

### Creating a Pull Request

```bash
# Push your branch
git push -u origin feature/issue-XX-description

# Create PR via GitHub CLI
gh pr create --title "Add feature X (issue #XX)" --body "..."
```

**PR Requirements:**
- Link to issue: `Closes #XX`
- All CI checks passing
- Working test examples in description
- Validation results if applicable

---

## 5. Testing Philosophy

### Test Categories

| Category | Location | Purpose |
|----------|----------|---------|
| Unit | `tests/unit/` | Individual function/class tests |
| Integration | `tests/integration/` | Component interaction tests |
| E2E | `tests/e2e/` | Full pipeline tests |
| Property | `tests/property/` | Invariant/property tests |

### Running Tests

```bash
# All unit tests
python -m pytest tests/unit/ -v

# Specific module
python -m pytest tests/unit/validation/ -v

# Single test file
python -m pytest tests/unit/test_asp_validators.py -v

# Single test
python -m pytest tests/unit/test_asp_validators.py::TestASPPatterns::test_variable_pattern -v

# With coverage
python -m pytest tests/unit/ --cov=loft --cov-report=html
open htmlcov/index.html
```

### Test Markers

```bash
# Skip slow tests
python -m pytest tests/unit/ -v -m "not slow"

# Run only integration tests
python -m pytest tests/integration/ -v
```

### Writing Tests

```python
import pytest
from loft.validation.asp_validators import check_unsafe_variables

class TestUnsafeVariables:
    """Tests for unsafe variable detection."""

    def test_safe_rule_returns_no_errors(self) -> None:
        """Safe rules should return empty error list."""
        rule = "result(X) :- input(X)."
        errors, warnings = check_unsafe_variables(rule)
        assert len(errors) == 0

    def test_unsafe_head_variable_detected(self) -> None:
        """Variables in head not bound in body should be flagged."""
        rule = "result(X, Y) :- input(X)."
        errors, warnings = check_unsafe_variables(rule)
        assert len(errors) == 1
        assert "Y" in errors[0]
```

### Key Test Files

- `tests/unit/test_asp_validators.py` - ASP syntax validation (110+ tests)
- `tests/unit/neural/` - LLM interface tests
- `tests/unit/translation/` - Translation layer tests (200+ tests)
- `tests/unit/validation/` - Validation pipeline tests (100+ tests)

---

## 6. Common Tasks

### Adding a New Validation Rule

```python
# 1. Add the validation function in loft/validation/asp_validators.py
def check_my_new_validation(rule_text: str) -> Tuple[List[str], List[str]]:
    """Check for my new validation rule."""
    errors = []
    warnings = []
    # Your validation logic here
    return errors, warnings

# 2. Add tests in tests/unit/test_asp_validators.py
class TestMyNewValidation:
    def test_valid_case(self) -> None:
        errors, warnings = check_my_new_validation("valid_rule(X) :- foo(X).")
        assert len(errors) == 0
```

### Adding LLM Provider Support

```python
# 1. Create provider in loft/neural/providers/
class MyProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "default"):
        self.api_key = api_key
        self.model = model

    def query(self, prompt: str) -> str:
        # Implementation here
        pass

# 2. Register in loft/neural/__init__.py
```

### Adding a New Legal Domain

1. Create dataset in `datasets/my_domain/`
2. Add domain predicates in `loft/legal/`
3. Create test cases following the JSON schema
4. Run `python scripts/validate_dataset.py datasets/my_domain/`

### Running Batch Processing

```bash
# Process a dataset
python scripts/run_batch_learning.py --dataset datasets/contracts/

# With custom config
python scripts/run_batch_learning.py \
    --dataset datasets/contracts/ \
    --max-cases 100 \
    --checkpoint-interval 25
```

### Running Autonomous Tests

```bash
# Start a 2-hour autonomous run
python -m loft.autonomous.cli start \
    --dataset datasets/contracts/ \
    --duration 2h \
    --checkpoint-interval 15

# Resume from checkpoint
python -m loft.autonomous.cli resume \
    --checkpoint data/autonomous_runs/run_001/checkpoints/latest.json
```

---

## 7. Validation and Review Process

### PR Validation Checklist

Every PR must validate:

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Linter passes (`ruff check loft/`)
- [ ] Type checker passes (`mypy loft/`)
- [ ] Coverage hasn't dropped below 70%
- [ ] Self-reflexivity preserved (if modifying core)
- [ ] Ontological bridge integrity (if modifying translation)

### PR Template

```markdown
## Summary
[1-2 sentence description]

Closes #XX

## Changes
- Added X
- Modified Y
- Fixed Z

## Validation Results
- [ ] All existing tests pass
- [ ] New tests: X added
- [ ] Coverage: XX%

## Test Examples

Run these commands to verify:

\`\`\`bash
# Run new tests
python -m pytest tests/unit/test_new_feature.py -v

# Run integration test
python -m pytest tests/integration/test_feature_integration.py -v
\`\`\`

## Testing Output

\`\`\`
[Paste test output here]
\`\`\`
```

### Code Review Standards

Reviewers check for:

1. **Correctness** - Does it work as intended?
2. **Tests** - Are edge cases covered?
3. **Simplicity** - Is it the simplest solution?
4. **Consistency** - Does it follow project patterns?
5. **Safety** - Any security or stability concerns?

---

## 8. Getting Help

### Documentation

- **README.md** - Project overview and examples
- **ROADMAP.md** - Development phases and MVP criteria
- **CLAUDE.md** - Architectural principles and guidelines
- **docs/TESTING.md** - Comprehensive testing guide
- **docs/MAINTAINABILITY.md** - LinkedASP strategy

### Communication

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - Questions and design discussions
- **Pull Request Comments** - Code-specific questions

### Before Asking Questions

- [ ] Search existing issues and discussions
- [ ] Check the troubleshooting section (Section 10)
- [ ] Review relevant documentation (README.md, ROADMAP.md)
- [ ] Prepare a minimal reproducible example

**Response time expectation:** Maintainers typically respond within 2-3 business days.

### Good vs Bad Questions

**Bad question:**
> "Tests are failing, help!"

**Good question:**
> When running `pytest tests/unit/test_asp_validators.py`, I get `ImportError` for `ValidationContext`. I'm on Python 3.11, installed via pip. Here's the full traceback: [...]

### Detailed Question Template

```markdown
## Question: How do I add a new validation rule?

**Goal:** I want to validate that ASP rules don't use reserved predicates.

**What I tried:**
- Added a function in asp_validators.py
- Tests pass locally but fail in CI

**Environment:**
- Python 3.11.4
- OS: macOS 14.0
- Installed via: pip

**Error:**
\`\`\`
ImportError: cannot import name 'check_reserved_predicates' from 'loft.validation.asp_validators'
\`\`\`

**Code:**
\`\`\`python
def check_reserved_predicates(rule_text: str) -> Tuple[List[str], List[str]]:
    ...
\`\`\`

**Steps to reproduce:**
1. Clone repo
2. Run `python -m pytest tests/unit/test_my_validator.py -v`
```

---

## 9. Project Principles

From CLAUDE.md, these principles guide all development:

### 1. Self-Reflexivity is Paramount

The symbolic core's ability to reason about and modify itself is the central innovation. Every change must preserve this capability.

**Validation:** Before merging, verify the core can still:
- Identify gaps in its own knowledge
- Question its own logic
- Generate candidates for self-improvement
- Validate proposed modifications

### 2. Validation at Every Level

This is a complex, multi-layered system. Validation must occur continuously:

- **Syntactic** - Rules are well-formed
- **Semantic** - No logical contradictions
- **Empirical** - Performance on test cases
- **Meta** - System can validate its own validators

### 3. Ontological Bridge Integrity

The symbolic-neural translation must maintain semantic fidelity:

- Roundtrip translation preserves meaning
- LLM outputs are grounded in symbolic terms
- Ambiguities are explicit, not silently resolved

### 4. Safety First

Self-modification is powerful but dangerous:

- **Stratified modification authority** - Constitutional layer is immutable
- **Confidence thresholds** - Rules below threshold are flagged
- **Rollback mechanisms** - All states are versioned
- **Human review** - High-impact changes require approval

### 5. Avoid Over-Engineering

Keep solutions simple and focused:

- Only make changes directly requested
- Don't add features beyond what was asked
- Don't create abstractions for one-time operations
- The right complexity is the minimum needed

---

## 10. Troubleshooting

### Common Issues

#### Tests Failing Locally

```bash
# Ensure dependencies are up to date
pip install -r requirements.txt -r requirements-dev.txt

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Run tests with verbose output
python -m pytest tests/unit/ -v --tb=long
```

#### Import Errors

```bash
# Ensure you're in the project root
cd /path/to/loft

# Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run with module syntax
python -m pytest tests/unit/
```

#### LLM API Errors

```bash
# Check API key is set
echo $ANTHROPIC_API_KEY

# Verify .env file exists
cat .env

# Test API connection
python -c "from loft.neural import AnthropicProvider; print('OK')"
```

#### Clingo Not Found

```bash
# Install clingo
pip install clingo

# Or via conda
conda install -c potassco clingo

# Verify installation
clingo --version
```

#### Memory Issues in Tests

```bash
# Run tests with limited parallelism
python -m pytest tests/unit/ -v -n 2

# Or run sequentially
python -m pytest tests/unit/ -v -n 0
```

#### CI Failures After Local Success

```bash
# Check CI logs for specific error
gh pr checks <pr-number>

# Ensure all files are committed
git status

# Check for environment-specific issues
python --version
pip list | grep -E "(pytest|ruff|mypy)"
```

### Getting Debug Information

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Print validation details
from loft.validation import ASPSyntaxValidator
validator = ASPSyntaxValidator()
result = validator.validate_generated_rule("your_rule_here.")
print(f"Valid: {result.is_valid}")
print(f"Details: {result.details}")
```

### Useful Debug Commands

```bash
# Check what tests exist for a module
python -m pytest tests/unit/ --collect-only | grep test_validation

# Run tests matching a pattern
python -m pytest tests/unit/ -v -k "unsafe_variable"

# Show test durations
python -m pytest tests/unit/ -v --durations=10
```

---

## Next Steps

1. **Read ROADMAP.md** to understand the current development phase
2. **Browse open issues** labeled `good first issue`
3. **Pick an issue** and comment that you're working on it
4. **Ask questions** if anything is unclear
5. **Submit your first PR!**

Welcome to the LOFT project. We're excited to have you contributing to building the ontological bridge between symbolic and neural AI!
