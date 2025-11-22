# Tests Directory

This directory contains the test suite for LOFT, organized by test type.

## Test Organization

### Unit Tests (`unit/`)
Tests for individual functions and classes in isolation.

**Examples:**
- ASP parsing functions
- Translation helpers
- Configuration validation
- Individual validators

**File naming:** `test_<module_name>.py`

### Integration Tests (`integration/`)
Tests for interactions between components.

**Examples:**
- ASP core + LLM interface
- Translation layer roundtrip (ASP → NL → ASP)
- Validation pipeline
- Full query flow

**File naming:** `test_<integration_scenario>.py`

### Property Tests (`property/`)
Property-based tests using Hypothesis framework.

**Examples:**
- Adding ASP facts preserves consistency
- Rule composition is associative
- Translation fidelity properties
- Invariants under operations

**File naming:** `test_<property_category>_properties.py`

## Running Tests

```bash
# Run all tests
pytest

# Run specific test type
pytest tests/unit
pytest tests/integration
pytest tests/property

# Run with coverage
pytest --cov=loft --cov-report=html

# Run specific test file
pytest tests/unit/test_config.py

# Run with verbose output
pytest -v
```

## Writing Tests

### Unit Test Example

```python
# tests/unit/test_config.py
from loft.config import Config

def test_config_defaults():
    """Test that config has sensible defaults"""
    config = Config()
    assert config.llm.provider == "anthropic"
    assert config.validation.consistency_check is True

def test_config_from_env(monkeypatch):
    """Test configuration from environment variables"""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    config = Config.from_env()
    assert config.llm.provider == "openai"
```

### Integration Test Example

```python
# tests/integration/test_translation_roundtrip.py
from loft.core import ASPCore
from loft.translation import ASPToNLTranslator, NLToASPTranslator

def test_asp_nl_asp_fidelity():
    """Test that ASP → NL → ASP preserves meaning"""
    original = "satisfies_statute_of_frauds(C) :- has_writing(C, W)."

    asp_to_nl = ASPToNLTranslator()
    nl_to_asp = NLToASPTranslator()

    nl = asp_to_nl.translate(original)
    reconstructed = nl_to_asp.translate(nl)

    # Check semantic equivalence
    assert semantically_equivalent(original, reconstructed)
```

### Property Test Example

```python
# tests/property/test_asp_properties.py
from hypothesis import given, strategies as st
from loft.core import ASPCore

@given(st.text(min_size=1, max_size=20))
def test_adding_fact_preserves_consistency(fact_name):
    """Property: Adding a fact should not break consistency"""
    core = ASPCore()
    assert core.is_consistent()

    core.add_fact(f"{fact_name}(test).")
    assert core.is_consistent()
```

## Test Coverage Goals

- **Phase 0**: 80%+ coverage for infrastructure
- **Phase 1**: 85%+ coverage for core functionality
- **Phase 2+**: 90%+ coverage overall

## Continuous Integration

Tests run automatically on:
- Every push to repository
- Every pull request
- Before merging to main

See `.github/workflows/` for CI configuration.
