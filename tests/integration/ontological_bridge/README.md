# Ontological Bridge Validation Suite

Comprehensive tests for the NL ↔ ASP translation layer (ontological bridge).

## Overview

The ontological bridge is critical to LOFT's architecture - it must maintain semantic fidelity when translating between natural language and ASP representations. This test suite validates that translations preserve meaning, handle edge cases, and adequately represent legal concepts.

## Test Modules

### 1. Translation Fidelity (`test_translation_fidelity.py`)
Tests round-trip translation (NL → ASP → NL) to ensure semantic preservation.

**Key Tests:**
- `test_round_trip_translation`: Individual round-trip tests
- `test_batch_translation_fidelity`: Aggregate fidelity metrics
- `test_information_loss`: Validates key concepts are preserved
- `test_translation_confidence`: Checks confidence scores are reasonable
- `test_all_translation_cases`: Comprehensive test of all cases (marked slow)

**Acceptance Criteria:**
- Semantic similarity > 0.7 (70%)
- Overall fidelity > 0.65 (65%)
- Information loss < 0.35 (35%)
- Hallucination rate < 0.3 (30%)

**Usage:**
```bash
# Run fidelity tests
pytest tests/integration/ontological_bridge/test_translation_fidelity.py -v

# Run with specific test
pytest tests/integration/ontological_bridge/test_translation_fidelity.py::TestTranslationFidelity::test_round_trip_translation -v

# Skip slow tests
pytest tests/integration/ontological_bridge/test_translation_fidelity.py -v -m "not slow"
```

### 2. Edge Cases (`test_edge_cases.py`)
Tests handling of challenging translation scenarios.

**Test Classes:**
- `TestAmbiguityHandling`: Scope ambiguity, quantifier ambiguity
- `TestNegationHandling`: Simple, double, implicit negation
- `TestComplexStructures`: Nested conditions, multiple conditions
- `TestContradictionHandling`: Direct and implicit contradictions
- `TestDomainSpecificEdgeCases`: Legal jargon, monetary amounts, temporal refs
- `TestEmptyAndMinimalInputs`: Edge cases like empty input, single words
- `TestRobustnessStress`: Malformed input stress tests

**Usage:**
```bash
# Run edge case tests
pytest tests/integration/ontological_bridge/test_edge_cases.py -v

# Run specific test class
pytest tests/integration/ontological_bridge/test_edge_cases.py::TestNegationHandling -v
```

### 3. Representational Adequacy (`test_representational_adequacy.py`)
Tests that ASP can adequately express legal concepts.

**Test Classes:**
- `TestConceptualDistinctions`: Necessary vs sufficient, universal vs existential
- `TestLegalConceptCoverage`: Contract formation, validity, statute of frauds concepts
- `TestRelationalConcepts`: Parties, temporal relations, causal relations
- `TestHierarchicalConcepts`: Type hierarchies, exception hierarchies
- `TestCompositionality`: Conjunction, disjunction, nested composition
- `TestNegationAdequacy`: Classical negation, negation as failure
- `TestConstraintRepresentation`: Numeric, cardinality, uniqueness constraints
- `TestComprehensiveCoverage`: Full statute of frauds coverage (marked slow)

**Usage:**
```bash
# Run representational tests
pytest tests/integration/ontological_bridge/test_representational_adequacy.py -v

# Run comprehensive coverage (slow)
pytest tests/integration/ontological_bridge/test_representational_adequacy.py::TestComprehensiveCoverage -v
```

## Utility Modules

### Semantic Similarity (`utils/semantic_similarity.py`)
Calculates semantic similarity between texts using sentence embeddings.

**Features:**
- Sentence-BERT embeddings (if available) with in-memory caching of embeddings
- Configurable toggle (`LOFT_USE_EMBEDDING_SIMILARITY=true|false`)
- Fallback to token-based Jaccard similarity
- Batch similarity calculation with cached vectors
- Cosine similarity utilities

**Usage:**
```python
from tests.integration.ontological_bridge.utils.semantic_similarity import semantic_similarity

score = semantic_similarity("Contract requires writing", "Agreement needs to be written")
# Returns: 0.85 (high similarity)
```

**Models:**
- Default: `all-MiniLM-L6-v2` (fast, 80MB)
- Alternative: `all-mpnet-base-v2` (slower, better quality)

### Fidelity Metrics (`utils/metrics.py`)
Calculates comprehensive translation fidelity metrics.

**Metrics:**
- `semantic_similarity`: Embedding-based similarity (0.0-1.0)
- `information_preservation`: How much information is preserved (0.0-1.0)
- `hallucination_rate`: Content not in original (0.0-1.0, lower is better)
- `structural_accuracy`: Structural similarity (0.0-1.0)
- `overall_fidelity`: Weighted average (0.0-1.0)

**Usage:**
```python
from tests.integration.ontological_bridge.utils.metrics import calculate_fidelity

metrics = calculate_fidelity(
    original_text="Contract requires writing",
    translated_text="Agreement needs written form",
    semantic_similarity_score=0.85
)

print(metrics.overall_fidelity)  # 0.82
print(metrics.hallucination_rate)  # 0.15
```

## Dependencies

Required:
- `pytest` - Test framework
- `numpy` - Numerical operations
- `sentence-transformers` - For semantic similarity

Optional (recommended):
- If `sentence-transformers` is not available, falls back to token-based similarity.

## Running Tests

**Quick validation:**
```bash
# Fast subset (core functionality)
pytest tests/integration/ontological_bridge/ -v --timeout=300 -m "not slow"
```

**Comprehensive validation:**
```bash
# All tests including slow ones
pytest tests/integration/ontological_bridge/ -v --timeout=600
```

**With coverage:**
```bash
pytest tests/integration/ontological_bridge/ \
  --cov=loft.translation \
  --cov-report=html \
  --cov-report=term
```

**Specific test patterns:**
```bash
# Only fidelity tests
pytest tests/integration/ontological_bridge/test_translation_fidelity.py -v

# Only edge cases
pytest tests/integration/ontological_bridge/test_edge_cases.py -v

# Only adequacy tests
pytest tests/integration/ontological_bridge/test_representational_adequacy.py -v
```

## Test Markers

- `@pytest.mark.slow` - Long-running tests (>30s)
- `@pytest.mark.parametrize` - Parameterized tests with multiple cases

## Expected Results

**Translation Fidelity:**
- Average semantic similarity: >0.70
- Average overall fidelity: >0.65
- Average hallucination rate: <0.30

**Edge Cases:**
- All edge case types should be handled gracefully
- No crashes on malformed input
- Confidence scores should reflect uncertainty

**Representational Adequacy:**
- Coverage of statute of frauds concepts: >80%
- All major legal concepts representable
- Conceptual distinctions preserved

## Troubleshooting

**Import errors:**
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=/path/to/loft:$PYTHONPATH
pytest tests/integration/ontological_bridge/ -v
```

**Sentence-transformers not available:**
```bash
# Install optional dependency
pip install sentence-transformers

# Or accept fallback to token-based similarity (lower quality but functional)
```

**API rate limits:**
```bash
# Run subset of tests
pytest tests/integration/ontological_bridge/test_translation_fidelity.py::TestTranslationFidelity::test_batch_translation_fidelity -v

# Use smaller batch sizes in test code
```

**Slow execution:**
```bash
# Skip slow tests
pytest tests/integration/ontological_bridge/ -v -m "not slow"

# Run in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest tests/integration/ontological_bridge/ -v -n auto
```

## Continuous Integration

Recommended CI configuration:

```yaml
test-ontological-bridge:
  script:
    - pip install sentence-transformers
    - pytest tests/integration/ontological_bridge/ -v -m "not slow" --timeout=300
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
```

## Future Enhancements

- [ ] Add multi-hop translation tests (NL→ASP→NL→ASP→NL)
- [ ] Test translation with context/dialogue history
- [ ] Measure translation latency and throughput
- [ ] Add adversarial test cases
- [ ] Test incremental translation (update vs regenerate)
- [ ] Compare different LLM models
- [ ] Add visualization of translation quality over time
