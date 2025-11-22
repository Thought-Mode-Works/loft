# Core: ASP-based Symbolic Reasoning

This module contains the ASP/Clingo-based symbolic core that:

## Responsibilities

- **Load and manage ASP programs** across stratified layers (constitutional, strategic, tactical, operational)
- **Perform logical reasoning and querying** using Answer Set Programming semantics
- **Detect knowledge gaps** that require LLM input for resolution
- **Support self-modification** with validation and rollback capabilities
- **Track rule provenance** and confidence scores
- **Version control** for symbolic core states

## Key Components (to be implemented)

- `asp_core.py` - Main interface to Clingo solver
- `stratified_core.py` - Manages 4-layer stratification architecture
- `rule.py` - ASP rule representation and metadata
- `query.py` - Query interface and result handling
- `gap_detector.py` - Identifies missing knowledge requiring LLM

## Example Usage (planned)

```python
from loft.core import ASPCore

# Initialize core
core = ASPCore()

# Load stratified programs
core.load_constitutional("programs/constitutional/base.lp")
core.load_strategic("programs/strategic/legal_reasoning.lp")

# Query
results = core.query("enforceable(C)")

# Detect gaps
gaps = core.detect_knowledge_gaps()
```

## Integration Points

- **Validation** (`loft.validation`): Consistency checking, metrics
- **Translation** (`loft.translation`): ASP ↔ NL conversion
- **Neural** (`loft.neural`): LLM queries for gap filling
- **Meta** (`loft.meta`): Self-modification orchestration
