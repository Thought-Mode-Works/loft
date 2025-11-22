# ASP Programs Directory

This directory contains Answer Set Programming (ASP) programs organized by stratification layer.

## Stratified Architecture

The symbolic core uses a 4-layer stratification with different modification authorities:

### Constitutional Layer (`constitutional/`)
- **Purpose**: Immutable core axioms, safety constraints, fundamental principles
- **Modification Authority**: Human approval required (threshold = 1.0)
- **Examples**:
  - Core logical axioms
  - Safety constraints (prevent catastrophic self-modification)
  - Fundamental legal principles

### Strategic Layer (`strategic/`)
- **Purpose**: High-level reasoning patterns that change slowly
- **Modification Authority**: High confidence threshold (0.9+), human review recommended
- **Examples**:
  - General legal reasoning frameworks
  - Meta-reasoning patterns
  - Cross-domain abstractions

### Tactical Layer (`tactical/`)
- **Purpose**: Domain-specific rules that update frequently
- **Modification Authority**: Moderate confidence threshold (0.8+), automated with validation
- **Examples**:
  - Statute of frauds rules
  - Contract formation requirements
  - Specific legal doctrines

### Operational Layer (`operational/`)
- **Purpose**: Immediate problem-solving heuristics that adapt rapidly
- **Modification Authority**: Low confidence threshold (0.6+), rapid iteration
- **Examples**:
  - Case-specific patterns
  - Optimization heuristics
  - Temporary working hypotheses

## File Organization

Each layer should contain `.lp` files (Logic Program files in ASP syntax):

```
programs/
├── constitutional/
│   └── base.lp               # Core axioms and constraints
├── strategic/
│   └── legal_reasoning.lp    # High-level legal patterns
├── tactical/
│   ├── statute_of_frauds.lp  # Specific legal rules
│   └── contract_formation.lp
└── operational/
    └── heuristics.lp         # Problem-solving patterns
```

## ASP File Format

Files should use standard ASP-Core-2 syntax with documentation:

```asp
%%% ============================================
%%% MODULE: Statute of Frauds Rules
%%% LAYER: Tactical
%%% CONFIDENCE: 0.85
%%% LAST_MODIFIED: 2025-01-15
%%% ============================================

% Contract within statute if it's a land sale
within_statute(C) :- land_sale_contract(C).

% Enforceable if has sufficient writing
satisfies_statute_of_frauds(C) :-
    within_statute(C),
    has_sufficient_writing(C).
```

## Loading Programs

Programs are loaded by the ASP core in order of stratification:

```python
from loft.core import ASPCore

core = ASPCore()
core.load_stratified_programs()  # Loads all layers in order
```

## Version Control

All ASP programs in this directory are version controlled via git.
The system also maintains its own internal version control for tracking
symbolic core evolution.
