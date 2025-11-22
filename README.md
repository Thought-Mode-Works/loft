# LOFT - Reflexive Neuro-Symbolic AI Architecture

> Building a self-reflexive symbolic core that autonomously reconstructs its own logic using LLM outputs, creating an ontological bridge between symbolic reasoning and neural pattern recognition.

## Vision

LOFT (Logical Ontological Framework for Thought) is an ambitious research project exploring the intersection of classical symbolic AI and modern large language models. The system features a **self-reflexive symbolic core** capable of:

- Questioning its own logic
- Identifying gaps in its knowledge
- Using LLMs to generate candidate logical rules
- Validating and incorporating new rules autonomously
- Reasoning about its own reasoning processes

This creates a continuous self-improvement loop where symbolic and neural components work together, each compensating for the other's limitations.

## Core Innovation: The Ontological Bridge

Traditional AI systems are either:
- **Symbolic**: Precise, explainable, but brittle and hand-coded
- **Neural**: Flexible, learning-capable, but opaque and inconsistent

LOFT bridges these approaches by:

1. **Symbolic Core**: Maintains logical consistency, compositional reasoning, explainability
2. **Neural Components**: LLMs provide pattern recognition, natural language understanding, creative rule generation
3. **Reflexive Orchestrator**: Meta-reasoning layer that questions and improves the system itself
4. **Validation Framework**: Rigorous multi-stage verification ensures quality and safety

The "bridge" is bidirectional translation that preserves semantic meaning across incompatible representational paradigms.

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│         Reflexive Meta-Reasoner                 │  ← Reasons about reasoning
├─────────────────────────────────────────────────┤
│         Symbolic Core (Self-Modifying)          │  ← Logical rules, compositional
│  - Constitutional Layer (immutable)             │
│  - Strategic Layer (slow change)                │
│  - Tactical Layer (frequent updates)            │
│  - Operational Layer (rapid adaptation)         │
├─────────────────────────────────────────────────┤
│      Translation Layer (Ontological Bridge)     │  ← Symbolic ↔ Natural Language
├─────────────────────────────────────────────────┤
│         Neural Interface & LLM Ensemble         │  ← Multiple specialized LLMs
│  - Logic Generator | Critic | Translator       │
│  - Meta-Reasoner | Synthesizer                  │
├─────────────────────────────────────────────────┤
│      Validation & Verification Framework        │  ← Multi-stage validation
│  - Syntactic | Semantic | Empirical | Meta     │
└─────────────────────────────────────────────────┘
```

## Initial Domain: Legal Reasoning

The first implementation targets legal document analysis because:

- **Structured + Uncertain**: Legal rules are symbolic but their application requires pattern recognition
- **Meta-Reasoning**: Legal analysis constantly reasons about reasoning (precedent, analogies, distinctions)
- **Verifiable**: Court decisions provide ground truth for validation
- **Natural Language Bridge**: Legal concepts naturally exist in structured language

## Development Phases

See [ROADMAP.md](ROADMAP.md) for detailed phased buildout plan.

**High-Level Phases:**

- **Phase 0-1** (Weeks 1-5): Foundation & Static Core
- **Phase 2-3** (Weeks 6-13): Logic Generation & Safe Self-Modification
- **Phase 4-5** (Weeks 14-22): Dialectical Reasoning & Meta-Reasoning
- **Phase 6-7** (Weeks 23-30): Neural Ensemble & Geometric Constraints
- **Phase 8-9** (Weeks 31-40): Multi-Domain & Production Hardening

Each phase has **MVP validation criteria** that must be met before proceeding.

## Key Technical Components

### Symbolic Core

- **Representation**: Datalog/Answer Set Programming for non-monotonic reasoning
- **Structure**: Stratified layers with different modification authorities
- **Composition**: Ring structure for principled rule combination
- **Versioning**: Git-like semantics for core state management

### Neural Components

- **Multi-LLM Strategy**: Specialized models for different tasks
- **Structured Output**: JSON schema constraints on LLM generation
- **Prompt Engineering**: Versioned, testable, optimized prompts
- **Uncertainty Tracking**: Confidence scores on all outputs

### Translation Layer

- **Bidirectional**: Symbolic → NL and NL → Symbolic
- **Fidelity Testing**: Roundtrip preservation of semantic meaning
- **Ambiguity Handling**: Explicit representation of unclear concepts
- **Provenance Tracking**: Every translation links to sources

### Validation Framework

- **Syntactic**: Well-formed rules, type safety
- **Semantic**: Logical consistency, no contradictions
- **Empirical**: Performance on test cases
- **Meta**: System validates its own validation

## Core Principles

### 1. Self-Reflexivity is Paramount

The symbolic core's ability to reason about and modify itself is the central innovation. All development preserves this capability.

### 2. Validation at Every Level

Complex builds require continuous validation to prove progress toward goals. Every claim must be backed by tests.

### 3. Ontological Bridge Integrity

The symbolic-neural translation must maintain semantic fidelity. Meaning cannot be lost crossing the bridge.

### 4. Experiential Learning

The system learns from experience and external validation, updating based on what works in practice.

### 5. Safety First

Self-modification is powerful but dangerous. Multiple safeguards prevent catastrophic changes.

## Installation

*Coming in Phase 0*

```bash
# Prerequisites
- Python 3.11+
- Access to LLM APIs (Anthropic, OpenAI, or local models)

# Setup
git clone https://github.com/yourusername/loft.git
cd loft
pip install -r requirements.txt
```

## Usage

*Coming in Phase 1*

```python
from loft import SymbolicCore, LLMInterface

# Initialize system
core = SymbolicCore(domain="contract_law")
llm = LLMInterface(model="claude-3-opus")

# Query the system
result = core.query("Does this contract satisfy statute of frauds?",
                    facts=contract_facts)

# System can explain its reasoning
explanation = result.explain()
```

## Validation & Testing

See [CLAUDE.md](CLAUDE.md) for comprehensive development guidelines.

**Testing Philosophy:**
- Unit tests for individual components
- Property tests for logical invariants
- Integration tests for symbolic-neural interaction
- Meta-tests for reflexive capabilities
- Validation tests for each phase's MVP criteria

**Key Metrics:**
- Prediction Accuracy: >85%
- Logical Consistency: 100%
- Translation Fidelity: >95%
- Confidence Calibration: ±5%

## Contributing

This is currently a research project. Contributions welcome once Phase 1 is complete.

**Development Workflow:**
1. Read ROADMAP.md to understand current phase
2. Follow guidelines in CLAUDE.md
3. All PRs must include validation results
4. Maintain self-reflexivity and ontological bridge integrity

## Research Foundation

This project builds on research in:

- **Neuro-Symbolic AI**: Hybrid architectures combining logic and learning
- **Meta-Reasoning**: Systems that reason about their own reasoning
- **Program Synthesis**: LLM-guided code/logic generation
- **Cognitive Architecture**: Dual-process reasoning systems
- **Lifelong Learning**: Continual autonomous improvement

Key papers and resources documented in [thoughts.md](thoughts.md).

## Project Status

**Current Phase**: Phase 0 - Foundation & Validation Framework

**Recent Updates:**
- ✅ Repository initialized
- ✅ Roadmap created with phased validation criteria
- ✅ Development guidelines established (CLAUDE.md)
- 🚧 Foundation infrastructure (in progress)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

Copyright 2025 LOFT Contributors

## Acknowledgments

Built on decades of research in symbolic AI, neural networks, cognitive science, and legal reasoning. Standing on the shoulders of giants.

---

**Note**: This is a research project exploring fundamental questions about AI, reasoning, and self-improvement. The goal is not just a working system, but understanding the nature of the ontological bridge between symbolic and neural computation.
