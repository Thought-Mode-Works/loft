# LOFT Documentation

Welcome to the **LOFT (Logical Ontological Framework Translator)** documentation!

LOFT is a neuro-symbolic AI system that combines Answer Set Programming (ASP) with large language models (LLMs) for legal reasoning and knowledge management.

## Documentation Structure

### 🏗️ Architecture

Comprehensive architecture diagrams and system design documentation.

- **[System Overview](architecture/system-overview.md)** - Complete system architecture with Mermaid diagrams
  - System architecture overview
  - Stratified core architecture
  - Validation pipeline
  - Translation flow
  - Version control flow
  - Data flow diagram
  - Module interactions

### 📐 Architecture Decision Records (ADRs)

Key design decisions and their rationale.

- **[ADR-001: Why ASP Over Datalog](adr/ADR-001-why-asp-over-datalog.md)**
  - Why we chose Answer Set Programming
  - Comparison with Datalog, Prolog, and Description Logics
  - Trade-offs and consequences

- **[ADR-002: Stratified Core Design](adr/ADR-002-stratified-core-design.md)**
  - Four-layer stratification (Constitutional → Strategic → Tactical → Operational)
  - Modification policies and validation requirements
  - Safe self-modification approach

- **[ADR-003: Multi-Stage Validation Framework](adr/ADR-003-validation-framework-approach.md)**
  - Syntactic → Semantic → Empirical → Consensus validation pipeline
  - Stage-by-stage filtering for efficiency
  - Performance monitoring and automatic rollback

### 📚 API Reference

Detailed API documentation for all modules.

#### Validation
- **[Validation Framework](api/validation/validation-framework.md)**
  - `ValidationPipeline`, `ASPSyntaxValidator`, `ASPSemanticValidator`
  - `TestCaseValidator`, `ValidationCase`, `TestResult`
  - Multi-stage validation workflow

#### Symbolic Core (Coming Soon)
- ASP rule management
- Stratification layers
- Rule execution with Clingo

#### Version Control (Coming Soon)
- Git-like workflow for rules
- Commit, branch, merge operations
- Diff computation and conflict resolution

#### Neural Interface (Coming Soon)
- LLM provider abstraction
- Structured input/output
- Cost tracking and caching

#### Translation (Coming Soon)
- ASP ↔ Natural Language conversion
- Fidelity checking
- Template-based generation

### 💡 Examples

Runnable code examples demonstrating key functionality.

- **[validate_asp_program.py](examples/validate_asp_program.py)**
  - Syntax validation
  - Semantic consistency checking
  - Empirical test case validation
  - Individual validator usage

### 📖 Additional Resources

- **[MAINTAINABILITY.md](MAINTAINABILITY.md)** - Long-term maintainability strategy and LinkedASP integration
- **[../ROADMAP.md](../ROADMAP.md)** - Development roadmap and phase planning
- **[../CLAUDE.md](../CLAUDE.md)** - Development guidelines and best practices
- **[../README.md](../README.md)** - Project overview and quick start

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/justin4957/loft.git
cd loft

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
pytest tests/
```

### Basic Usage

```python
from loft.validation import validate_asp_program, ValidationPipeline

# Quick syntax check
is_valid, error = validate_asp_program("contract(X) :- offer(X), acceptance(X).")
print(f"Valid: {is_valid}")

# Full validation pipeline
pipeline = ValidationPipeline()
report = pipeline.validate(
    asp_program="enforceable(C) :- contract(C), has_writing(C).",
    test_cases=[...],
    threshold=0.80
)
print(f"Validation passed: {report.passed}")
```

See [examples/](examples/) for more detailed examples.

## System Architecture Overview

LOFT combines symbolic reasoning (ASP) with neural reasoning (LLMs):

```
User Query
    ↓
Meta-Reasoning Layer
    ↓
Symbolic Core (ASP) ←→ Neural Interface (LLMs)
    ↓
Validation Framework
    ↓
Version Control & Logging
```

Key features:
- **Stratified rule base**: Constitutional → Strategic → Tactical → Operational
- **Multi-stage validation**: Syntax → Semantic → Empirical → Consensus
- **Git-like versioning**: Commit, branch, merge, rollback
- **Comprehensive logging**: Structured JSON for observability
- **Self-modifying**: System can propose and incorporate new rules

See [Architecture Overview](architecture/system-overview.md) for detailed diagrams.

## Core Concepts

### Answer Set Programming (ASP)

LOFT uses ASP (via Clingo) for symbolic reasoning:

```asp
% Rules with negation as failure
enforceable(C) :- contract(C), not void(C).

% Exceptions
void(C) :- contract(C), fraud(C).

% Constraints
:- enforceable(C), void(C).  % Cannot be both!
```

**Why ASP?** Nonmonotonic reasoning, default logic, constraints. See [ADR-001](adr/ADR-001-why-asp-over-datalog.md).

### Stratification

Rules are organized into four confidence-based layers:

1. **Constitutional (Stratum 0)**: Immutable axioms (confidence = 1.0)
2. **Strategic (Stratum 1)**: High-confidence rules (≥0.90)
3. **Tactical (Stratum 2)**: Medium-confidence rules (≥0.75)
4. **Operational (Stratum 3)**: Low-confidence rules (≥0.50)

Higher strata override lower strata in conflicts. See [ADR-002](adr/ADR-002-stratified-core-design.md).

### Validation Pipeline

Four sequential validation stages:

1. **Syntactic**: Parse with Clingo (~10ms)
2. **Semantic**: Check consistency (~50-200ms)
3. **Empirical**: Run test cases (~100-500ms)
4. **Consensus**: Multi-agent agreement (~5-10s, Phase 2+)

See [ADR-003](adr/ADR-003-validation-framework-approach.md) and [Validation API](api/validation/validation-framework.md).

### Neuro-Symbolic Integration

LOFT combines:
- **Symbolic (ASP)**: Logical reasoning, provable guarantees
- **Neural (LLMs)**: Natural language understanding, knowledge extraction

LLMs are used for:
- Translating legal text → ASP rules
- Explaining ASP conclusions → natural language
- Proposing new rules (validated before incorporation)
- Multi-agent consensus validation

## Development Phases

LOFT is being developed in phases:

- **Phase 0** ✅: Foundational infrastructure (validation, versioning, logging)
- **Phase 1**: Translation layer (ASP ↔ Natural Language)
- **Phase 1.5**: LinkedASP integration (RDF queryability)
- **Phase 2**: Advanced validation (consensus, dialectical reasoning)
- **Phase 3**: Self-modification (autonomous improvement)
- **Phase 4**: Distributed deployment

See [ROADMAP.md](../ROADMAP.md) for details.

## Contributing

### Documentation Guidelines

- **Architecture diagrams**: Use Mermaid (embeddable in markdown)
- **API docs**: Include signature, parameters, return values, examples
- **ADRs**: Follow template (Context → Decision → Rationale → Consequences)
- **Examples**: Must be runnable and tested

### Running Examples

```bash
# Set PYTHONPATH
export PYTHONPATH=/path/to/loft:$PYTHONPATH

# Run example
python3 docs/examples/validate_asp_program.py
```

### Testing Documentation

```bash
# Test all examples
for example in docs/examples/*.py; do
    echo "Testing $example"
    python3 "$example" || echo "FAILED: $example"
done
```

## Additional Documentation

### External Resources

- [Clingo Documentation](https://potassco.org/clingo/) - ASP solver
- [ASP Tutorial](https://potassco.org/doc/start/) - Learn Answer Set Programming
- [Legal Reasoning with ASP](https://arxiv.org/abs/1805.09925) - Research paper

### Related Projects

- **Potassco**: ASP tools and solvers (Clingo, Gringo, Clasp)
- **s(CASP)**: Constraint ASP with justifications
- **ASPIC+**: Argumentation framework for defeasible reasoning

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/justin4957/loft/issues)
- **Discussions**: [GitHub Discussions](https://github.com/justin4957/loft/discussions)
- **Email**: [Maintainer contact info]

## License

[License information]

---

**Last Updated**: 2025-11-27
**Documentation Version**: 1.0
**LOFT Version**: Phase 0 (Foundational Infrastructure)
