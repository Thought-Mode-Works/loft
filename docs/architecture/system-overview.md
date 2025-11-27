# LOFT System Architecture Overview

This document provides comprehensive architecture diagrams and explanations for the LOFT (Logical Ontological Framework Translator) neuro-symbolic AI system.

## 1. System Architecture Overview

```mermaid
graph TB
    User[User Query/NL Input] --> Meta[Meta-Reasoning Layer]
    Meta --> Symbolic[Symbolic Core<br/>ASP Rule Base]
    Meta --> Neural[Neural Interface Layer]

    Symbolic --> Translation[Translation Layer<br/>ASP ↔ NL]
    Translation --> Neural

    Neural --> LLM[LLM Providers<br/>Claude/GPT/Local]

    Symbolic --> Validation[Validation Framework<br/>Syntax/Semantic/Empirical]
    Validation --> Symbolic

    Symbolic --> VersionControl[Version Control<br/>Git-like Workflow]
    Symbolic --> Logging[Logging & Observability<br/>Structured JSON]
    Symbolic --> Consistency[Consistency Checker<br/>SAT/Answer Set]

    Consistency --> Validation
    VersionControl --> Logging

    style Symbolic fill:#e1f5ff
    style Neural fill:#fff4e1
    style Meta fill:#f0e1ff
    style Validation fill:#e1ffe1
```

**Key Components:**

- **Meta-Reasoning Layer**: High-level decision making and strategy selection
- **Symbolic Core**: Stratified ASP rule base (Constitutional → Strategic → Tactical → Operational)
- **Neural Interface**: LLM integration with structured I/O
- **Translation Layer**: Bidirectional ASP ↔ Natural Language conversion
- **Validation Framework**: Multi-stage validation pipeline
- **Version Control**: Git-like versioning for rule evolution
- **Consistency Checker**: Ensures logical consistency across rule base
- **Logging**: Comprehensive observability and debugging

## 2. Stratified Core Architecture

```mermaid
graph TD
    subgraph Constitutional["Constitutional Layer (Stratum 0)"]
        C1[Immutable Legal Axioms]
        C2[Foundational Definitions]
        C3[Core Legal Concepts]
    end

    subgraph Strategic["Strategic Layer (Stratum 1)"]
        S1[High-Confidence Rules]
        S2[Legal Doctrines]
        S3[Established Precedents]
    end

    subgraph Tactical["Tactical Layer (Stratum 2)"]
        T1[Medium-Confidence Rules]
        T2[Case-Specific Logic]
        T3[Contextual Rules]
    end

    subgraph Operational["Operational Layer (Stratum 3)"]
        O1[Low-Confidence Rules]
        O2[Experimental Rules]
        O3[Temporary Overrides]
    end

    Constitutional -->|Authority: Immutable| Strategic
    Strategic -->|Authority: High| Tactical
    Tactical -->|Authority: Medium| Operational

    Operational -.->|Self-Modification| Operational
    Tactical -.->|Proposed Changes| Tactical
    Strategic -.->|Requires Validation| Strategic
    Constitutional -.->|Forbidden| Constitutional

    style Constitutional fill:#ff6b6b
    style Strategic fill:#ffd93d
    style Tactical fill:#6bcf7f
    style Operational fill:#4d96ff
```

**Modification Authority:**

| Layer | Confidence Threshold | Self-Modification | Validation Required |
|-------|---------------------|-------------------|---------------------|
| Constitutional | 1.0 (Immutable) | ❌ Forbidden | N/A |
| Strategic | ≥ 0.90 | ⚠️ Restricted | Rigorous (All stages) |
| Tactical | ≥ 0.75 | ✅ Allowed | Standard (Syntax + Semantic) |
| Operational | ≥ 0.50 | ✅ Encouraged | Lightweight (Syntax only) |

## 3. Validation Pipeline

```mermaid
graph LR
    Input[Proposed Rule] --> Syntax[Syntactic Validation<br/>ASP Parser]
    Syntax -->|Valid| Semantic[Semantic Validation<br/>Consistency Check]
    Syntax -->|Invalid| Reject1[Reject]

    Semantic -->|Consistent| Empirical[Empirical Validation<br/>Test Cases]
    Semantic -->|Inconsistent| Reject2[Reject]

    Empirical -->|Pass| Consensus[Consensus Validation<br/>Multi-Agent Agreement]
    Empirical -->|Fail| Reject3[Reject]

    Consensus -->|Agreement| Accept[Accept & Incorporate]
    Consensus -->|Disagreement| Meta[Meta-Validation<br/>Human Review]

    Meta -->|Approve| Accept
    Meta -->|Deny| Reject4[Reject]

    Accept --> Monitor[Performance Monitoring]
    Monitor -->|Regression| Rollback[Automatic Rollback]
    Rollback --> Reject5[Revert]

    style Syntax fill:#e3f2fd
    style Semantic fill:#f3e5f5
    style Empirical fill:#e8f5e9
    style Consensus fill:#fff3e0
    style Accept fill:#c8e6c9
    style Reject1 fill:#ffcdd2
    style Reject2 fill:#ffcdd2
    style Reject3 fill:#ffcdd2
    style Reject4 fill:#ffcdd2
    style Reject5 fill:#ffcdd2
```

**Validation Stages:**

1. **Syntactic Validation**: Ensures valid ASP syntax using Clingo parser
2. **Semantic Validation**: Checks logical consistency with existing rules
3. **Empirical Validation**: Tests against curated test cases (ground truth)
4. **Consensus Validation**: Multi-agent debate for agreement
5. **Performance Monitoring**: Tracks accuracy and triggers rollback on regression

## 4. Translation Flow

```mermaid
graph TB
    subgraph "ASP to Natural Language"
        ASP1[ASP Rule] --> Parse1[Parse ASP<br/>Extract Predicates]
        Parse1 --> Template1[Apply Templates<br/>Legal Language]
        Template1 --> LLM1[LLM Enhancement<br/>Fluent Expression]
        LLM1 --> NL1[Natural Language Output]
    end

    subgraph "Natural Language to ASP"
        NL2[Natural Language Input] --> LLM2[LLM Extraction<br/>Structured Output]
        LLM2 --> Elements[Legal Elements<br/>Predicates, Entities]
        Elements --> Generate[Generate ASP<br/>Rule Templates]
        Generate --> ASP2[ASP Rule Output]
    end

    NL1 -.->|Fidelity Check| Fidelity1{Translation Fidelity}
    Fidelity1 -->|High| Pass1[Accept]
    Fidelity1 -->|Low| Retry1[Refine & Retry]

    ASP2 -.->|Fidelity Check| Fidelity2{Translation Fidelity}
    Fidelity2 -->|High| Pass2[Accept]
    Fidelity2 -->|Low| Retry2[Refine & Retry]

    Retry1 -.-> Template1
    Retry2 -.-> LLM2

    style ASP1 fill:#e1f5ff
    style ASP2 fill:#e1f5ff
    style NL1 fill:#fff4e1
    style NL2 fill:#fff4e1
    style LLM1 fill:#f0e1ff
    style LLM2 fill:#f0e1ff
    style Pass1 fill:#c8e6c9
    style Pass2 fill:#c8e6c9
```

**Fidelity Checkpoints:**

- **ASP → NL**: Verify natural language preserves logical meaning
- **NL → ASP**: Verify ASP captures all legal elements from text
- **Round-trip**: Test ASP → NL → ASP produces equivalent rule

## 5. Version Control Flow

```mermaid
graph TD
    Working[Working State<br/>In-Memory Rules] -->|Commit| Commit1[Create Commit<br/>Snapshot + Metadata]
    Commit1 --> History[Commit History<br/>Immutable Log]

    History --> Branch{Branch}
    Branch -->|New Feature| Feature[Feature Branch]
    Branch -->|Bug Fix| Fix[Fix Branch]
    Branch -->|Experiment| Exp[Experimental Branch]

    Feature --> WorkF[Develop Feature]
    Fix --> WorkFix[Develop Fix]
    Exp --> WorkExp[Test Hypothesis]

    WorkF --> Merge1{Merge}
    WorkFix --> Merge1
    WorkExp --> Merge1

    Merge1 -->|Fast-Forward| Main1[Main Branch]
    Merge1 -->|Conflicts| Resolve[Resolve Conflicts<br/>Prefer Higher Stratum]
    Resolve --> Main1

    Main1 --> Tag[Tag Release]
    Tag --> Deploy[Deploy to Production]

    Deploy -->|Regression| Rollback[Rollback<br/>Revert to Previous Tag]
    Rollback --> Main1

    style Working fill:#e3f2fd
    style History fill:#f3e5f5
    style Main1 fill:#c8e6c9
    style Deploy fill:#81c784
    style Rollback fill:#ffcdd2
```

**Version Control Features:**

- **Commit**: Snapshot of entire rule base with metadata
- **Branch**: Isolated development environments
- **Merge**: Combine branches with automatic conflict resolution
- **Diff**: Compute rule additions, deletions, modifications
- **Tag**: Mark stable releases
- **Rollback**: Revert to any previous state

## 6. Data Flow Diagram

```mermaid
flowchart TD
    User[User Query] -->|1. Input| System[LOFT System]

    System -->|2. Parse| Parser[Query Parser]
    Parser -->|3. Extract Intent| Meta[Meta-Reasoner]

    Meta -->|4a. Query Rules| Symbolic[Symbolic Core]
    Meta -->|4b. Request Translation| Neural[Neural Interface]

    Symbolic -->|5. Execute ASP| Clingo[Clingo Solver]
    Clingo -->|6. Answer Sets| Results[Intermediate Results]

    Neural -->|7. LLM Query| LLM[Language Model]
    LLM -->|8. Structured Response| Parse[Response Parser]

    Results -->|9. Combine| Synthesizer[Result Synthesizer]
    Parse -->|9. Combine| Synthesizer

    Synthesizer -->|10. Generate Explanation| Explainer[Explanation Generator]
    Explainer -->|11. Natural Language| User

    subgraph "Cross-Cutting Concerns"
        Logging[Logger] -.->|Log| Parser
        Logging -.->|Log| Meta
        Logging -.->|Log| Symbolic
        Logging -.->|Log| Neural
        Logging -.->|Log| Clingo
        Logging -.->|Log| LLM

        Validation[Validator] -.->|Validate| Symbolic
        Validation -.->|Validate| Results

        VC[Version Control] -.->|Track| Symbolic
    end

    style User fill:#fff4e1
    style System fill:#e1f5ff
    style Symbolic fill:#e1f5ff
    style Neural fill:#f0e1ff
    style Results fill:#c8e6c9
```

**Data Flow Steps:**

1. User submits natural language query
2. System parses query to extract legal question
3. Meta-reasoner determines reasoning strategy
4. Query symbolic core (ASP) and/or neural interface (LLM)
5. Execute ASP program with Clingo solver
6. Retrieve answer sets (legal conclusions)
7. Query LLM for translation/explanation
8. Parse structured LLM response
9. Synthesize symbolic and neural results
10. Generate human-readable explanation
11. Return natural language response

**Cross-Cutting Concerns:**

- **Logging**: All steps logged with structured JSON
- **Validation**: Rules and results validated at each stage
- **Version Control**: Rule modifications tracked

## 7. Module Interactions

```mermaid
graph TD
    subgraph "Core Modules"
        Symbolic[loft.symbolic<br/>ASP Rule Base]
        Validation[loft.validation<br/>Multi-Stage Validation]
        VersionControl[loft.core<br/>Version Control]
        Logging[loft.logging<br/>Observability]
    end

    subgraph "Neural Modules"
        Neural[loft.neural<br/>LLM Interface]
        Translation[loft.translation<br/>ASP ↔ NL]
    end

    subgraph "Analysis Modules"
        Consistency[loft.consistency<br/>Consistency Checker]
        Meta[loft.meta<br/>Meta-Reasoning]
    end

    Symbolic <-->|Validate Rules| Validation
    Symbolic <-->|Version Rules| VersionControl
    Symbolic -->|Log Operations| Logging
    Symbolic <-->|Check Consistency| Consistency

    Validation <-->|Empirical Tests| Consistency
    VersionControl -->|Log Commits| Logging

    Neural <-->|Translate| Translation
    Translation <-->|Extract/Generate| Symbolic
    Neural -->|Log Queries| Logging

    Meta -->|Query| Symbolic
    Meta -->|Query| Neural
    Meta -->|Validate Strategy| Validation
    Meta -->|Analyze| Consistency

    style Symbolic fill:#e1f5ff
    style Validation fill:#e1ffe1
    style Neural fill:#f0e1ff
    style Meta fill:#fff4e1
```

**Key Integration Points:**

- **Symbolic ↔ Validation**: All rule modifications validated
- **Symbolic ↔ Version Control**: Every change committed to history
- **Symbolic ↔ Consistency**: Periodic consistency checks
- **Translation ↔ Neural**: ASP ↔ NL conversion via LLMs
- **Meta ↔ All**: Meta-reasoning coordinates all components
- **Logging → All**: Universal observability

## Architecture Principles

### 1. Separation of Concerns
- **Symbolic reasoning** (ASP) separate from **neural reasoning** (LLMs)
- Clear boundaries between validation, versioning, and core logic

### 2. Stratification
- Rules organized by confidence and modifiability
- Higher strata constrain lower strata

### 3. Validation Gates
- No rule enters system without passing validation pipeline
- Automatic rollback on performance regression

### 4. Observability First
- Comprehensive logging at all decision points
- Structured JSON for machine-readable analysis

### 5. Immutable History
- Version control preserves all changes
- Easy rollback to any previous state

### 6. Fail-Safe Defaults
- Conservative validation thresholds
- Constitutional layer cannot be modified
- Automatic rollback beats manual intervention

## Performance Characteristics

| Component | Typical Latency | Scalability | Bottleneck |
|-----------|----------------|-------------|------------|
| ASP Solver | 10-100ms | 1000s of rules | Grounding phase |
| LLM Query | 1-5s | Token limit | API rate limits |
| Validation | 100-500ms | Test suite size | Empirical tests |
| Version Control | <10ms | Commit history | Diff computation |
| Consistency Check | 50-200ms | Rule count | SAT solving |

## Future Architecture Evolution

See [MAINTAINABILITY.md](../MAINTAINABILITY.md) for planned enhancements:

- **Phase 1.5**: LinkedASP integration for queryable documentation
- **Phase 2**: Advanced validation and consensus mechanisms
- **Phase 3**: Self-modifying system with autonomous improvement
- **Phase 4**: Distributed deployment and collaborative reasoning

---

**Last Updated**: 2025-11-27
**Maintained By**: LOFT Development Team
