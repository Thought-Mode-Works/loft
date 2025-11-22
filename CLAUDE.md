# LOFT Project: AI Development Guidelines

## Project Mission

Build a self-reflexive neuro-symbolic AI system where the symbolic core autonomously reconstructs its own logic using LLM outputs, creating a validated ontological bridge between symbolic reasoning and neural pattern recognition.

---

## Core Architectural Principles

### 1. Self-Reflexive Symbolic Core is Paramount

The symbolic core's ability to reason about and modify its own reasoning processes is the central innovation. All development must preserve and enhance this reflexivity.

**Requirements:**
- Every code change must consider impact on the symbolic core's self-modification capabilities
- The core must always be able to explain its own state and reasoning
- Modifications to the core must be traceable, reversible, and justifiable
- Meta-reasoning capabilities cannot be compromised for performance or convenience

**Validation:**
- Before any PR merge, verify that the symbolic core can still:
  1. Identify gaps in its own knowledge
  2. Question its own logic
  3. Generate candidates for self-improvement
  4. Validate proposed modifications
  5. Explain why modifications were accepted or rejected

### 2. Validation Oversight for Complex Build

This is an ambitious, multi-layered system. Validation must occur continuously at every level to prove we're building the ontological bridge correctly.

**Validation Layers:**

#### Syntactic Validation
- All symbolic rules are well-formed
- LLM outputs parse correctly into structured representations
- Type safety is maintained across symbolic-neural boundaries

#### Semantic Validation
- Generated rules don't contradict existing knowledge
- Logical consistency is preserved across modifications
- Compositional rule combinations follow formal properties (ring structure)

#### Empirical Validation
- All claims about performance must be backed by test results
- Every new rule or modification must be tested against labeled cases
- Regression testing ensures previous capabilities are not lost

#### Meta-Validation
- The system can validate its own validation processes
- Confidence scores are calibrated against actual performance
- Self-assessment aligns with external evaluation

**Code Requirements:**
- No code merged without corresponding tests
- All validation functions must have clear pass/fail criteria
- Validation results must be logged and traceable
- Failed validations must trigger rollback or human review

### 3. Ontological Bridge Integrity

The symbolic-neural translation layer is where meaning crosses between incompatible representational paradigms. This bridge must maintain semantic fidelity.

**Requirements:**
- Symbolic → Natural Language → Symbolic roundtrip must preserve meaning
- LLM outputs must be grounded in symbolic terms
- Translation fidelity metrics must be continuously monitored
- Ambiguities in translation must be explicitly represented, not silently resolved

**Validation:**
- Roundtrip translation tests for all symbolic predicates
- Semantic similarity measurements for NL translations
- Edge case testing for boundary concepts
- Regular audits of translation quality on held-out examples

### 4. Experiential Learning Integration

The system must learn from its own experience and external validation, updating its core based on what works in practice.

**Requirements:**
- Track performance of all rules over time
- Identify which rules improve/degrade accuracy
- Learn from failures: analyze why predictions were wrong
- Update rule confidence based on empirical success
- Incorporate external feedback (case law, expert corrections)

**Validation:**
- Demonstrate learning curves: performance improves with experience
- Show that failed predictions lead to rule refinements
- Prove that experiential learning doesn't cause instability
- Verify that learned knowledge generalizes to new cases

---

## Development Workflow Requirements

### Before Starting Any Task

1. **Read ROADMAP.md** to understand current phase and validation criteria
2. **Check current symbolic core state** to understand what you're modifying
3. **Review recent validation results** to ensure system is stable
4. **Identify validation requirements** for your specific task

### During Development

1. **Incremental Validation**: Test continuously, not just at the end
2. **Logging**: Capture all LLM interactions, core modifications, validation results
3. **Explainability**: Every function should be able to explain what it's doing
4. **Version Control**: Symbolic core states must be versioned and comparable

### Before Creating Pull Requests

1. **Run Full Test Suite**: All existing tests must pass
2. **Add New Tests**: Your changes must have corresponding validation tests
3. **Validate Ontological Bridge**: If you modified translation layers, prove fidelity
4. **Check Self-Reflexivity**: Ensure core can still reason about itself
5. **Performance Metrics**: Document impact on accuracy, consistency, latency
6. **Explainability**: System must explain new capabilities in natural language

### Pull Request Requirements

```markdown
## Summary
[Brief description of changes]

## Phase & Validation Criteria
- Phase: [0-9]
- Relevant MVP criteria: [specific criteria from ROADMAP.md]

## Validation Results
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Validation metrics: [specific numbers]
  - Accuracy: [before → after]
  - Consistency score: [value]
  - Translation fidelity: [value]
- [ ] Self-reflexivity preserved: [explanation]
- [ ] Ontological bridge integrity: [test results]

## Tested Examples
[Working, tested code examples demonstrating functionality]

## Impact on Symbolic Core
- Rules added/modified: [count and description]
- Meta-reasoning changes: [description]
- Stability impact: [analysis]

## Risks & Mitigations
[Any potential issues and how they're addressed]

## Rollback Plan
[How to revert if this causes problems]
```

---

## Coding Standards

### Symbolic Core Code

**Composability & Reusability:**
- Functions must compose cleanly - avoid side effects
- Build abstractions that work across domains (legal → other)
- Design for future transposition to new problem areas
- Prefer small, single-purpose functions over monolithic implementations

**Descriptive Naming:**
- Variable names must describe semantic meaning, not just types
- `rule_confidence_score` not `rcs`
- `symbolic_to_nl_translation` not `s2n`
- Functions named for what they accomplish, not how

**Validation-First:**
- Every symbolic operation must validate its inputs and outputs
- Validation functions must explain WHY validation failed
- Confidence scores must be propagated through all operations

### Neural Interface Code

**Explicit Uncertainty:**
- LLM outputs must include confidence/uncertainty measures
- Prompts must be versioned and traceable
- All LLM interactions must be logged for analysis
- Failures must be captured and analyzed

**Prompt Engineering Standards:**
- Prompts must be testable and version-controlled
- Include examples and constraints in prompts
- Use structured output formats (JSON schema)
- Validate LLM outputs before using them

### Translation Layer Code

**Bidirectional Fidelity:**
- Every translation function must have a roundtrip test
- Measure and log translation quality metrics
- Handle ambiguity explicitly, never silently
- Preserve provenance: track where meaning came from

---

## Testing Philosophy

### Test Pyramid for Neuro-Symbolic Systems

```
                    /\
                   /  \
                  /Meta\          ← Does system reason about itself?
                 /------\
                /        \
               /Integration\      ← Do symbolic + neural work together?
              /------------\
             /              \
            / Validation     \   ← Do components validate correctly?
           /------------------\
          /                    \
         /   Unit + Property    \  ← Do individual functions work?
        /________________________\
```

### Required Test Types

**Unit Tests:**
- Individual symbolic operations
- LLM output parsing
- Translation functions
- Validation predicates

**Property Tests:**
- Logical consistency is preserved under operations
- Ring structure properties hold
- Geometric invariances are maintained
- Confidence scores are calibrated

**Integration Tests:**
- Symbolic core + LLM queries work end-to-end
- Rule generation → validation → incorporation pipeline
- Translation roundtrips preserve meaning
- Version control and rollback mechanisms

**Meta-Tests:**
- System can evaluate its own performance
- Self-improvement cycles converge
- Reflexive reasoning produces accurate self-assessments
- Failure analysis correctly diagnoses errors

**Validation Tests:**
- All validation criteria from ROADMAP.md
- Regression tests: previous capabilities maintained
- Performance benchmarks: accuracy, consistency, latency
- Safety tests: bad rules are rejected or rolled back

---

## Critical Safeguards

### Preventing Catastrophic Self-Modification

The system's ability to modify itself is powerful but dangerous. Safeguards:

1. **Constitutional Layer Immutability**
   - Core safety constraints, logical axioms never change
   - Modifications to constitutional layer require human approval
   - System must prove proposed changes preserve safety properties

2. **Stratified Modification Authority**
   - Operational layer: autonomous modification allowed
   - Tactical layer: requires validation threshold >0.8
   - Strategic layer: requires validation threshold >0.9
   - Constitutional layer: requires human approval

3. **Rollback Mechanisms**
   - All core states versioned with git-like semantics
   - Performance degradation triggers automatic rollback
   - Logical inconsistency triggers immediate rollback
   - Human can manually rollback at any time

4. **Validation Thresholds**
   - Rules below confidence threshold are flagged, not incorporated
   - High-impact changes require multi-LLM consensus
   - Novel rule types require human review
   - Modifications affecting >10% of core require careful analysis

5. **Monitoring & Alerts**
   - Continuous monitoring of accuracy, consistency, stability
   - Alerts for performance degradation
   - Alerts for unusual modification patterns
   - Regular human review of system evolution

### Preventing Infinite Reflection Loops

Meta-reasoning can spiral. Safeguards:

1. **Depth Limits**: Maximum recursion depth for reflexive reasoning
2. **Termination Conditions**: Clear criteria for when to stop reflecting
3. **Progress Metrics**: Reflection must improve something measurable
4. **Timeout Mechanisms**: Hard time limits on meta-reasoning cycles

### Maintaining Epistemic Humility

The system must know what it doesn't know:

1. **Explicit Uncertainty**: All outputs include confidence scores
2. **Knowledge Boundaries**: System tracks edge of its competence
3. **Graceful Degradation**: Outside competence, request human help
4. **Confidence Calibration**: Scores must align with actual accuracy

---

## Validation Metrics

### Core Metrics (Track Continuously)

| Metric | How to Measure | Target | Alert Threshold |
|--------|---------------|--------|-----------------|
| Prediction Accuracy | % correct on test cases | >85% | <80% |
| Logical Consistency | Automated consistency checker | 100% | <100% (immediate alert) |
| Rule Stability | Modifications per 100 queries | <5 | >20 |
| Translation Fidelity | Roundtrip semantic similarity | >95% | <90% |
| Confidence Calibration | Predicted vs. actual accuracy | ±5% | ±15% |
| Self-Assessment Accuracy | System's self-evaluation vs. reality | >80% | <70% |
| Coverage | % queries system can handle | >60% | <50% |
| Latency | Time to response | <5s | >10s |

### Phase-Specific Validation

Each phase in ROADMAP.md has specific MVP criteria. Those must be validated before proceeding.

### Meta-Validation

Validate the validators:
- Do validation functions correctly identify problems?
- Are validation thresholds calibrated appropriately?
- Does the validation framework itself need improvement?

---

## Communication & Explainability

### System Must Explain Itself

At every level, the system must be able to generate natural language explanations:

**Symbolic Core:**
- "I applied rule X because elements A, B, C were satisfied"
- "I am uncertain about this case because rule Y conflicts with rule Z"
- "I modified rule W because it failed on 15% of test cases"

**Neural Components:**
- "LLM proposed rule R with confidence 0.87 based on these precedents"
- "Translation from symbolic to NL: [show mapping]"
- "This prompt produced better results than previous version because..."

**Meta-Reasoning:**
- "I identified that my reasoning about contracts is less reliable than torts"
- "I improved my prompts by analyzing 50 failed predictions"
- "My self-modification strategy changed because performance stagnated"

### Documentation Requirements

**Code Comments:**
- Explain WHY, not what (code shows what)
- Document assumptions and invariants
- Flag areas where self-reflexivity is critical
- Note validation requirements

**System Documentation:**
- Architecture diagrams showing symbolic-neural interactions
- Data flow for self-modification cycles
- Validation pipeline explanations
- Failure mode analyses

**Research Log:**
- Document experiments and results
- Track what works and what doesn't
- Maintain bibliography of relevant research
- Note open questions and future directions

---

## Validation Oversight Checklist

Before any significant change, verify:

- [ ] **Self-Reflexivity Preserved**: Core can still reason about itself
- [ ] **Ontological Bridge Intact**: Symbolic-neural translation maintains fidelity
- [ ] **Validation Framework Functional**: All validators work correctly
- [ ] **Experiential Learning Enabled**: System can learn from outcomes
- [ ] **Safety Mechanisms Active**: Rollback, limits, alerts all functional
- [ ] **Explainability Maintained**: System can explain all behaviors
- [ ] **Performance Metrics Stable**: No unexpected degradation
- [ ] **Consistency Preserved**: No logical contradictions introduced
- [ ] **Tests Comprehensive**: New functionality has corresponding tests
- [ ] **Documentation Updated**: Changes are documented and explained

---

## Research Integration

This project sits at cutting edge of neuro-symbolic AI. Stay connected to research:

1. **Monitor Key Venues**: NeurIPS, ICML, AAAI (neuro-symbolic track), IJCAI
2. **Track Techniques**:
   - Program synthesis via LLMs
   - Automated theorem proving
   - Meta-reasoning and meta-learning
   - Lifelong learning systems
   - Cognitive architectures
3. **Evaluate Tools**: As new frameworks emerge, assess fit for LOFT
4. **Document Learnings**: Integrate relevant research into system design

---

## When in Doubt

**Ask These Questions:**

1. Does this preserve the self-reflexive symbolic core?
2. Can this be validated programmatically?
3. Will this help build the ontological bridge?
4. Can the system explain this to a human?
5. Is this moving toward the vision in thoughts.md?

**Prioritize:**

1. **Correctness** over speed
2. **Validation** over feature addition
3. **Simplicity** over premature optimization
4. **Explainability** over black-box performance
5. **Safety** over capability

**Remember:**

The goal is not just to build a system that works, but to build a system that understands and improves itself - a genuine ontological bridge between symbolic and neural reasoning. Every line of code should serve that vision.
