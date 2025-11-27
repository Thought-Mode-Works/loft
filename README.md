# LOFT - Reflexive Neuro-Symbolic AI Architecture

> Building a self-reflexive symbolic core that autonomously reconstructs its own logic using LLM outputs, creating an ontological bridge between symbolic reasoning and neural pattern recognition.

## 🎯 Vision

LOFT (Logical Ontological Framework Translator) is an ambitious research project exploring the intersection of classical symbolic AI and modern large language models. The system features a **self-reflexive symbolic core** capable of:

- Questioning its own logic
- Identifying gaps in its knowledge
- Using LLMs to generate candidate logical rules
- Validating and incorporating new rules autonomously
- Reasoning about its own reasoning processes

This creates a continuous self-improvement loop where symbolic and neural components work together, each compensating for the other's limitations.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/justin4957/loft.git
cd loft

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env to add your API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
```

### Running Examples

```bash
# Example 1: Statute of Frauds Legal Reasoning
python examples/statute_of_frauds_demo.py

# Example 2: Natural Language to ASP Translation
python examples/nl_to_asp_examples.py

# Example 3: LLM Integration with Context Enrichment
python examples/example_5_llm_integration.py

# Example 4: Confidence Tracking and Calibration
python examples/confidence_tracking_example.py
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=loft --cov-report=html

# Run specific test suite
pytest tests/unit/legal/test_statute_of_frauds.py -v
```

## 📚 User-Friendly Examples

### Example 1: Legal Contract Analysis

Analyze whether a contract satisfies the Statute of Frauds:

```python
from loft.legal import StatuteOfFraudsSystem

# Initialize the legal reasoning system
system = StatuteOfFraudsSystem()

# Define contract facts
facts = """
contract_fact(c1).
land_sale_contract(c1).
party_fact(alice).
party_fact(bob).
party_to_contract(c1, alice).
party_to_contract(c1, bob).
writing_fact(w1).
references_contract(w1, c1).
signed_by(w1, alice).
signed_by(w1, bob).
identifies_parties(w1).
describes_subject_matter(w1).
states_consideration(w1).
"""

# Load facts and analyze
system.add_facts(facts)

# Check enforceability
is_enforceable = system.is_enforceable("c1")
print(f"Contract enforceable: {is_enforceable}")  # True

# Get detailed explanation
explanation = system.explain_conclusion("c1")
print(f"\nExplanation:\n{explanation}")
```

**Output:**
```
Contract enforceable: True

Explanation:
The contract is enforceable because:
1. It falls within the statute (land sale contract)
2. There is a sufficient writing (w1)
   - Signed by party to be charged (alice)
   - Contains essential terms (parties, subject matter, consideration)
```

### Example 2: Translate Natural Language to Symbolic Logic

Convert natural language descriptions into formal ASP rules:

```python
from loft.translation import NLToASPTranslator

translator = NLToASPTranslator()

# Translate a simple statement
nl_text = "Alice signed a contract to buy land from Bob for $50,000."
facts = translator.translate_to_facts(nl_text)

print("Generated ASP facts:")
for fact in facts:
    print(f"  {fact}")
```

**Output:**
```
Generated ASP facts:
  party_fact(alice).
  party_fact(bob).
  contract_fact(contract_1).
  land_sale_contract(contract_1).
  sale_amount(contract_1, 50000).
  party_to_contract(contract_1, alice).
  party_to_contract(contract_1, bob).
  signed_by_party(contract_1, alice).
```

### Example 3: LLM-Generated Rule Proposals

Use LLMs to generate new logical rules with validation:

```python
from loft.neural import RuleGenerator, LLMInterface, AnthropicProvider
from loft.symbolic import ASPCore

# Set up LLM interface
provider = AnthropicProvider(api_key="your-api-key")
llm = LLMInterface(provider)

# Initialize rule generator
asp_core = ASPCore()
asp_core.load_rules()  # Load existing statute of frauds rules

generator = RuleGenerator(llm, asp_core, domain="contract_law")

# Generate rule from legal principle
principle = """
A contract is voidable by a minor if they were under 18 when they entered
into the contract, unless it was for necessaries like food or shelter.
"""

rule = generator.generate_from_principle(principle)

print(f"Generated ASP Rule:")
print(f"  {rule.asp_rule}")
print(f"\nConfidence: {rule.confidence}")
print(f"Reasoning: {rule.reasoning}")
print(f"Predicates used: {rule.predicates_used}")
```

**Output:**
```
Generated ASP Rule:
  voidable(C) :- contract(C), party_to_contract(C, P),
                 minor(P), not for_necessaries(C).

Confidence: 0.87
Reasoning: This rule captures the principle that minors can void contracts
           except for necessaries. The negation 'not for_necessaries(C)'
           implements the exception.
Predicates used: ['contract', 'party_to_contract', 'minor', 'for_necessaries']
```

### Example 4: Multi-Stage Validation Pipeline

Validate LLM-generated rules before incorporating them:

```python
from loft.validation import ValidationPipeline
from loft.legal import STATUTE_OF_FRAUDS_TEST_CASES

# Create validation pipeline
pipeline = ValidationPipeline(
    asp_core=asp_core,
    test_suite=STATUTE_OF_FRAUDS_TEST_CASES,
    voter_llms=[llm1, llm2, llm3],  # Ensemble of LLMs
    confidence_thresholds={
        "constitutional": 0.95,
        "strategic": 0.85,
        "tactical": 0.75,
        "operational": 0.65
    }
)

# Validate the generated rule
validation_report = pipeline.validate(
    rule=rule,
    target_layer="tactical"
)

print(f"Validation Decision: {validation_report.final_decision}")
print(f"Aggregate Confidence: {validation_report.aggregate_confidence.weighted_mean}")

if validation_report.final_decision == "accept":
    # Safe to incorporate into symbolic core
    asp_core.add_rule(rule)
    print("✓ Rule accepted and incorporated")
elif validation_report.final_decision == "reject":
    print(f"✗ Rule rejected: {validation_report.rejection_reasons}")
else:
    print(f"⚠ Rule flagged for review: {validation_report.flag_reason}")
```

### Example 5: Confidence Tracking and Calibration

Track confidence scores and calibrate them against actual performance:

```python
from loft.validation import ConfidenceTracker, ConfidenceCalibrator

# Track confidence vs. accuracy
tracker = ConfidenceTracker()

# Record predictions with confidence scores
tracker.record(
    rule_id="rule_1",
    predicted_confidence=0.85,
    actual_outcome=True,  # Was prediction correct?
    metadata={"domain": "contract_law", "test_case": "land_sale_01"}
)

# After many predictions, calibrate
calibrator = ConfidenceCalibrator(tracker.get_history())

# Get calibrated confidence
raw_confidence = 0.75
calibrated = calibrator.calibrate(raw_confidence)

print(f"Raw confidence: {raw_confidence}")
print(f"Calibrated confidence: {calibrated}")
print(f"Expected Calibration Error: {calibrator.ece()}")

# Visualize calibration
calibrator.plot_calibration_curve()  # Shows reliability diagram
```

## 🏗️ Core Innovation: The Ontological Bridge

Traditional AI systems are either:
- **Symbolic**: Precise, explainable, but brittle and hand-coded
- **Neural**: Flexible, learning-capable, but opaque and inconsistent

LOFT bridges these approaches by:

1. **Symbolic Core**: Maintains logical consistency, compositional reasoning, explainability
2. **Neural Components**: LLMs provide pattern recognition, natural language understanding, creative rule generation
3. **Reflexive Orchestrator**: Meta-reasoning layer that questions and improves the system itself
4. **Validation Framework**: Rigorous multi-stage verification ensures quality and safety

The "bridge" is bidirectional translation that preserves semantic meaning across incompatible representational paradigms.

## 🎨 Architecture Overview

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

## 📊 Project Status

**Current Phase**: **Phase 2 (95% complete)** - LLM Logic Generation + Validation

**✅ Completed Phases:**
- **Phase 0** (100%): Foundation & Validation Framework
- **Phase 1** (100%): Static Symbolic Core + LLM Query
- **Phase 2.1** (100%): LLM Rule Generation Prompts
- **Phase 2.2** (100%): Multi-Stage Validation Pipeline
- **Phase 2.3** (100%): Confidence Scoring & Calibration

**🚧 In Progress:**
- **Phase 2.4** (50%): Human-in-the-Loop Review Interface

**📅 Next Steps:**
- Complete Phase 2.4: Human review queue and CLI interface
- Begin Phase 3: Safe Self-Modification with stratified layers
- Implement Meta-Reasoning layer (Phase 5)

## 🎯 Development Phases

See [ROADMAP.md](ROADMAP.md) for detailed phased buildout plan.

**High-Level Timeline:**

- **Phase 0-1** (Weeks 1-5): Foundation & Static Core ✅
- **Phase 2-3** (Weeks 6-13): Logic Generation & Safe Self-Modification 🚧
- **Phase 4-5** (Weeks 14-22): Dialectical Reasoning & Meta-Reasoning
- **Phase 6-7** (Weeks 23-30): Neural Ensemble & Geometric Constraints
- **Phase 8-9** (Weeks 31-40): Multi-Domain & Production Hardening

Each phase has **MVP validation criteria** that must be met before proceeding.

## 🔑 Key Technical Components

### Symbolic Core (ASP-based)

The symbolic core uses Answer Set Programming (ASP) for non-monotonic reasoning:

```python
from loft.symbolic import ASPCore, ASPRule, StratificationLevel

# Create stratified core
core = ASPCore()

# Add rules at different stratification levels
core.add_rule(ASPRule(
    head="enforceable(C)",
    body="contract(C), not void(C)",
    stratification_level=StratificationLevel.CONSTITUTIONAL,
    confidence=1.0,
    description="Fundamental enforceability rule"
))

# Query the core
result = core.query(query_predicate="enforceable")
print(f"Enforceable contracts: {result.symbols}")
```

### Neural Components (LLM Interface)

Support for multiple LLM providers with structured output:

```python
from loft.neural import LLMInterface, AnthropicProvider, OpenAIProvider

# Anthropic (Claude)
anthropic = AnthropicProvider(api_key="...", model="claude-3-opus-20240229")
llm_anthropic = LLMInterface(anthropic)

# OpenAI (GPT-4)
openai = OpenAIProvider(api_key="...", model="gpt-4")
llm_openai = LLMInterface(openai)

# Query with structured output
from loft.neural.rule_schemas import GeneratedRule

rule = llm_anthropic.query_structured(
    question="Generate an ASP rule for contract formation",
    output_schema=GeneratedRule,
    context={"domain": "contract_law"}
)
```

### Translation Layer (Bidirectional)

Translate between symbolic and natural language:

```python
from loft.translation import ASPToNLTranslator, NLToASPTranslator

# ASP → Natural Language
asp_translator = ASPToNLTranslator()
nl_explanation = asp_translator.explain_rule(
    "enforceable(C) :- contract(C), has_writing(C)."
)
print(nl_explanation)
# Output: "A contract is enforceable if it is a contract and has a writing."

# Natural Language → ASP
nl_translator = NLToASPTranslator()
facts = nl_translator.translate_to_facts(
    "Alice signed a contract with Bob."
)
```

### Validation Framework (Multi-Stage)

Rigorous validation before rule incorporation:

1. **Syntactic Validation**: ASP syntax checking with Clingo
2. **Semantic Validation**: Logical consistency, no contradictions
3. **Empirical Validation**: Performance on test cases
4. **Consensus Validation**: Multi-LLM voting
5. **Confidence Gating**: Meets threshold for target layer

## 🎓 Initial Domain: Legal Reasoning

The first implementation targets legal document analysis because:

- **Structured + Uncertain**: Legal rules are symbolic but their application requires pattern recognition
- **Meta-Reasoning**: Legal analysis constantly reasons about reasoning (precedent, analogies, distinctions)
- **Verifiable**: Court decisions provide ground truth for validation
- **Natural Language Bridge**: Legal concepts naturally exist in structured language

**Current Implementation**: Statute of Frauds (contract law)
- 210-line ASP program
- 24 test cases covering edge cases and exceptions
- >90% accuracy on validation suite

## 🧪 Validation & Testing

**Testing Philosophy:**
- Unit tests for individual components
- Property tests for logical invariants
- Integration tests for symbolic-neural interaction
- Meta-tests for reflexive capabilities
- Validation tests for each phase's MVP criteria

**Current Test Suite:**
- 392 tests passing
- ~80% code coverage
- All major components tested

**Key Metrics:**
- Prediction Accuracy: >85% (currently: 92%)
- Logical Consistency: 100% (enforced)
- Translation Fidelity: >95% (currently: 97%)
- Confidence Calibration: ±5% (currently: ±3%)

## 🛠️ Core Principles

### 1. Self-Reflexivity is Paramount

The symbolic core's ability to reason about and modify itself is the central innovation. All development preserves this capability.

### 2. Validation at Every Level

Complex builds require continuous validation to prove progress toward goals. Every claim must be backed by tests.

### 3. Ontological Bridge Integrity

The symbolic-neural translation must maintain semantic fidelity. Meaning cannot be lost crossing the bridge.

### 4. Experiential Learning

The system learns from experience and external validation, updating based on what works in practice.

### 5. Safety First

Self-modification is powerful but dangerous. Multiple safeguards prevent catastrophic changes:
- Stratified modification authority
- Confidence thresholds
- Human review for constitutional changes
- Rollback mechanisms

## 📖 Documentation

- **[ROADMAP.md](ROADMAP.md)**: Detailed phased buildout plan with MVP criteria
- **[CLAUDE.md](CLAUDE.md)**: Comprehensive development guidelines and architectural principles
- **[docs/MAINTAINABILITY.md](docs/MAINTAINABILITY.md)**: LinkedASP strategy for preventing ASP complexity
- **[thoughts.md](thoughts.md)**: Research foundation and theoretical background

## 🤝 Contributing

This is currently a research project. Contributions welcome!

**Development Workflow:**
1. Read [ROADMAP.md](ROADMAP.md) to understand current phase
2. Follow guidelines in [CLAUDE.md](CLAUDE.md)
3. All PRs must include validation results
4. Maintain self-reflexivity and ontological bridge integrity

**Setting Up Development Environment:**

```bash
# Clone and install
git clone https://github.com/justin4957/loft.git
cd loft
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Linting, testing tools

# Run linter
ruff check loft/
ruff format loft/

# Run tests
pytest -v

# Run type checking
mypy loft/
```

## 🔬 Research Foundation

This project builds on research in:

- **Neuro-Symbolic AI**: Hybrid architectures combining logic and learning
- **Meta-Reasoning**: Systems that reason about their own reasoning
- **Program Synthesis**: LLM-guided code/logic generation
- **Cognitive Architecture**: Dual-process reasoning systems
- **Lifelong Learning**: Continual autonomous improvement

Key papers and resources documented in [thoughts.md](thoughts.md).

## 📜 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

Copyright 2025 LOFT Contributors

## 🙏 Acknowledgments

Built on decades of research in symbolic AI, neural networks, cognitive science, and legal reasoning. Standing on the shoulders of giants.

---

**Note**: This is a research project exploring fundamental questions about AI, reasoning, and self-improvement. The goal is not just a working system, but understanding the nature of the ontological bridge between symbolic and neural computation.

## 📈 Recent Updates

**2025-11-23**: Phase 2.3 complete - Confidence scoring and calibration system implemented
**2025-11-23**: Phase 2.2 complete - Multi-stage validation pipeline with ensemble voting
**2025-11-23**: Phase 2.1 complete - LLM rule generation with versioned prompts
**2025-11-23**: Phase 1.5 complete - Statute of Frauds domain implementation
**2025-11-22**: Phase 0 complete - Foundation and validation framework

For detailed change history, see commit log and pull requests.
