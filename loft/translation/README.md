# Translation: ASP ↔ Natural Language Bridge

This module implements the ontological bridge between symbolic (ASP) and neural (NL) representations.

## Responsibilities

- **ASP → Natural Language** conversion for LLM queries
- **Natural Language → ASP** parsing for incorporating LLM outputs
- **Bidirectional fidelity** preservation (roundtrip testing)
- **Ambiguity detection** and handling
- **Domain-specific templates** for legal reasoning
- **Semantic grounding** in ASP core context

## The Ontological Bridge Challenge

Symbolic systems (ASP) operate with discrete, compositional structures and formal semantics.
Neural systems (LLMs) operate with probabilistic patterns over continuous embeddings.

This translation layer bridges incompatible ways of representing meaning, ensuring semantic fidelity across the boundary.

## Key Components (to be implemented)

- `asp_to_nl.py` - ASP predicates/rules → Natural language
- `nl_to_asp.py` - Natural language → ASP facts/rules
- `templates.py` - Domain-specific translation templates
- `fidelity_tester.py` - Roundtrip testing framework
- `grounding.py` - Context-aware grounding in ASP core

## Example Usage (planned)

```python
from loft.translation import ASPToNLTranslator, NLToASPTranslator

# ASP to NL
asp_to_nl = ASPToNLTranslator(domain="legal")
nl_query = asp_to_nl.translate("satisfies_statute_of_frauds(C)?")
# Output: "Which contracts satisfy the statute of frauds requirements?"

# NL to ASP
nl_to_asp = NLToASPTranslator(domain="legal")
asp_facts = nl_to_asp.translate("The contract was signed by John.")
# Output: ["signed_by(contract_1, john)."]

# Test fidelity
from loft.translation import test_roundtrip_fidelity
fidelity_score = test_roundtrip_fidelity(original_asp, translators)
```

## Integration Points

- **Core** (`loft.core`): Receives ASP, provides context for grounding
- **Neural** (`loft.neural`): Provides NL for LLMs, receives NL responses
- **Validation** (`loft.validation`): Fidelity measurement
