"""
Symbolic rule representation using Answer Set Programming (ASP).

Provides the foundational symbolic core with Clingo integration,
stratified architecture, and legal domain primitives.

## Main Components

- **ASPRule**: Python wrapper for ASP rules with metadata
- **ASPProgram**: Collection of rules and facts
- **StratifiedASPCore**: Four-layer stratified architecture
- **ASPCore**: Main engine with Clingo integration

## Example Usage

```python
from loft.symbolic import (
    ASPCore,
    ASPRule,
    StratificationLevel,
    RuleMetadata,
    create_statute_of_frauds_rules,
)
from datetime import datetime

# Create ASP core
core = ASPCore()

# Load statute of frauds rules
sof_program = create_statute_of_frauds_rules()
for rule in sof_program.rules:
    core.add_rule(rule)

# Add facts
facts = [
    "contract(c1).",
    "land_sale(c1).",
    "writing(w1).",
    "signed(w1, john).",
    "party_to_contract(c1, john).",
    "references(w1, c1).",
    "has_term(w1, parties).",
    "has_term(w1, subject_matter).",
    "has_term(w1, consideration).",
]
core.add_facts(facts)

# Load and query
core.load_rules()
result = core.query("satisfies_statute_of_frauds")

print(f"Satisfiable: {result.satisfiable}")
print(f"Answer sets: {result.answer_set_count}")
for symbol in result.symbols:
    print(f"  {symbol}")
```
"""

from .asp_rule import (
    ASPRule,
    StratificationLevel,
    RuleMetadata,
    create_rule_id,
)

from .asp_program import (
    ASPProgram,
    StratifiedASPCore,
    compose_programs,
)

from .asp_core import (
    ASPCore,
    QueryResult,
)

from .legal_primitives import (
    create_statute_of_frauds_rules,
    create_contract_basics_rules,
    create_meta_reasoning_rules,
)

from .stratification import (
    MODIFICATION_POLICIES,
    ModificationPolicy,
    get_policy,
    infer_stratification_level,
    print_all_policies,
)

__all__ = [
    # Core classes
    "ASPRule",
    "StratificationLevel",
    "RuleMetadata",
    "ASPProgram",
    "StratifiedASPCore",
    "ASPCore",
    "QueryResult",
    # Utilities
    "create_rule_id",
    "compose_programs",
    # Legal primitives
    "create_statute_of_frauds_rules",
    "create_contract_basics_rules",
    "create_meta_reasoning_rules",
    # Stratification and policies
    "MODIFICATION_POLICIES",
    "ModificationPolicy",
    "get_policy",
    "infer_stratification_level",
    "print_all_policies",
]
