# Dataset Design Guidelines for Transfer Learning

This document provides requirements and best practices for designing datasets that enable effective transfer learning in the LOFT neuro-symbolic pipeline.

## Table of Contents

1. [Key Principles](#key-principles)
2. [Structure Requirements](#structure-requirements)
3. [Content Requirements](#content-requirements)
4. [File Format](#file-format)
5. [Predicate Naming Conventions](#predicate-naming-conventions)
6. [Validation Checklist](#validation-checklist)
7. [Example Dataset](#example-dataset)
8. [Cross-Domain Design](#cross-domain-design)

## Key Principles

Through validation testing, we discovered that dataset structure critically impacts transfer learning success:

| Dataset Design | Coverage | Accuracy | Why |
|---------------|----------|----------|-----|
| 10 scenarios, 10 doctrines | 10% | 0% | Predicate mismatch between doctrines |
| 10 scenarios, 1 doctrine | 100% | 100% | Shared predicate vocabulary |

### 1. Shared Predicate Vocabulary is Essential

**Ineffective Design** - Each scenario uses different predicates:
```
prop_001 (adverse possession): occupation_continuous, occupation_years
prop_002 (easement): is_landlocked, common_ownership_history
prop_003 (recording): first_deed_recorded, second_buyer_notice
```

**Effective Design** - All scenarios share predicates:
```
ap_001: occupation_continuous, occupation_years, statutory_period
ap_002: occupation_continuous, occupation_years, statutory_period
ap_003: occupation_continuous, occupation_years, statutory_period
```

### 2. Vary Facts, Not Schema

Transfer learning works when scenarios vary in **fact values** while sharing **predicate schema**:

```json
// ap_001: All elements met, 25 > 20 years
{"occupation_years": 25, "statutory_period": 20, "ground_truth": "enforceable"}

// ap_002: Insufficient time, 12 < 20 years
{"occupation_years": 12, "statutory_period": 20, "ground_truth": "unenforceable"}

// ap_003: Permission given (not hostile)
{"occupation_hostile": "no", "ground_truth": "unenforceable"}
```

### 3. Cover Positive and Negative Cases

Dataset should include both outcomes with clear distinguishing factors:

| Scenario | Key Difference | Ground Truth |
|----------|---------------|--------------|
| ap_001 | All elements met | enforceable |
| ap_002 | Time insufficient | unenforceable |
| ap_003 | Permissive use | unenforceable |
| ap_004 | Interrupted possession | unenforceable |
| ap_005 | All elements + tacking | enforceable |

## Structure Requirements

- **Single doctrine per dataset**: All scenarios test the same legal principle
- **Shared predicate vocabulary**: Consistent predicate names across scenarios
- **Consistent argument types**: Same arity and argument semantics
- **Minimum 10 scenarios**: Enough for train/test split (7/3 or 8/2)

## Content Requirements

- **Balanced outcomes**: Both enforceable and unenforceable cases (at least 30% minority class)
- **Distinct failure modes**: Different reasons for negative outcomes
- **Edge cases**: Boundary conditions (exactly at threshold, just under, just over)
- **Clear rationales**: Each scenario explains the legal reasoning

## File Format

Each scenario should be a JSON file with this structure:

```json
{
  "id": "unique_scenario_id",
  "description": "Human-readable description of the scenario",
  "facts": [
    "Natural language fact 1",
    "Natural language fact 2"
  ],
  "asp_facts": "predicate1(entity). predicate2(entity, value).",
  "question": "Legal question being answered",
  "ground_truth": "enforceable|unenforceable",
  "rationale": "Explanation of why this is the correct answer",
  "legal_citations": ["Source 1", "Source 2"],
  "difficulty": "easy|medium|hard"
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., "ap_001") |
| `description` | string | Human-readable summary |
| `asp_facts` | string | ASP facts for the scenario |
| `ground_truth` | string | Expected outcome ("enforceable" or "unenforceable") |
| `rationale` | string | Explanation of the legal reasoning |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `facts` | array | Natural language facts |
| `question` | string | Legal question being asked |
| `legal_citations` | array | Source citations |
| `difficulty` | string | Scenario difficulty level |

## Predicate Naming Conventions

Use consistent naming patterns across all scenarios:

| Pattern | Example | Use For |
|---------|---------|---------|
| `concept(Entity)` | `claim(c1)` | Entity type declarations |
| `property(Entity, Value)` | `occupation_years(c1, 25)` | Numeric properties |
| `boolean_prop(Entity, yes/no)` | `occupation_continuous(c1, yes)` | Boolean properties |
| `relationship(Entity, Other)` | `claimant(c1, alice)` | Entity relationships |
| `threshold(Entity, Value)` | `statutory_period(c1, 20)` | Threshold values |

### Boolean Values

Always use `yes`/`no` for boolean predicates (not `true`/`false` or `1`/`0`):

```asp
occupation_continuous(c1, yes).
occupation_hostile(c1, no).
```

### Entity Identifiers

Use short, consistent entity identifiers:

- Claims: `c1`, `c2`, `c3`, ...
- Contracts: `contract1`, `contract2`, ...
- Parties: `alice`, `bob`, `seller`, `buyer`
- Property: `lot42`, `parcel1`, `land1`

## Validation Checklist

### Automated Validation

Run the dataset validation script before committing:

```bash
python scripts/validate_dataset.py datasets/your_dataset/
```

The script checks:

1. **Minimum size**: At least 10 scenarios
2. **Shared predicates**: At least 3 common predicates across all scenarios
3. **Balanced outcomes**: Both outcomes represented with at least 30% minority
4. **Unique scenarios**: No duplicate fact combinations
5. **Required fields**: All required JSON fields present
6. **Valid ASP syntax**: Facts parse correctly

### Manual Review

Before finalizing a dataset, verify:

- [ ] Each scenario has a unique combination of facts
- [ ] Rationales correctly explain the legal reasoning
- [ ] Ground truths are legally accurate
- [ ] Predicates are consistently named
- [ ] No scenario is ambiguous (could go either way)
- [ ] Edge cases are represented

## Example Dataset

A well-designed adverse possession dataset:

```
datasets/adverse_possession/
├── ap_001_successful_claim.json      # All elements met
├── ap_002_insufficient_time.json     # N < statutory_period
├── ap_003_permissive_use.json        # occupation_hostile = no
├── ap_004_interrupted_possession.json # occupation_continuous = no
├── ap_005_successful_with_tacking.json # Tacking allowed
├── ap_006_hidden_use.json            # occupation_open = no
├── ap_007_successful_short_period.json # Different jurisdiction
├── ap_008_exactly_at_period.json     # Boundary case (success)
├── ap_009_no_taxes_paid.json         # Tax requirement not met
└── ap_010_just_under_period.json     # Boundary case (fail)
```

### Example Scenario File

```json
{
  "id": "ap_001",
  "description": "Successful adverse possession claim with all elements satisfied",
  "facts": [
    "Alice has occupied the property for 25 years",
    "The statutory period is 20 years",
    "Occupation was continuous without interruption",
    "Occupation was hostile (without permission)",
    "Occupation was open and notorious",
    "Occupation was exclusive"
  ],
  "asp_facts": "claim(c1). claimant(c1, alice). occupation_years(c1, 25). statutory_period(c1, 20). occupation_continuous(c1, yes). occupation_hostile(c1, yes). occupation_open(c1, yes). occupation_exclusive(c1, yes).",
  "question": "Has Alice acquired title through adverse possession?",
  "ground_truth": "enforceable",
  "rationale": "All elements of adverse possession are satisfied: (1) continuous occupation for 25 years exceeds the 20-year statutory period, (2) occupation was hostile (without permission), (3) occupation was open and notorious, and (4) occupation was exclusive.",
  "legal_citations": ["Restatement (Third) of Property"],
  "difficulty": "easy"
}
```

## Cross-Domain Design

For cross-domain transfer studies, additional considerations apply:

### 1. Use Canonical Concepts

When designing datasets for cross-domain transfer, map predicates to the canonical ontology:

```
Source Domain (Adverse Possession):
  occupation_continuous -> use_continuous (canonical)
  occupation_hostile -> use_adverse (canonical)

Target Domain (Statute of Frauds):
  contract_in_writing -> writing_exists (canonical)
  signed_by_party -> signed_by (canonical)
```

### 2. Document Predicate Mappings

Create a mapping file for cross-domain datasets:

```json
{
  "source_domain": "adverse_possession",
  "target_domain": "statute_of_frauds",
  "mappings": {
    "occupation_continuous": "performance_started",
    "occupation_years": "contract_duration"
  }
}
```

### 3. Validate Same-Domain First

Always test with same-domain transfer before attempting cross-domain:

1. Train on ap_001-007, test on ap_008-010 (same domain)
2. Only if same-domain works, try cross-domain transfer

## Common Mistakes to Avoid

### 1. Mixed Doctrines

**Wrong**: One dataset with scenarios from different legal areas
```
scenarios/
├── adverse_possession_001.json
├── easement_001.json
├── statute_of_frauds_001.json
```

**Right**: Separate datasets per doctrine
```
datasets/
├── adverse_possession/
├── easements/
└── statute_of_frauds/
```

### 2. Inconsistent Predicates

**Wrong**: Different predicate names for same concept
```asp
% Scenario 1
occupation_duration(c1, 25).

% Scenario 2
years_of_occupation(c2, 15).
```

**Right**: Consistent naming
```asp
% Scenario 1
occupation_years(c1, 25).

% Scenario 2
occupation_years(c2, 15).
```

### 3. Missing Negative Cases

**Wrong**: Only successful claims
```
ap_001: ground_truth = enforceable
ap_002: ground_truth = enforceable
ap_003: ground_truth = enforceable
```

**Right**: Balanced outcomes
```
ap_001: ground_truth = enforceable
ap_002: ground_truth = unenforceable (time insufficient)
ap_003: ground_truth = unenforceable (permissive use)
```

### 4. Duplicate Scenarios

**Wrong**: Same facts with different IDs
```json
// ap_001
{"occupation_years": 25, "statutory_period": 20}

// ap_002 (duplicate!)
{"occupation_years": 25, "statutory_period": 20}
```

**Right**: Each scenario is unique
```json
// ap_001
{"occupation_years": 25, "statutory_period": 20}

// ap_002
{"occupation_years": 12, "statutory_period": 20}
```

## Related Documentation

- [Canonical Predicate Ontology](../ontology/canonical_predicates.md) - Cross-domain concept mappings
- [Transfer Study Guide](../experiments/transfer_study.md) - Running transfer experiments
- [ASP Integration](../architecture/asp-integration.md) - ASP reasoning integration
