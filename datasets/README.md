# LOFT Test Datasets

This directory contains legal scenario datasets for testing LOFT's learning and transfer capabilities.

## Datasets

### Statute of Frauds (`statute_of_frauds/`)

10 scenarios covering the Statute of Frauds doctrine, focusing on contract enforceability requirements:

- **Topics:** Land sales, goods over $500, services, marriage consideration, executor promises, surety agreements, multi-year employment, partial performance, merchant confirmations
- **Difficulty Distribution:** 2 easy, 5 medium, 3 hard
- **Ground Truth:** 6 enforceable, 4 unenforceable

**Key Legal Concepts:**
- Writing requirements for different contract types
- Exceptions (partial performance, merchant confirmations)
- Special categories (land, goods, marriage, surety)

### Property Law (`property_law/`)

10 scenarios covering various property law doctrines:

- **Topics:** Adverse possession, easements (necessity & prescriptive), recording acts, fixtures, partition, restrictive covenants, life estates, implied warranties, equitable servitudes
- **Difficulty Distribution:** 2 easy, 5 medium, 3 hard
- **Ground Truth:** 9 enforceable, 1 unenforceable

**Key Legal Concepts:**
- Property acquisition and rights
- Land use restrictions
- Property transfers and priorities
- Co-ownership and division

## Scenario Format

Each scenario is a JSON file with the following structure:

```json
{
  "id": "unique_scenario_id",
  "description": "Brief description of the scenario",
  "facts": ["List", "of", "natural", "language", "facts"],
  "asp_facts": "ASP representation of facts for symbolic reasoning",
  "question": "Legal question to be answered",
  "ground_truth": "enforceable or unenforceable",
  "rationale": "Legal reasoning for the ground truth",
  "legal_citations": ["Relevant", "legal", "sources"],
  "difficulty": "easy, medium, or hard"
}
```

## Usage

### Dataset Verification

Verify datasets are properly formatted:

```bash
python3 scripts/verify_transfer_datasets.py
```

### Dataset Loading

Load datasets programmatically:

```python
from pathlib import Path
from experiments.casework.dataset_loader import DatasetLoader

# Load all scenarios
loader = DatasetLoader(Path("datasets/statute_of_frauds"))
scenarios = loader.load_all()

# Load by difficulty
easy_scenarios = loader.load_by_difficulty("easy")

# Get statistics
stats = loader.get_statistics()
print(f"Total scenarios: {stats['total_scenarios']}")
```

### Transfer Study Experiments

Test cross-domain knowledge transfer:

```bash
python3 experiments/casework/transfer_study.py \
  --source-domain datasets/statute_of_frauds \
  --target-domain datasets/property_law \
  --few-shot 10 \
  --output reports/transfer_test.json
```

This will:
1. Train on the source domain (Statute of Frauds)
2. Test zero-shot transfer to target domain (Property Law)
3. Perform few-shot learning on target domain
4. Compare against from-scratch baseline
5. Generate detailed transfer metrics report

### Casework Explorer

Process scenarios with automated learning:

```bash
python3 experiments/casework/explorer.py \
  --dataset datasets/statute_of_frauds \
  --max-cases 20 \
  --output reports/casework_results.json
```

## Design Notes

### Domain Relationships

The **Statute of Frauds** and **Property Law** datasets are designed to test transfer learning:

- **Shared Concepts:** Both involve contracts, property rights, enforceability
- **Distinct Focus:** SOF focuses on writing requirements; Property Law on ownership/use
- **Transfer Hypothesis:** Rules about contract formation may partially transfer to property rights analysis

### Scenario Design Principles

1. **Balanced Distribution:** Each dataset has a mix of easy, medium, and hard scenarios
2. **Realistic Complexity:** Scenarios reflect real-world legal complexity
3. **Clear Ground Truth:** Each scenario has a definitive correct answer
4. **ASP-Ready:** All scenarios include ASP fact representations
5. **Educational Value:** Scenarios cover core doctrines taught in law school

### Adding New Datasets

To create a new dataset:

1. Create a directory: `datasets/your_domain_name/`
2. Add scenario JSON files following the format above
3. Include 10+ scenarios with varied difficulty
4. Run verification: `python3 scripts/verify_transfer_datasets.py`
5. Document in this README

Recommended domains for future datasets:
- Torts (negligence, strict liability, causation)
- Criminal law (mens rea, actus reus, defenses)
- Constitutional law (due process, equal protection)
- Evidence (hearsay, privileges, relevance)

## Citations

Scenarios are based on common law principles and the following sources:
- Restatement (Second) of Contracts
- Uniform Commercial Code (UCC)
- Restatement of Property
- Standard law school case law

All scenarios are simplified for educational/testing purposes and should not be used as legal advice.
