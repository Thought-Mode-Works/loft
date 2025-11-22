"""
Logical consistency checking framework for symbolic core.

Provides comprehensive consistency checking, property-based testing,
test data generation, and reporting capabilities.

## Components

- **checker**: Core consistency checking engine
  - ConsistencyChecker: Main checking interface
  - InconsistencyType: Types of inconsistencies
  - Inconsistency: Representation of an inconsistency
  - ConsistencyReport: Results of consistency check

- **generators**: Test data generation
  - TestFixtures: Deterministic test fixtures
  - hypothesis strategies: Property-based test data generation

- **properties**: Property-based tests
  - ConsistencyProperties: Property test suite
  - run_property_tests: Execute all property tests

- **reports**: Reporting and analysis
  - ConsistencyReporter: Enhanced reporting with history
  - ConsistencyHistory: Track consistency over time
  - ConsistencyMetrics: Metrics for analysis

## Example Usage

```python
from loft.consistency import ConsistencyChecker, TestFixtures
from loft.version_control import CoreState

# Create checker
checker = ConsistencyChecker(strict=False)

# Check a state
state = TestFixtures.simple_consistent_state()
report = checker.check(state)

# View results
print(report.summary())
print(report.format())

# Check specific inconsistency types
for inconsistency in report.inconsistencies:
    print(f"{inconsistency.type}: {inconsistency.message}")
```
"""

from .checker import (
    ConsistencyChecker,
    InconsistencyType,
    Inconsistency,
    ConsistencyReport,
)

from .generators import (
    TestFixtures,
    rule_strategy,
    core_state_strategy,
    asp_fact_strategy,
    asp_rule_strategy,
    stratification_level_strategy,
)

from .properties import ConsistencyProperties, run_property_tests

from .reports import (
    ConsistencyReporter,
    ConsistencyHistory,
    ConsistencyMetrics,
)

__all__ = [
    # Checker
    "ConsistencyChecker",
    "InconsistencyType",
    "Inconsistency",
    "ConsistencyReport",
    # Generators
    "TestFixtures",
    "rule_strategy",
    "core_state_strategy",
    "asp_fact_strategy",
    "asp_rule_strategy",
    "stratification_level_strategy",
    # Properties
    "ConsistencyProperties",
    "run_property_tests",
    # Reports
    "ConsistencyReporter",
    "ConsistencyHistory",
    "ConsistencyMetrics",
]
