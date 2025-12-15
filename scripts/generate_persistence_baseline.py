import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Assuming these are available in the loft environment
from loft.persistence.asp_persistence import ASPPersistenceManager
from loft.persistence.metrics import PersistenceMetricsCollector
from loft.symbolic.stratification import StratificationLevel
from loft.symbolic.asp_program import StratifiedASPCore
from loft.symbolic.asp_rule import ASPRule, RuleMetadata

# --- Helper Functions (copied from tests/integration/persistence/test_asp_persistence_validation.py) ---


def create_test_asp_core_with_rules(
    num_rules: int, start_idx: int = 0
) -> StratifiedASPCore:
    """Creates a StratifiedASPCore with a specified number of dummy rules."""
    core = StratifiedASPCore()
    for i in range(start_idx, start_idx + num_rules):
        confidence_val = 0.8 + (i % 10) / 100
        rule = ASPRule(
            rule_id=f"rule_{i}",
            asp_text=f"p({i}) :- q({i}).",
            stratification_level=StratificationLevel.TACTICAL,
            confidence=confidence_val,
            metadata=RuleMetadata(
                provenance="test_gen",
                timestamp=f"2025-01-01T{i%24:02d}:00:00Z",
                validation_score=confidence_val,
                tags=[f"case_{i%5}"],
                notes=f"Generated for test, author: TestRunner{i%2}",
            ),
        )
        core.add_rule(rule)
    return core


def add_rules_to_core(
    core: StratifiedASPCore, num_rules: int, start_idx: int = 0
) -> None:
    """Adds a specified number of dummy rules to an existing core."""
    for i in range(start_idx, start_idx + num_rules):
        confidence_val = 0.7 + (i % 10) / 100
        rule = ASPRule(
            rule_id=f"new_rule_{i}",
            asp_text=f"new_p({i}) :- new_q({i}).",
            stratification_level=StratificationLevel.OPERATIONAL,
            confidence=confidence_val,
            metadata=RuleMetadata(
                provenance="test_incremental",
                timestamp=f"2025-01-02T{i%24:02d}:00:00Z",
                validation_score=confidence_val,
                tags=[f"new_case_{i%3}"],
                notes=f"Incremental rule, author: TestRunner{i%2}",
            ),
        )
        core.add_rule(rule)


def create_or_modify_core(
    cycle: int, base_rules: int = 10, new_rules_per_cycle: int = 2
) -> StratifiedASPCore:
    """Creates a new core or modifies an existing one based on the cycle number."""
    # Start with a base set of rules
    core = create_test_asp_core_with_rules(base_rules, start_idx=0)

    # Add new rules based on the cycle number
    # This ensures the core grows over time in a long-running test
    if cycle > 0:
        add_rules_to_core(core, new_rules_per_cycle * cycle, start_idx=base_rules)

    return core


# --- Main Script ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate persistence baseline metrics for ASPPersistenceManager."
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
        help="Number of save/load cycles to run for baseline.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=10,
        help="Interval (in cycles) for creating snapshots.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./asp_rules_baseline_test",
        help="Temporary directory for persistence operations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reports",
        help="Directory to save the baseline report.",
    )
    parser.add_argument(
        "--initial-rules",
        type=int,
        default=10,
        help="Number of initial rules in the core.",
    )
    parser.add_argument(
        "--rules-per-cycle",
        type=int,
        default=2,
        help="Number of new rules added per cycle.",
    )
    args = parser.parse_args()

    test_base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up previous test_base_dir if it exists
    if test_base_dir.exists():
        shutil.rmtree(test_base_dir)
    test_base_dir.mkdir(parents=True, exist_ok=True)

    manager = ASPPersistenceManager(str(test_base_dir), enable_git=False)
    metrics_collector = PersistenceMetricsCollector()

    print(f"Running {args.cycles} persistence cycles...")
    print(f"Persistence directory: {test_base_dir}")

    for i in range(args.cycles):
        if i % 10 == 0:
            print(f"  Cycle {i}/{args.cycles}")

        # Create or modify rules - progressively larger core
        current_core = create_or_modify_core(
            i, base_rules=args.initial_rules, new_rules_per_cycle=args.rules_per_cycle
        )

        # Save with metrics
        rules_by_layer_current = {
            level: current_core.get_program(level).rules
            for level in StratificationLevel
        }
        metrics_collector.measure_save_cycle(manager, rules_by_layer_current)

        # Periodic snapshot
        if i % args.snapshot_interval == 0:
            metrics_collector.measure_snapshot_cycle(manager, i, f"cycle_{i}")

        # Load (and measure load metrics)
        metrics_collector.measure_load_cycle(manager)

    print("\nGenerating baseline report...")
    baseline_report = metrics_collector.generate_baseline_report()

    # Correct total cycles for report, as each loop adds multiple metrics.
    # We want total_cycles to reflect the number of original loops/updates.
    baseline_report.total_cycles = args.cycles

    report_filename = (
        f"persistence_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    report_path = output_dir / report_filename
    report_path.write_text(baseline_report.to_markdown())

    print(f"Baseline report generated: {report_path}")
    print(f"Total rules processed: {baseline_report.total_rules_processed}")
    print(f"Average save time: {baseline_report.avg_save_time_ms:.2f} ms")
    print(f"Average load time: {baseline_report.avg_load_time_ms:.2f} ms")

    if metrics_collector.error_log:
        print("\n--- Errors during run ---")
        for error in metrics_collector.error_log:
            print(f"- {error['type']}: {error['error']}")
        print("Please check logs for more details.")

    # Clean up the test_base_dir
    if test_base_dir.exists():
        shutil.rmtree(test_base_dir)
        print(f"Cleaned up temporary persistence directory: {test_base_dir}")


if __name__ == "__main__":
    main()
