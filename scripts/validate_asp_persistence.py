import argparse
from pathlib import Path

from loft.persistence.asp_persistence import ASPPersistenceManager, LoadResult
from loft.symbolic.stratification import StratificationLevel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the integrity of persisted ASP rules in a directory."
    )
    parser.add_argument(
        "asp_rules_dir",
        type=str,
        help="Path to the directory containing ASP rule files (e.g., ./asp_rules).",
    )
    parser.add_argument(
        "--recover-on-error",
        action="store_true",
        help="Attempt to recover from corrupted files by skipping invalid rules/layers.",
    )
    args = parser.parse_args()

    rules_dir = Path(args.asp_rules_dir)

    if not rules_dir.is_dir():
        print(f"Error: Directory '{rules_dir}' not found or is not a directory.")
        exit(1)

    print(f"Validating ASP rule persistence in: {rules_dir}\n")

    manager = ASPPersistenceManager(str(rules_dir), enable_git=False)
    load_result: LoadResult

    try:
        load_result = manager.load_all_rules(recover_on_error=args.recover_on_error)
    except Exception as e:
        print(
            f"FATAL ERROR: Could not initialize persistence manager or load rules: {e}"
        )
        print("Validation failed prematurely.")
        exit(1)

    print("\n--- Validation Report ---")
    if load_result.had_errors:
        print("Integrity Status: 🔴 FAILED (Errors detected)")
        if load_result.parsing_errors:
            print("\nParsing Errors:")
            for error in load_result.parsing_errors:
                print(f"- {error}")
        if load_result.recovered_layers:
            print("\nLayers Recovered (rules skipped):")
            for layer in load_result.recovered_layers:
                print(f"- {layer}.lp")
    else:
        print("Integrity Status: 🟢 PASSED (No errors detected)")

    print("\n--- Loaded Rules Summary ---")
    total_rules = 0
    for level in StratificationLevel:
        rules_count = len(load_result.rules_by_layer.get(level, []))
        total_rules += rules_count
        print(f"- {level.value.capitalize()} layer: {rules_count} rules")

    print(f"\nTotal rules loaded: {total_rules}")

    print("\n--- Persistence Manager Stats ---")
    try:
        stats = manager.get_stats()
        print(f"- Total snapshots: {stats.get('snapshots_count', 0)}")
        print(f"- Total backups: {stats.get('backups_count', 0)}")
        print("- Last modified times per layer:")
        for layer_name, layer_stats in stats.get("layers", {}).items():
            print(f"  - {layer_name}: {layer_stats.get('last_modified', 'N/A')}")
    except Exception as e:
        print(f"Error retrieving manager stats: {e}")

    if load_result.had_errors:
        exit(1)  # Indicate failure via exit code
    else:
        exit(0)  # Indicate success


if __name__ == "__main__":
    main()
