#!/usr/bin/env python3
"""
View coverage progression for an experiment.

Displays coverage metrics and trends from experiment state.
"""

import json
import sys
from pathlib import Path

import click


@click.command()
@click.argument("experiment_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format", "output_format", type=click.Choice(["text", "json"]), default="text"
)
def view_coverage(experiment_dir: Path, output_format: str):
    """
    View coverage progression for an experiment.

    EXPERIMENT_DIR: Path to experiment directory
    """
    # Look for coverage history file
    coverage_file = experiment_dir / "coverage_history.json"

    if not coverage_file.exists():
        click.echo(f"Error: No coverage history found at {coverage_file}", err=True)
        sys.exit(1)

    # Load coverage data
    with open(coverage_file, "r") as f:
        data = json.load(f)

    if output_format == "json":
        click.echo(json.dumps(data, indent=2))
        return

    # Display as text
    history = data.get("history", [])

    if not history:
        click.echo("No coverage history available.")
        return

    click.echo("# Coverage Progression\n")
    click.echo(f"Total snapshots: {len(history)}")
    click.echo(f"Domain predicates: {data.get('domain_predicates_count', 'N/A')}\n")

    # Show progression
    click.echo("| Snapshot | Timestamp | Predicates | Cases | Rules |")
    click.echo("|----------|-----------|------------|-------|-------|")

    for i, snapshot in enumerate(history):
        timestamp = snapshot.get("timestamp", "")[:16]  # Truncate timestamp
        pred_cov = snapshot.get("predicates_covered", 0)
        pred_tot = snapshot.get("predicates_total", 1)
        pred_pct = (pred_cov / pred_tot * 100) if pred_tot > 0 else 0

        case_cov = snapshot.get("cases_with_predictions", 0)
        case_tot = snapshot.get("cases_total", 1)
        case_pct = (case_cov / case_tot * 100) if case_tot > 0 else 0

        total_rules = snapshot.get("total_rules", 0)

        click.echo(
            f"| {i+1:8d} | {timestamp:>9s} | {pred_pct:5.1f}% | {case_pct:5.1f}% | {total_rules:5d} |"
        )

    # Show current coverage
    current = history[-1]
    click.echo("\n## Current Coverage")
    click.echo(f"Predicate coverage: {pred_pct:.1f}%")
    click.echo(f"Case coverage: {case_pct:.1f}%")
    click.echo(f"Total rules: {current.get('total_rules', 0)}")

    # Check monotonicity
    is_monotonic = True
    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        prev_cov = prev.get("predicates_covered", 0) / max(
            prev.get("predicates_total", 1), 1
        )
        curr_cov = curr.get("predicates_covered", 0) / max(
            curr.get("predicates_total", 1), 1
        )
        if curr_cov < prev_cov:
            is_monotonic = False
            break

    mono_status = "✓ Maintained" if is_monotonic else "✗ Violated"
    click.echo(f"Monotonicity: {mono_status}")


if __name__ == "__main__":
    view_coverage()
