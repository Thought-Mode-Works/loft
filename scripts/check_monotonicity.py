#!/usr/bin/env python3
"""
Check monotonicity of coverage progression.

Verifies that coverage never decreases during learning.
"""

import json
import sys
from pathlib import Path

import click


@click.command()
@click.argument("experiment_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--tolerance", type=float, default=0.0, help="Allowed decrease tolerance")
def check_monotonicity(experiment_dir: Path, tolerance: float):
    """
    Check monotonicity of coverage progression.

    EXPERIMENT_DIR: Path to experiment directory

    Exit code 0: Monotonicity maintained
    Exit code 1: Monotonicity violated
    """
    # Look for coverage history file
    coverage_file = experiment_dir / "coverage_history.json"

    if not coverage_file.exists():
        click.echo(f"Error: No coverage history found at {coverage_file}", err=True)
        sys.exit(1)

    # Load coverage data
    with open(coverage_file, "r") as f:
        data = json.load(f)

    history = data.get("history", [])

    if len(history) < 2:
        click.echo("✓ Monotonicity check: PASSED (insufficient data)")
        return

    # Check each transition
    violations = []

    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]

        prev_cov = prev.get("predicates_covered", 0) / max(
            prev.get("predicates_total", 1), 1
        )
        curr_cov = curr.get("predicates_covered", 0) / max(
            curr.get("predicates_total", 1), 1
        )

        decrease = prev_cov - curr_cov

        if decrease > tolerance:
            violations.append(
                {
                    "snapshot": i,
                    "prev_coverage": prev_cov,
                    "curr_coverage": curr_cov,
                    "decrease": decrease,
                    "timestamp": curr.get("timestamp", "unknown"),
                }
            )

    if not violations:
        click.echo(f"✓ Monotonicity check: PASSED (tolerance: {tolerance})")
        sys.exit(0)
    else:
        click.echo(f"✗ Monotonicity check: FAILED ({len(violations)} violations)")
        click.echo()
        click.echo("Violations:")
        for v in violations:
            click.echo(
                f"  Snapshot {v['snapshot']}: {v['prev_coverage']:.2%} → {v['curr_coverage']:.2%} "
                f"(decrease: {v['decrease']:.2%}) at {v['timestamp'][:16]}"
            )

        sys.exit(1)


if __name__ == "__main__":
    check_monotonicity()
