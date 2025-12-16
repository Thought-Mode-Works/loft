#!/usr/bin/env python3
"""
View experiment status.

Issue #256: Long-Running Experiment Runner

Example:
    python scripts/experiment_status.py baseline_001
"""

import sys
from pathlib import Path

import click

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loft.experiments import ExperimentState


@click.command()
@click.argument("experiment_id")
@click.option(
    "--state-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/experiments/"),
    help="Directory containing experiment state",
)
def experiment_status(experiment_id: str, state_dir: Path):
    """View status of an experiment."""

    state_file = state_dir / f"{experiment_id}_state.json"

    if not state_file.exists():
        click.echo(f"Error: Experiment '{experiment_id}' not found", err=True)
        click.echo(f"Looked in: {state_file}", err=True)
        sys.exit(1)

    # Load state
    state = ExperimentState.load(state_file)

    # Get summary
    summary = state.get_summary()

    # Display status
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Experiment: {state.experiment_id}")
    click.echo(f"{'=' * 60}\n")

    click.echo(f"Started: {state.started_at}")
    click.echo(f"Last Updated: {state.last_updated}")
    click.echo(f"Elapsed Time: {summary['elapsed_time'] / 3600:.2f} hours\n")

    click.echo(f"Cycles Completed: {summary['cycles_completed']}")
    click.echo(f"Cases Processed: {summary['cases_processed']}")
    click.echo(f"Rules Incorporated: {summary['rules_incorporated']}\n")

    click.echo(f"Current Accuracy: {summary['current_accuracy']:.2%}")
    click.echo(f"Current Coverage: {summary['current_coverage']:.2%}\n")

    click.echo("Goals:")
    for goal, achieved in summary["goals_achieved"].items():
        status = "✓" if achieved else "✗"
        click.echo(f"  {status} {goal}")

    click.echo(f"\nState file: {state_file}")


if __name__ == "__main__":
    experiment_status()
