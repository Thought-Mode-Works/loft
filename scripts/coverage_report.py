#!/usr/bin/env python3
"""
Generate comprehensive coverage report for an experiment.

Creates detailed markdown report with coverage metrics,
trends, and uncovered predicates.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import click


@click.command()
@click.argument("experiment_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "text"]),
    default="markdown",
)
def coverage_report(experiment_dir: Path, output: Path, output_format: str):
    """
    Generate coverage report for an experiment.

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

    # Generate report
    report = _generate_markdown_report(data, experiment_dir)

    # Output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report)
        click.echo(f"Report written to: {output}")
    else:
        click.echo(report)


def _generate_markdown_report(data: dict, experiment_dir: Path) -> str:
    """Generate markdown coverage report."""
    history = data.get("history", [])

    if not history:
        return "# Coverage Report\n\nNo coverage data available.\n"

    current = history[-1]
    domain_predicates = data.get("domain_predicates", [])

    # Calculate current metrics
    pred_total = current.get("predicates_total", 0)
    pred_covered = current.get("predicates_covered", 0)
    pred_pct = (pred_covered / pred_total * 100) if pred_total > 0 else 0

    case_total = current.get("cases_total", 0)
    case_covered = current.get("cases_with_predictions", 0)
    case_pct = (case_covered / case_total * 100) if case_total > 0 else 0

    scenario_total = current.get("scenarios_total", 0)
    scenario_covered = current.get("scenarios_covered", 0)
    scenario_pct = (
        (scenario_covered / scenario_total * 100) if scenario_total > 0 else 0
    )

    # Build report
    report = f"""# Coverage Report

**Experiment**: {experiment_dir.name}
**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Snapshots**: {len(history)}

---

## Current Coverage

| Metric | Coverage | Count |
|--------|----------|-------|
| **Predicates** | {pred_pct:.1f}% | {pred_covered}/{pred_total} |
| **Cases** | {case_pct:.1f}% | {case_covered}/{case_total} |
| **Scenarios** | {scenario_pct:.1f}% | {scenario_covered}/{scenario_total} |
| **Total Rules** | - | {current.get('total_rules', 0)} |

### Rules by Layer

"""

    rules_by_layer = current.get("rules_by_layer", {})
    for layer, count in sorted(rules_by_layer.items()):
        report += f"- **{layer}**: {count} rules\n"

    # Coverage trend
    report += "\n## Coverage Trend\n\n"

    if len(history) >= 5:
        # Calculate trend over last 5 snapshots
        recent = history[-5:]
        first_cov = recent[0].get("predicates_covered", 0) / max(
            recent[0].get("predicates_total", 1), 1
        )
        last_cov = recent[-1].get("predicates_covered", 0) / max(
            recent[-1].get("predicates_total", 1), 1
        )

        change = last_cov - first_cov

        if change > 0.01:
            trend = "📈 **Increasing**"
        elif change < -0.01:
            trend = "📉 **Decreasing**"
        else:
            trend = "📊 **Stable**"

        report += f"Recent trend (last 5 snapshots): {trend}\n\n"
    else:
        report += "Insufficient data for trend analysis.\n\n"

    # Monotonicity check
    is_monotonic = True
    for i in range(1, len(history)):
        prev_cov = history[i - 1].get("predicates_covered", 0) / max(
            history[i - 1].get("predicates_total", 1), 1
        )
        curr_cov = history[i].get("predicates_covered", 0) / max(
            history[i].get("predicates_total", 1), 1
        )
        if curr_cov < prev_cov:
            is_monotonic = False
            break

    mono_status = "✓ Maintained" if is_monotonic else "✗ Violated"
    report += f"**Monotonicity**: {mono_status}\n\n"

    # Uncovered predicates
    covered_predicates = set(current.get("covered_predicates", []))
    all_predicates = set(domain_predicates)
    uncovered = sorted(list(all_predicates - covered_predicates))

    report += f"## Uncovered Predicates ({len(uncovered)})\n\n"

    if uncovered:
        for pred in uncovered[:30]:  # Show first 30
            report += f"- `{pred}`\n"

        if len(uncovered) > 30:
            report += f"\n... and {len(uncovered) - 30} more\n"
    else:
        report += "🎉 All domain predicates are covered!\n"

    # Coverage history table
    if len(history) > 1:
        report += "\n## Coverage History\n\n"
        report += "| Snapshot | Timestamp | Predicates | Cases | Rules |\n"
        report += "|----------|-----------|------------|-------|-------|\n"

        # Show last 15 snapshots
        for i, snapshot in enumerate(history[-15:], start=max(1, len(history) - 14)):
            timestamp = snapshot.get("timestamp", "")[:16]
            pred_cov = snapshot.get("predicates_covered", 0)
            pred_tot = snapshot.get("predicates_total", 1)
            pred_pct = (pred_cov / pred_tot * 100) if pred_tot > 0 else 0

            case_cov = snapshot.get("cases_with_predictions", 0)
            case_tot = snapshot.get("cases_total", 1)
            case_pct = (case_cov / case_tot * 100) if case_tot > 0 else 0

            total_rules = snapshot.get("total_rules", 0)

            report += (
                f"| {i:8d} | {timestamp} | {pred_pct:5.1f}% | "
                f"{case_pct:5.1f}% | {total_rules:5d} |\n"
            )

    report += "\n---\n\n"
    report += "*Generated by LOFT Coverage Report Tool*\n"

    return report


if __name__ == "__main__":
    coverage_report()
