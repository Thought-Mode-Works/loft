#!/usr/bin/env python3
"""
Run LOFT infrastructure benchmarks.

Issue #257: Baseline Validation Benchmarks

Examples:
    # Run full benchmark suite
    python scripts/run_benchmarks.py --output reports/baseline_001.json

    # Compare against previous
    python scripts/run_benchmarks.py \\
      --output reports/baseline_002.json \\
      --compare reports/baseline_001.json

    # Quick validation (no LLM)
    python scripts/run_benchmarks.py --sample-size 20 --no-llm
"""

import logging
import subprocess
import sys
from pathlib import Path

import click

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loft.benchmarks import (
    BaselineMetrics,
    BenchmarkConfig,
    BenchmarkSuite,
    compare_baselines,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


@click.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("reports/baseline_metrics.json"),
    help="Output path for baseline metrics",
)
@click.option(
    "--sample-size",
    default=50,
    type=int,
    help="Number of samples for benchmarks",
)
@click.option(
    "--enable-llm/--no-llm",
    default=False,
    help="Enable LLM-based benchmarks",
)
@click.option(
    "--compare",
    type=click.Path(exists=True, path_type=Path),
    help="Compare against previous baseline",
)
@click.option(
    "--format",
    type=click.Choice(["json", "markdown", "both"]),
    default="both",
    help="Output format",
)
def run_benchmarks(
    output: Path,
    sample_size: int,
    enable_llm: bool,
    compare: Path,
    format: str,
):
    """Run comprehensive benchmarks and establish baseline."""

    click.echo("=" * 60)
    click.echo("LOFT Infrastructure Benchmarks")
    click.echo("=" * 60 + "\n")

    # Get git commit
    commit_hash = get_git_commit()
    click.echo(f"Commit: {commit_hash[:12]}")
    click.echo(f"Sample size: {sample_size}")
    click.echo(f"LLM enabled: {enable_llm}\n")

    # Create config
    config = BenchmarkConfig(
        sample_size=sample_size,
        enable_llm=enable_llm,
        commit_hash=commit_hash,
        description=f"Baseline with {sample_size} samples",
    )

    # Run benchmarks
    click.echo("Running benchmark suite...")
    suite = BenchmarkSuite(config)
    baseline = suite.run_all()

    # Save baseline
    baseline.save(output)
    click.echo(f"\n✓ Baseline saved to: {output}")

    # Output formats
    if format in ["markdown", "both"]:
        click.echo("\n" + "=" * 60)
        click.echo(baseline.to_markdown())
        click.echo("=" * 60 + "\n")

        # Save markdown
        md_path = output.with_suffix(".md")
        md_path.write_text(baseline.to_markdown())
        click.echo(f"✓ Markdown report saved to: {md_path}")

    # Compare if requested
    if compare:
        click.echo(f"\nLoading previous baseline from: {compare}")
        previous = BaselineMetrics.load(compare)

        comparison = compare_baselines(previous, baseline)

        click.echo("\n" + "=" * 60)
        click.echo(comparison.to_markdown())
        click.echo("=" * 60 + "\n")

        # Save comparison
        comp_path = output.parent / f"comparison_{output.stem}.md"
        comp_path.write_text(comparison.to_markdown())
        click.echo(f"✓ Comparison saved to: {comp_path}")

        # Warn on regressions
        if comparison.regression_count > 0:
            click.echo(
                f"\n⚠️  WARNING: {comparison.regression_count} regressions detected!",
                err=True,
            )
            sys.exit(1)


if __name__ == "__main__":
    run_benchmarks()
