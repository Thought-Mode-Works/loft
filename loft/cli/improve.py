"""
CLI commands for self-modifying system.

Provides commands to:
- Run improvement cycles
- View self-analysis reports
- Check system health
"""

import click

from loft.core.self_modifying_system import SelfModifyingSystem
from loft.symbolic.stratification import StratificationLevel


@click.group()
def improve():
    """Self-modification and improvement commands."""
    pass


@improve.command()
@click.option("--max-gaps", default=5, help="Maximum knowledge gaps to process")
@click.option(
    "--target-layer",
    type=click.Choice(["constitutional", "strategic", "tactical", "operational"]),
    default="tactical",
    help="Target stratification layer for new rules",
)
@click.option(
    "--enable-llm", is_flag=True, help="Enable LLM integration for real rule generation"
)
@click.option(
    "--enable-dialectical",
    is_flag=True,
    help="Enable Phase 4.1 dialectical validation (critique system)",
)
@click.option(
    "--persistence-dir",
    default="./asp_rules",
    help="Directory for persisting ASP rules",
)
def run(max_gaps, target_layer, enable_llm, enable_dialectical, persistence_dir):
    """Run self-improvement cycle."""
    click.echo("=" * 80)
    click.echo("STARTING SELF-IMPROVEMENT CYCLE")
    click.echo("=" * 80)

    # Map string to enum
    layer_map = {
        "constitutional": StratificationLevel.CONSTITUTIONAL,
        "strategic": StratificationLevel.STRATEGIC,
        "tactical": StratificationLevel.TACTICAL,
        "operational": StratificationLevel.OPERATIONAL,
    }

    # Initialize system
    click.echo("\nInitializing self-modifying system...")
    click.echo(
        f"  LLM Integration: {'Enabled' if enable_llm else 'Disabled (Mock Mode)'}"
    )
    click.echo(
        f"  Dialectical Validation: {'Enabled' if enable_dialectical else 'Disabled'}"
    )
    click.echo(f"  Persistence: {persistence_dir}")
    system = SelfModifyingSystem(
        enable_llm=enable_llm,
        enable_dialectical=enable_dialectical,
        persistence_dir=persistence_dir,
    )

    # Run cycle
    click.echo(
        f"\nRunning improvement cycle (max_gaps={max_gaps}, layer={target_layer})..."
    )
    result = system.run_improvement_cycle(
        max_gaps=max_gaps, target_layer=layer_map[target_layer]
    )

    # Display results
    click.echo("\n" + "=" * 80)
    click.echo("CYCLE COMPLETE")
    click.echo("=" * 80)
    click.echo(f"\nCycle #{result.cycle_number}")
    click.echo(f"Status: {result.status}")
    click.echo("\nMetrics:")
    click.echo(f"  Gaps identified: {result.gaps_identified}")
    click.echo(f"  Variants generated: {result.variants_generated}")
    click.echo(f"  Rules incorporated: {result.rules_incorporated}")
    click.echo(f"  Rules pending review: {result.rules_pending_review}")
    click.echo("\nPerformance:")
    click.echo(f"  Baseline accuracy: {result.baseline_accuracy:.2%}")
    click.echo(f"  Final accuracy: {result.final_accuracy:.2%}")
    click.echo(f"  Improvement: {result.overall_improvement:+.2%}")
    click.echo("\n" + "=" * 80)


@improve.command()
def analyze():
    """Generate self-analysis report."""
    click.echo("=" * 80)
    click.echo("SELF-ANALYSIS REPORT")
    click.echo("=" * 80)

    # Initialize system
    system = SelfModifyingSystem()

    # Run a cycle first for meaningful analysis
    click.echo("\nRunning improvement cycle for analysis...")
    system.run_improvement_cycle(max_gaps=3)

    # Generate self-report
    report = system.get_self_report()

    # Display markdown report
    click.echo("\n" + report.to_markdown())


@improve.command()
def health():
    """Check system health."""
    click.echo("=" * 80)
    click.echo("SYSTEM HEALTH CHECK")
    click.echo("=" * 80)

    # Initialize system
    system = SelfModifyingSystem()

    # Generate health report
    health_report = system.get_health_report()

    # Display markdown report
    click.echo("\n" + health_report.to_markdown())


@improve.command()
def history():
    """View improvement cycle history."""
    click.echo("=" * 80)
    click.echo("IMPROVEMENT CYCLE HISTORY")
    click.echo("=" * 80)

    # Initialize system
    system = SelfModifyingSystem()

    # Run a couple cycles for demonstration
    click.echo("\nRunning cycles...")
    system.run_improvement_cycle(max_gaps=2)
    system.run_improvement_cycle(max_gaps=2)

    # Get history
    cycle_history = system.get_cycle_history()

    if not cycle_history:
        click.echo("\nNo improvement cycles found.")
        return

    click.echo(f"\nFound {len(cycle_history)} cycle(s):\n")

    for cycle in cycle_history:
        click.echo(f"Cycle #{cycle.cycle_number}")
        click.echo(f"  Timestamp: {cycle.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo(f"  Status: {cycle.status}")
        click.echo(f"  Gaps: {cycle.gaps_identified}")
        click.echo(f"  Rules incorporated: {cycle.rules_incorporated}")
        click.echo(f"  Improvement: {cycle.overall_improvement:+.2%}")
        click.echo()


@improve.command()
@click.option(
    "--persistence-dir",
    default="./asp_rules",
    help="Directory containing persisted rules",
)
@click.option("--output", default=None, help="Output path for living document")
def document(persistence_dir, output):
    """Generate living document of ASP core evolution."""
    click.echo("=" * 80)
    click.echo("GENERATING LIVING DOCUMENT")
    click.echo("=" * 80)

    # Initialize system
    click.echo("\nInitializing system...")
    system = SelfModifyingSystem(persistence_dir=persistence_dir)

    # Run a cycle if no history exists
    if not system.improvement_cycles:
        click.echo("\nNo improvement cycles found. Running initial cycle...")
        system.run_improvement_cycle(max_gaps=3)

    # Generate document
    click.echo("\nGenerating living document...")
    doc = system.generate_living_document(output_path=output)

    # Display location
    output_path = output or f"{persistence_dir}/LIVING_DOCUMENT.md"
    click.echo(f"\n✓ Living document generated at: {output_path}")
    click.echo(f"  Total size: {len(doc)} characters")
    click.echo()


if __name__ == "__main__":
    improve()
