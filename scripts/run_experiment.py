#!/usr/bin/env python3
"""
Run long-running learning experiment.

Issue #256: Long-Running Experiment Runner

Examples:
    # Start new experiment (30 minutes)
    python scripts/run_experiment.py \\
      --experiment-id baseline_001 \\
      --dataset datasets/contracts/ \\
      --duration 30m \\
      --cases-per-cycle 10 \\
      --report-interval 5m

    # Resume interrupted experiment
    python scripts/run_experiment.py \\
      --experiment-id baseline_001 \\
      --resume
"""

import logging
import sys
from pathlib import Path

import click

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loft.experiments import ExperimentConfig, ExperimentRunner, parse_duration

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_experiment_runner(
    config: ExperimentConfig,
) -> ExperimentRunner:
    """
    Create experiment runner with appropriate processor.

    Args:
        config: Experiment configuration

    Returns:
        Configured ExperimentRunner
    """
    # Import components
    from loft.persistence import ASPPersistenceManager

    # Create persistence manager
    persistence = ASPPersistenceManager(base_dir=str(config.rules_path))

    # Create processor based on config
    if config.enable_llm:
        try:
            from loft.batch.full_pipeline import create_full_pipeline_processor

            processor = create_full_pipeline_processor(
                model=config.model,
                rules_dir=str(config.rules_path),
                enable_persistence=True,
            )
            logger.info(f"Created full pipeline processor with model {config.model}")
        except ImportError as e:
            logger.error(f"Failed to import full pipeline processor: {e}")
            logger.info("Falling back to simple processor")
            from loft.batch.simple_processor import create_simple_processor

            processor = create_simple_processor()
    else:
        # Simple processor without LLM
        from loft.batch.simple_processor import create_simple_processor

        processor = create_simple_processor()
        logger.info("Created simple processor (no LLM)")

    # Wrap with meta-awareness if enabled
    if config.enable_meta:
        try:
            from loft.batch.meta_aware_processor import MetaAwareBatchProcessor
            from loft.batch.meta_state import create_meta_state_manager

            state_manager = create_meta_state_manager(config.meta_state_dir)
            processor = MetaAwareBatchProcessor(
                underlying_processor=processor, state_manager=state_manager
            )
            logger.info("Enabled meta-awareness")
        except ImportError as e:
            logger.warning(f"Failed to enable meta-awareness: {e}")

    # Create runner
    runner = ExperimentRunner(
        config=config, processor=processor, persistence=persistence
    )

    return runner


@click.command()
@click.option("--experiment-id", required=True, help="Unique experiment identifier")
@click.option("--description", default="", help="Experiment description")
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    help="Path to dataset directory",
)
@click.option(
    "--duration",
    default="4h",
    help="Max duration (e.g., 30m, 2h, 4h)",
)
@click.option("--max-cycles", default=100, type=int, help="Maximum number of cycles")
@click.option(
    "--cases-per-cycle", default=20, type=int, help="Cases to process per cycle"
)
@click.option("--report-interval", default="30m", help="Interval for interim reports")
@click.option(
    "--resume/--no-resume", default=True, help="Resume existing experiment if found"
)
@click.option("--enable-llm/--no-llm", default=True, help="Enable LLM processing")
@click.option("--enable-meta/--no-meta", default=True, help="Enable meta-reasoning")
@click.option(
    "--model",
    default="claude-3-5-haiku-20241022",
    help="LLM model to use",
)
@click.option(
    "--target-accuracy", default=0.85, type=float, help="Target accuracy goal"
)
@click.option(
    "--target-coverage", default=0.80, type=float, help="Target coverage goal"
)
@click.option("--target-rules", default=100, type=int, help="Target rule count goal")
def run_experiment(
    experiment_id: str,
    description: str,
    dataset: Path,
    duration: str,
    max_cycles: int,
    cases_per_cycle: int,
    report_interval: str,
    resume: bool,
    enable_llm: bool,
    enable_meta: bool,
    model: str,
    target_accuracy: float,
    target_coverage: float,
    target_rules: int,
):
    """Run a long-running learning experiment."""

    # Build config
    config = ExperimentConfig(
        experiment_id=experiment_id,
        description=description,
        max_duration_seconds=parse_duration(duration),
        max_cycles=max_cycles,
        cases_per_cycle=cases_per_cycle,
        dataset_path=dataset if dataset else Path("datasets/contracts/"),
        report_interval_seconds=parse_duration(report_interval),
        model=model,
        enable_llm=enable_llm,
        enable_meta=enable_meta,
        target_accuracy=target_accuracy,
        target_coverage=target_coverage,
        target_rule_count=target_rules,
    )

    # Check for existing state
    state_file = config.state_path / f"{experiment_id}_state.json"
    if state_file.exists() and resume:
        click.echo(f"Resuming experiment {experiment_id}")
        click.echo(f"State file: {state_file}")
    else:
        click.echo(f"Starting new experiment {experiment_id}")
        click.echo(f"Dataset: {config.dataset_path}")
        click.echo(f"Duration: {duration} ({config.max_duration_seconds}s)")
        click.echo(f"Max cycles: {max_cycles}")

    # Create runner
    click.echo("Creating experiment runner...")
    runner = create_experiment_runner(config)

    # Run experiment
    click.echo("\n" + "=" * 60)
    click.echo("EXPERIMENT STARTED")
    click.echo("=" * 60 + "\n")

    try:
        report = runner.run()

        click.echo("\n" + "=" * 60)
        click.echo("EXPERIMENT COMPLETED")
        click.echo("=" * 60 + "\n")

        # Show final report
        click.echo(report.to_markdown())

        # Show report location
        final_report_path = config.reports_path / f"{experiment_id}_final.md"
        click.echo(f"\nFinal report saved to: {final_report_path}")

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        click.echo(f"\nExperiment failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    run_experiment()
