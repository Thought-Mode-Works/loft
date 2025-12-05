"""
CLI Interface for Autonomous Test Harness.

This module provides a Click-based command-line interface for
managing autonomous test runs.

Commands:
- start: Start a new autonomous run
- resume: Resume from checkpoint
- status: Check run status
- report: Generate report from run
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click

from loft.autonomous.config import AutonomousRunConfig
from loft.autonomous.health import create_health_server
from loft.autonomous.logging_config import (
    create_log_summary,
    setup_autonomous_logging,
)
from loft.autonomous.notifications import create_notification_manager
from loft.autonomous.persistence import create_persistence_manager
from loft.autonomous.runner import AutonomousTestRunner


def setup_logging(
    log_level: str,
    log_file: Optional[Path] = None,
    enable_clingo_filter: bool = True,
    progress_interval_seconds: float = 300.0,
) -> Dict[str, Any]:
    """Configure logging with improved readability.

    Uses the enhanced logging configuration that:
    - Filters Clingo "info" messages that flood logs
    - Provides summarized error logging for LLM failures
    - Standardizes log format across all modules
    - Adds periodic progress indicators

    Args:
        log_level: Logging level string
        log_file: Optional log file path
        enable_clingo_filter: Whether to filter Clingo info messages
        progress_interval_seconds: Seconds between progress logs

    Returns:
        Dictionary with logging components for customization
    """
    return setup_autonomous_logging(
        log_level=log_level,
        log_file=log_file,
        enable_clingo_filter=enable_clingo_filter,
        progress_interval_seconds=progress_interval_seconds,
    )


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """LOFT Autonomous Test Harness.

    Run long-duration autonomous test experiments with meta-reasoning
    integration, checkpointing, and Slack notifications.
    """
    pass


@cli.command()
@click.option(
    "--dataset",
    "-d",
    multiple=True,
    type=click.Path(exists=True),
    help="Dataset directory or file path (can specify multiple)",
)
@click.option(
    "--source",
    "-s",
    type=click.Choice(["local", "courtlistener"]),
    default="local",
    help="Data source: 'local' for files, 'courtlistener' for API",
)
@click.option(
    "--search-queries",
    multiple=True,
    help="Search queries for CourtListener (e.g., 'statute of frauds')",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="YAML configuration file",
)
@click.option(
    "--duration",
    "-t",
    default="4h",
    help="Maximum duration (e.g., 4h, 30m, 1.5h)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/autonomous_runs",
    help="Output directory for run data",
)
@click.option(
    "--run-id",
    help="Custom run ID (auto-generated if not provided)",
)
@click.option(
    "--checkpoint-interval",
    default=15,
    type=int,
    help="Minutes between checkpoints",
)
@click.option(
    "--max-cases",
    default=0,
    type=int,
    help="Maximum cases to process (0 = unlimited)",
)
@click.option(
    "--model",
    default="claude-3-5-haiku-20241022",
    help="LLM model to use",
)
@click.option(
    "--slack-webhook",
    envvar="SLACK_WEBHOOK_URL",
    help="Slack webhook URL for notifications",
)
@click.option(
    "--health-port",
    default=8080,
    type=int,
    help="Port for health endpoint",
)
@click.option(
    "--no-health",
    is_flag=True,
    help="Disable health endpoint",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level",
)
@click.option(
    "--no-clingo-filter",
    is_flag=True,
    help="Disable filtering of Clingo info messages",
)
@click.option(
    "--progress-interval",
    default=300,
    type=int,
    help="Seconds between progress summary logs (0 to disable)",
)
def start(
    dataset: tuple,
    source: str,
    search_queries: tuple,
    config: Optional[str],
    duration: str,
    output: str,
    run_id: Optional[str],
    checkpoint_interval: int,
    max_cases: int,
    model: str,
    slack_webhook: Optional[str],
    health_port: int,
    no_health: bool,
    log_level: str,
    no_clingo_filter: bool,
    progress_interval: int,
) -> None:
    """Start a new autonomous test run.

    Examples:

        # Basic run with local dataset
        loft-autonomous start --dataset datasets/contracts/ --duration 4h

        # Run with CourtListener API (real legal cases)
        loft-autonomous start \\
            --source courtlistener \\
            --search-queries "statute of frauds" \\
            --search-queries "adverse possession" \\
            --duration 2h \\
            --max-cases 100

        # Custom configuration with multiple datasets
        loft-autonomous start \\
            --dataset datasets/contracts/ \\
            --dataset datasets/torts/ \\
            --config config/autonomous.yaml \\
            --duration 2h \\
            --max-cases 500

        # With Slack notifications
        loft-autonomous start \\
            --dataset datasets/contracts/ \\
            --slack-webhook https://hooks.slack.com/... \\
            --duration 4h

        # With verbose logging (no Clingo filtering)
        loft-autonomous start \\
            --dataset datasets/contracts/ \\
            --no-clingo-filter \\
            --progress-interval 60 \\
            --duration 1h
    """
    # Set up logging file path
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / "run.log" if run_id else None

    # Set up logging with improved readability
    logging_components = setup_logging(
        log_level=log_level,
        log_file=log_file,
        enable_clingo_filter=not no_clingo_filter,
        progress_interval_seconds=float(progress_interval),
    )
    logger = logging.getLogger(__name__)
    progress_indicator = logging_components.get("progress_indicator")
    clingo_filter = logging_components.get("clingo_filter")
    error_summarizer = logging_components.get("error_summarizer")

    # Validate arguments based on source
    if source == "local" and not dataset:
        raise click.UsageError("--dataset is required when using --source local")

    duration_hours = _parse_duration(duration)

    if config:
        run_config = AutonomousRunConfig.from_yaml(Path(config))
    else:
        run_config = AutonomousRunConfig()

    run_config.max_duration_hours = duration_hours
    run_config.checkpoint_interval_minutes = checkpoint_interval
    run_config.max_cases = max_cases
    run_config.llm_model = model
    run_config.output_dir = output
    run_config.dataset_paths = list(dataset) if dataset else []

    if slack_webhook:
        run_config.notification.slack_webhook_url = slack_webhook

    run_config.health.enabled = not no_health
    run_config.health.port = health_port

    # Set up data source adapter based on source type
    data_source_adapter = None
    if source == "courtlistener":
        from loft.autonomous.data_sources import create_courtlistener_adapter

        queries = list(search_queries) if search_queries else None
        data_source_adapter = create_courtlistener_adapter(
            search_queries=queries,
            max_cases_per_query=max(50, max_cases // 10) if max_cases > 0 else 50,
        )
        logger.info(f"Using CourtListener API with queries: {queries or 'default'}")
    else:
        logger.info(f"Using local datasets: {dataset}")

    logger.info(f"Starting autonomous run with source: {source}")
    logger.info(f"Max duration: {duration_hours} hours")
    logger.info(f"Output directory: {output}")

    health_server = None
    notification_manager = None

    try:
        if run_config.health.enabled:
            health_server = create_health_server(run_config.health)
            health_server.start()

        notification_manager = create_notification_manager(run_config.notification)

        runner = AutonomousTestRunner(run_config, output_dir=Path(output))

        # Set data source adapter if using CourtListener
        if data_source_adapter:
            runner.set_data_source(data_source_adapter)

        def on_progress(progress):
            if health_server:
                health_server.update_state(runner.state)

        runner.set_callbacks(on_progress=on_progress)

        notification_manager.notify_started(
            run_id=run_id or "auto",
            config=run_config.to_dict(),
        )

        result = runner.start(dataset_paths=list(dataset), run_id=run_id)

        notification_manager.notify_completion(result)

        # Log final summary
        summary = create_log_summary(
            clingo_filter=clingo_filter,
            error_summarizer=error_summarizer,
            progress_indicator=progress_indicator,
        )
        logger.info("\n" + summary)

        if result.was_successful:
            click.echo("\nRun completed successfully!")
            click.echo(f"Run ID: {result.run_id}")
            click.echo(f"Duration: {result.duration_hours:.2f} hours")
            click.echo(f"Final accuracy: {result.final_metrics.overall_accuracy:.2%}")
            click.echo(f"Report: {Path(output) / result.run_id / 'reports' / 'final_report.md'}")
        else:
            click.echo(f"\nRun ended with status: {result.status.value}")
            if result.error_message:
                click.echo(f"Error: {result.error_message}")
            if result.checkpoint_path:
                click.echo(f"Checkpoint: {result.checkpoint_path}")

    except KeyboardInterrupt:
        click.echo("\nInterrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.exception("Run failed")
        if notification_manager:
            notification_manager.notify_error(
                run_id=run_id or "unknown",
                error=e,
                context={},
            )
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    finally:
        if health_server:
            health_server.stop()


@cli.command()
@click.option(
    "--checkpoint",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Checkpoint file to resume from",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level",
)
def resume(checkpoint: str, log_level: str) -> None:
    """Resume an autonomous run from checkpoint.

    Examples:

        # Resume from latest checkpoint
        loft-autonomous resume --checkpoint data/autonomous_runs/run_001/checkpoints/latest.json

        # Resume from specific checkpoint
        loft-autonomous resume --checkpoint data/autonomous_runs/run_001/checkpoints/checkpoint_0005.json
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    checkpoint_path = Path(checkpoint)
    run_dir = checkpoint_path.parent.parent

    config_path = run_dir / "config.json"
    if not config_path.exists():
        click.echo(f"Config not found at {config_path}", err=True)
        sys.exit(1)

    import json

    with open(config_path) as f:
        config_dict = json.load(f)

    run_config = AutonomousRunConfig(**config_dict)

    logger.info(f"Resuming from checkpoint: {checkpoint}")

    health_server = None

    try:
        if run_config.health.enabled:
            health_server = create_health_server(run_config.health)
            health_server.start()

        runner = AutonomousTestRunner(run_config)

        def on_progress(progress):
            if health_server:
                health_server.update_state(runner.state)

        runner.set_callbacks(on_progress=on_progress)

        result = runner.resume(checkpoint_path)

        if result.was_successful:
            click.echo("\nRun completed successfully!")
            click.echo(f"Run ID: {result.run_id}")
            click.echo(f"Duration: {result.duration_hours:.2f} hours")
            click.echo(f"Final accuracy: {result.final_metrics.overall_accuracy:.2%}")
        else:
            click.echo(f"\nRun ended with status: {result.status.value}")
            if result.checkpoint_path:
                click.echo(f"Checkpoint: {result.checkpoint_path}")

    except KeyboardInterrupt:
        click.echo("\nInterrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.exception("Resume failed")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    finally:
        if health_server:
            health_server.stop()


@cli.command()
@click.option(
    "--run-id",
    "-r",
    required=True,
    help="Run ID to check status",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/autonomous_runs",
    help="Output directory containing runs",
)
def status(run_id: str, output: str) -> None:
    """Check status of an autonomous run.

    Examples:

        loft-autonomous status --run-id run_20240101_120000_abc123
    """
    run_dir = Path(output) / run_id

    if not run_dir.exists():
        click.echo(f"Run not found: {run_dir}", err=True)
        sys.exit(1)

    persistence = create_persistence_manager(Path(output), run_id)

    state = persistence.load_state()
    if state:
        click.echo(f"Run ID: {state.run_id}")
        click.echo(f"Status: {state.status.value}")

        progress = state.progress
        click.echo("\nProgress:")
        click.echo(f"  Cases processed: {progress.cases_processed}/{progress.total_cases}")
        click.echo(f"  Accuracy: {progress.current_accuracy:.2%}")
        click.echo(f"  Elapsed: {progress.elapsed_seconds / 3600:.2f} hours")
        click.echo(f"  Improvement cycles: {progress.total_cycles}")

        if state.last_updated:
            click.echo(f"\nLast updated: {state.last_updated.isoformat()}")

    result = persistence.load_result()
    if result:
        click.echo("\nFinal Result:")
        click.echo(f"  Status: {result.status.value}")
        click.echo(f"  Duration: {result.duration_hours:.2f} hours")
        click.echo(f"  Accuracy: {result.final_metrics.overall_accuracy:.2%}")

    checkpoints = persistence.list_checkpoints()
    if checkpoints:
        click.echo(f"\nCheckpoints: {len(checkpoints)}")
        click.echo(f"  Latest: {checkpoints[-1].name}")


@cli.command()
@click.option(
    "--run-id",
    "-r",
    required=True,
    help="Run ID to generate report for",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/autonomous_runs",
    help="Output directory containing runs",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "markdown", "both"]),
    default="both",
    help="Report format",
)
def report(run_id: str, output: str, format: str) -> None:
    """Generate report from a completed run.

    Examples:

        loft-autonomous report --run-id run_20240101_120000_abc123 --format markdown
    """
    run_dir = Path(output) / run_id

    if not run_dir.exists():
        click.echo(f"Run not found: {run_dir}", err=True)
        sys.exit(1)

    persistence = create_persistence_manager(Path(output), run_id)
    result = persistence.load_result()

    if not result:
        click.echo("No result found for this run", err=True)
        sys.exit(1)

    reports_dir = run_dir / "reports"

    if format in ["json", "both"]:
        json_path = reports_dir / "final_report.json"
        click.echo(f"JSON report: {json_path}")

    if format in ["markdown", "both"]:
        md_path = reports_dir / "final_report.md"
        click.echo(f"Markdown report: {md_path}")

        if md_path.exists():
            click.echo("\n" + "=" * 60)
            with open(md_path) as f:
                click.echo(f.read())


@cli.command()
def list_runs() -> None:
    """List all autonomous runs.

    Examples:

        loft-autonomous list-runs
    """
    output_dir = Path("data/autonomous_runs")

    if not output_dir.exists():
        click.echo("No runs found")
        return

    runs = sorted(output_dir.iterdir())
    if not runs:
        click.echo("No runs found")
        return

    click.echo(f"Found {len(runs)} run(s):\n")

    for run_dir in runs:
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        state_file = run_dir / "state.json"

        status_str = "unknown"
        if state_file.exists():
            import json

            with open(state_file) as f:
                state_data = json.load(f)
                status_str = state_data.get("status", "unknown")

        click.echo(f"  {run_id}: {status_str}")


def _parse_duration(duration: str) -> float:
    """Parse duration string to hours.

    Args:
        duration: Duration string (e.g., "4h", "30m", "1.5h")

    Returns:
        Duration in hours
    """
    duration = duration.strip().lower()

    if duration.endswith("h"):
        return float(duration[:-1])
    elif duration.endswith("m"):
        return float(duration[:-1]) / 60
    elif duration.endswith("s"):
        return float(duration[:-1]) / 3600
    else:
        return float(duration)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
