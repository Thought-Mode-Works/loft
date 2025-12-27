"""
CLI interface for knowledge database operations.

Provides command-line tools for database management and queries.

Issue #271: Persistent Legal Knowledge Database
"""

from typing import Optional

import click

from loft.knowledge.coverage_dashboard import CoverageDashboard
from loft.knowledge.coverage_tracker import CoverageTracker
from loft.knowledge.database import KnowledgeDatabase
from loft.knowledge.migration import migrate_asp_files_to_database


@click.group()
def cli():
    """Legal Knowledge Database CLI."""
    pass


@cli.command()
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def init(database_url: str):
    """Initialize the knowledge database."""
    db = KnowledgeDatabase(database_url)
    click.echo(f"Initialized database: {database_url}")
    db.close()


@cli.command()
@click.option(
    "--rules-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing ASP files",
)
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
@click.option("--domain", help="Default domain for imported rules")
def migrate(rules_dir: str, database_url: str, domain: Optional[str]):
    """Migrate ASP files to database."""
    click.echo(f"Migrating ASP files from {rules_dir}...")

    result = migrate_asp_files_to_database(
        rules_dir=rules_dir,
        database_url=database_url,
        default_domain=domain,
    )

    click.echo("\nMigration complete:")
    click.echo(f"  Files processed: {result.files_processed}")
    click.echo(f"  Rules imported: {result.rules_imported}")
    click.echo(f"  Rules skipped: {result.rules_skipped}")
    click.echo(f"  Errors: {result.errors}")

    if result.error_messages:
        click.echo("\nErrors:")
        for error in result.error_messages:
            click.echo(f"  - {error}")


@cli.command()
@click.argument("asp_rule")
@click.option("--domain", help="Legal domain")
@click.option("--confidence", type=float, help="Confidence score (0-1)")
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def add(
    asp_rule: str, domain: Optional[str], confidence: Optional[float], database_url: str
):
    """Add a rule to the database."""
    db = KnowledgeDatabase(database_url)

    try:
        rule_id = db.add_rule(
            asp_rule=asp_rule,
            domain=domain,
            confidence=confidence,
        )
        click.echo(f"Added rule: {rule_id}")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
    finally:
        db.close()


@cli.command()
@click.option("--domain", help="Filter by domain")
@click.option("--min-confidence", type=float, default=0.0, help="Minimum confidence")
@click.option("--limit", type=int, default=10, help="Maximum results")
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def search(
    domain: Optional[str],
    min_confidence: float,
    limit: int,
    database_url: str,
):
    """Search for rules in the database."""
    db = KnowledgeDatabase(database_url)

    rules = db.search_rules(
        domain=domain,
        min_confidence=min_confidence,
        limit=limit,
    )

    click.echo(f"Found {len(rules)} rules:\n")

    for rule in rules:
        click.echo(f"ID: {rule.rule_id}")
        click.echo(f"Rule: {rule.asp_rule}")
        if rule.domain:
            click.echo(f"Domain: {rule.domain}")
        if rule.confidence:
            click.echo(f"Confidence: {rule.confidence:.2f}")
        if rule.reasoning:
            click.echo(f"Reasoning: {rule.reasoning}")
        click.echo()

    db.close()


@cli.command()
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def stats(database_url: str):
    """Show database statistics."""
    db = KnowledgeDatabase(database_url)

    stats = db.get_database_stats()

    click.echo("Database Statistics")
    click.echo("=" * 50)
    click.echo(f"Total Rules: {stats.total_rules}")
    click.echo(f"Active Rules: {stats.active_rules}")
    click.echo(f"Archived Rules: {stats.archived_rules}")
    click.echo(f"Total Questions: {stats.total_questions}")

    if stats.avg_confidence:
        click.echo(f"Average Confidence: {stats.avg_confidence:.2%}")

    click.echo(f"\nDomains: {', '.join(stats.domains)}")

    click.echo("\nCoverage by Domain:")
    for domain, count in stats.coverage_by_domain.items():
        click.echo(f"  {domain}: {count} rules")

    db.close()


@cli.command()
@click.option("--domain", help="Show coverage for specific domain")
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def coverage(domain: Optional[str], database_url: str):
    """Show knowledge coverage statistics."""
    db = KnowledgeDatabase(database_url)

    if domain:
        cov = db.get_coverage_stats(domain)
        if cov:
            click.echo(f"Coverage for {cov.domain}:")
            click.echo(f"  Rules: {cov.rule_count}")
            click.echo(f"  Questions: {cov.question_count}")
            if cov.accuracy:
                click.echo(f"  Accuracy: {cov.accuracy:.2%}")
            if cov.avg_confidence:
                click.echo(f"  Avg Confidence: {cov.avg_confidence:.2%}")
        else:
            click.echo(f"No coverage data for domain: {domain}")
    else:
        coverages = db.get_all_coverage_stats()
        click.echo("Coverage by Domain:")
        click.echo("=" * 50)
        for cov in coverages:
            click.echo(f"\n{cov.domain}:")
            click.echo(f"  Rules: {cov.rule_count}")
            click.echo(f"  Questions: {cov.question_count}")
            if cov.accuracy:
                click.echo(f"  Accuracy: {cov.accuracy:.2%}")

    db.close()


@cli.command()
@click.option(
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Output directory for ASP files",
)
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def export(output_dir: str, database_url: str):
    """Export database rules to ASP files."""
    db = KnowledgeDatabase(database_url)

    stats = db.export_to_asp_files(output_dir)

    click.echo("Export complete:")
    click.echo(f"  Files created: {stats['files_created']}")
    click.echo(f"  Rules exported: {stats['rules_exported']}")
    click.echo(f"  Output directory: {output_dir}")

    db.close()


# Coverage Metrics Commands (Issue #274)


@cli.command(name="metrics-summary")
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def metrics_summary(database_url: str):
    """Display coverage metrics summary dashboard."""
    db = KnowledgeDatabase(database_url)
    dashboard = CoverageDashboard(db)

    summary = dashboard.display_summary()
    click.echo(summary)

    db.close()


@cli.command(name="metrics-domain")
@click.argument("domain")
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def metrics_domain(domain: str, database_url: str):
    """Display detailed metrics for a specific domain."""
    db = KnowledgeDatabase(database_url)
    dashboard = CoverageDashboard(db)

    details = dashboard.display_domain_details(domain)
    click.echo(details)

    db.close()


@cli.command(name="metrics-quality")
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def metrics_quality(database_url: str):
    """Display quality metrics report."""
    db = KnowledgeDatabase(database_url)
    dashboard = CoverageDashboard(db)

    quality = dashboard.display_quality_report()
    click.echo(quality)

    db.close()


@cli.command(name="metrics-gaps")
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def metrics_gaps(database_url: str):
    """Display identified coverage gaps."""
    db = KnowledgeDatabase(database_url)
    dashboard = CoverageDashboard(db)

    gaps = dashboard.display_gaps()
    click.echo(gaps)

    db.close()


@cli.command(name="metrics-report")
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output file for report (Markdown)",
)
def metrics_report(database_url: str, output: Optional[str]):
    """Generate comprehensive coverage report."""
    db = KnowledgeDatabase(database_url)
    dashboard = CoverageDashboard(db)

    report = dashboard.generate_full_report()

    if output:
        with open(output, "w") as f:
            f.write(report)
        click.echo(f"Report saved to: {output}")
    else:
        click.echo(report)

    db.close()


@cli.command(name="metrics-trends")
@click.option("--metric", default="total_rules", help="Metric name to track")
@click.option("--days", type=int, default=30, help="Number of days to display")
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def metrics_trends(metric: str, days: int, database_url: str):
    """Display metric trends over time."""
    db = KnowledgeDatabase(database_url)
    dashboard = CoverageDashboard(db)

    trends = dashboard.display_trends(metric_name=metric, days=days)
    click.echo(trends)

    db.close()


@cli.command(name="metrics-snapshot")
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def metrics_snapshot(database_url: str):
    """Take a snapshot of current metrics."""
    db = KnowledgeDatabase(database_url)
    tracker = CoverageTracker(db)

    metrics = tracker.take_snapshot()

    click.echo("Snapshot taken successfully!")
    click.echo(f"  Total Rules: {metrics.total_rules}")
    click.echo(f"  Domains: {metrics.domain_count}")
    click.echo(f"  Snapshots Stored: {tracker.get_snapshot_count()}")

    db.close()


@cli.command(name="metrics-compare")
@click.option(
    "--days-ago",
    type=int,
    default=7,
    help="Days ago to compare from",
)
@click.option(
    "--comparison-days",
    type=int,
    default=7,
    help="Duration of comparison period",
)
@click.option(
    "--database-url",
    default="sqlite:///legal_knowledge.db",
    help="Database connection URL",
)
def metrics_compare(days_ago: int, comparison_days: int, database_url: str):
    """Compare metrics between two time periods."""
    db = KnowledgeDatabase(database_url)
    dashboard = CoverageDashboard(db)

    comparison = dashboard.compare_periods(
        days_ago=days_ago, comparison_days=comparison_days
    )
    click.echo(comparison)

    db.close()


if __name__ == "__main__":
    cli()
