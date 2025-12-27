"""
CLI commands for rule accumulation pipeline.

Provides commands for processing cases and accumulating rules.

Issue #273: Continuous Rule Accumulation Pipeline
"""

import json
import logging
from pathlib import Path
from typing import Optional

import click

from loft.accumulation.conflict_detection import ConflictDetector
from loft.accumulation.pipeline import RuleAccumulationPipeline
from loft.accumulation.schemas import CaseData
from loft.knowledge.database import KnowledgeDatabase
from loft.neural.rule_generator import RuleGenerator

logger = logging.getLogger(__name__)


@click.group(name="accumulate")
def cli():
    """Rule accumulation commands."""
    pass


@cli.command()
@click.argument("case_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--database-url",
    default="sqlite:///loft_knowledge.db",
    help="Database URL for knowledge storage",
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.7,
    help="Minimum confidence threshold for rules",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def process_case(
    case_file: Path,
    database_url: str,
    min_confidence: float,
    verbose: bool,
):
    """
    Process a single case to extract and accumulate rules.

    CASE_FILE: Path to JSON case file
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    click.echo(f"Processing case: {case_file}")

    # Initialize components
    db = KnowledgeDatabase(database_url)
    pipeline = RuleAccumulationPipeline(
        knowledge_db=db,
        min_rule_confidence=min_confidence,
    )

    # Load and process case
    with open(case_file) as f:
        case_dict = json.load(f)

    case = CaseData.from_dict(case_dict)
    result = pipeline.process_case(case)

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo(f"Case: {result.case_id}")
    click.echo(f"Rules Added: {result.rules_added}")
    click.echo(f"Rules Skipped: {result.rules_skipped}")
    click.echo(f"Conflicts Found: {len(result.conflicts_found)}")
    click.echo(f"Success Rate: {result.success_rate:.1%}")
    click.echo(f"Processing Time: {result.processing_time_ms:.0f}ms")

    if result.rules_skipped > 0:
        click.echo("\nSkipped Reasons:")
        for reason in result.skipped_reasons:
            click.echo(f"  - {reason}")

    if result.conflicts_found:
        click.echo("\nConflicts:")
        for conflict in result.conflicts_found:
            click.echo(f"  - {conflict}")

    if result.rule_ids:
        click.echo("\nAdded Rule IDs:")
        for rule_id in result.rule_ids:
            click.echo(f"  - {rule_id}")

    click.echo("=" * 60)


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--database-url",
    default="sqlite:///loft_knowledge.db",
    help="Database URL for knowledge storage",
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.7,
    help="Minimum confidence threshold for rules",
)
@click.option(
    "--max-cases",
    type=int,
    default=None,
    help="Maximum number of cases to process",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def process_dataset(
    dataset_path: Path,
    database_url: str,
    min_confidence: float,
    max_cases: Optional[int],
    verbose: bool,
):
    """
    Process all cases in a dataset directory.

    DATASET_PATH: Path to directory containing JSON case files
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    click.echo(f"Processing dataset: {dataset_path}")

    # Initialize components
    db = KnowledgeDatabase(database_url)
    pipeline = RuleAccumulationPipeline(
        knowledge_db=db,
        min_rule_confidence=min_confidence,
    )

    # Process dataset
    report = pipeline.process_dataset(
        dataset_path=dataset_path,
        max_cases=max_cases,
    )

    # Display report
    click.echo("\n" + report.to_string())


@cli.command()
@click.option(
    "--database-url",
    default="sqlite:///loft_knowledge.db",
    help="Database URL for knowledge storage",
)
@click.option(
    "--domain",
    default=None,
    help="Filter by legal domain",
)
def stats(database_url: str, domain: Optional[str]):
    """
    Display accumulation statistics.
    """
    db = KnowledgeDatabase(database_url)
    pipeline = RuleAccumulationPipeline(knowledge_db=db)

    stats_dict = pipeline.get_accumulation_stats()

    click.echo("\nAccumulation Statistics")
    click.echo("=" * 60)
    click.echo(f"Total Rules: {stats_dict['total_rules']}")
    click.echo(f"Active Rules: {stats_dict['active_rules']}")
    click.echo(f"Archived Rules: {stats_dict['archived_rules']}")
    click.echo(f"Average Confidence: {stats_dict['avg_confidence']:.2f}")

    if stats_dict["domains"]:
        click.echo("\nBy Domain:")
        for domain_name, count in sorted(stats_dict["domains"].items()):
            if domain is None or domain_name == domain:
                click.echo(f"  {domain_name}: {count} rules")

    click.echo("=" * 60)


@cli.command()
@click.argument("case_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--database-url",
    default="sqlite:///loft_knowledge.db",
    help="Database URL for knowledge storage",
)
@click.option(
    "--domain",
    default=None,
    help="Legal domain to check",
)
def check_conflicts(
    case_file: Path,
    database_url: str,
    domain: Optional[str],
):
    """
    Check a case for conflicts without adding rules.

    CASE_FILE: Path to JSON case file
    """
    click.echo(f"Checking conflicts for: {case_file}")

    # Initialize components
    db = KnowledgeDatabase(database_url)
    rule_generator = RuleGenerator()
    conflict_detector = ConflictDetector(knowledge_db=db)

    # Load case
    with open(case_file) as f:
        case_dict = json.load(f)

    case = CaseData.from_dict(case_dict)
    domain = domain or case.domain or "general"

    # Extract rule candidates
    click.echo("\nExtracting rules from case...")
    generated_rules = rule_generator.generate_rules_from_principle(
        principle=case.rationale,
        domain=domain,
    )

    click.echo(f"Found {len(generated_rules)} candidate rules")

    # Check each for conflicts
    all_conflicts = []

    for i, gen_rule in enumerate(generated_rules, 1):
        click.echo(f"\nCandidate {i}:")
        click.echo(f"  Rule: {gen_rule.asp_rule}")
        click.echo(f"  Confidence: {gen_rule.confidence:.2f}")

        # Convert to RuleCandidate
        from loft.accumulation.schemas import RuleCandidate

        candidate = RuleCandidate(
            asp_rule=gen_rule.asp_rule,
            domain=domain,
            confidence=gen_rule.confidence,
            reasoning=gen_rule.reasoning,
            source_case_id=case.case_id,
        )

        # Find conflicts
        conflicts = conflict_detector.find_conflicts(candidate, domain=domain)

        if conflicts:
            click.echo(f"  Conflicts: {len(conflicts)}")
            for conflict in conflicts:
                click.echo(f"    - Type: {conflict.conflict_type}")
                click.echo(f"      Severity: {conflict.severity:.2f}")
                click.echo(f"      Existing: {conflict.existing_rule_id[:8]}...")
                click.echo(f"      Reason: {conflict.explanation}")

                # Show resolution suggestion
                suggestion = conflict_detector.suggest_resolution(conflict)
                click.echo(f"      Suggestion: {suggestion}")

            all_conflicts.extend(conflicts)
        else:
            click.echo("  Conflicts: None")

    # Summary
    click.echo("\n" + "=" * 60)
    click.echo(f"Total Conflicts: {len(all_conflicts)}")
    if all_conflicts:
        conflict_types = {}
        for c in all_conflicts:
            conflict_types[c.conflict_type] = conflict_types.get(c.conflict_type, 0) + 1

        click.echo("\nBy Type:")
        for ctype, count in sorted(conflict_types.items()):
            click.echo(f"  {ctype}: {count}")

    click.echo("=" * 60)


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file for report (default: stdout)",
)
def validate_dataset(dataset_path: Path, output: Optional[Path]):
    """
    Validate that dataset cases are properly formatted.

    DATASET_PATH: Path to directory containing JSON case files
    """
    click.echo(f"Validating dataset: {dataset_path}")

    case_files = sorted(dataset_path.glob("*.json"))
    click.echo(f"Found {len(case_files)} case files\n")

    errors = []
    valid_count = 0

    for case_file in case_files:
        try:
            with open(case_file) as f:
                case_dict = json.load(f)

            # Validate required fields
            case = CaseData.from_dict(case_dict)

            # Check for empty fields
            issues = []
            if not case.case_id:
                issues.append("Missing case_id")
            if not case.facts:
                issues.append("Empty facts")
            if not case.asp_facts:
                issues.append("Empty asp_facts")
            if not case.rationale:
                issues.append("Empty rationale")

            if issues:
                errors.append((case_file.name, issues))
                click.echo(f"❌ {case_file.name}: {', '.join(issues)}")
            else:
                valid_count += 1
                click.echo(f"✓ {case_file.name}")

        except Exception as e:
            errors.append((case_file.name, [str(e)]))
            click.echo(f"❌ {case_file.name}: {e}")

    # Summary
    click.echo("\n" + "=" * 60)
    click.echo(f"Valid Cases: {valid_count}/{len(case_files)}")
    click.echo(f"Invalid Cases: {len(errors)}")
    click.echo("=" * 60)

    # Save report if requested
    if output and errors:
        report = {
            "dataset": str(dataset_path),
            "total_cases": len(case_files),
            "valid_cases": valid_count,
            "invalid_cases": len(errors),
            "errors": [{"file": fname, "issues": issues} for fname, issues in errors],
        }

        with open(output, "w") as f:
            json.dump(report, f, indent=2)

        click.echo(f"\nValidation report saved to: {output}")


if __name__ == "__main__":
    cli()
