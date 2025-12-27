"""
CLI commands for legal question answering.

Provides command-line interface for asking legal questions, batch evaluation,
and interactive QA sessions.

Issue #272: Legal Question Answering Interface
"""

import json
from typing import Optional

import click

from loft.knowledge.database import KnowledgeDatabase
from loft.qa.interface import LegalQAInterface


@click.group()
def cli():
    """Legal Question Answering CLI."""
    pass


@cli.command()
@click.argument("question")
@click.option(
    "--database-url", default="sqlite:///legal_knowledge.db", help="Database URL"
)
@click.option("--domain", default=None, help="Legal domain (contracts, torts, etc.)")
@click.option("--expected", default=None, help="Expected answer for validation")
@click.option("--verbose", is_flag=True, help="Show detailed output")
def ask(
    question: str,
    database_url: str,
    domain: Optional[str],
    expected: Optional[str],
    verbose: bool,
):
    """
    Ask a legal question and get an answer.

    Example:
        python -m loft.qa.cli ask "Is a contract valid without consideration?"
    """
    # Initialize database and QA interface
    db = KnowledgeDatabase(database_url)
    qa = LegalQAInterface(knowledge_db=db)

    # Ask question
    click.echo(f"\nQuestion: {question}\n")

    answer = qa.ask(question, domain=domain, expected_answer=expected)

    # Display answer
    click.echo(answer.to_natural_language())

    # Show ASP query if verbose
    if verbose and answer.asp_query:
        click.echo("\n" + "=" * 50)
        click.echo("ASP Query:")
        click.echo(answer.asp_query.to_asp_program())

    # Show correctness if expected answer provided
    if expected:
        correct = answer.is_correct(expected)
        if correct:
            click.echo(f"\n✓ Correct! (expected: {expected})")
        else:
            click.echo(f"\n✗ Incorrect (expected: {expected}, got: {answer.answer})")

    db.close()


@cli.command()
@click.argument("test_file", type=click.Path(exists=True))
@click.option(
    "--database-url", default="sqlite:///legal_knowledge.db", help="Database URL"
)
@click.option("--domain", default=None, help="Legal domain")
@click.option("--output", default=None, help="Output file for results (JSON)")
def evaluate(
    test_file: str, database_url: str, domain: Optional[str], output: Optional[str]
):
    """
    Evaluate QA system on a batch of test questions.

    Test file should be JSON with format:
    [
        {"question": "...", "answer": "yes/no"},
        {"question": "...", "answer": "yes/no", "domain": "contracts"}
    ]

    Example:
        python -m loft.qa.cli evaluate tests/data/legal_questions.json
    """
    # Load test questions
    with open(test_file) as f:
        test_data = json.load(f)

    # Convert to question/answer tuples
    questions = []
    for item in test_data:
        q = item["question"]
        a = item["answer"]
        questions.append((q, a))

    # Initialize QA interface
    db = KnowledgeDatabase(database_url)
    qa = LegalQAInterface(knowledge_db=db)

    # Evaluate
    click.echo(f"\nEvaluating on {len(questions)} questions...")
    click.echo("=" * 50 + "\n")

    report = qa.batch_eval(questions, domain=domain)

    # Display report
    click.echo(report.to_string())

    # Save detailed results if output specified
    if output:
        results_data = {
            "summary": {
                "total": report.total_questions,
                "correct": report.correct_count,
                "incorrect": report.incorrect_count,
                "unknown": report.unknown_count,
                "accuracy": report.accuracy,
                "avg_confidence": report.avg_confidence,
            },
            "by_domain": report.by_domain,
            "detailed_results": [
                {
                    "question": r.question,
                    "expected": r.expected_answer,
                    "actual": r.actual_answer.answer,
                    "correct": r.correct,
                    "confidence": r.actual_answer.confidence,
                    "explanation": r.actual_answer.explanation,
                }
                for r in report.results
            ],
        }

        with open(output, "w") as f:
            json.dump(results_data, f, indent=2)

        click.echo(f"\nDetailed results saved to: {output}")

    db.close()


@cli.command()
@click.option(
    "--database-url", default="sqlite:///legal_knowledge.db", help="Database URL"
)
@click.option("--domain", default=None, help="Default legal domain")
def interactive(database_url: str, domain: Optional[str]):
    """
    Start an interactive QA session.

    Example:
        python -m loft.qa.cli interactive
    """
    # Initialize QA interface
    db = KnowledgeDatabase(database_url)
    qa = LegalQAInterface(knowledge_db=db)

    click.echo("=" * 60)
    click.echo("Legal Question Answering - Interactive Mode")
    click.echo("=" * 60)
    click.echo("\nType your legal questions. Commands:")
    click.echo("  /quit or /exit - Exit interactive mode")
    click.echo("  /domain <name> - Set domain (contracts, torts, etc.)")
    click.echo("  /stats - Show performance statistics")
    click.echo("")

    current_domain = domain

    while True:
        try:
            # Get question
            question = click.prompt("\n>", type=str, prompt_suffix=" ")

            # Handle commands
            if question.lower() in ["/quit", "/exit", "quit", "exit"]:
                click.echo("Goodbye!")
                break

            if question.lower().startswith("/domain"):
                parts = question.split()
                if len(parts) > 1:
                    current_domain = parts[1]
                    click.echo(f"Domain set to: {current_domain}")
                else:
                    click.echo(f"Current domain: {current_domain or 'none'}")
                continue

            if question.lower() == "/stats":
                stats = qa.get_performance_summary()
                click.echo("\nPerformance Statistics:")
                click.echo(f"  Total rules: {stats.get('total_rules', 0)}")
                click.echo(f"  Active rules: {stats.get('active_rules', 0)}")
                click.echo(f"  Questions asked: {stats.get('total_questions', 0)}")
                if stats.get("avg_confidence"):
                    click.echo(f"  Avg confidence: {stats['avg_confidence']:.1%}")
                continue

            # Answer question
            answer = qa.ask(question, domain=current_domain)

            # Display answer
            click.echo("\n" + "-" * 60)
            click.echo(answer.to_natural_language())
            click.echo("-" * 60)

        except (KeyboardInterrupt, EOFError):
            click.echo("\nGoodbye!")
            break
        except Exception as e:
            click.echo(f"\nError: {e}", err=True)

    db.close()


@cli.command()
@click.option(
    "--database-url", default="sqlite:///legal_knowledge.db", help="Database URL"
)
def stats(database_url: str):
    """
    Show QA system statistics.

    Example:
        python -m loft.qa.cli stats
    """
    db = KnowledgeDatabase(database_url)
    qa = LegalQAInterface(knowledge_db=db)

    stats = qa.get_performance_summary()

    click.echo("\nLegal QA System Statistics")
    click.echo("=" * 50)
    click.echo("\nKnowledge Base:")
    click.echo(f"  Total rules: {stats.get('total_rules', 0)}")
    click.echo(f"  Active rules: {stats.get('active_rules', 0)}")
    click.echo(f"  Domains: {', '.join(stats.get('domains', []))}")

    click.echo("\nQuestions:")
    click.echo(f"  Total asked: {stats.get('total_questions', 0)}")

    if stats.get("avg_confidence"):
        click.echo("\nPerformance:")
        click.echo(f"  Avg confidence: {stats['avg_confidence']:.1%}")

    if stats.get("coverage_by_domain"):
        click.echo("\nCoverage by Domain:")
        for domain, count in stats["coverage_by_domain"].items():
            click.echo(f"  {domain}: {count} rules")

    click.echo("")
    db.close()


if __name__ == "__main__":
    cli()
