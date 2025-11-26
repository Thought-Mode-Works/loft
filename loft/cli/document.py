"""
CLI commands for living document generation.

Provides commands to generate and manage living documentation
of the self-modifying system's ASP core.
"""

import click
from pathlib import Path

from loft.core.self_modifying_system import SelfModifyingSystem
from loft.documentation.living_document import LivingDocumentGenerator


@click.group()
def document():
    """Living document generation commands."""
    pass


@document.command()
@click.option(
    "--output",
    "-o",
    default="./LIVING_DOCUMENT.md",
    help="Output path for generated document",
)
@click.option(
    "--persistence-dir",
    default="./asp_rules",
    help="Directory for loading persisted ASP rules",
)
@click.option(
    "--enable-llm",
    is_flag=True,
    help="Enable LLM integration (for system initialization)",
)
@click.option(
    "--enable-dialectical",
    is_flag=True,
    help="Enable dialectical validation (for system initialization)",
)
@click.option(
    "--no-metadata",
    is_flag=True,
    help="Exclude generation metadata from document",
)
def generate(output, persistence_dir, enable_llm, enable_dialectical, no_metadata):
    """Generate living document for current ASP core state."""
    click.echo("=" * 80)
    click.echo("GENERATING LIVING DOCUMENT")
    click.echo("=" * 80)

    # Initialize system to access ASP core
    click.echo("\nInitializing system...")
    click.echo(f"  Persistence directory: {persistence_dir}")

    system = SelfModifyingSystem(
        enable_llm=enable_llm,
        enable_dialectical=enable_dialectical,
        persistence_dir=persistence_dir,
    )

    # Create generator
    generator = LivingDocumentGenerator(system=system)

    # Generate document
    click.echo("\nGenerating document...")
    click.echo(f"  Output: {output}")
    click.echo(f"  Include metadata: {not no_metadata}")

    document_content = generator.generate(
        output_path=output,
        include_metadata=not no_metadata,
    )

    # Display summary
    output_path = Path(output)
    file_size = output_path.stat().st_size if output_path.exists() else 0

    click.echo("\n" + "=" * 80)
    click.echo("DOCUMENT GENERATED")
    click.echo("=" * 80)
    click.echo(f"\nLocation: {output}")
    click.echo(f"Size: {file_size:,} bytes")
    click.echo(f"Lines: {len(document_content.splitlines())}")

    click.echo("\nDocument sections:")
    # Count sections by looking for ## headers
    sections = [line for line in document_content.split("\n") if line.startswith("## ")]
    for section in sections:
        click.echo(f"  - {section[3:]}")  # Remove "## " prefix

    click.echo("\n✓ Living document successfully generated!")


@document.command()
@click.option(
    "--output",
    "-o",
    default="./LIVING_DOCUMENT.md",
    help="Output path for generated document",
)
@click.option(
    "--persistence-dir",
    default="./asp_rules",
    help="Directory for loading persisted ASP rules",
)
def preview(output, persistence_dir):
    """Preview living document content without saving."""
    click.echo("=" * 80)
    click.echo("PREVIEWING LIVING DOCUMENT")
    click.echo("=" * 80)

    # Initialize system
    click.echo("\nInitializing system...")
    system = SelfModifyingSystem(persistence_dir=persistence_dir)

    # Create generator
    generator = LivingDocumentGenerator(system=system)

    # Generate document (still saves to temp location)
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
        tmp_path = tmp.name

    document_content = generator.generate(output_path=tmp_path, include_metadata=False)

    # Display preview
    click.echo("\n" + "=" * 80)
    click.echo("DOCUMENT PREVIEW")
    click.echo("=" * 80 + "\n")

    # Show first 50 lines
    lines = document_content.split("\n")
    preview_lines = lines[:50]

    for line in preview_lines:
        click.echo(line)

    if len(lines) > 50:
        click.echo(f"\n... ({len(lines) - 50} more lines)")

    # Clean up temp file
    Path(tmp_path).unlink(missing_ok=True)


@document.command()
@click.option(
    "--document",
    "-d",
    default="./LIVING_DOCUMENT.md",
    help="Path to living document",
)
def stats(document):
    """Show statistics about the living document."""
    click.echo("=" * 80)
    click.echo("LIVING DOCUMENT STATISTICS")
    click.echo("=" * 80)

    doc_path = Path(document)

    if not doc_path.exists():
        click.echo(f"\n✗ Document not found: {document}")
        click.echo("  Run 'loft document generate' to create it.")
        return

    content = doc_path.read_text()
    lines = content.split("\n")

    # Count various elements
    total_lines = len(lines)
    headers = [line for line in lines if line.startswith("#")]
    code_blocks = len([line for line in lines if line.startswith("```")])
    tables = len([line for line in lines if line.startswith("|")])
    links = sum(line.count("](") for line in lines)

    # File stats
    file_size = doc_path.stat().st_size
    last_modified = doc_path.stat().st_mtime

    import datetime

    last_modified_str = datetime.datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M:%S")

    click.echo(f"\nDocument: {document}")
    click.echo("\nFile Statistics:")
    click.echo(f"  Size: {file_size:,} bytes")
    click.echo(f"  Lines: {total_lines:,}")
    click.echo(f"  Last modified: {last_modified_str}")

    click.echo("\nContent Statistics:")
    click.echo(f"  Headers: {len(headers)}")
    click.echo(f"  Code blocks: {code_blocks // 2}")  # Divide by 2 (opening and closing)
    click.echo(f"  Tables: {tables}")
    click.echo(f"  Links: {links}")

    click.echo("\nDocument Structure:")
    for header in headers[:10]:  # Show first 10 headers
        level = len(header) - len(header.lstrip("#"))
        indent = "  " * (level - 1)
        text = header.lstrip("#").strip()
        click.echo(f"{indent}- {text}")

    if len(headers) > 10:
        click.echo(f"  ... ({len(headers) - 10} more headers)")


if __name__ == "__main__":
    document()
