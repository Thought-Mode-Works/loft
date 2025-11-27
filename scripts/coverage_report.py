#!/usr/bin/env python3
"""
Generate comprehensive test coverage report and identify testing gaps.

Usage:
    # Generate HTML coverage report
    python scripts/coverage_report.py --format html --output coverage/

    # Show summary in terminal
    python scripts/coverage_report.py --summary

    # Identify coverage gaps
    python scripts/coverage_report.py --gaps --threshold 80

    # Generate all reports
    python scripts/coverage_report.py --all
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def run_coverage(output_format: str = "term") -> int:
    """
    Run pytest with coverage.

    Args:
        output_format: Output format (term, html, xml, json)

    Returns:
        Exit code from pytest
    """
    cmd = [
        "pytest",
        "--cov=loft",
        "--cov-report=" + output_format,
        "tests/",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_coverage_html(output_dir: str = "coverage") -> int:
    """Generate HTML coverage report."""
    Path(output_dir).mkdir(exist_ok=True)
    cmd = [
        "pytest",
        "--cov=loft",
        f"--cov-report=html:{output_dir}",
        "tests/",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        index_path = Path(output_dir) / "index.html"
        print(f"\n✓ Coverage report generated: {index_path.absolute()}")
        print(f"  Open in browser: file://{index_path.absolute()}")

    return result.returncode


def run_coverage_json() -> Tuple[int, Dict]:
    """
    Run coverage and return JSON data.

    Returns:
        Tuple of (exit_code, coverage_data)
    """
    cmd = [
        "pytest",
        "--cov=loft",
        "--cov-report=json:coverage.json",
        "--cov-report=term",
        "tests/",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    coverage_data = {}
    if Path("coverage.json").exists():
        with open("coverage.json") as f:
            coverage_data = json.load(f)

    return result.returncode, coverage_data


def analyze_gaps(coverage_data: Dict, threshold: float = 80.0) -> List[Dict]:
    """
    Identify modules with coverage below threshold.

    Args:
        coverage_data: Coverage JSON data
        threshold: Minimum acceptable coverage percentage

    Returns:
        List of modules with gaps
    """
    gaps = []
    files = coverage_data.get("files", {})

    for file_path, file_data in files.items():
        coverage_pct = file_data.get("summary", {}).get("percent_covered", 0)

        if coverage_pct < threshold:
            gaps.append(
                {
                    "file": file_path,
                    "coverage": coverage_pct,
                    "gap": threshold - coverage_pct,
                    "missing_lines": file_data.get("summary", {}).get("missing_lines", 0),
                    "total_statements": file_data.get("summary", {}).get("num_statements", 0),
                }
            )

    # Sort by gap size (largest gaps first)
    gaps.sort(key=lambda x: x["gap"], reverse=True)
    return gaps


def print_summary(coverage_data: Dict):
    """Print coverage summary."""
    totals = coverage_data.get("totals", {})

    print("\n" + "=" * 60)
    print("Coverage Summary")
    print("=" * 60)
    print(f"Total Coverage: {totals.get('percent_covered', 0):.2f}%")
    print(f"Total Statements: {totals.get('num_statements', 0)}")
    print(f"Covered: {totals.get('covered_lines', 0)}")
    print(f"Missing: {totals.get('missing_lines', 0)}")
    print("=" * 60)


def print_gaps(gaps: List[Dict], threshold: float):
    """Print coverage gaps."""
    if not gaps:
        print(f"\n✓ All modules meet {threshold}% coverage threshold!")
        return

    print("\n" + "=" * 60)
    print(f"Coverage Gaps (below {threshold}%)")
    print("=" * 60)

    for gap in gaps:
        print(f"\n📁 {gap['file']}")
        print(f"   Coverage: {gap['coverage']:.2f}%")
        print(f"   Gap: {gap['gap']:.2f}%")
        print(f"   Missing: {gap['missing_lines']} / {gap['total_statements']} statements")

    print("\n" + "=" * 60)
    print(f"Total modules below threshold: {len(gaps)}")
    print("=" * 60)


def generate_markdown_report(coverage_data: Dict, gaps: List[Dict], output_path: str):
    """Generate markdown coverage report."""
    totals = coverage_data.get("totals", {})
    total_coverage = totals.get("percent_covered", 0)

    content = f"""# Test Coverage Report

**Generated**: {coverage_data.get("meta", {}).get("timestamp", "N/A")}

## Overall Coverage

- **Total Coverage**: {total_coverage:.2f}%
- **Total Statements**: {totals.get("num_statements", 0)}
- **Covered Lines**: {totals.get("covered_lines", 0)}
- **Missing Lines**: {totals.get("missing_lines", 0)}

## Coverage by Module

"""

    # Sort files by coverage (lowest first)
    files = coverage_data.get("files", {})
    sorted_files = sorted(
        files.items(),
        key=lambda x: x[1].get("summary", {}).get("percent_covered", 0),
    )

    for file_path, file_data in sorted_files:
        summary = file_data.get("summary", {})
        coverage_pct = summary.get("percent_covered", 0)
        status = "✓" if coverage_pct >= 80 else "⚠️" if coverage_pct >= 60 else "❌"

        content += f"\n### {status} {file_path}\n"
        content += f"- **Coverage**: {coverage_pct:.2f}%\n"
        content += f"- **Statements**: {summary.get('num_statements', 0)}\n"
        content += f"- **Missing**: {summary.get('missing_lines', 0)}\n"

    if gaps:
        content += "\n## Coverage Gaps (< 80%)\n\n"
        content += f"**Total modules below threshold**: {len(gaps)}\n\n"

        for gap in gaps:
            content += f"\n### {gap['file']}\n"
            content += f"- Coverage: {gap['coverage']:.2f}%\n"
            content += f"- Gap: {gap['gap']:.2f}%\n"
            content += f"- Missing: {gap['missing_lines']} / {gap['total_statements']} statements\n"

    Path(output_path).write_text(content)
    print(f"\n✓ Markdown report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate test coverage reports")
    parser.add_argument(
        "--format",
        choices=["term", "html", "xml", "json"],
        default="term",
        help="Output format",
    )
    parser.add_argument("--output", default="coverage", help="Output directory for HTML reports")
    parser.add_argument("--summary", action="store_true", help="Show coverage summary")
    parser.add_argument("--gaps", action="store_true", help="Identify coverage gaps")
    parser.add_argument(
        "--threshold", type=float, default=80.0, help="Coverage threshold for gap analysis"
    )
    parser.add_argument("--markdown", help="Generate markdown report to specified file")
    parser.add_argument("--all", action="store_true", help="Generate all reports")

    args = parser.parse_args()

    if args.all:
        # Generate all reports
        print("Generating comprehensive coverage reports...")

        # 1. HTML report
        print("\n1. Generating HTML report...")
        run_coverage_html(args.output)

        # 2. JSON + analysis
        print("\n2. Running coverage analysis...")
        exit_code, coverage_data = run_coverage_json()

        if coverage_data:
            print_summary(coverage_data)

            gaps = analyze_gaps(coverage_data, args.threshold)
            print_gaps(gaps, args.threshold)

            # 3. Markdown report
            markdown_path = args.markdown or "docs/TEST_COVERAGE.md"
            generate_markdown_report(coverage_data, gaps, markdown_path)

        return exit_code

    # Individual operations
    if args.format == "html":
        return run_coverage_html(args.output)

    elif args.summary or args.gaps:
        exit_code, coverage_data = run_coverage_json()

        if args.summary and coverage_data:
            print_summary(coverage_data)

        if args.gaps and coverage_data:
            gaps = analyze_gaps(coverage_data, args.threshold)
            print_gaps(gaps, args.threshold)

        if args.markdown and coverage_data:
            gaps = analyze_gaps(coverage_data, args.threshold)
            generate_markdown_report(coverage_data, gaps, args.markdown)

        return exit_code

    else:
        # Default: just run coverage with specified format
        return run_coverage(args.format)


if __name__ == "__main__":
    sys.exit(main())
