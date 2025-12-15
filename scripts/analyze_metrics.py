#!/usr/bin/env python3
"""
Metrics analysis script for batch processing.

Analyzes batch metrics and generates reports on performance,
trends, and anomalies.

Usage:
    python scripts/analyze_metrics.py metrics.json
    python scripts/analyze_metrics.py metrics.json --output report.json
    python scripts/analyze_metrics.py metrics.json --format text
    python scripts/analyze_metrics.py --compare batch1/metrics.json batch2/metrics.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loft.metrics.analyzer import ScaleReport, load_and_analyze_metrics


def format_report_text(report: ScaleReport) -> str:
    """Format report as readable text.

    Args:
        report: Scale report

    Returns:
        Formatted text
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"SCALE PERFORMANCE REPORT: {report.batch_id}")
    lines.append("=" * 70)
    lines.append(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Volume summary
    lines.append("VOLUME SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total Cases Processed: {report.total_cases}")
    lines.append(f"  Total Rules Generated: {report.total_rules}")
    lines.append(f"  Total Runtime: {report.total_runtime_seconds:.1f} seconds")
    lines.append("")

    # Performance summary
    lines.append("PERFORMANCE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Average Latency: {report.avg_case_latency_ms:.2f} ms")
    lines.append(f"  P50 Latency: {report.p50_latency_ms:.2f} ms")
    lines.append(f"  P95 Latency: {report.p95_latency_ms:.2f} ms")
    lines.append(f"  P99 Latency: {report.p99_latency_ms:.2f} ms")
    lines.append(f"  Peak Memory: {report.peak_memory_mb:.1f} MB")
    lines.append("")

    # Quality summary
    lines.append("QUALITY SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Initial Accuracy: {report.initial_accuracy * 100:.1f}%")
    lines.append(f"  Final Accuracy: {report.final_accuracy * 100:.1f}%")
    change_symbol = "+" if report.accuracy_change >= 0 else ""
    lines.append(
        f"  Accuracy Change: {change_symbol}{report.accuracy_change * 100:.1f}%"
    )
    lines.append(f"  Consistency Score: {report.consistency_score:.2f}")
    lines.append("")

    # Trend analysis
    lines.append("TREND ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"  Overall Trend: {report.trends.trend_direction.upper()}")
    lines.append(f"  Accuracy Trend: {report.trends.accuracy_trend}")
    lines.append(f"  Latency Trend: {report.trends.latency_trend}")
    lines.append(
        f"  Rules Growth Rate: {report.trends.rules_growth_rate:.2f} rules/min"
    )
    lines.append(
        f"  Accuracy vs Rules Correlation: {report.trends.accuracy_vs_rules_correlation:.3f}"
    )
    lines.append(
        f"  Latency vs Rules Correlation: {report.trends.latency_vs_rules_correlation:.3f}"
    )
    lines.append("")

    # Anomalies
    lines.append("ANOMALY DETECTION")
    lines.append("-" * 40)
    ar = report.anomaly_report
    lines.append(f"  Total Anomalies: {ar.total_anomalies}")
    lines.append(f"    High Severity: {ar.high_severity_count}")
    lines.append(f"    Medium Severity: {ar.medium_severity_count}")
    lines.append(f"    Low Severity: {ar.low_severity_count}")

    if ar.anomalies:
        lines.append("")
        lines.append("  Anomalies Detected:")
        for anomaly in ar.anomalies[:5]:  # Show top 5
            lines.append(f"    [{anomaly.severity.upper()}] {anomaly.description}")
    lines.append("")

    # Bottlenecks
    if report.bottlenecks:
        lines.append("IDENTIFIED BOTTLENECKS")
        lines.append("-" * 40)
        for bottleneck in report.bottlenecks[:5]:
            lines.append(
                f"  - {bottleneck['operation']}: "
                f"{bottleneck['avg_time_ms']:.0f}ms avg "
                f"({bottleneck['count']} calls)"
            )
        lines.append("")

    # Recommendations
    if report.recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def compare_reports(reports: List[ScaleReport]) -> str:
    """Compare multiple batch reports.

    Args:
        reports: List of reports to compare

    Returns:
        Comparison text
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BATCH COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Header row
    headers = ["Metric"] + [r.batch_id for r in reports]
    col_width = max(20, max(len(h) for h in headers) + 2)

    header_row = "".join(h.ljust(col_width) for h in headers)
    lines.append(header_row)
    lines.append("-" * len(header_row))

    # Metrics rows
    metrics = [
        ("Cases Processed", [r.total_cases for r in reports]),
        ("Rules Generated", [r.total_rules for r in reports]),
        ("Runtime (s)", [f"{r.total_runtime_seconds:.1f}" for r in reports]),
        ("Avg Latency (ms)", [f"{r.avg_case_latency_ms:.1f}" for r in reports]),
        ("P95 Latency (ms)", [f"{r.p95_latency_ms:.1f}" for r in reports]),
        ("Peak Memory (MB)", [f"{r.peak_memory_mb:.1f}" for r in reports]),
        ("Initial Accuracy", [f"{r.initial_accuracy * 100:.1f}%" for r in reports]),
        ("Final Accuracy", [f"{r.final_accuracy * 100:.1f}%" for r in reports]),
        ("Accuracy Change", [f"{r.accuracy_change * 100:+.1f}%" for r in reports]),
        ("Trend Direction", [r.trends.trend_direction for r in reports]),
        ("Anomalies", [r.anomaly_report.total_anomalies for r in reports]),
    ]

    for metric_name, values in metrics:
        row = metric_name.ljust(col_width)
        for v in values:
            row += str(v).ljust(col_width)
        lines.append(row)

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze batch processing metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "metrics_file",
        nargs="*",
        help="Path to metrics JSON file(s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple metrics files",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=2.0,
        help="Standard deviations for anomaly detection (default: 2.0)",
    )

    args = parser.parse_args()

    if not args.metrics_file:
        parser.error("At least one metrics file is required")

    # Load and analyze metrics
    reports: List[ScaleReport] = []
    for metrics_path in args.metrics_file:
        path = Path(metrics_path)
        if not path.exists():
            print(f"Error: File not found: {metrics_path}", file=sys.stderr)
            sys.exit(1)

        try:
            report = load_and_analyze_metrics(path)
            reports.append(report)
        except Exception as e:
            print(f"Error analyzing {metrics_path}: {e}", file=sys.stderr)
            sys.exit(1)

    # Generate output
    if args.compare and len(reports) > 1:
        output = compare_reports(reports)
    elif args.format == "json":
        if len(reports) == 1:
            output = json.dumps(reports[0].to_dict(), indent=2)
        else:
            output = json.dumps([r.to_dict() for r in reports], indent=2)
    else:
        if len(reports) == 1:
            output = format_report_text(reports[0])
        else:
            output = "\n\n".join(format_report_text(r) for r in reports)

    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Report saved to: {output_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
