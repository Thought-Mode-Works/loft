#!/usr/bin/env python3
"""
Fidelity Trends Script

Analyzes fidelity trends from experiment data and generates trend reports.

Usage:
    python scripts/fidelity_trends.py data/experiments/ \\
        --output reports/fidelity_trends.html
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loft.translation.fidelity_tracker import (
    FidelityTracker,
    FidelitySnapshot,
)


def load_fidelity_snapshots(data_dir: Path) -> List[FidelitySnapshot]:
    """
    Load fidelity snapshots from experiment data.

    Args:
        data_dir: Directory containing experiment data

    Returns:
        List of FidelitySnapshot objects
    """
    snapshots = []

    if not data_dir.exists():
        print(f"Warning: Data directory {data_dir} does not exist")
        return snapshots

    # Look for fidelity history files
    for json_file in data_dir.rglob("fidelity_history.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

                if "snapshots" in data:
                    for snapshot_data in data["snapshots"]:
                        snapshot = FidelitySnapshot.from_dict(snapshot_data)
                        snapshots.append(snapshot)

        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    # Sort by timestamp
    snapshots.sort(key=lambda s: s.timestamp)

    return snapshots


def generate_html_report(tracker: FidelityTracker, output_path: Path) -> None:
    """
    Generate HTML trend report with visualizations.

    Args:
        tracker: FidelityTracker with history
        output_path: Path to save HTML report
    """
    if not tracker.history:
        print("No fidelity data to visualize")
        return

    # Extract data for charts
    timestamps = [s.timestamp[:19] for s in tracker.history]  # Truncate to YYYY-MM-DDTHH:MM:SS
    avg_fidelities = [s.avg_fidelity for s in tracker.history]
    perfect_rates = [s.perfect_rate for s in tracker.history]
    min_fidelities = [s.min_fidelity for s in tracker.history]

    # Generate HTML with inline charting (using Chart.js CDN)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Translation Fidelity Trends</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .chart-container {{
            margin: 30px 0;
            height: 400px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #4CAF50;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        .trend-indicator {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }}
        .trend-improving {{ background: #4CAF50; color: white; }}
        .trend-stable {{ background: #2196F3; color: white; }}
        .trend-degrading {{ background: #f44336; color: white; }}
        .alert {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            color: #856404;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Translation Fidelity Trends</h1>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Current Avg Fidelity</div>
                <div class="stat-value">{avg_fidelities[-1]:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Perfect Roundtrips</div>
                <div class="stat-value">{perfect_rates[-1]:.1%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Snapshots</div>
                <div class="stat-value">{len(tracker.history)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Trend</div>
                <div class="stat-value">
                    <span class="trend-indicator trend-{tracker.get_trend().lower().replace("_", "-")}">
                        {tracker.get_trend().upper()}
                    </span>
                </div>
            </div>
        </div>
"""

    # Add regression alert if detected
    regression = tracker.detect_regression()
    if regression:
        html += f"""
        <div class="alert">
            <strong>⚠ Regression Detected!</strong><br>
            Baseline: {regression.baseline_fidelity:.2%} → Current: {regression.current_fidelity:.2%}
            (degradation: {regression.degradation:.2%})
        </div>
"""

    html += (
        """
        <div class="chart-container">
            <canvas id="fidelityChart"></canvas>
        </div>

        <div class="chart-container">
            <canvas id="distributionChart"></canvas>
        </div>

        <script>
            // Fidelity Trend Chart
            const fidelityCtx = document.getElementById('fidelityChart').getContext('2d');
            new Chart(fidelityCtx, {
                type: 'line',
                data: {
                    labels: """
        + str(timestamps)
        + """,
                    datasets: [
                        {
                            label: 'Average Fidelity',
                            data: """
        + str(avg_fidelities)
        + """,
                            borderColor: '#4CAF50',
                            backgroundColor: 'rgba(76, 175, 80, 0.1)',
                            tension: 0.4,
                            fill: true
                        },
                        {
                            label: 'Perfect Roundtrips',
                            data: """
        + str(perfect_rates)
        + """,
                            borderColor: '#2196F3',
                            backgroundColor: 'rgba(33, 150, 243, 0.1)',
                            tension: 0.4,
                            fill: true
                        },
                        {
                            label: 'Min Fidelity',
                            data: """
        + str(min_fidelities)
        + """,
                            borderColor: '#f44336',
                            backgroundColor: 'transparent',
                            borderDash: [5, 5],
                            tension: 0.4,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Translation Fidelity Over Time'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            ticks: {
                                callback: function(value) {
                                    return (value * 100).toFixed(0) + '%';
                                }
                            }
                        }
                    }
                }
            });

            // Distribution Chart (last snapshot)
            const distributionCtx = document.getElementById('distributionChart').getContext('2d');
            new Chart(distributionCtx, {
                type: 'bar',
                data: {
                    labels: ['< 0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0', '1.0'],
                    datasets: [{
                        label: 'Number of Translations',
                        data: """
        + str(list(tracker.history[-1].fidelity_distribution.values()))
        + """,
                        backgroundColor: [
                            '#f44336',
                            '#ff9800',
                            '#ffc107',
                            '#8bc34a',
                            '#4caf50'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Current Fidelity Distribution'
                        }
                    }
                }
            });
        </script>
    </div>
</body>
</html>
"""
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze fidelity trends and generate reports")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing experiment data with fidelity history",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/fidelity_trends.html",
        help="Output path for trend report (default: reports/fidelity_trends.html)",
    )
    parser.add_argument(
        "--format",
        choices=["html", "markdown"],
        default="html",
        help="Output format (default: html)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    print(f"Loading fidelity snapshots from {data_dir}...")
    snapshots = load_fidelity_snapshots(data_dir)

    if not snapshots:
        print("No fidelity snapshots found. Exiting.")
        return 1

    print(f"Loaded {len(snapshots)} snapshots")

    # Create tracker and load snapshots
    tracker = FidelityTracker()
    tracker.history = snapshots

    # Analyze trends
    trend = tracker.get_trend()
    regression = tracker.detect_regression()

    print("\n=== Fidelity Trend Analysis ===")
    print(f"Current fidelity: {tracker.current_fidelity:.2%}")
    print(f"Trend: {trend.upper()}")

    if regression:
        print("\n⚠ Regression detected!")
        print(f"  Baseline: {regression.baseline_fidelity:.2%}")
        print(f"  Current: {regression.current_fidelity:.2%}")
        print(f"  Degradation: {regression.degradation:.2%}")

    # Generate report
    if args.format == "html":
        print("\nGenerating HTML report...")
        generate_html_report(tracker, output_path)
        print(f"✓ HTML report generated: {output_path}")
    else:
        print("\nGenerating Markdown report...")
        report = tracker.generate_trend_report()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"✓ Markdown report generated: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
