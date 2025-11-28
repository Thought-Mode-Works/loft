"""
Report generation for casework exploration results.

Generates JSON and simple text reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class ReportGenerator:
    """Generate reports from casework exploration results."""

    def __init__(self, report_data: Dict[str, Any]):
        """Initialize with report data."""
        self.data = report_data

    def generate_json(self, output_path: Path) -> None:
        """Generate JSON report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def generate_text(self, output_path: Path) -> None:
        """Generate human-readable text report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("LOFT Casework Exploration Report")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Configuration
        lines.append("Configuration:")
        lines.append("-" * 80)
        config = self.data.get("config", {})
        for key, value in config.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # Dataset Statistics
        lines.append("Dataset Statistics:")
        lines.append("-" * 80)
        dataset_stats = self.data.get("dataset_stats", {})
        lines.append(f"  Total scenarios: {dataset_stats.get('total_scenarios', 0)}")

        if "by_difficulty" in dataset_stats:
            lines.append("  By difficulty:")
            for diff, count in dataset_stats["by_difficulty"].items():
                lines.append(f"    {diff}: {count}")

        if "by_ground_truth" in dataset_stats:
            lines.append("  By ground truth:")
            for truth, count in dataset_stats["by_ground_truth"].items():
                lines.append(f"    {truth}: {count}")
        lines.append("")

        # Metrics Summary
        lines.append("Results Summary:")
        lines.append("-" * 80)
        metrics = self.data.get("metrics", {})
        summary = metrics.get("summary", {})

        lines.append(f"  Total cases: {summary.get('total_cases', 0)}")
        lines.append(f"  Correct: {summary.get('cases_correct', 0)}")
        lines.append(f"  Final accuracy: {summary.get('final_accuracy', 0):.1%}")
        lines.append("")
        lines.append(f"  Gaps identified: {summary.get('total_gaps_identified', 0)}")
        lines.append(f"  Rules generated: {summary.get('total_rules_generated', 0)}")
        lines.append(f"  Rules accepted: {summary.get('total_rules_accepted', 0)}")
        lines.append(f"  Acceptance rate: {summary.get('acceptance_rate', 0):.1%}")
        lines.append("")
        lines.append(f"  Total time: {summary.get('total_time_seconds', 0):.2f}s")
        lines.append(f"  Avg time per case: {summary.get('avg_time_per_case', 0):.2f}s")
        lines.append("")

        # Learning Curve
        lines.append("Learning Curve:")
        lines.append("-" * 80)
        learning_curve = metrics.get("learning_curve", [])
        for point in learning_curve:
            lines.append(f"  Case {point['case_number']}: {point['accuracy']:.1%}")
        lines.append("")

        # Knowledge Base
        lines.append("Knowledge Base:")
        lines.append("-" * 80)
        kb = self.data.get("knowledge_base", {})
        lines.append(f"  Total rules learned: {kb.get('total_rules', 0)}")
        lines.append("")

        # Write report
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def generate_html(self, output_path: Path) -> None:
        """Generate HTML report with charts (simplified)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = self.data.get("metrics", {})
        summary = metrics.get("summary", {})
        learning_curve = metrics.get("learning_curve", [])

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>LOFT Casework Exploration Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #666; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
    </style>
</head>
<body>
    <h1>LOFT Casework Exploration Report</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Cases</td><td>{summary.get("total_cases", 0)}</td></tr>
        <tr><td>Cases Correct</td><td>{summary.get("cases_correct", 0)}</td></tr>
        <tr><td>Final Accuracy</td><td class="metric">{summary.get("final_accuracy", 0):.1%}</td></tr>
        <tr><td>Gaps Identified</td><td>{summary.get("total_gaps_identified", 0)}</td></tr>
        <tr><td>Rules Generated</td><td>{summary.get("total_rules_generated", 0)}</td></tr>
        <tr><td>Rules Accepted</td><td>{summary.get("total_rules_accepted", 0)}</td></tr>
        <tr><td>Acceptance Rate</td><td>{summary.get("acceptance_rate", 0):.1%}</td></tr>
        <tr><td>Total Time</td><td>{summary.get("total_time_seconds", 0):.2f}s</td></tr>
    </table>

    <h2>Learning Curve</h2>
    <table>
        <tr><th>Case Number</th><th>Accuracy</th></tr>
"""

        for point in learning_curve:
            html += f"        <tr><td>{point['case_number']}</td><td>{point['accuracy']:.1%}</td></tr>\n"

        html += """    </table>
</body>
</html>"""

        with open(output_path, "w") as f:
            f.write(html)
