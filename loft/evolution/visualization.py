"""
Visualization helpers for rule evolution.

Provides text-based visualization for rule history, genealogy trees,
performance charts, and A/B test dashboards.
"""

from typing import Dict, List, Optional

from .tracking import (
    RuleMetadata,
    ABTestResult,
    StratificationLayer,
)


def format_rule_history(
    rules: List[RuleMetadata],
    show_validation: bool = True,
    show_dialectical: bool = True,
) -> str:
    """
    Format rule version history as text.

    Args:
        rules: List of rule versions (ordered by version)
        show_validation: Include validation results
        show_dialectical: Include dialectical history

    Returns:
        Formatted string showing rule evolution
    """
    if not rules:
        return "No rule history available."

    lines = []
    root = rules[0]

    # Header
    lines.append(f"Rule Evolution: {root.rule_id[:20]}...")
    lines.append("")

    for i, rule in enumerate(rules):
        # Version line
        status_str = f"[{rule.status.value.upper()}]"
        layer_str = rule.current_layer.value.upper()
        lines.append(
            f"v{rule.version} ({rule.created_at.strftime('%Y-%m-%d %H:%M')}) "
            f"- {layer_str} {status_str}"
        )

        # Rule description
        lines.append(
            f'│   "{rule.natural_language[:60]}..."'
            if len(rule.natural_language) > 60
            else f'│   "{rule.natural_language}"'
        )

        # Accuracy
        if rule.validation_results and show_validation:
            latest = rule.validation_results[-1]
            if len(rule.validation_results) > 1:
                prev = rule.validation_results[-2]
                diff = latest.accuracy - prev.accuracy
                diff_str = f" ({diff:+.0%})" if diff != 0 else ""
            else:
                diff_str = ""
            lines.append(f"│   Accuracy: {latest.accuracy:.0%}{diff_str}")

        # Superseded info
        if rule.superseded_by:
            lines.append(f"│   Superseded by: {rule.superseded_by[:16]}...")

        # Dialectical history
        if show_dialectical and rule.dialectical.cycles_completed > 0:
            lines.append(f"│   Dialectical cycles: {rule.dialectical.cycles_completed}")
            lines.append("│")
            lines.append("└─Dialectical History:")
            if rule.dialectical.thesis_rule:
                lines.append(f"   [Thesis] {rule.dialectical.thesis_rule[:50]}...")
            for critique in rule.dialectical.antithesis_critiques[:2]:
                lines.append(f"   [Antithesis] {critique[:50]}...")
            if rule.dialectical.synthesis_rule:
                lines.append(
                    f"   [Synthesis] {rule.dialectical.synthesis_rule[:50]}..."
                )

        lines.append("│")

        # Add connector to next version
        if i < len(rules) - 1:
            lines.append("├─" + "─" * 40)
        else:
            lines.append("└─" + "─" * 40)

    return "\n".join(lines)


def format_genealogy_tree(
    root_rule: RuleMetadata,
    all_rules: Dict[str, RuleMetadata],
    max_depth: int = 5,
) -> str:
    """
    Format rule dependency graph as a tree.

    Args:
        root_rule: Root rule to start from
        all_rules: Dictionary of all rules by ID
        max_depth: Maximum tree depth

    Returns:
        Formatted tree string
    """
    lines = []

    def format_node(rule_id: str, prefix: str, is_last: bool, depth: int) -> None:
        if depth > max_depth:
            return

        rule = all_rules.get(rule_id)
        if not rule:
            return

        # Node connector
        connector = "└── " if is_last else "├── "
        status = rule.status.value[:3].upper()

        lines.append(
            f"{prefix}{connector}{rule_id[:12]}... [{status}] "
            f"v{rule.version} ({rule.current_accuracy:.0%})"
        )

        # Child prefix for next level
        child_prefix = prefix + ("    " if is_last else "│   ")

        # Add downstream rules
        downstream = rule.downstream_rules
        for i, child_id in enumerate(downstream):
            is_child_last = i == len(downstream) - 1
            format_node(child_id, child_prefix, is_child_last, depth + 1)

    # Root node
    lines.append(
        f"{root_rule.rule_id[:12]}... [{root_rule.status.value[:3].upper()}] v{root_rule.version}"
    )

    # Add children
    for i, child_id in enumerate(root_rule.downstream_rules):
        is_last = i == len(root_rule.downstream_rules) - 1
        format_node(child_id, "", is_last, 1)

    return "\n".join(lines)


def format_performance_chart(
    rule: RuleMetadata,
    width: int = 50,
    height: int = 10,
) -> str:
    """
    Format accuracy history as ASCII chart.

    Args:
        rule: Rule with accuracy history
        width: Chart width in characters
        height: Chart height in lines

    Returns:
        ASCII chart string
    """
    if not rule.accuracy_history:
        return "No accuracy history available."

    lines = []
    accuracies = [acc for _, acc in rule.accuracy_history]

    # Chart header
    lines.append(f"Accuracy Over Time: {rule.rule_id[:20]}...")
    lines.append("")

    # Determine scale
    min_acc = max(0, min(accuracies) - 0.1)
    max_acc = min(1.0, max(accuracies) + 0.1)
    acc_range = max_acc - min_acc

    # Create chart
    chart_lines = []
    for row in range(height):
        row_acc = max_acc - (row / height) * acc_range
        label = f"{row_acc:.0%}│" if row % 2 == 0 else "   │"

        row_chars = []
        for col in range(len(accuracies)):
            # Map accuracy to row
            if accuracies[col] >= row_acc - (acc_range / height / 2):
                if accuracies[col] < row_acc + (acc_range / height / 2):
                    row_chars.append("●")
                else:
                    row_chars.append(" ")
            else:
                row_chars.append(" ")

        # Pad or truncate to width
        row_str = "".join(row_chars)
        if len(row_str) < width:
            row_str = row_str + " " * (width - len(row_str))
        else:
            row_str = row_str[:width]

        chart_lines.append(f"{label}{row_str}")

    lines.extend(chart_lines)

    # X-axis
    lines.append("   └" + "─" * width)

    # Version labels (if we can infer them)
    if len(accuracies) <= 10:
        labels = " " * 4
        for i in range(len(accuracies)):
            labels += f"v{i + 1}".ljust(width // len(accuracies))
        lines.append(labels[: width + 4])

    return "\n".join(lines)


def format_ab_test_dashboard(
    tests: List[ABTestResult],
    rules: Optional[Dict[str, RuleMetadata]] = None,
) -> str:
    """
    Format A/B test status as dashboard.

    Args:
        tests: List of A/B tests to display
        rules: Optional dictionary of rules for additional info

    Returns:
        Formatted dashboard string
    """
    if not tests:
        return "No A/B tests to display."

    lines = []
    lines.append("Active A/B Tests")
    lines.append("=" * 50)
    lines.append("")

    for test in tests:
        # Test header
        lines.append(f"Test #{test.test_id}")

        # Variant A
        a_label = (
            f"v{rules[test.variant_a_id].version}"
            if rules and test.variant_a_id in rules
            else test.variant_a_id[:12]
        )
        lines.append(
            f"├─ Variant A ({a_label}): "
            f"{test.variant_a_accuracy:.0%} accuracy ({test.cases_evaluated} cases)"
        )

        # Variant B
        b_label = (
            f"v{rules[test.variant_b_id].version}"
            if rules and test.variant_b_id in rules
            else test.variant_b_id[:12]
        )
        lines.append(
            f"├─ Variant B ({b_label}): "
            f"{test.variant_b_accuracy:.0%} accuracy ({test.cases_evaluated} cases)"
        )

        # Statistical significance
        if test.p_value is not None:
            if test.p_value < 0.001:
                sig_str = "highly significant"
            elif test.p_value < 0.05:
                sig_str = "significant"
            else:
                sig_str = "not significant"
            lines.append(
                f"├─ Statistical significance: p={test.p_value:.3f} ({sig_str})"
            )

        # Recommendation
        if test.winner:
            winner_label = a_label if test.winner == "a" else b_label
            lines.append(f"└─ Recommendation: PROMOTE {winner_label}")
        elif test.completed_at:
            lines.append("└─ Recommendation: RETAIN current (no improvement)")
        else:
            lines.append("└─ Status: IN PROGRESS")

        lines.append("")

    return "\n".join(lines)


def format_stratification_timeline(
    rule: RuleMetadata,
    width: int = 40,
) -> str:
    """
    Format stratification layer changes as timeline.

    Args:
        rule: Rule with layer history
        width: Timeline width in characters

    Returns:
        Formatted timeline string
    """
    if not rule.layer_history:
        return "No stratification history available."

    lines = []
    lines.append(f"Stratification History: {rule.rule_id[:20]}...")
    lines.append("")

    # Get unique layers from history
    layers = [
        StratificationLayer.OPERATIONAL,
        StratificationLayer.TACTICAL,
        StratificationLayer.STRATEGIC,
        StratificationLayer.CONSTITUTIONAL,
    ]

    # Calculate timeline
    if len(rule.layer_history) > 1:
        start_time = rule.layer_history[0][0]
        end_time = rule.layer_history[-1][0]
        total_duration = (end_time - start_time).total_seconds() or 1

        for layer in layers:
            layer_name = layer.value.upper().ljust(14)

            # Build timeline bar
            bar = ""
            current_pos = 0

            for i, (timestamp, hist_layer) in enumerate(rule.layer_history):
                if hist_layer == layer:
                    # Calculate position
                    pos = int(
                        (timestamp - start_time).total_seconds()
                        / total_duration
                        * width
                    )

                    # Add empty space before
                    bar += "░" * (pos - current_pos)

                    # Calculate end position
                    if i + 1 < len(rule.layer_history):
                        next_time = rule.layer_history[i + 1][0]
                        end_pos = int(
                            (next_time - start_time).total_seconds()
                            / total_duration
                            * width
                        )
                    else:
                        end_pos = width

                    # Add filled section
                    bar += "█" * (end_pos - pos)
                    current_pos = end_pos

            # Pad to full width
            bar += "░" * (width - len(bar))
            lines.append(f"{layer_name}{bar}")
    else:
        # Only one entry
        for layer in layers:
            layer_name = layer.value.upper().ljust(14)
            if layer == rule.current_layer:
                lines.append(f"{layer_name}{'█' * width}")
            else:
                lines.append(f"{layer_name}{'░' * width}")

    return "\n".join(lines)


def format_rule_diff(
    rule_a: RuleMetadata,
    rule_b: RuleMetadata,
) -> str:
    """
    Format comparison between two rule versions.

    Args:
        rule_a: First rule version
        rule_b: Second rule version

    Returns:
        Formatted diff string
    """
    lines = []

    lines.append(f"Comparing: v{rule_a.version} vs v{rule_b.version}")
    lines.append("")

    # Version A
    lines.append(f"v{rule_a.version} ({rule_a.current_layer.value.upper()}):")
    for line in rule_a.rule_text.split("\n"):
        if line.strip():
            lines.append(f"  {line}")

    lines.append("")

    # Version B with diff markers
    lines.append(f"v{rule_b.version} ({rule_b.current_layer.value.upper()}):")
    a_lines = set(rule_a.rule_text.strip().split("\n"))
    for line in rule_b.rule_text.split("\n"):
        if line.strip():
            if line not in a_lines:
                lines.append(f"+ {line}")
            else:
                lines.append(f"  {line}")

    # Show removed lines
    b_lines = set(rule_b.rule_text.strip().split("\n"))
    for line in rule_a.rule_text.split("\n"):
        if line.strip() and line not in b_lines:
            lines.append(f"- {line}")

    lines.append("")

    # Impact analysis
    lines.append("Impact Analysis:")

    # Accuracy comparison
    if rule_a.validation_results and rule_b.validation_results:
        acc_a = rule_a.validation_results[-1].accuracy
        acc_b = rule_b.validation_results[-1].accuracy
        diff = acc_b - acc_a
        lines.append(
            f"  - Accuracy: {acc_a:.0%} (v{rule_a.version}) vs "
            f"{acc_b:.0%} (v{rule_b.version}) [{diff:+.0%}]"
        )

        if diff > 0:
            lines.append(f"  - Recommendation: ACCEPT v{rule_b.version} (improvement)")
        elif diff < 0:
            lines.append(f"  - Recommendation: KEEP v{rule_a.version} (regression)")
        else:
            lines.append("  - Recommendation: NEUTRAL (no accuracy change)")

    return "\n".join(lines)
