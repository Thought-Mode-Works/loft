"""
Interactive CLI for LOFT Playground

A rich, user-friendly CLI for exploring LOFT's capabilities interactively.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from experiments.playground.session import PlaygroundSession


console = Console()


class PlaygroundCLI:
    """Interactive CLI for LOFT Playground."""

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        """Initialize CLI with session."""
        self.session = PlaygroundSession(model=model)
        self.running = True

        console.print(
            Panel.fit(
                "[bold cyan]LOFT Interactive Playground[/bold cyan]\n"
                f"Model: {model}\n"
                "Type 'help' for commands, 'exit' to quit",
                border_style="cyan",
            )
        )

    def run(self) -> None:
        """Run the interactive CLI."""
        while self.running:
            try:
                command = console.input("\n[bold green]loft>[/bold green] ").strip()

                if not command:
                    continue

                self.execute_command(command)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except EOFError:
                break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")

    def execute_command(self, command: str) -> None:
        """Execute a command."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "help":
            self.cmd_help()
        elif cmd == "exit" or cmd == "quit":
            self.cmd_exit()
        elif cmd == "load":
            self.cmd_load(args)
        elif cmd == "translate":
            self.cmd_translate(args)
        elif cmd == "identify-gaps":
            self.cmd_identify_gaps()
        elif cmd == "generate-rule":
            self.cmd_generate_rule(args)
        elif cmd == "validate-rule":
            self.cmd_validate_rule(args)
        elif cmd == "incorporate-rule":
            self.cmd_incorporate_rule(args)
        elif cmd == "predict":
            self.cmd_predict()
        elif cmd == "status":
            self.cmd_status()
        elif cmd == "export":
            self.cmd_export(args)
        elif cmd == "show-rules":
            self.cmd_show_rules()
        elif cmd == "explain":
            self.cmd_explain()
        elif cmd == "rollback":
            self.cmd_rollback()
        elif cmd == "metrics":
            self.cmd_metrics()
        elif cmd == "history":
            self.cmd_history()
        else:
            console.print(f"[red]Unknown command:[/red] {cmd}")
            console.print("Type 'help' for available commands")

    def cmd_help(self) -> None:
        """Display help information."""
        table = Table(
            title="Available Commands", show_header=True, header_style="bold magenta"
        )
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")

        commands = [
            ("load <file>", "Load a test scenario from JSON file"),
            ("translate <text>", "Translate natural language to ASP (and back)"),
            ("identify-gaps", "Identify knowledge gaps in loaded scenario"),
            ("generate-rule <gap-id>", "Generate candidate rule for a gap"),
            ("validate-rule <rule-id>", "Validate a generated rule"),
            ("incorporate-rule <rule-id>", "Incorporate validated rule into KB"),
            ("show-rules", "Show all generated and incorporated rules"),
            ("predict", "Make prediction on loaded scenario"),
            ("explain", "Show reasoning trace for current prediction"),
            ("rollback", "Undo last rule incorporation"),
            ("metrics", "Display detailed performance metrics"),
            ("history", "Show command history"),
            ("status", "Show current session status"),
            ("export <file>", "Export session data to JSON"),
            ("help", "Show this help message"),
            ("exit", "Exit playground"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        console.print(table)

    def cmd_exit(self) -> None:
        """Exit the playground."""
        console.print("[yellow]Exiting playground...[/yellow]")
        self.running = False

    def cmd_load(self, args: str) -> None:
        """Load a scenario."""
        if not args:
            console.print("[red]Usage:[/red] load <scenario-file>")
            return

        try:
            scenario_path = Path(args)
            scenario = self.session.load_scenario(scenario_path)

            console.print(
                Panel(
                    f"[bold]Scenario:[/bold] {scenario.scenario_id}\n"
                    f"[bold]Description:[/bold] {scenario.description}\n"
                    f"[bold]Question:[/bold] {scenario.question}",
                    title="Loaded Scenario",
                    border_style="green",
                )
            )

            if scenario.facts:
                syntax = Syntax(
                    scenario.facts, "prolog", theme="monokai", line_numbers=True
                )
                console.print(Panel(syntax, title="Facts", border_style="blue"))

        except Exception as e:
            console.print(f"[red]Error loading scenario:[/red] {e}")

    def cmd_translate(self, args: str) -> None:
        """Translate natural language to/from ASP."""
        if not args:
            console.print("[red]Usage:[/red] translate <natural language text>")
            return

        try:
            # Translate NL -> ASP
            console.print("[cyan]Translating NL -> ASP...[/cyan]")
            result = self.session.translate_nl_to_asp(args)

            asp_syntax = Syntax(
                result["asp_code"], "prolog", theme="monokai", line_numbers=True
            )
            console.print(
                Panel(
                    asp_syntax,
                    title=f"ASP Translation (confidence: {result['confidence']:.2f})",
                    border_style="green",
                )
            )

            # Translate back ASP -> NL
            console.print("\n[cyan]Translating ASP -> NL (round-trip test)...[/cyan]")
            back = self.session.translate_asp_to_nl(result["asp_code"])

            console.print(
                Panel(
                    back["natural_language"],
                    title=f"Natural Language (confidence: {back['confidence']:.2f})",
                    border_style="blue",
                )
            )

        except Exception as e:
            console.print(f"[red]Error translating:[/red] {e}")

    def cmd_identify_gaps(self) -> None:
        """Identify knowledge gaps."""
        try:
            gaps = self.session.identify_gaps()

            table = Table(
                title="Identified Gaps", show_header=True, header_style="bold magenta"
            )
            table.add_column("Gap ID", style="cyan")
            table.add_column("Description", style="white")

            for gap in gaps:
                table.add_row(gap["gap_id"], gap["description"][:80])

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error identifying gaps:[/red] {e}")

    def cmd_generate_rule(self, args: str) -> None:
        """Generate a rule for a gap."""
        if not args:
            console.print("[red]Usage:[/red] generate-rule <gap-id>")
            return

        try:
            console.print(f"[cyan]Generating rule for gap: {args}...[/cyan]")
            rule = self.session.generate_rule(args)

            asp_syntax = Syntax(
                rule.asp_rule, "prolog", theme="monokai", line_numbers=True
            )

            console.print(
                Panel(
                    asp_syntax,
                    title=f"Generated Rule: {rule.rule_id} (confidence: {rule.confidence:.2f})",
                    border_style="green",
                )
            )

            console.print(Panel(rule.reasoning, title="Reasoning", border_style="blue"))

            console.print(f"\n[green]Rule generated:[/green] {rule.rule_id}")
            console.print(f"Next steps: validate-rule {rule.rule_id}")

        except Exception as e:
            console.print(f"[red]Error generating rule:[/red] {e}")

    def cmd_validate_rule(self, args: str) -> None:
        """Validate a rule."""
        if not args:
            console.print("[red]Usage:[/red] validate-rule <rule-id>")
            return

        try:
            console.print(f"[cyan]Validating rule: {args}...[/cyan]")
            validation = self.session.validate_rule(args)

            decision_color = "green" if validation.decision == "accept" else "red"

            console.print(
                Panel(
                    f"[bold]Decision:[/bold] [{decision_color}]{validation.decision.upper()}[/{decision_color}]\n"
                    f"[bold]Confidence:[/bold] {validation.confidence:.2f}\n"
                    f"[bold]Summary:[/bold] {validation.report_summary}",
                    title=f"Validation Result: {args}",
                    border_style=decision_color,
                )
            )

            if validation.decision == "accept":
                console.print("\n[green]Rule accepted![/green]")
                console.print(f"Next step: incorporate-rule {args}")
            else:
                console.print(
                    "\n[red]Rule rejected.[/red] Try generating a different rule."
                )

        except Exception as e:
            console.print(f"[red]Error validating rule:[/red] {e}")

    def cmd_incorporate_rule(self, args: str) -> None:
        """Incorporate a rule."""
        if not args:
            console.print("[red]Usage:[/red] incorporate-rule <rule-id>")
            return

        try:
            result = self.session.incorporate_rule(args)

            # Show the actual rule that was incorporated
            rule_asp = result.get("asp_rule", "N/A")
            asp_syntax = Syntax(rule_asp, "prolog", theme="monokai", line_numbers=False)

            console.print(
                Panel(
                    f"[bold green]Successfully incorporated rule: {result['rule_id']}[/bold green]\n\n"
                    f"The rule has been added to the knowledge base.\n\n"
                    f"[bold]Rule:[/bold]",
                    title="Rule Incorporated",
                    border_style="green",
                )
            )
            console.print(asp_syntax)
            console.print(
                "\n[cyan]Tip:[/cyan] Use 'show-rules' to see all rules, or 'export' to save session data"
            )

        except Exception as e:
            console.print(f"[red]Error incorporating rule:[/red] {e}")

    def cmd_predict(self) -> None:
        """Make a prediction."""
        try:
            prediction = self.session.make_prediction()

            console.print(
                Panel(
                    f"[bold]Prediction:[/bold] {prediction['prediction']}\n"
                    f"[bold]Confidence:[/bold] {prediction['confidence']:.2f}\n"
                    f"[bold]Reasoning:[/bold] {prediction['reasoning']}",
                    title=f"Prediction for {prediction['scenario_id']}",
                    border_style="cyan",
                )
            )

        except Exception as e:
            console.print(f"[red]Error making prediction:[/red] {e}")

    def cmd_status(self) -> None:
        """Show session status."""
        status = self.session.get_status()

        table = Table(
            title="Session Status", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        for key, value in status.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)

    def cmd_show_rules(self) -> None:
        """Show all generated and incorporated rules."""
        if not self.session.generated_rules:
            console.print("[yellow]No rules generated yet.[/yellow]")
            console.print(
                "Use 'identify-gaps' and 'generate-rule <gap-id>' to create rules."
            )
            return

        # Create table of all rules
        table = Table(
            title="Generated Rules", show_header=True, header_style="bold magenta"
        )
        table.add_column("Rule ID", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Confidence", style="white")
        table.add_column("ASP Rule", style="green")

        for rule_id, rule in self.session.generated_rules.items():
            # Determine status
            if rule_id in self.session.incorporated_rules:
                status = "[bold green]Incorporated[/bold green]"
            elif rule.validation_status == "accept":
                status = "[green]Accepted[/green]"
            elif rule.validation_status == "reject":
                status = "[red]Rejected[/red]"
            elif rule.validation_status:
                status = f"[yellow]{rule.validation_status.title()}[/yellow]"
            else:
                status = "[dim]Not validated[/dim]"

            # Truncate long rules
            rule_text = rule.asp_rule
            if len(rule_text) > 60:
                rule_text = rule_text[:57] + "..."

            table.add_row(
                rule_id,
                status,
                f"{rule.confidence:.2f}",
                rule_text,
            )

        console.print(table)

        # Show incorporated rules summary
        if self.session.incorporated_rules:
            console.print(
                f"\n[green]✓ {len(self.session.incorporated_rules)} rule(s) incorporated into knowledge base[/green]"
            )
            console.print("[cyan]Tip:[/cyan] Use 'export' to save all rules to a file")

    def cmd_export(self, args: str) -> None:
        """Export session data."""
        if not args:
            args = f"playground_session_{self.session.session_start.strftime('%Y%m%d_%H%M%S')}.json"

        try:
            output_path = Path(args)
            self.session.export_session(output_path)
            console.print(f"[green]Session exported to:[/green] {output_path}")

        except Exception as e:
            console.print(f"[red]Error exporting session:[/red] {e}")

    def cmd_explain(self) -> None:
        """Show reasoning trace for current prediction."""
        try:
            explanation = self.session.explain_prediction()

            console.print(
                Panel(
                    f"[bold]Scenario:[/bold] {explanation['scenario']}\n"
                    f"[bold]Question:[/bold] {explanation['question']}\n\n"
                    f"[bold]Description:[/bold] {explanation['description']}",
                    title="Explanation",
                    border_style="cyan",
                )
            )

            if explanation["applicable_rules"]:
                console.print("\n[bold]Applicable Rules:[/bold]")
                for rule_info in explanation["applicable_rules"]:
                    syntax = Syntax(
                        rule_info["rule"], "prolog", theme="monokai", line_numbers=False
                    )
                    console.print(f"  [{rule_info['rule_id']}]")
                    console.print(syntax)
            else:
                console.print(
                    "\n[yellow]No incorporated rules apply to this scenario.[/yellow]"
                )

            console.print(f"\n[bold]Prediction:[/bold] {explanation['prediction']}")
            console.print(f"[bold]Confidence:[/bold] {explanation['confidence']:.2f}")

        except Exception as e:
            console.print(f"[red]Error generating explanation:[/red] {e}")

    def cmd_rollback(self) -> None:
        """Undo last rule incorporation."""
        try:
            result = self.session.rollback_last_incorporation()

            console.print(
                Panel(
                    f"[bold yellow]Rolled back rule: {result['rule_id']}[/bold yellow]\n\n"
                    f"Remaining incorporated rules: {result['remaining_incorporated']}\n"
                    f"Can undo rollback: {'Yes' if result['can_undo'] else 'No'}",
                    title="Rollback Complete",
                    border_style="yellow",
                )
            )

        except ValueError as e:
            console.print(f"[yellow]{e}[/yellow]")
        except Exception as e:
            console.print(f"[red]Error during rollback:[/red] {e}")

    def cmd_metrics(self) -> None:
        """Display detailed performance metrics."""
        metrics = self.session.get_metrics()

        table = Table(
            title="Performance Metrics", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Session Duration", metrics["session_duration"])
        table.add_row("Commands Executed", str(metrics["commands_executed"]))
        table.add_row("", "")  # Separator

        table.add_row("[bold]Rule Generation", "")
        table.add_row("  Gaps Identified", str(metrics["gaps_identified"]))
        table.add_row("  Rules Generated", str(metrics["rules_generated"]))
        table.add_row("  Rules Validated", str(metrics["rules_validated"]))
        table.add_row("  Rules Incorporated", str(metrics["rules_incorporated"]))
        table.add_row("", "")  # Separator

        table.add_row("[bold]Success Rates", "")
        table.add_row("  Acceptance Rate", f"{metrics['acceptance_rate']:.1%}")
        table.add_row("  Incorporation Rate", f"{metrics['incorporation_rate']:.1%}")
        table.add_row("", "")  # Separator

        table.add_row("[bold]Confidence Scores", "")
        table.add_row(
            "  Avg Generation Confidence", f"{metrics['avg_generation_confidence']:.2f}"
        )
        table.add_row(
            "  Avg Validation Confidence", f"{metrics['avg_validation_confidence']:.2f}"
        )

        console.print(table)

    def cmd_history(self) -> None:
        """Show command history."""
        history = self.session.get_command_history()

        if not history:
            console.print("[yellow]No commands executed yet.[/yellow]")
            return

        table = Table(
            title="Command History", show_header=True, header_style="bold magenta"
        )
        table.add_column("Time", style="cyan")
        table.add_column("Command", style="white")
        table.add_column("Details", style="dim")

        # Show last 20 commands
        for entry in history[-20:]:
            timestamp = entry["timestamp"].split("T")[1].split(".")[0]  # Get time part
            details = str(entry.get("details", ""))
            if len(details) > 50:
                details = details[:47] + "..."

            table.add_row(timestamp, entry["command"], details)

        console.print(table)

        if len(history) > 20:
            console.print(f"\n[dim]Showing last 20 of {len(history)} commands[/dim]")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LOFT Interactive Playground")
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-haiku-20241022",
        help="LLM model to use",
    )

    args = parser.parse_args()

    cli = PlaygroundCLI(model=args.model)
    cli.run()


if __name__ == "__main__":
    main()
