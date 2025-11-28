"""
Playground session management.

Maintains state for interactive exploration including loaded scenarios,
generated rules, validation results, and system state.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

from loft.symbolic.asp_core import ASPCore
from loft.legal import StatuteOfFraudsSystem
from loft.neural.llm_interface import LLMInterface
from loft.neural.providers import AnthropicProvider
from loft.neural.rule_generator import RuleGenerator
from loft.validation.validation_pipeline import ValidationPipeline
from loft.translation.nl_to_asp import NLToASPTranslator


@dataclass
class LoadedScenario:
    """A loaded test scenario."""

    scenario_id: str
    description: str
    facts: str
    question: str
    ground_truth: Optional[str] = None
    rationale: Optional[str] = None


@dataclass
class GeneratedRuleRecord:
    """Record of a generated rule."""

    rule_id: str
    gap_id: str
    asp_rule: str
    confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    validation_status: Optional[str] = None


@dataclass
class ValidationRecord:
    """Record of a validation result."""

    rule_id: str
    decision: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    report_summary: str = ""


class PlaygroundSession:
    """
    Interactive session for exploring LOFT capabilities.

    Maintains state across commands and provides access to all LOFT components.
    """

    def __init__(self, model: str = "claude-haiku-3-5-20241022"):
        """Initialize playground session."""
        self.model = model
        self.session_start = datetime.now()

        # Core components
        self.asp_core = ASPCore()
        self.sof_system = StatuteOfFraudsSystem()

        # LLM components
        import os

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
            )
        provider = AnthropicProvider(api_key=api_key, model=model)
        self.llm = LLMInterface(provider=provider)
        self.rule_generator = RuleGenerator(llm=self.llm, asp_core=self.asp_core)

        # Translation components
        self.nl_to_asp = NLToASPTranslator(llm_interface=self.llm)
        # asp_to_nl is a function, not a class - it doesn't need llm parameter for basic usage
        from loft.translation.asp_to_nl import asp_to_nl as asp_to_nl_func

        self.asp_to_nl = asp_to_nl_func

        # Validation pipeline
        self.validation_pipeline = ValidationPipeline(min_confidence=0.6)

        # Session state
        self.loaded_scenario: Optional[LoadedScenario] = None
        self.identified_gaps: List[Dict[str, Any]] = []
        self.generated_rules: Dict[str, GeneratedRuleRecord] = {}
        self.validation_results: Dict[str, ValidationRecord] = {}
        self.incorporated_rules: List[str] = []

        # Command history
        self.command_history: List[Dict[str, Any]] = []

    def load_scenario(self, scenario_path: Path) -> LoadedScenario:
        """Load a test scenario from JSON file."""
        with open(scenario_path) as f:
            data = json.load(f)

        scenario = LoadedScenario(
            scenario_id=data.get("id", "unknown"),
            description=data.get("description", ""),
            facts=data.get("facts", ""),
            question=data.get("question", ""),
            ground_truth=data.get("ground_truth"),
            rationale=data.get("rationale"),
        )

        self.loaded_scenario = scenario
        self._log_command("load", {"scenario_id": scenario.scenario_id})

        return scenario

    def translate_nl_to_asp(self, text: str) -> Dict[str, Any]:
        """Translate natural language to ASP."""
        result = self.nl_to_asp.translate(text)

        self._log_command("translate_nl_to_asp", {"text": text[:100]})

        return {
            "asp_code": result.asp_code,
            "confidence": result.confidence,
            "predicates": result.predicates_used,
        }

    def translate_asp_to_nl(self, asp_code: str) -> Dict[str, Any]:
        """Translate ASP to natural language."""
        # asp_to_nl is a simple function that returns a string
        nl_text = self.asp_to_nl(asp_code, context=self.asp_core)

        self._log_command("translate_asp_to_nl", {"asp_code": asp_code[:100]})

        return {
            "natural_language": nl_text,
            "confidence": 1.0,  # Simple function doesn't provide confidence scores
        }

    def identify_gaps(self) -> List[Dict[str, Any]]:
        """Identify knowledge gaps in current scenario."""
        if not self.loaded_scenario:
            raise ValueError("No scenario loaded. Use 'load <scenario>' first.")

        # Simple gap identification based on failed prediction
        self.sof_system.reset()
        self.sof_system.add_facts(self.loaded_scenario.facts)

        # For MVP, create a simple gap
        gap = {
            "gap_id": f"gap_{len(self.identified_gaps)}",
            "scenario_id": self.loaded_scenario.scenario_id,
            "description": f"Gap in reasoning for: {self.loaded_scenario.description}",
            "question": self.loaded_scenario.question,
        }

        self.identified_gaps.append(gap)
        self._log_command("identify_gaps", {"gap_count": len(self.identified_gaps)})

        return self.identified_gaps

    def generate_rule(self, gap_id: str) -> GeneratedRuleRecord:
        """Generate a candidate rule for a gap."""
        gap = next((g for g in self.identified_gaps if g["gap_id"] == gap_id), None)
        if not gap:
            raise ValueError(f"Gap {gap_id} not found")

        # Generate rule using RuleGenerator
        principle = gap["description"]
        generated = self.rule_generator.generate_from_principle(principle)

        rule_id = f"rule_{len(self.generated_rules)}"
        record = GeneratedRuleRecord(
            rule_id=rule_id,
            gap_id=gap_id,
            asp_rule=generated.asp_rule,
            confidence=generated.confidence,
            reasoning=generated.reasoning,
        )

        self.generated_rules[rule_id] = record
        self._log_command("generate_rule", {"rule_id": rule_id, "gap_id": gap_id})

        return record

    def validate_rule(self, rule_id: str) -> ValidationRecord:
        """Validate a generated rule."""
        if rule_id not in self.generated_rules:
            raise ValueError(f"Rule {rule_id} not found")

        rule = self.generated_rules[rule_id]

        # Run validation pipeline
        report = self.validation_pipeline.validate_rule(
            rule_asp=rule.asp_rule,
            rule_id=rule_id,
            proposer_reasoning=rule.reasoning,
        )

        validation = ValidationRecord(
            rule_id=rule_id,
            decision=report.overall_decision,
            confidence=report.final_confidence,
            report_summary=f"Syntax: {report.syntax_valid}, Semantic: {report.semantic_score:.2f}",
        )

        self.validation_results[rule_id] = validation
        rule.validation_status = report.overall_decision

        self._log_command("validate_rule", {"rule_id": rule_id, "decision": validation.decision})

        return validation

    def incorporate_rule(self, rule_id: str) -> Dict[str, Any]:
        """Incorporate a validated rule into the knowledge base."""
        if rule_id not in self.generated_rules:
            raise ValueError(f"Rule {rule_id} not found")

        if rule_id not in self.validation_results:
            raise ValueError(f"Rule {rule_id} not validated. Run 'validate-rule {rule_id}' first.")

        validation = self.validation_results[rule_id]
        if validation.decision != "accept":
            raise ValueError(f"Cannot incorporate rejected rule {rule_id}")

        rule = self.generated_rules[rule_id]

        # For MVP, just track incorporation
        self.incorporated_rules.append(rule_id)

        self._log_command("incorporate_rule", {"rule_id": rule_id})

        return {
            "rule_id": rule_id,
            "status": "incorporated",
            "asp_rule": rule.asp_rule,
        }

    def make_prediction(self) -> Dict[str, Any]:
        """Make a prediction on the loaded scenario."""
        if not self.loaded_scenario:
            raise ValueError("No scenario loaded")

        self.sof_system.reset()
        self.sof_system.add_facts(self.loaded_scenario.facts)

        # Make prediction (simplified)
        prediction = {
            "scenario_id": self.loaded_scenario.scenario_id,
            "prediction": "enforceable",  # Simplified
            "confidence": 0.75,
            "reasoning": "Based on current knowledge base",
        }

        self._log_command("predict", {"scenario_id": self.loaded_scenario.scenario_id})

        return prediction

    def get_status(self) -> Dict[str, Any]:
        """Get current session status."""
        return {
            "session_duration": str(datetime.now() - self.session_start),
            "model": self.model,
            "loaded_scenario": self.loaded_scenario.scenario_id if self.loaded_scenario else None,
            "identified_gaps": len(self.identified_gaps),
            "generated_rules": len(self.generated_rules),
            "validated_rules": len(self.validation_results),
            "incorporated_rules": len(self.incorporated_rules),
            "commands_executed": len(self.command_history),
        }

    def _log_command(self, command: str, details: Dict[str, Any]) -> None:
        """Log a command to history."""
        self.command_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "details": details,
            }
        )

    def export_session(self, output_path: Path) -> None:
        """Export session data to JSON."""
        data = {
            "session_info": {
                "start_time": self.session_start.isoformat(),
                "model": self.model,
            },
            "scenario": self.loaded_scenario.__dict__ if self.loaded_scenario else None,
            "gaps": self.identified_gaps,
            "rules": {
                rule_id: {
                    "asp_rule": rule.asp_rule,
                    "confidence": rule.confidence,
                    "reasoning": rule.reasoning,
                    "validation_status": rule.validation_status,
                }
                for rule_id, rule in self.generated_rules.items()
            },
            "validations": {
                rule_id: {
                    "decision": val.decision,
                    "confidence": val.confidence,
                    "summary": val.report_summary,
                }
                for rule_id, val in self.validation_results.items()
            },
            "incorporated_rules": self.incorporated_rules,
            "command_history": self.command_history,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
