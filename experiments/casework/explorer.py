"""
Casework Explorer - Main pipeline for batch case exploration.

Processes multiple legal scenarios, identifies gaps, generates rules,
validates, and incorporates them while tracking learning metrics.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loft.symbolic.asp_core import ASPCore
from loft.symbolic.asp_reasoner import ASPReasoner
from loft.neural.llm_interface import LLMInterface
from loft.neural.providers import AnthropicProvider
from loft.neural.rule_generator import (
    RuleGenerator,
    RuleGenerationError,
    extract_predicates_from_asp_facts,
)
from loft.validation.validation_pipeline import ValidationPipeline

from experiments.casework.dataset_loader import DatasetLoader, LegalScenario
from experiments.casework.metrics import LearningMetrics, CaseResult


class CaseworkExplorer:
    """
    Automated exploration pipeline for legal casework.

    Processes scenarios sequentially, learning from each case.
    """

    def __init__(
        self,
        dataset_dir: Path,
        model: str = "claude-3-5-haiku-20241022",
        enable_learning: bool = True,
    ):
        """
        Initialize casework explorer.

        Args:
            dataset_dir: Directory containing test scenario JSON files
            model: LLM model to use
            enable_learning: Whether to incorporate learned rules
        """
        self.dataset_dir = Path(dataset_dir)
        self.model = model
        self.enable_learning = enable_learning

        # Initialize components
        self.asp_core = ASPCore()
        self.asp_reasoner = ASPReasoner()  # For ASP-based predictions
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
        self.validation_pipeline = ValidationPipeline(min_confidence=0.6)

        # Dataset and metrics
        self.dataset_loader = DatasetLoader(dataset_dir)
        self.metrics = LearningMetrics()

        # Knowledge base (track incorporated rules)
        self.knowledge_base_rules: List[str] = []

        # Collect dataset predicates for aligned rule generation
        self.dataset_predicates: List[str] = self._collect_dataset_predicates()

        logger.info(
            f"Initialized CaseworkExplorer with model={model}, learning={enable_learning}, "
            f"dataset_predicates={len(self.dataset_predicates)}"
        )

    def _collect_dataset_predicates(self) -> List[str]:
        """
        Collect all unique predicates from dataset scenarios.

        Scans all scenarios in the dataset and extracts predicate patterns
        for use in aligned rule generation.

        Returns:
            List of predicate patterns found in the dataset
        """
        all_predicates: set = set()

        try:
            scenarios = self.dataset_loader.load_all()
            for scenario in scenarios:
                if scenario.asp_facts:
                    predicates = extract_predicates_from_asp_facts(scenario.asp_facts)
                    all_predicates.update(predicates)

            logger.info(f"Collected {len(all_predicates)} unique predicates from dataset")
            return sorted(all_predicates)

        except Exception as e:
            logger.warning(f"Failed to collect dataset predicates: {e}")
            return []

    def explore_dataset(
        self,
        max_cases: Optional[int] = None,
        difficulty: Optional[str] = None,
    ) -> LearningMetrics:
        """
        Explore all cases in the dataset.

        Args:
            max_cases: Maximum number of cases to process (None = all)
            difficulty: Filter by difficulty level ("easy", "medium", "hard")

        Returns:
            Learning metrics
        """
        logger.info("Starting casework exploration...")

        # Load scenarios
        if difficulty:
            scenarios = self.dataset_loader.load_by_difficulty(difficulty)
        else:
            scenarios = self.dataset_loader.load_all()

        if max_cases:
            scenarios = scenarios[:max_cases]

        logger.info(f"Loaded {len(scenarios)} scenarios")

        # Process each scenario
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Processing case {i}/{len(scenarios)}: {scenario.scenario_id}")
            logger.info(f"{'=' * 80}")

            result = self.process_scenario(scenario)
            self.metrics.add_case_result(result)

            logger.info(
                f"Result: {'CORRECT' if result.correct else 'INCORRECT'} "
                f"(confidence: {result.confidence:.2f})"
            )
            logger.info(
                f"Current accuracy: {self.metrics.get_current_accuracy():.1%} "
                f"({self.metrics.cases_correct}/{self.metrics.total_cases})"
            )

        self.metrics.end_time = datetime.now()

        # Calculate final accuracy with complete knowledge base
        if self.knowledge_base_rules:
            final_accuracy = self._calculate_final_accuracy(scenarios)
            logger.info(f"\nFinal accuracy with complete KB: {final_accuracy:.1%}")
            self.metrics.final_kb_accuracy = final_accuracy

        logger.info("\n" + "=" * 80)
        logger.info("Casework exploration complete")
        logger.info("=" * 80)

        return self.metrics

    def _calculate_final_accuracy(self, scenarios: List[LegalScenario]) -> float:
        """
        Calculate accuracy using the complete knowledge base.

        Re-evaluates all scenarios with all learned rules to see what
        accuracy would be achieved with the final knowledge base.

        Args:
            scenarios: All scenarios to evaluate

        Returns:
            Accuracy as a fraction (0.0-1.0)
        """
        correct = 0
        total = 0

        for scenario in scenarios:
            asp_facts = scenario.asp_facts or ""
            if not asp_facts:
                continue

            result = self.asp_reasoner.reason(self.knowledge_base_rules, asp_facts)

            if result.prediction != "unknown":
                total += 1
                if result.prediction == scenario.ground_truth:
                    correct += 1

        return correct / total if total > 0 else 0.0

    def process_scenario(self, scenario: LegalScenario) -> CaseResult:
        """
        Process a single scenario.

        Args:
            scenario: Legal scenario to process

        Returns:
            Case result
        """
        start_time = time.time()

        # Step 1: Make initial prediction
        prediction, confidence = self._make_prediction(scenario)

        # Step 2: Check if correct
        correct = prediction == scenario.ground_truth

        # Step 3: If incorrect and learning enabled, try to learn
        gaps_identified = 0
        rules_generated = 0
        rules_accepted = 0

        if not correct and self.enable_learning:
            logger.info("Prediction incorrect - identifying gaps and generating rules...")

            # Identify gap
            _gap = self._identify_gap(scenario, prediction)  # noqa: F841
            gaps_identified = 1

            # Generate candidate rule using aligned generation
            try:
                # Combine dataset predicates with scenario-specific predicates
                scenario_predicates = []
                if scenario.asp_facts:
                    scenario_predicates = extract_predicates_from_asp_facts(scenario.asp_facts)

                all_predicates = list(set(self.dataset_predicates + scenario_predicates))

                # Use aligned generation for better predicate matching
                if all_predicates:
                    candidate = self.rule_generator.generate_from_principle_aligned(
                        principle_text=scenario.rationale,
                        dataset_predicates=all_predicates,
                    )
                else:
                    # Fall back to standard generation if no predicates available
                    candidate = self.rule_generator.generate_from_principle(
                        principle_text=scenario.rationale,
                    )
                rules_generated = 1

                # Validate
                report = self.validation_pipeline.validate_rule(
                    rule_asp=candidate.asp_rule,
                    rule_id=f"rule_{scenario.scenario_id}",
                    proposer_reasoning=candidate.reasoning,
                )

                # Accept if validated
                if report.final_decision == "accept":
                    rules_accepted = 1
                    self.knowledge_base_rules.append(candidate.asp_rule)
                    logger.info(f"Learned new rule from {scenario.scenario_id}")

            except RuleGenerationError as e:
                logger.warning(f"Rule generation failed for {scenario.scenario_id}: {e}")
            except Exception as e:
                logger.warning(f"Error learning from scenario: {e}")

        processing_time = time.time() - start_time

        return CaseResult(
            scenario_id=scenario.scenario_id,
            prediction=prediction,
            ground_truth=scenario.ground_truth,
            correct=correct,
            confidence=confidence,
            gaps_identified=gaps_identified,
            rules_generated=rules_generated,
            rules_accepted=rules_accepted,
            processing_time=processing_time,
        )

    def _make_prediction(self, scenario: LegalScenario) -> tuple[str, float]:
        """
        Make a prediction for a scenario using ASP reasoning.

        Uses the current knowledge base rules combined with scenario facts
        to derive predictions through actual ASP solving.

        Returns:
            Tuple of (prediction, confidence)
        """
        # Get ASP facts from scenario
        asp_facts = scenario.asp_facts or ""

        if not asp_facts:
            # Fall back to unknown if no ASP facts available
            logger.debug(f"No ASP facts for scenario {scenario.scenario_id}")
            return "unknown", 0.0

        # Use ASP reasoning with current knowledge base
        result = self.asp_reasoner.reason(self.knowledge_base_rules, asp_facts)

        # Log reasoning details for debugging
        if result.prediction == "unknown":
            logger.debug(
                f"ASP reasoning for {scenario.scenario_id}: unknown "
                f"(derived atoms: {len(result.derived_atoms)}, "
                f"rules fired: {len(result.rules_fired)})"
            )
        else:
            logger.debug(
                f"ASP reasoning for {scenario.scenario_id}: {result.prediction} "
                f"(confidence: {result.confidence:.2f})"
            )

        return result.prediction, result.confidence

    def _identify_gap(self, scenario: LegalScenario, prediction: str) -> Dict[str, Any]:
        """Identify knowledge gap from failed prediction."""
        return {
            "scenario_id": scenario.scenario_id,
            "description": f"Failed to predict {scenario.ground_truth} for: {scenario.description}",
            "incorrect_prediction": prediction,
            "correct_answer": scenario.ground_truth,
            "rationale": scenario.rationale,
        }

    def get_report_data(self) -> Dict[str, Any]:
        """Get data for reporting."""
        return {
            "config": {
                "model": self.model,
                "enable_learning": self.enable_learning,
                "dataset_dir": str(self.dataset_dir),
            },
            "dataset_stats": self.dataset_loader.get_statistics(),
            "metrics": self.metrics.to_dict(),
            "knowledge_base": {
                "total_rules": len(self.knowledge_base_rules),
                "rules": self.knowledge_base_rules,
            },
        }


def main():
    """Main entry point for casework explorer CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated Casework Explorer - Process legal scenarios and learn rules"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory containing JSON scenario files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file for results (default: reports/casework_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-haiku-20241022",
        help="LLM model to use (default: claude-3-5-haiku-20241022)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of cases to process (default: all)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter by difficulty level",
    )
    parser.add_argument(
        "--no-learning",
        action="store_true",
        help="Disable rule incorporation (exploration only)",
    )

    args = parser.parse_args()

    # Create output path if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        args.output = str(output_dir / f"casework_{timestamp}.json")

    # Run explorer
    logger.info("Starting casework exploration")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Learning: {not args.no_learning}")

    explorer = CaseworkExplorer(
        dataset_dir=Path(args.dataset),
        model=args.model,
        enable_learning=not args.no_learning,
    )

    metrics = explorer.explore_dataset(
        max_cases=args.max_cases,
        difficulty=args.difficulty,
    )

    # Generate and save reports
    from experiments.casework.reporting import ReportGenerator

    report_data = explorer.get_report_data()
    report_gen = ReportGenerator(report_data)

    # Save JSON report
    output_path = Path(args.output)
    report_gen.generate_json(output_path)
    logger.info(f"JSON report saved to: {args.output}")

    # Generate text report
    text_output = output_path.with_suffix(".txt")
    report_gen.generate_text(text_output)
    logger.info(f"Text report saved to: {text_output}")

    # Generate HTML report
    html_output = output_path.with_suffix(".html")
    report_gen.generate_html(html_output)
    logger.info(f"HTML report saved to: {html_output}")

    # Print summary
    summary = metrics.get_summary()
    print("\n" + "=" * 80)
    print("Casework Exploration Summary")
    print("=" * 80)
    print(f"Cases processed: {summary['total_cases']}")
    print(f"Cases correct: {summary['cases_correct']}")
    print(f"Final accuracy: {summary['final_accuracy']:.1%}")
    print(f"Gaps identified: {summary['total_gaps_identified']}")
    print(f"Rules generated: {summary['total_rules_generated']}")
    print(f"Rules accepted: {summary['total_rules_accepted']}")
    print(f"Rules incorporated: {len(explorer.knowledge_base_rules)}")
    print(f"Acceptance rate: {summary['acceptance_rate']:.1%}")
    print(f"Total processing time: {summary['total_time_seconds']:.2f}s")
    print(f"Avg time per case: {summary['avg_time_per_case']:.2f}s")
    print(f"\nFull results: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
