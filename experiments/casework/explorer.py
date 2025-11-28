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
from loft.neural.llm_interface import LLMInterface
from loft.neural.providers import AnthropicProvider
from loft.neural.rule_generator import RuleGenerator
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
        model: str = "claude-haiku-3-5-20241022",
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

        logger.info(f"Initialized CaseworkExplorer with model={model}, learning={enable_learning}")

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
        logger.info("\n" + "=" * 80)
        logger.info("Casework exploration complete")
        logger.info("=" * 80)

        return self.metrics

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

            # Generate candidate rule
            try:
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
                if report.overall_decision == "accept":
                    rules_accepted = 1
                    self.knowledge_base_rules.append(candidate.asp_rule)
                    logger.info(f"Learned new rule from {scenario.scenario_id}")

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
        Make a prediction for a scenario.

        Returns:
            Tuple of (prediction, confidence)
        """
        # Simplified prediction logic for MVP
        # In full implementation, would use ASP core with current KB

        # For now, use a simple heuristic
        if "writing" in scenario.description.lower() or "written" in scenario.description.lower():
            return "enforceable", 0.7
        else:
            return "unenforceable", 0.6

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
