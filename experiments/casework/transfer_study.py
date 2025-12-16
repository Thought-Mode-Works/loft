"""
Cross-Domain Transfer Study for Legal Knowledge.

Tests how well knowledge learned in one legal domain transfers to another.
Implements zero-shot and few-shot transfer learning experiments.

Uses actual ASP (Answer Set Programming) reasoning instead of heuristics
to make predictions based on learned rules and scenario facts.
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
from dataclasses import dataclass
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.casework.explorer import CaseworkExplorer
from experiments.casework.dataset_loader import DatasetLoader, LegalScenario
from loft.symbolic.asp_reasoner import ASPReasoner

# Optional canonical translator for cross-domain translation
try:
    from loft.ontology.canonical_translator import CanonicalTranslator

    CANONICAL_TRANSLATOR_AVAILABLE = True
except ImportError:
    CANONICAL_TRANSLATOR_AVAILABLE = False
    CanonicalTranslator = None

# Optional hybrid translator with LLM fallback
try:
    from loft.ontology.hybrid_translator import HybridTranslator

    HYBRID_TRANSLATOR_AVAILABLE = True
except ImportError:
    HYBRID_TRANSLATOR_AVAILABLE = False
    HybridTranslator = None


@dataclass
class TransferResult:
    """Results from a transfer learning experiment."""

    source_domain: str
    target_domain: str
    source_cases: int
    target_cases: int

    # Source domain training results
    source_baseline_accuracy: float
    source_final_accuracy: float
    source_rules_learned: int

    # Zero-shot transfer (no target training)
    zero_shot_accuracy: float

    # Few-shot transfer (limited target training)
    few_shot_cases: int
    few_shot_accuracy: float
    few_shot_rules_learned: int

    # From-scratch baseline (train on target only)
    scratch_accuracy: float
    scratch_rules_learned: int

    # Transfer effectiveness metrics
    transfer_rate: float = 0.0  # (zero_shot - baseline) / (final - baseline)
    few_shot_advantage: float = 0.0  # few_shot_accuracy - scratch_accuracy

    # ASP reasoning statistics (new)
    zero_shot_coverage: float = 0.0  # % of scenarios with ASP predictions
    zero_shot_unknown: int = 0  # Scenarios where ASP returned "unknown"
    reasoning_stats: Optional[dict] = None  # Detailed ASP reasoning stats

    def __post_init__(self):
        """Calculate derived metrics."""
        # Transfer rate: how much of the learning transferred
        source_improvement = self.source_final_accuracy - self.source_baseline_accuracy
        if source_improvement > 0:
            zero_shot_improvement = (
                self.zero_shot_accuracy - self.source_baseline_accuracy
            )
            self.transfer_rate = zero_shot_improvement / source_improvement
        else:
            self.transfer_rate = 0.0

        # Few-shot advantage: how much few-shot with transfer beats from-scratch
        self.few_shot_advantage = self.few_shot_accuracy - self.scratch_accuracy


@dataclass
class SameDomainResult:
    """Results from a same-domain learning experiment."""

    domain: str
    total_cases: int
    train_cases: int
    test_cases: int

    # Training results
    rules_learned: int
    train_accuracy: float

    # Test results (rules applied to held-out test set)
    test_accuracy: float
    test_coverage: float
    test_unknown: int

    # Final KB accuracy (re-evaluate all with complete KB)
    final_kb_accuracy: float

    # Learned rules for inspection
    learned_rules: List[str]

    # ASP reasoning statistics
    reasoning_stats: Optional[dict] = None


class TransferStudy:
    """
    Cross-domain transfer learning experiments.

    Tests how well knowledge learned in one legal domain applies to another.
    Uses actual ASP reasoning for predictions instead of heuristics.
    Supports canonical translation for cross-domain predicate mapping.
    """

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        use_canonical_translation: bool = False,
        use_hybrid_translation: bool = False,
        min_llm_confidence: float = 0.6,
    ):
        """
        Initialize transfer study.

        Args:
            model: LLM model to use for learning
            use_canonical_translation: If True, use canonical ontology for
                cross-domain predicate translation (deterministic only)
            use_hybrid_translation: If True, use hybrid translator with LLM
                fallback for unmapped predicates
            min_llm_confidence: Minimum confidence for LLM translations (hybrid mode)
        """
        self.model = model
        self.asp_reasoner = ASPReasoner()
        self._unknown_predictions: List[str] = (
            []
        )  # Track scenarios with unknown predictions
        self.use_canonical_translation = use_canonical_translation
        self.use_hybrid_translation = use_hybrid_translation
        self._translator: Optional["CanonicalTranslator"] = None
        self._hybrid_translator: Optional["HybridTranslator"] = None

        # Hybrid translation takes precedence
        if use_hybrid_translation:
            if not HYBRID_TRANSLATOR_AVAILABLE:
                raise ImportError(
                    "Hybrid translation requires rdflib and anthropic. "
                    "Install with: pip install rdflib anthropic"
                )
            self._hybrid_translator = HybridTranslator(
                model=model,
                min_llm_confidence=min_llm_confidence,
                enable_llm=True,
            )
            self._translator = self._hybrid_translator.canonical
            logger.info(
                f"Hybrid translation enabled (LLM fallback). "
                f"Domains: {self._translator.get_domains()}, "
                f"min_confidence: {min_llm_confidence}"
            )
        elif use_canonical_translation:
            if not CANONICAL_TRANSLATOR_AVAILABLE:
                raise ImportError(
                    "Canonical translation requires rdflib. Install with: pip install rdflib"
                )
            self._translator = CanonicalTranslator()
            logger.info(
                f"Canonical translation enabled. Domains: {self._translator.get_domains()}"
            )

    def run_same_domain_experiment(
        self,
        dataset_path: Path,
        train_ratio: float = 0.7,
    ) -> SameDomainResult:
        """
        Run same-domain learning experiment with train/test split.

        This validates the pipeline by testing on the same domain,
        eliminating predicate ontology mismatch issues.

        Args:
            dataset_path: Path to dataset directory
            train_ratio: Fraction of data for training (default: 0.7)

        Returns:
            SameDomainResult with metrics
        """
        loader = DatasetLoader(dataset_path)
        domain = dataset_path.name

        logger.info(f"Starting same-domain experiment: {domain}")

        # Load and split scenarios
        all_scenarios = loader.load_all()
        split_idx = int(len(all_scenarios) * train_ratio)
        train_scenarios = all_scenarios[:split_idx]
        test_scenarios = all_scenarios[split_idx:]

        logger.info(
            f"Split: {len(train_scenarios)} train, {len(test_scenarios)} test "
            f"({train_ratio:.0%}/{1 - train_ratio:.0%})"
        )

        # Step 1: Train on training set
        logger.info("Step 1: Training on training set...")
        explorer = CaseworkExplorer(
            dataset_dir=dataset_path, model=self.model, enable_learning=True
        )

        # Only process training scenarios
        train_metrics = explorer.explore_dataset(max_cases=len(train_scenarios))
        train_accuracy = train_metrics.get_current_accuracy()
        rules_learned = len(explorer.knowledge_base_rules)

        logger.info(
            f"Training complete: {train_accuracy:.1%} accuracy, {rules_learned} rules"
        )

        # Log the learned rules for inspection
        logger.info("Learned rules:")
        for i, rule in enumerate(explorer.knowledge_base_rules, 1):
            logger.info(f"  {i}. {rule[:100]}...")

        # Step 2: Test on held-out test set
        logger.info("Step 2: Testing on held-out test set...")
        self.asp_reasoner.reset_stats()
        self._unknown_predictions = []

        test_accuracy, test_coverage = self._test_knowledge_base_asp(
            explorer.knowledge_base_rules, test_scenarios
        )
        test_unknown = len(self._unknown_predictions)

        logger.info(
            f"Test accuracy: {test_accuracy:.1%} "
            f"(coverage: {test_coverage:.1%}, unknown: {test_unknown})"
        )

        # Step 3: Calculate final KB accuracy on ALL scenarios
        logger.info("Step 3: Re-evaluating all scenarios with complete KB...")
        self.asp_reasoner.reset_stats()
        self._unknown_predictions = []

        final_accuracy, final_coverage = self._test_knowledge_base_asp(
            explorer.knowledge_base_rules, all_scenarios
        )

        logger.info(f"Final KB accuracy (all scenarios): {final_accuracy:.1%}")

        return SameDomainResult(
            domain=domain,
            total_cases=len(all_scenarios),
            train_cases=len(train_scenarios),
            test_cases=len(test_scenarios),
            rules_learned=rules_learned,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            test_coverage=test_coverage,
            test_unknown=test_unknown,
            final_kb_accuracy=final_accuracy,
            learned_rules=explorer.knowledge_base_rules.copy(),
            reasoning_stats=self.asp_reasoner.get_stats().to_dict(),
        )

    def run_transfer_experiment(
        self,
        source_dataset: Path,
        target_dataset: Path,
        few_shot_count: int = 10,
    ) -> TransferResult:
        """
        Run complete transfer learning experiment.

        Args:
            source_dataset: Path to source domain dataset
            target_dataset: Path to target domain dataset
            few_shot_count: Number of target cases for few-shot learning

        Returns:
            TransferResult with all metrics
        """
        source_loader = DatasetLoader(source_dataset)
        target_loader = DatasetLoader(target_dataset)

        source_domain = source_dataset.name
        target_domain = target_dataset.name

        logger.info(f"Starting transfer study: {source_domain} → {target_domain}")

        # Check canonical translation availability
        if self.use_canonical_translation and self._translator:
            coverage = self._translator.get_translation_coverage(
                source_domain, target_domain
            )
            logger.info(
                f"Canonical translation coverage: {coverage['coverage_ratio']:.1%} "
                f"({coverage['translatable_count']}/{coverage['source_predicates']} predicates)"
            )

        # Step 1: Train on source domain
        logger.info(f"Step 1: Training on source domain ({source_domain})...")
        source_explorer = CaseworkExplorer(
            dataset_dir=source_dataset, model=self.model, enable_learning=True
        )

        source_scenarios = source_loader.load_all()
        source_baseline_acc = self._measure_baseline(
            source_explorer, source_scenarios[:10]
        )

        source_metrics = source_explorer.explore_dataset()
        source_final_acc = source_metrics.get_current_accuracy()
        source_rules = len(source_explorer.knowledge_base_rules)

        logger.info(
            f"Source training complete: {source_baseline_acc:.1%} → {source_final_acc:.1%} "
            f"({source_rules} rules)"
        )

        # Step 2: Test zero-shot on target domain (using ASP reasoning)
        logger.info(f"Step 2: Testing zero-shot transfer to {target_domain}...")
        target_scenarios = target_loader.load_all()

        # Reset ASP reasoner stats and unknown predictions for this test
        self.asp_reasoner.reset_stats()
        self._unknown_predictions = []

        # Translate rules if canonical translation is enabled
        transfer_rules = source_explorer.knowledge_base_rules
        translation_stats = None

        if self.use_canonical_translation and self._translator:
            logger.info("Translating rules using canonical predicates...")
            transfer_rules, translation_stats = self._translate_knowledge_base(
                source_explorer.knowledge_base_rules,
                source_domain,
                target_domain,
            )
            logger.info(
                f"Translated {len(transfer_rules)} rules: "
                f"{translation_stats['total_translated']} predicates translated, "
                f"{translation_stats['total_untranslatable']} untranslatable"
            )

        zero_shot_acc, zero_shot_coverage = self._test_knowledge_base_asp(
            transfer_rules, target_scenarios
        )
        zero_shot_unknown = len(self._unknown_predictions)

        logger.info(
            f"Zero-shot accuracy: {zero_shot_acc:.1%} "
            f"(coverage: {zero_shot_coverage:.1%}, unknown: {zero_shot_unknown})"
        )

        # Step 3: Few-shot learning on target
        logger.info(f"Step 3: Few-shot learning ({few_shot_count} cases)...")
        few_shot_explorer = CaseworkExplorer(
            dataset_dir=target_dataset, model=self.model, enable_learning=True
        )

        # Pre-load source knowledge
        few_shot_explorer.knowledge_base_rules = (
            source_explorer.knowledge_base_rules.copy()
        )

        few_shot_metrics = few_shot_explorer.explore_dataset(max_cases=few_shot_count)
        few_shot_acc = few_shot_metrics.get_current_accuracy()
        few_shot_rules = len(few_shot_explorer.knowledge_base_rules) - source_rules

        logger.info(
            f"Few-shot accuracy: {few_shot_acc:.1%} ({few_shot_rules} new rules)"
        )

        # Step 4: From-scratch baseline on target
        logger.info(f"Step 4: Training from scratch on {target_domain}...")
        scratch_explorer = CaseworkExplorer(
            dataset_dir=target_dataset, model=self.model, enable_learning=True
        )

        scratch_metrics = scratch_explorer.explore_dataset(max_cases=few_shot_count)
        scratch_acc = scratch_metrics.get_current_accuracy()
        scratch_rules = len(scratch_explorer.knowledge_base_rules)

        logger.info(f"From-scratch accuracy: {scratch_acc:.1%} ({scratch_rules} rules)")

        # Create result with ASP reasoning statistics
        result = TransferResult(
            source_domain=source_domain,
            target_domain=target_domain,
            source_cases=len(source_scenarios),
            target_cases=len(target_scenarios),
            source_baseline_accuracy=source_baseline_acc,
            source_final_accuracy=source_final_acc,
            source_rules_learned=source_rules,
            zero_shot_accuracy=zero_shot_acc,
            few_shot_cases=few_shot_count,
            few_shot_accuracy=few_shot_acc,
            few_shot_rules_learned=few_shot_rules,
            scratch_accuracy=scratch_acc,
            scratch_rules_learned=scratch_rules,
            zero_shot_coverage=zero_shot_coverage,
            zero_shot_unknown=zero_shot_unknown,
            reasoning_stats=self.asp_reasoner.get_stats().to_dict(),
        )

        return result

    def _translate_knowledge_base(
        self,
        rules: List[str],
        source_domain: str,
        target_domain: str,
    ) -> tuple[List[str], dict]:
        """
        Translate knowledge base rules from source to target domain.

        Uses hybrid translator (with LLM fallback) if available, otherwise
        falls back to canonical-only translation.

        Args:
            rules: List of ASP rules in source domain
            source_domain: Source domain name
            target_domain: Target domain name

        Returns:
            Tuple of:
            - List of translated rules
            - Statistics dictionary with translation metrics
        """
        if not self._translator:
            return rules, {"total_translated": 0, "total_untranslatable": 0}

        # Use hybrid translator if available
        if self._hybrid_translator:
            return self._translate_with_hybrid(rules, source_domain, target_domain)

        # Fall back to canonical-only translation
        translated_rules = []
        total_translated = 0
        total_untranslatable = 0

        for rule in rules:
            translated, success, failed = self._translator.translate_rule(
                rule, source_domain, target_domain
            )
            translated_rules.append(translated)
            total_translated += len(success)
            total_untranslatable += len(failed)

            if success:
                logger.debug(f"Translated: {success}")
            if failed:
                logger.debug(f"Untranslatable: {failed}")

        stats = {
            "total_translated": total_translated,
            "total_untranslatable": total_untranslatable,
            "rules_processed": len(rules),
        }

        return translated_rules, stats

    def _translate_with_hybrid(
        self,
        rules: List[str],
        source_domain: str,
        target_domain: str,
    ) -> tuple[List[str], dict]:
        """
        Translate rules using hybrid translator (canonical + LLM fallback).

        Args:
            rules: List of ASP rules to translate
            source_domain: Source domain name
            target_domain: Target domain name

        Returns:
            Tuple of (translated_rules, stats)
        """
        translated_rules = []
        successful_translations = 0
        failed_translations = 0
        total_confidence = 0.0

        for rule in rules:
            result = self._hybrid_translator.translate_rule(
                rule, source_domain, target_domain
            )

            if result.translated is not None:
                translated_rules.append(result.translated)
                successful_translations += 1
                total_confidence += result.confidence

                if result.translations:
                    logger.debug(
                        f"Hybrid translated ({result.method}): {result.reasoning}"
                    )
            else:
                # Keep original rule if translation fails
                translated_rules.append(rule)
                failed_translations += 1
                logger.debug(f"Translation failed: {result.reasoning}")

        # Get hybrid translator stats
        hybrid_stats = self._hybrid_translator.get_stats()

        stats = {
            "total_translated": successful_translations,
            "total_untranslatable": failed_translations,
            "rules_processed": len(rules),
            "average_confidence": (
                total_confidence / successful_translations
                if successful_translations > 0
                else 0.0
            ),
            "canonical_translations": hybrid_stats.canonical_translations,
            "llm_translations": hybrid_stats.llm_translations,
            "canonical_rate": hybrid_stats.canonical_rate,
            "llm_rate": hybrid_stats.llm_rate,
        }

        return translated_rules, stats

    def _measure_baseline(
        self, explorer: CaseworkExplorer, scenarios: List[Any]
    ) -> float:
        """Measure baseline accuracy before learning."""
        # Use simple heuristic prediction for baseline
        correct = 0
        for scenario in scenarios:
            prediction, _ = explorer._make_prediction(scenario)
            if prediction == scenario.ground_truth:
                correct += 1

        return correct / len(scenarios) if scenarios else 0.0

    def _test_knowledge_base_asp(
        self, knowledge_base: List[str], scenarios: List[LegalScenario]
    ) -> tuple[float, float]:
        """
        Test knowledge base on scenarios using actual ASP reasoning.

        Args:
            knowledge_base: List of ASP rules
            scenarios: List of legal scenarios to test

        Returns:
            Tuple of (accuracy, coverage) where:
            - accuracy: Fraction of definitive predictions that were correct
            - coverage: Fraction of scenarios with definitive (non-unknown) predictions
        """
        correct = 0
        total_definitive = 0

        for scenario in scenarios:
            # Get ASP facts from scenario
            asp_facts = scenario.asp_facts or ""

            if not asp_facts:
                # No ASP facts available - log and skip
                logger.debug(
                    f"No ASP facts for scenario {scenario.scenario_id}, skipping"
                )
                self._unknown_predictions.append(scenario.scenario_id)
                continue

            # Make prediction using ASP reasoning
            result = self.asp_reasoner.reason(knowledge_base, asp_facts)

            if result.prediction == "unknown":
                # Track unknown predictions separately
                self._unknown_predictions.append(scenario.scenario_id)
                logger.debug(
                    f"ASP returned unknown for {scenario.scenario_id}: "
                    f"derived atoms = {result.derived_atoms}"
                )
            else:
                # Count definitive predictions
                total_definitive += 1
                if result.prediction == scenario.ground_truth:
                    correct += 1
                else:
                    logger.debug(
                        f"Incorrect prediction for {scenario.scenario_id}: "
                        f"predicted {result.prediction}, expected {scenario.ground_truth}"
                    )

            # Update reasoner stats
            is_correct = (
                result.prediction == scenario.ground_truth
                if result.prediction != "unknown"
                else None
            )
            self.asp_reasoner.stats.update(result, is_correct)

        # Calculate metrics
        accuracy = correct / total_definitive if total_definitive > 0 else 0.0
        coverage = total_definitive / len(scenarios) if scenarios else 0.0

        return accuracy, coverage

    def _make_asp_prediction(
        self, knowledge_base: List[str], scenario: LegalScenario
    ) -> tuple[str, float]:
        """
        Make a single prediction using ASP reasoning.

        Args:
            knowledge_base: List of ASP rules
            scenario: Legal scenario to predict

        Returns:
            Tuple of (prediction, confidence)
        """
        asp_facts = scenario.asp_facts or ""

        if not asp_facts:
            logger.warning(f"No ASP facts for scenario {scenario.scenario_id}")
            return "unknown", 0.0

        result = self.asp_reasoner.reason(knowledge_base, asp_facts)
        return result.prediction, result.confidence

    def generate_report(self, result: TransferResult, output_path: Path) -> None:
        """Generate transfer study report."""
        report = {
            "experiment": {
                "type": "cross_domain_transfer",
                "timestamp": datetime.now().isoformat(),
                "source_domain": result.source_domain,
                "target_domain": result.target_domain,
                "reasoning_method": "asp",  # Explicitly note ASP reasoning is used
            },
            "source_training": {
                "cases": result.source_cases,
                "baseline_accuracy": result.source_baseline_accuracy,
                "final_accuracy": result.source_final_accuracy,
                "improvement": result.source_final_accuracy
                - result.source_baseline_accuracy,
                "rules_learned": result.source_rules_learned,
            },
            "transfer_results": {
                "zero_shot_accuracy": result.zero_shot_accuracy,
                "zero_shot_coverage": result.zero_shot_coverage,
                "zero_shot_unknown": result.zero_shot_unknown,
                "transfer_rate": result.transfer_rate,
                "few_shot_cases": result.few_shot_cases,
                "few_shot_accuracy": result.few_shot_accuracy,
                "few_shot_rules_learned": result.few_shot_rules_learned,
                "few_shot_advantage": result.few_shot_advantage,
            },
            "baseline_comparison": {
                "from_scratch_accuracy": result.scratch_accuracy,
                "from_scratch_rules": result.scratch_rules_learned,
                "few_shot_vs_scratch": result.few_shot_advantage,
            },
            "asp_reasoning_stats": result.reasoning_stats,
        }

        # Save JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Transfer study report saved to: {output_path}")

    def generate_same_domain_report(
        self, result: SameDomainResult, output_path: Path
    ) -> None:
        """Generate same-domain study report."""
        report = {
            "experiment": {
                "type": "same_domain_learning",
                "timestamp": datetime.now().isoformat(),
                "domain": result.domain,
                "reasoning_method": "asp",
            },
            "dataset": {
                "total_cases": result.total_cases,
                "train_cases": result.train_cases,
                "test_cases": result.test_cases,
            },
            "training_results": {
                "rules_learned": result.rules_learned,
                "train_accuracy": result.train_accuracy,
            },
            "test_results": {
                "test_accuracy": result.test_accuracy,
                "test_coverage": result.test_coverage,
                "test_unknown": result.test_unknown,
            },
            "final_evaluation": {
                "final_kb_accuracy": result.final_kb_accuracy,
            },
            "learned_rules": result.learned_rules,
            "asp_reasoning_stats": result.reasoning_stats,
        }

        # Save JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Same-domain study report saved to: {output_path}")


def main():
    """Main entry point for transfer study CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transfer Learning Study for Legal Knowledge (Cross-Domain or Same-Domain)"
    )

    # Mode selection
    parser.add_argument(
        "--same-domain",
        type=str,
        default=None,
        metavar="PATH",
        help="Run same-domain experiment with train/test split on this dataset",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train/test split ratio for same-domain mode (default: 0.7)",
    )

    # Cross-domain arguments
    parser.add_argument(
        "--source-domain",
        type=str,
        default=None,
        help="Path to source domain dataset (for cross-domain mode)",
    )
    parser.add_argument(
        "--target-domain",
        type=str,
        default=None,
        help="Path to target domain dataset (for cross-domain mode)",
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=10,
        help="Number of target cases for few-shot learning (default: 10)",
    )

    # Common arguments
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-haiku-20241022",
        help="LLM model to use (default: claude-3-5-haiku-20241022)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for report (default: auto-generated)",
    )
    parser.add_argument(
        "--use-canonical",
        action="store_true",
        help="Use canonical predicate translation for cross-domain transfer (deterministic only)",
    )
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Use hybrid translation with LLM fallback for unmapped predicates",
    )
    parser.add_argument(
        "--min-llm-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold for LLM translations (default: 0.6)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.same_domain:
        # Same-domain mode
        mode = "same_domain"
    elif args.source_domain and args.target_domain:
        # Cross-domain mode
        mode = "cross_domain"
    else:
        parser.error(
            "Either --same-domain PATH or both --source-domain and --target-domain are required"
        )

    # Create output path if not specified
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    if not args.output:
        if mode == "same_domain":
            domain_name = Path(args.same_domain).name
            args.output = str(
                output_dir / f"same_domain_{domain_name}_{timestamp}.json"
            )
        else:
            args.output = str(output_dir / f"transfer_{timestamp}.json")

    # Run experiment
    study = TransferStudy(
        model=args.model,
        use_canonical_translation=args.use_canonical,
        use_hybrid_translation=args.use_hybrid,
        min_llm_confidence=args.min_llm_confidence,
    )

    if mode == "same_domain":
        logger.info("=" * 80)
        logger.info("Same-Domain Learning Experiment")
        logger.info("=" * 80)

        result = study.run_same_domain_experiment(
            dataset_path=Path(args.same_domain),
            train_ratio=args.train_ratio,
        )

        # Generate report
        study.generate_same_domain_report(result, Path(args.output))

        # Print summary
        print("\n" + "=" * 80)
        print("Same-Domain Learning Results (ASP Reasoning)")
        print("=" * 80)
        print(f"Domain: {result.domain}")
        print(f"Dataset: {result.train_cases} train, {result.test_cases} test")
        print()
        print("Training Results:")
        print(f"  Rules Learned: {result.rules_learned}")
        print(f"  Train Accuracy: {result.train_accuracy:.1%}")
        print()
        print("Test Results (Held-out Set):")
        print(f"  Test Accuracy: {result.test_accuracy:.1%}")
        print(f"  Test Coverage: {result.test_coverage:.1%}")
        print(f"  Unknown Predictions: {result.test_unknown}")
        print()
        print("Final KB Evaluation (All Scenarios):")
        print(f"  Final Accuracy: {result.final_kb_accuracy:.1%}")
        print()
        print("Learned Rules:")
        for i, rule in enumerate(result.learned_rules[:5], 1):
            print(f"  {i}. {rule[:80]}...")
        if len(result.learned_rules) > 5:
            print(f"  ... and {len(result.learned_rules) - 5} more")
        print()
        if result.reasoning_stats:
            print("ASP Reasoning Statistics:")
            print(
                f"  Total Scenarios: {result.reasoning_stats.get('total_scenarios', 0)}"
            )
            print(
                f"  Correct Predictions: {result.reasoning_stats.get('correct_predictions', 0)}"
            )
            print(
                f"  Unknown Predictions: {result.reasoning_stats.get('unknown_predictions', 0)}"
            )
            print()
        print(f"Full report: {args.output}")
        print("=" * 80)

    else:
        # Cross-domain mode
        logger.info("=" * 80)
        logger.info("Cross-Domain Transfer Learning Experiment")
        logger.info("=" * 80)

        result = study.run_transfer_experiment(
            source_dataset=Path(args.source_domain),
            target_dataset=Path(args.target_domain),
            few_shot_count=args.few_shot,
        )

        # Generate report
        study.generate_report(result, Path(args.output))

        # Print summary
        print("\n" + "=" * 80)
        print("Transfer Study Results (ASP Reasoning)")
        print("=" * 80)
        print(f"Source Domain: {result.source_domain}")
        print(f"Target Domain: {result.target_domain}")
        print()
        print("Source Training:")
        print(
            f"  Baseline → Final: {result.source_baseline_accuracy:.1%} → {result.source_final_accuracy:.1%}"
        )
        print(f"  Rules Learned: {result.source_rules_learned}")
        print()
        print("Transfer Results (ASP Reasoning):")
        print(f"  Zero-shot Accuracy: {result.zero_shot_accuracy:.1%}")
        print(f"  Zero-shot Coverage: {result.zero_shot_coverage:.1%}")
        print(f"  Unknown Predictions: {result.zero_shot_unknown}")
        print(f"  Transfer Rate: {result.transfer_rate:.1%}")
        print()
        print("Few-shot Learning:")
        print(
            f"  Few-shot Accuracy: {result.few_shot_accuracy:.1%} ({args.few_shot} cases)"
        )
        print(f"  From-scratch Accuracy: {result.scratch_accuracy:.1%}")
        print(f"  Advantage: {result.few_shot_advantage:+.1%}")
        print()
        if result.reasoning_stats:
            print("ASP Reasoning Statistics:")
            print(
                f"  Total Scenarios: {result.reasoning_stats.get('total_scenarios', 0)}"
            )
            print(
                f"  Correct Predictions: {result.reasoning_stats.get('correct_predictions', 0)}"
            )
            print(
                f"  Unknown Predictions: {result.reasoning_stats.get('unknown_predictions', 0)}"
            )
            print(
                f"  Reasoning Errors: {result.reasoning_stats.get('reasoning_errors', 0)}"
            )
            print()
        print(f"Full report: {args.output}")
        print("=" * 80)


if __name__ == "__main__":
    main()
