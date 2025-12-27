"""
Rule accumulation pipeline for continuous learning.

Processes legal cases to extract, validate, and accumulate rules in the knowledge database.

Issue #273: Continuous Rule Accumulation Pipeline
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from loft.accumulation.conflict_detection import ConflictDetector
from loft.accumulation.schemas import (
    AccumulationResult,
    BatchAccumulationReport,
    CaseData,
    Conflict,
    ConflictResolution,
    RuleCandidate,
)
from loft.knowledge.database import KnowledgeDatabase
from loft.neural.rule_generator import RuleGenerator

logger = logging.getLogger(__name__)


class RuleAccumulationPipeline:
    """
    Pipeline for continuously accumulating rules from legal cases.

    Workflow:
    1. Load case from dataset
    2. Extract rules using neural rule generator
    3. Validate rule quality
    4. Check for conflicts with existing rules
    5. Resolve conflicts and decide whether to add rule
    6. Add rule to knowledge database
    7. Track results and statistics
    """

    def __init__(
        self,
        knowledge_db: KnowledgeDatabase,
        rule_generator: Optional[RuleGenerator] = None,
        conflict_detector: Optional[ConflictDetector] = None,
        min_rule_confidence: float = 0.7,
        auto_resolve_conflicts: bool = True,
    ):
        """
        Initialize accumulation pipeline.

        Args:
            knowledge_db: Knowledge database for storing rules
            rule_generator: Neural rule generator (creates if None)
            conflict_detector: Conflict detector (creates if None)
            min_rule_confidence: Minimum confidence for accepting rules
            auto_resolve_conflicts: Automatically resolve conflicts without human input
        """
        self.knowledge_db = knowledge_db
        self.rule_generator = rule_generator  # Optional, can be None
        self.conflict_detector = conflict_detector or ConflictDetector(
            knowledge_db=knowledge_db
        )
        self.min_rule_confidence = min_rule_confidence
        self.auto_resolve_conflicts = auto_resolve_conflicts

    def process_case(self, case: CaseData) -> AccumulationResult:
        """
        Process a single case to extract and accumulate rules.

        Args:
            case: Case data to process

        Returns:
            Result of accumulation
        """
        start_time = time.time()

        logger.info(f"Processing case: {case.case_id}")

        # Initialize result
        result = AccumulationResult(
            case_id=case.case_id,
            rules_added=0,
            rules_skipped=0,
        )

        # Extract rule candidates from case
        candidates = self._extract_rule_candidates(case)

        if not candidates:
            logger.warning(f"No rules extracted from case {case.case_id}")
            result.skipped_reasons.append("No rules extracted")
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result

        logger.info(f"Extracted {len(candidates)} rule candidates from {case.case_id}")

        # Process each candidate
        for candidate in candidates:
            # Validate rule quality
            if not self._validate_rule_quality(candidate):
                result.rules_skipped += 1
                result.skipped_reasons.append(
                    f"Low confidence: {candidate.confidence:.2f}"
                )
                continue

            # Check for conflicts
            conflicts = self.conflict_detector.find_conflicts(
                new_rule=candidate,
                domain=case.domain,
            )

            if conflicts:
                result.conflicts_found.extend(conflicts)
                logger.info(f"Found {len(conflicts)} conflicts for candidate rule")

                # Resolve conflicts
                resolution = self._resolve_conflicts(candidate, conflicts)

                if not resolution.should_add:
                    result.rules_skipped += 1
                    result.skipped_reasons.append(resolution.reason)
                    continue

                # Archive conflicting rules if needed
                for rule_id in resolution.rules_to_archive:
                    self.knowledge_db.archive_rule(rule_id)
                    logger.info(f"Archived conflicting rule: {rule_id}")

                # Use modified rule if merging
                if resolution.modified_rule:
                    candidate.asp_rule = resolution.modified_rule

            # Add rule to database
            rule_id = self._add_rule_to_database(candidate, case)

            if rule_id:
                result.rules_added += 1
                result.rule_ids.append(rule_id)
                logger.info(f"Added rule {rule_id} from case {case.case_id}")
            else:
                result.rules_skipped += 1
                result.skipped_reasons.append("Failed to add to database")

        # Calculate processing time
        result.processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Completed case {case.case_id}: "
            f"+{result.rules_added} rules, -{result.rules_skipped} skipped"
        )

        return result

    def process_dataset(
        self,
        dataset_path: Path,
        max_cases: Optional[int] = None,
    ) -> BatchAccumulationReport:
        """
        Process all cases in a dataset directory.

        Args:
            dataset_path: Path to dataset directory containing JSON case files
            max_cases: Maximum number of cases to process (None = all)

        Returns:
            Batch accumulation report
        """
        logger.info(f"Processing dataset: {dataset_path}")

        results = []
        case_files = sorted(dataset_path.glob("*.json"))

        if max_cases:
            case_files = case_files[:max_cases]

        logger.info(f"Found {len(case_files)} case files")

        for case_file in case_files:
            try:
                # Load case
                case = self._load_case(case_file)

                # Process case
                result = self.process_case(case)
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process case {case_file}: {e}")
                # Add failed result
                results.append(
                    AccumulationResult(
                        case_id=case_file.stem,
                        rules_added=0,
                        rules_skipped=0,
                        skipped_reasons=[f"Processing error: {e}"],
                    )
                )

        # Create report
        report = BatchAccumulationReport(results=results)

        logger.info(
            f"Dataset processing complete: "
            f"{report.total_cases} cases, "
            f"+{report.total_rules_added} rules, "
            f"{report.total_conflicts} conflicts"
        )

        return report

    def _extract_rule_candidates(self, case: CaseData) -> List[RuleCandidate]:
        """
        Extract rule candidates from case using neural rule generator.

        Args:
            case: Case data

        Returns:
            List of rule candidates
        """
        candidates = []

        if self.rule_generator is None:
            logger.warning("No rule generator available - cannot extract rules")
            return candidates

        try:
            # Use rule generator to extract rules from case
            # The rationale contains the legal principle/reasoning
            generated_rules = self.rule_generator.generate_rules_from_principle(
                principle=case.rationale,
                domain=case.domain or "general",
                predicates_from_case=self._extract_predicates_from_case(case),
            )

            # Convert to rule candidates
            for gen_rule in generated_rules:
                candidate = RuleCandidate(
                    asp_rule=gen_rule.asp_rule,
                    domain=case.domain or "general",
                    confidence=gen_rule.confidence,
                    reasoning=gen_rule.reasoning,
                    source_case_id=case.case_id,
                    principle=case.rationale[:200],  # Truncate long rationales
                )
                candidates.append(candidate)

        except Exception as e:
            logger.error(f"Failed to extract rules from case {case.case_id}: {e}")

        return candidates

    def _extract_predicates_from_case(self, case: CaseData) -> List[str]:
        """
        Extract predicates used in case facts.

        Args:
            case: Case data

        Returns:
            List of predicate names
        """
        predicates = set()

        # Parse asp_facts to extract predicates
        # Split by both newlines and periods to handle different formats
        statements = []
        for line in case.asp_facts.split("\n"):
            # Further split by periods for single-line facts
            statements.extend([s.strip() for s in line.split(".") if s.strip()])

        for statement in statements:
            if not statement or statement.startswith("%"):
                continue

            # Extract predicate name (text before opening paren)
            if "(" in statement:
                predicate = statement.split("(")[0].strip()
                predicates.add(predicate)

        return sorted(predicates)

    def _validate_rule_quality(self, candidate: RuleCandidate) -> bool:
        """
        Validate that rule meets quality thresholds.

        Args:
            candidate: Rule candidate to validate

        Returns:
            True if rule passes validation
        """
        # Check confidence threshold
        if candidate.confidence < self.min_rule_confidence:
            logger.debug(
                f"Rule rejected: confidence {candidate.confidence:.2f} "
                f"< threshold {self.min_rule_confidence:.2f}"
            )
            return False

        # Check that rule is not empty
        if not candidate.asp_rule or not candidate.asp_rule.strip():
            logger.debug("Rule rejected: empty rule text")
            return False

        # Check basic ASP syntax (has :- or ends with .)
        rule_text = candidate.asp_rule.strip()
        if not (":-" in rule_text or rule_text.endswith(".")):
            logger.debug("Rule rejected: invalid ASP syntax")
            return False

        return True

    def _resolve_conflicts(
        self,
        candidate: RuleCandidate,
        conflicts: List[Conflict],
    ) -> ConflictResolution:
        """
        Resolve conflicts and decide whether to add rule.

        Args:
            candidate: Rule candidate
            conflicts: List of conflicts

        Returns:
            Resolution decision
        """
        if not self.auto_resolve_conflicts:
            # Manual resolution required
            return ConflictResolution(
                should_add=False,
                action="skip",
                reason="Manual conflict resolution required",
            )

        # Analyze conflict severity
        max_severity = max(c.severity for c in conflicts)
        conflict_types = {c.conflict_type for c in conflicts}

        # Handle contradictions (most severe)
        if "contradiction" in conflict_types:
            # If new rule has high confidence, consider replacing
            if candidate.confidence >= 0.9:
                conflicting_ids = [
                    c.existing_rule_id
                    for c in conflicts
                    if c.conflict_type == "contradiction"
                ]
                return ConflictResolution(
                    should_add=True,
                    action="replace",
                    reason=f"New rule has high confidence ({candidate.confidence:.2f}), replacing contradicting rules",
                    rules_to_archive=conflicting_ids,
                )
            else:
                return ConflictResolution(
                    should_add=False,
                    action="skip",
                    reason="Contradiction detected with existing rule",
                )

        # Handle subsumption
        if "subsumption" in conflict_types and max_severity >= 0.8:
            return ConflictResolution(
                should_add=False,
                action="skip",
                reason="Rule is substantially subsumed by existing rule",
            )

        # Handle inconsistency
        if "inconsistency" in conflict_types:
            return ConflictResolution(
                should_add=False,
                action="skip",
                reason="Rule creates logical inconsistency with existing rules",
            )

        # Minor conflicts - add anyway
        return ConflictResolution(
            should_add=True,
            action="add",
            reason="Conflicts are minor, adding rule",
        )

    def _add_rule_to_database(
        self,
        candidate: RuleCandidate,
        case: CaseData,
    ) -> Optional[str]:
        """
        Add rule to knowledge database.

        Args:
            candidate: Rule candidate to add
            case: Source case

        Returns:
            Rule ID if added successfully, None otherwise
        """
        try:
            rule_id = self.knowledge_db.add_rule(
                asp_rule=candidate.asp_rule,
                domain=candidate.domain,
                confidence=candidate.confidence,
                reasoning=candidate.reasoning,
                source_case_id=candidate.source_case_id,
                principle=candidate.principle,
            )

            return rule_id

        except Exception as e:
            logger.error(f"Failed to add rule to database: {e}")
            return None

    def _load_case(self, case_file: Path) -> CaseData:
        """
        Load case from JSON file.

        Args:
            case_file: Path to JSON case file

        Returns:
            Parsed case data
        """
        with open(case_file) as f:
            case_dict = json.load(f)

        return CaseData.from_dict(case_dict)

    def get_accumulation_stats(self) -> dict:
        """
        Get statistics about accumulated rules.

        Returns:
            Statistics dictionary
        """
        stats = self.knowledge_db.get_database_stats()

        return {
            "total_rules": stats.total_rules,
            "active_rules": stats.active_rules,
            "archived_rules": stats.archived_rules,
            "domains": stats.domains,
            "avg_confidence": stats.avg_confidence,
        }
