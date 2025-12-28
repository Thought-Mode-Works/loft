"""
Case analyzer: orchestrates parsing, extraction, and rule generation.

Main interface for analyzing case documents and generating rules.

Issue #276: Case Analysis and Rule Extraction
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

from loft.case_analysis.extractor import MetadataExtractor, PrincipleExtractor
from loft.case_analysis.parser import CaseDocumentParser
from loft.case_analysis.schemas import (
    CaseAnalysisResult,
    CaseDocument,
    CaseMetadata,
    ExtractedRule,
    LegalPrinciple,
)
from loft.neural.llm_interface import LLMInterface
from loft.neural.rule_generator import RuleGenerator

logger = logging.getLogger(__name__)


class CaseAnalyzer:
    """
    Main case analysis orchestrator.

    Coordinates parsing, metadata extraction, principle identification,
    and rule generation for legal case documents.
    """

    def __init__(
        self,
        llm: LLMInterface,
        min_principle_confidence: float = 0.5,
        min_rule_confidence: float = 0.6,
        max_principles_per_case: int = 10,
        validate_rules: bool = True,
    ):
        """
        Initialize case analyzer.

        Args:
            llm: Language model interface
            min_principle_confidence: Minimum confidence for principles
            min_rule_confidence: Minimum confidence for generated rules
            max_principles_per_case: Max principles to extract per case
            validate_rules: Whether to validate generated ASP rules
        """
        self.llm = llm
        self.min_principle_confidence = min_principle_confidence
        self.min_rule_confidence = min_rule_confidence
        self.max_principles_per_case = max_principles_per_case
        self.validate_rules = validate_rules

        # Initialize components
        self.parser = CaseDocumentParser()
        self.metadata_extractor = MetadataExtractor(llm)
        self.principle_extractor = PrincipleExtractor(
            llm,
            min_confidence=min_principle_confidence,
            max_principles=max_principles_per_case,
        )
        self.rule_generator = RuleGenerator(llm, domain="legal")

    def analyze_file(self, file_path: Path) -> CaseAnalysisResult:
        """
        Analyze a case document file.

        Args:
            file_path: Path to case document

        Returns:
            Complete analysis result
        """
        start_time = time.time()

        try:
            # Parse document
            logger.info(f"Parsing {file_path}")
            case_doc = self.parser.parse_file(file_path)

            # Analyze document
            result = self.analyze_document(case_doc)

            # Update timing
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time

            logger.info(
                f"Analysis complete: {result.principle_count} principles, "
                f"{result.rule_count} rules ({processing_time:.0f}ms)"
            )

            return result

        except Exception as e:
            logger.error(f"Analysis failed for {file_path}: {e}")
            processing_time = (time.time() - start_time) * 1000

            return CaseAnalysisResult(
                case_id=file_path.stem,
                metadata=CaseMetadata(case_id=file_path.stem),
                principles=[],
                rules=[],
                processing_time_ms=processing_time,
                errors=[str(e)],
            )

    def analyze_document(self, case_doc: CaseDocument) -> CaseAnalysisResult:
        """
        Analyze a parsed case document.

        Args:
            case_doc: Parsed case document

        Returns:
            Complete analysis result
        """
        start_time = time.time()
        errors = []
        case_id = case_doc.case_id or "unknown"

        try:
            # Extract metadata
            logger.info(f"Extracting metadata for {case_id}")
            metadata = self.metadata_extractor.extract_metadata(case_doc)

        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            metadata = CaseMetadata(case_id=case_id)
            errors.append(f"Metadata extraction failed: {e}")

        try:
            # Extract principles
            logger.info(f"Extracting principles for {case_id}")
            principles = self.principle_extractor.extract_principles(case_doc, metadata)

        except Exception as e:
            logger.error(f"Principle extraction failed: {e}")
            principles = []
            errors.append(f"Principle extraction failed: {e}")

        # Generate rules from principles
        rules = []
        for principle in principles:
            try:
                logger.debug(
                    f"Generating rule from principle: {principle.principle_text[:50]}..."
                )
                rule = self._generate_rule_from_principle(principle, case_id, metadata)

                if rule and rule.confidence >= self.min_rule_confidence:
                    rules.append(rule)
                else:
                    logger.debug(
                        f"Filtered low-confidence rule: {rule.confidence if rule else 'None'}"
                    )

            except Exception as e:
                logger.warning(f"Rule generation failed for principle: {e}")
                errors.append(f"Rule generation failed: {e}")
                continue

        processing_time = (time.time() - start_time) * 1000

        return CaseAnalysisResult(
            case_id=case_id,
            metadata=metadata,
            principles=principles,
            rules=rules,
            processing_time_ms=processing_time,
            errors=errors,
        )

    def _generate_rule_from_principle(
        self,
        principle: LegalPrinciple,
        case_id: str,
        metadata: CaseMetadata,
    ) -> Optional[ExtractedRule]:
        """
        Generate ASP rule from legal principle.

        Args:
            principle: Legal principle
            case_id: Source case ID
            metadata: Case metadata

        Returns:
            Extracted rule or None if generation fails
        """
        try:
            # Generate rule using RuleGenerator
            generated = self.rule_generator.generate_from_principle(
                principle_text=principle.principle_text,
                existing_predicates=None,  # Could pass domain predicates here
                constraints=None,
            )

            if not generated.asp_rule:
                return None

            # Validate if requested
            validation_passed = True
            if self.validate_rules:
                validation_passed = self._validate_asp_syntax(generated.asp_rule)

            # Extract predicates from rule
            predicates = self._extract_predicates(generated.asp_rule)

            return ExtractedRule(
                asp_rule=generated.asp_rule,
                principle=principle,
                domain=principle.domain,
                confidence=generated.confidence,
                reasoning=generated.reasoning or principle.reasoning or "",
                predicates_used=predicates,
                source_case_id=case_id,
                validation_passed=validation_passed,
            )

        except Exception as e:
            logger.warning(f"Failed to generate rule: {e}")
            return None

    def _validate_asp_syntax(self, asp_rule: str) -> bool:
        """
        Validate ASP rule syntax.

        Basic syntax checking - looks for common patterns.

        Args:
            asp_rule: ASP rule text

        Returns:
            True if syntax appears valid
        """
        # Basic checks
        if not asp_rule or not asp_rule.strip():
            return False

        # Should contain :- for rules or end with . for facts
        if ":-" not in asp_rule and not asp_rule.rstrip().endswith("."):
            return False

        # Should have balanced parentheses
        if asp_rule.count("(") != asp_rule.count(")"):
            return False

        # Check for rule structure: head :- body.
        if ":-" in asp_rule:
            parts = asp_rule.split(":-")
            if len(parts) != 2:
                return False

            head, body = parts
            # Head should have predicate
            if not head.strip() or "(" not in head:
                return False

            # Body should end with .
            if not body.rstrip().endswith("."):
                return False

        return True

    def _extract_predicates(self, asp_rule: str) -> List[str]:
        """
        Extract predicate names from ASP rule.

        Args:
            asp_rule: ASP rule text

        Returns:
            List of predicate names
        """
        import re

        # Find all predicate patterns: word(
        pattern = r"\b([a-z_][a-z0-9_]*)\s*\("
        matches = re.findall(pattern, asp_rule.lower())

        # Return unique predicates
        return list(set(matches))

    def analyze_batch(self, file_paths: List[Path]) -> List[CaseAnalysisResult]:
        """
        Analyze multiple case files.

        Args:
            file_paths: List of paths to case documents

        Returns:
            List of analysis results
        """
        results = []

        for file_path in file_paths:
            logger.info(
                f"Analyzing {file_path.name} ({len(results) + 1}/{len(file_paths)})"
            )
            result = self.analyze_file(file_path)
            results.append(result)

        # Log summary
        successful = sum(1 for r in results if r.success)
        total_rules = sum(r.rule_count for r in results)

        logger.info(
            f"Batch analysis complete: {successful}/{len(results)} successful, "
            f"{total_rules} total rules extracted"
        )

        return results
