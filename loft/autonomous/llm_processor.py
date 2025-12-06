"""
LLM Case Processor for Autonomous Runs.

Provides LLM-powered case processing for the autonomous test harness,
including fact extraction from legal cases, rule generation, validation,
and incorporation.
"""

import logging
import os
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from loft.batch.schemas import CaseResult, CaseStatus

if TYPE_CHECKING:
    from loft.neural.llm_interface import LLMInterface
    from loft.neural.rule_generator import RuleGenerator
    from loft.symbolic.asp_core import ASPCore
    from loft.validation.validation_pipeline import ValidationPipeline

logger = logging.getLogger(__name__)


class LLMCaseProcessor:
    """
    LLM-powered case processor for autonomous runs.

    Processes legal cases by:
    1. Extracting facts and legal issues from case text
    2. Generating ASP rules for the case domain
    3. Validating generated rules
    4. Tracking metrics for metareasoning analysis
    """

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        extraction_prompt_template: Optional[str] = None,
    ):
        """
        Initialize LLM case processor.

        Args:
            model: LLM model to use for processing
            extraction_prompt_template: Optional custom prompt template
        """
        self.model = model
        self._llm: Optional["LLMInterface"] = None
        self._rule_generator: Optional["RuleGenerator"] = None
        self._validation_pipeline: Optional["ValidationPipeline"] = None
        self._asp_core: Optional["ASPCore"] = None
        self._extract_predicates: Optional[Callable[[str], List[str]]] = None
        self._initialized = False

        # Metrics tracking
        self._total_llm_calls = 0
        self._total_tokens_used = 0
        self._total_cost_usd = 0.0
        self._processing_times: List[float] = []

        # Extraction prompt template
        self._extraction_template = (
            extraction_prompt_template or self._default_extraction_template()
        )

    def _default_extraction_template(self) -> str:
        """Return default extraction prompt template."""
        return """Analyze the following legal case and extract:

1. **Case Facts**: Key factual elements relevant to legal analysis
2. **Legal Issues**: Primary legal questions or doctrines involved
3. **Domain**: Legal domain (e.g., contracts, torts, property, procedural)
4. **Key Terms**: Important legal terms and their relationships

Case Text:
{case_text}

Domain Hint: {domain}

Respond in JSON format:
{{
    "facts": ["fact1", "fact2", ...],
    "legal_issues": ["issue1", "issue2", ...],
    "domain": "domain_name",
    "key_terms": {{"term": "definition", ...}},
    "asp_predicates": ["predicate1(arg1, arg2)", ...]
}}
"""

    def initialize(self) -> None:
        """Initialize LLM components (lazy initialization)."""
        if self._initialized:
            return

        logger.info(f"Initializing LLM components with model={self.model}")

        # Check for API key
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
            )

        # Initialize LLM interface
        from loft.neural.providers import AnthropicProvider
        from loft.neural.llm_interface import LLMInterface
        from loft.neural.rule_generator import RuleGenerator, extract_predicates_from_asp_facts
        from loft.symbolic.asp_core import ASPCore
        from loft.validation.validation_pipeline import ValidationPipeline
        from loft.validation.asp_validators import ASPSyntaxValidator
        from loft.validation.semantic_validator import SemanticValidator
        from loft.validation.empirical_validator import EmpiricalValidator

        provider = AnthropicProvider(api_key=api_key, model=self.model)
        self._llm = LLMInterface(provider=provider)
        self._asp_core = ASPCore()
        self._rule_generator = RuleGenerator(llm=self._llm, asp_core=self._asp_core)
        # Store predicate extraction function for gap-filling alignment (issue #166)
        self._extract_predicates = extract_predicates_from_asp_facts

        # Initialize validation pipeline
        self._validation_pipeline = ValidationPipeline(
            syntax_validator=ASPSyntaxValidator(),
            semantic_validator=SemanticValidator(),
            empirical_validator=EmpiricalValidator(),
            min_confidence=0.6,
        )

        self._initialized = True
        logger.info("LLM components initialized successfully")

    def process_case(
        self,
        case: Dict[str, Any],
        accumulated_rules: List[str],
    ) -> CaseResult:
        """
        Process a single legal case using LLM.

        Args:
            case: Case data dictionary with 'id', 'facts', 'text', 'domain', etc.
            accumulated_rules: List of rule IDs already accumulated

        Returns:
            CaseResult with processing outcome
        """
        start_time = time.time()
        case_id = case.get("id", str(uuid.uuid4())[:8])

        try:
            # Ensure initialization
            if not self._initialized:
                self.initialize()

            # Extract case information
            case_text = case.get("text", "") or case.get("facts", "")
            domain = case.get("domain", "general")

            if not case_text:
                logger.warning(f"Case {case_id} has no text or facts, skipping")
                return CaseResult(
                    case_id=case_id,
                    status=CaseStatus.SKIPPED,
                    processed_at=datetime.now(),
                    processing_time_ms=(time.time() - start_time) * 1000,
                    error_message="No case text or facts available",
                )

            # Step 1: Extract facts using LLM
            extraction = self._extract_facts(case_text, domain, case_id)
            self._total_llm_calls += 1

            # Step 2: Generate candidate rules if gaps identified
            rules_generated = 0
            rules_accepted = 0
            rules_rejected = 0
            generated_rule_ids: List[str] = []

            if extraction.get("asp_predicates"):
                # Extract dataset predicates from case facts for alignment (issue #166)
                facts_text = "\n".join(extraction.get("facts", []))
                assert self._extract_predicates is not None
                dataset_predicates = self._extract_predicates(facts_text)
                logger.debug(
                    f"Extracted {len(dataset_predicates)} predicates from case "
                    f"{case_id} facts: {dataset_predicates[:5]}..."
                    if len(dataset_predicates) > 5
                    else f"Extracted predicates: {dataset_predicates}"
                )

                # Try to generate rules from extracted predicates
                assert self._rule_generator is not None
                assert self._validation_pipeline is not None
                for predicate in extraction.get("asp_predicates", [])[:3]:
                    try:
                        # Use fill_knowledge_gap to generate candidate rules
                        # Pass dataset_predicates for predicate alignment (issue #166)
                        gap_response = self._rule_generator.fill_knowledge_gap(
                            gap_description=f"Case {case_id}: Need rule for {predicate}",
                            missing_predicate=predicate,
                            context={
                                "facts": facts_text,
                                "domain": extraction.get("domain", "general"),
                                "legal_issues": ", ".join(extraction.get("legal_issues", [])),
                            },
                            dataset_predicates=dataset_predicates,
                        )
                        rules_generated += 1
                        self._total_llm_calls += 1

                        # Get the recommended candidate
                        if gap_response.candidates:
                            candidate = gap_response.candidates[gap_response.recommended_index]

                            # Validate the candidate rule
                            report = self._validation_pipeline.validate_rule(
                                rule_asp=candidate.rule.asp_rule,
                                rule_id=f"rule_{case_id}_{rules_generated}",
                                proposer_reasoning=candidate.rule.reasoning,
                                test_cases=[],
                            )

                            if report.final_decision == "accept":
                                rules_accepted += 1
                                generated_rule_ids.append(f"rule_{case_id}_{rules_generated}")
                                logger.info(
                                    f"Accepted rule from case {case_id}: "
                                    f"{candidate.rule.asp_rule[:50]}..."
                                )
                            else:
                                rules_rejected += 1
                                logger.debug(f"Rejected rule from case {case_id}")
                        else:
                            logger.warning(f"No candidates generated for {case_id}/{predicate}")
                            rules_rejected += 1

                    except Exception as e:
                        logger.warning(f"Error generating rule for {case_id}: {e}")
                        rules_rejected += 1

            processing_time_ms = (time.time() - start_time) * 1000
            self._processing_times.append(processing_time_ms)

            return CaseResult(
                case_id=case_id,
                status=CaseStatus.SUCCESS,
                processed_at=datetime.now(),
                processing_time_ms=processing_time_ms,
                rules_generated=rules_generated,
                rules_accepted=rules_accepted,
                rules_rejected=rules_rejected,
                prediction_correct=True,  # Assume success for now
                confidence=extraction.get("confidence", 0.8),
                generated_rule_ids=generated_rule_ids,
            )

        except Exception as e:
            logger.error(f"Error processing case {case_id}: {e}")
            return CaseResult(
                case_id=case_id,
                status=CaseStatus.FAILED,
                processed_at=datetime.now(),
                processing_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )

    def _extract_facts(
        self,
        case_text: str,
        domain: str,
        case_id: str,
    ) -> Dict[str, Any]:
        """
        Extract facts and legal issues from case text using LLM.

        Args:
            case_text: Raw case text
            domain: Domain hint
            case_id: Case identifier

        Returns:
            Dictionary with extracted facts, issues, predicates
        """
        # Truncate very long texts
        max_length = 4000
        if len(case_text) > max_length:
            case_text = case_text[:max_length] + "..."

        prompt = self._extraction_template.format(
            case_text=case_text,
            domain=domain,
        )

        try:
            assert self._llm is not None
            response = self._llm.query(
                question=prompt,
                max_tokens=1000,
            )

            # Parse JSON response
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response.raw_text)
            if json_match:
                extraction = json.loads(json_match.group())
                extraction["confidence"] = response.confidence
                return extraction
            else:
                logger.warning(f"Could not parse JSON from LLM response for case {case_id}")
                return {
                    "facts": [],
                    "legal_issues": [],
                    "domain": domain,
                    "key_terms": {},
                    "asp_predicates": [],
                    "confidence": 0.5,
                }

        except Exception as e:
            logger.error(f"Error extracting facts from case {case_id}: {e}")
            return {
                "facts": [],
                "legal_issues": [],
                "domain": domain,
                "key_terms": {},
                "asp_predicates": [],
                "confidence": 0.0,
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        avg_time = (
            sum(self._processing_times) / len(self._processing_times)
            if self._processing_times
            else 0.0
        )

        return {
            "total_llm_calls": self._total_llm_calls,
            "total_tokens_used": self._total_tokens_used,
            "total_cost_usd": self._total_cost_usd,
            "cases_processed": len(self._processing_times),
            "avg_processing_time_ms": avg_time,
            "model": self.model,
        }

    def create_process_fn(self) -> Callable[[Dict[str, Any], List[str]], CaseResult]:
        """
        Create a process function compatible with BatchLearningHarness.

        Returns:
            Callable that processes a case and returns CaseResult
        """
        return self.process_case


def create_llm_processor(
    model: str = "claude-3-5-haiku-20241022",
) -> LLMCaseProcessor:
    """
    Factory function to create an LLM case processor.

    Args:
        model: LLM model to use

    Returns:
        Configured LLMCaseProcessor instance
    """
    return LLMCaseProcessor(model=model)
