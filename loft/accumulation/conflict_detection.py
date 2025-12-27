"""
Conflict detection for rule accumulation pipeline.

Detects contradictions and inconsistencies between new and existing rules.

Issue #273: Continuous Rule Accumulation Pipeline
"""

import logging
from typing import List, Optional

import clingo

from loft.accumulation.schemas import Conflict, RuleCandidate
from loft.symbolic.asp_core import ASPCore
from loft.knowledge.database import KnowledgeDatabase
from loft.knowledge.models import LegalRule

logger = logging.getLogger(__name__)


class ConflictDetector:
    """
    Detect contradictions between new and existing rules.

    Uses ASP solver to check for logical conflicts:
    - Direct contradiction: Rules directly contradict each other
    - Subsumption: One rule makes the other redundant
    - Inconsistency: Rules lead to inconsistent conclusions
    """

    def __init__(
        self,
        knowledge_db: Optional[KnowledgeDatabase] = None,
        subsumption_threshold: float = 0.9,
    ):
        """
        Initialize conflict detector.

        Args:
            knowledge_db: Knowledge database for retrieving rules
            subsumption_threshold: Similarity threshold for subsumption detection (0.0-1.0)
        """
        self.knowledge_db = knowledge_db
        self.subsumption_threshold = subsumption_threshold

    def find_conflicts(
        self,
        new_rule: RuleCandidate,
        domain: Optional[str] = None,
        existing_rules: Optional[List[LegalRule]] = None,
    ) -> List[Conflict]:
        """
        Detect if new rule conflicts with existing rules.

        Args:
            new_rule: Candidate rule to check
            domain: Legal domain to check (if None, uses new_rule.domain)
            existing_rules: Existing rules to check against (if None, retrieves from DB)

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Get existing rules if not provided
        if existing_rules is None:
            if self.knowledge_db is None:
                logger.warning("No knowledge database or existing rules provided")
                return conflicts

            domain = domain or new_rule.domain
            existing_rules = self.knowledge_db.search_rules(
                domain=domain,
                is_active=True,
                min_confidence=0.5,  # Check against all reasonably confident rules
            )

        if not existing_rules:
            logger.info("No existing rules to check conflicts against")
            return conflicts

        # Check each existing rule for conflicts
        for existing_rule in existing_rules:
            conflict = self._check_conflict_pair(new_rule, existing_rule)
            if conflict:
                conflicts.append(conflict)

        logger.info(f"Found {len(conflicts)} conflicts for new rule")
        return conflicts

    def _check_conflict_pair(
        self, new_rule: RuleCandidate, existing_rule: LegalRule
    ) -> Optional[Conflict]:
        """
        Check if two rules conflict.

        Args:
            new_rule: New candidate rule
            existing_rule: Existing rule from database

        Returns:
            Conflict object if conflict detected, None otherwise
        """
        # Check for direct contradiction
        contradiction = self._check_contradiction(new_rule, existing_rule)
        if contradiction:
            return contradiction

        # Check for subsumption
        subsumption = self._check_subsumption(new_rule, existing_rule)
        if subsumption:
            return subsumption

        # Check for inconsistent conclusions
        inconsistency = self._check_inconsistency(new_rule, existing_rule)
        if inconsistency:
            return inconsistency

        return None

    def _check_contradiction(
        self, new_rule: RuleCandidate, existing_rule: LegalRule
    ) -> Optional[Conflict]:
        """
        Check for direct logical contradiction.

        A contradiction occurs when rules directly negate each other,
        like: rule1: p(X) :- q(X). and rule2: not p(X) :- q(X).

        Args:
            new_rule: New candidate rule
            existing_rule: Existing rule

        Returns:
            Conflict if contradiction found, None otherwise
        """
        # Extract head predicates from both rules
        new_head = self._extract_head(new_rule.asp_rule)
        existing_head = self._extract_head(existing_rule.asp_rule)

        if not new_head or not existing_head:
            return None

        # Check if one is negation of the other
        is_contradiction = False
        explanation = ""

        if new_head.startswith("not ") and new_head[4:] == existing_head:
            is_contradiction = True
            explanation = (
                f"New rule concludes 'not {existing_head}' while "
                f"existing rule concludes '{existing_head}'"
            )
        elif existing_head.startswith("not ") and existing_head[4:] == new_head:
            is_contradiction = True
            explanation = (
                f"New rule concludes '{new_head}' while "
                f"existing rule concludes 'not {new_head}'"
            )

        if is_contradiction:
            return Conflict(
                conflict_type="contradiction",
                new_rule=new_rule.asp_rule,
                existing_rule_id=existing_rule.rule_id,
                existing_rule=existing_rule.asp_rule,
                explanation=explanation,
                severity=1.0,  # Contradictions are always severe
            )

        return None

    def _check_subsumption(
        self, new_rule: RuleCandidate, existing_rule: LegalRule
    ) -> Optional[Conflict]:
        """
        Check if one rule makes the other redundant.

        Subsumption occurs when one rule is more general than another,
        making the more specific rule unnecessary.

        Args:
            new_rule: New candidate rule
            existing_rule: Existing rule

        Returns:
            Conflict if subsumption found, None otherwise
        """
        # Calculate similarity between rules
        similarity = self._calculate_rule_similarity(
            new_rule.asp_rule, existing_rule.asp_rule
        )

        if similarity >= self.subsumption_threshold:
            # Determine which rule is more general based on confidence
            # Higher confidence rule is considered more authoritative
            new_more_general = new_rule.confidence > existing_rule.confidence

            explanation = (
                f"Rules are {similarity:.1%} similar. "
                f"{'New rule' if new_more_general else 'Existing rule'} "
                f"appears to subsume the other."
            )

            # Calculate severity: scales from threshold (0.5) to perfect match (1.0)
            # For identical rules (1.0 similarity), severity is 1.0
            severity_range = 1.0 - self.subsumption_threshold
            normalized_diff = (similarity - self.subsumption_threshold) / severity_range
            severity = 0.5 + (normalized_diff * 0.5)

            return Conflict(
                conflict_type="subsumption",
                new_rule=new_rule.asp_rule,
                existing_rule_id=existing_rule.rule_id,
                existing_rule=existing_rule.asp_rule,
                explanation=explanation,
                severity=severity,
            )

        return None

    def _check_inconsistency(
        self, new_rule: RuleCandidate, existing_rule: LegalRule
    ) -> Optional[Conflict]:
        """
        Check if rules lead to inconsistent conclusions.

        Uses ASP solver to check if combining both rules leads to
        logical inconsistency (UNSAT).

        Args:
            new_rule: New candidate rule
            existing_rule: Existing rule

        Returns:
            Conflict if inconsistency found, None otherwise
        """
        try:
            # Create ASP program with both rules
            asp_core = ASPCore()
            asp_core.control = clingo.Control()

            program = f"{existing_rule.asp_rule}\n{new_rule.asp_rule}"
            asp_core.control.add("base", [], program)
            asp_core.control.ground([("base", [])])

            # Check if program is satisfiable
            is_satisfiable = False

            def on_model(model):
                nonlocal is_satisfiable
                is_satisfiable = True

            solve_result = asp_core.control.solve(on_model=on_model)

            # If UNSAT, rules are inconsistent
            if solve_result.unsatisfiable:
                return Conflict(
                    conflict_type="inconsistency",
                    new_rule=new_rule.asp_rule,
                    existing_rule_id=existing_rule.rule_id,
                    existing_rule=existing_rule.asp_rule,
                    explanation="Rules together produce unsatisfiable program (UNSAT)",
                    severity=0.9,  # Very severe
                )

        except RuntimeError as e:
            logger.warning(f"Failed to check inconsistency: {e}")
            # Don't treat parsing errors as conflicts
            return None

        return None

    def _extract_head(self, rule: str) -> Optional[str]:
        """
        Extract head predicate from ASP rule.

        Args:
            rule: ASP rule text

        Returns:
            Head predicate or None if not found
        """
        rule = rule.strip()
        if ":-" in rule:
            head = rule.split(":-")[0].strip()
            # Remove trailing period if present
            if head.endswith("."):
                head = head[:-1].strip()
            return head
        elif rule.endswith("."):
            # Fact, not a rule
            return rule[:-1].strip()
        return None

    def _calculate_rule_similarity(self, rule1: str, rule2: str) -> float:
        """
        Calculate similarity between two rules.

        Uses simple token-based Jaccard similarity.

        Args:
            rule1: First rule
            rule2: Second rule

        Returns:
            Similarity score (0.0-1.0)
        """
        # Tokenize rules
        tokens1 = set(self._tokenize_rule(rule1))
        tokens2 = set(self._tokenize_rule(rule2))

        if not tokens1 or not tokens2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union) if union else 0.0

    def _tokenize_rule(self, rule: str) -> List[str]:
        """
        Tokenize ASP rule into meaningful parts.

        Args:
            rule: ASP rule text

        Returns:
            List of tokens
        """
        # Remove common ASP syntax
        rule = rule.replace(":-", " ").replace(".", " ").replace(",", " ")
        rule = rule.replace("(", " ").replace(")", " ")

        # Split into tokens
        tokens = [t.strip() for t in rule.split() if t.strip()]

        return tokens

    def suggest_resolution(self, conflict: Conflict) -> str:
        """
        Suggest how to resolve a conflict.

        Args:
            conflict: Detected conflict

        Returns:
            Resolution suggestion
        """
        if conflict.conflict_type == "contradiction":
            return (
                "Contradiction detected. Review both rules carefully. "
                "Consider: (1) Skip new rule, (2) Replace existing rule if "
                "new rule has higher confidence, (3) Request human review."
            )

        elif conflict.conflict_type == "subsumption":
            if conflict.severity < 0.7:
                return (
                    "Minor subsumption. Consider adding new rule anyway "
                    "as it may provide additional nuance."
                )
            else:
                return (
                    "Significant subsumption. Consider skipping new rule "
                    "to avoid redundancy."
                )

        elif conflict.conflict_type == "inconsistency":
            return (
                "Rules produce logical inconsistency. This is serious. "
                "Recommend human review to determine which rule is correct."
            )

        return "Unknown conflict type - manual review recommended."
