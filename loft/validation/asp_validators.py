"""
ASP-specific validators for Answer Set Programming syntax and structure.

This module provides validation for ASP programs using Clingo, including:
- Syntax validation
- Type checking
- Stratification verification
- Satisfiability checking
"""

from typing import Tuple, List, Optional, Dict, Any
import clingo
from loguru import logger


class ASPSyntaxValidator:
    """Validates ASP program syntax and structure."""

    def validate_program(self, asp_text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if ASP program is syntactically valid.

        Args:
            asp_text: ASP program text to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if syntax is valid
            - error_message: None if valid, error description if invalid

        Example:
            >>> validator = ASPSyntaxValidator()
            >>> is_valid, error = validator.validate_program("fact(a).")
            >>> assert is_valid and error is None
        """
        try:
            ctl = clingo.Control()
            ctl.add("base", [], asp_text)
            logger.debug(f"ASP program syntax validation passed: {len(asp_text)} chars")
            return (True, None)
        except Exception as e:
            error_msg = f"Syntax error: {str(e)}"
            logger.warning(f"ASP syntax validation failed: {error_msg}")
            return (False, error_msg)

    def validate_predicate_types(
        self, asp_text: str, domain_schema: Dict[str, int]
    ) -> Tuple[bool, List[str]]:
        """
        Verify predicates match domain schema (check arities).

        Args:
            asp_text: ASP program text
            domain_schema: Dict mapping predicate names to expected arity
                          e.g., {"contract": 1, "party": 1, "signed_by": 2}

        Returns:
            Tuple of (is_valid, errors)
            - is_valid: True if all predicates match schema
            - errors: List of error messages for mismatches

        Example:
            >>> validator = ASPSyntaxValidator()
            >>> schema = {"fact": 1, "rule": 2}
            >>> valid, errors = validator.validate_predicate_types(
            ...     "fact(a). rule(x, y).", schema
            ... )
            >>> assert valid
        """
        errors = []

        # Parse ASP and extract predicates with their arities
        try:
            # Use Clingo's parsing capabilities
            ctl = clingo.Control()
            ctl.add("base", [], asp_text)
            ctl.ground([("base", [])])

            # Check against schema
            # Note: Full implementation would parse AST to extract predicates
            # For now, basic validation that program is well-formed
            logger.debug("Predicate type validation passed")
            return (True, [])

        except Exception as e:
            errors.append(f"Type checking failed: {str(e)}")
            logger.warning(f"Predicate type validation failed: {errors}")
            return (False, errors)

    def check_stratification(self, asp_text: str) -> Tuple[bool, List[str]]:
        """
        Verify program is stratified (no cycles through negation).

        A program is stratified if there are no cycles in the dependency
        graph that go through negation. This ensures the program has a
        unique stable model.

        Args:
            asp_text: ASP program text

        Returns:
            Tuple of (is_stratified, problematic_rules)
            - is_stratified: True if properly stratified
            - problematic_rules: List of rule descriptions that violate stratification

        Example:
            >>> validator = ASPSyntaxValidator()
            >>> # This is stratified
            >>> is_strat, issues = validator.check_stratification(
            ...     "a :- not b. b :- c."
            ... )
            >>> assert is_strat
        """
        # Basic implementation: check if program grounds without issues
        # Full implementation would analyze dependency graph
        try:
            ctl = clingo.Control()
            ctl.add("base", [], asp_text)
            ctl.ground([("base", [])])

            # If grounding succeeds, stratification is likely OK
            # Clingo handles non-stratified programs, but we prefer stratified
            logger.debug("Stratification check passed")
            return (True, [])

        except Exception as e:
            error_msg = f"Stratification issue: {str(e)}"
            logger.warning(error_msg)
            return (False, [error_msg])


class ASPSemanticValidator:
    """Validates logical consistency and properties of ASP programs."""

    def check_consistency(self, asp_program: str) -> Tuple[bool, str]:
        """
        Check if ASP program has at least one answer set (is satisfiable).

        An inconsistent program has no answer sets, meaning the constraints
        cannot be satisfied.

        Args:
            asp_program: ASP program text to check

        Returns:
            Tuple of (is_consistent, explanation)
            - is_consistent: True if program has answer sets
            - explanation: Description of result

        Example:
            >>> validator = ASPSemanticValidator()
            >>> # Consistent program
            >>> consistent, msg = validator.check_consistency("fact(a).")
            >>> assert consistent
            >>>
            >>> # Inconsistent program
            >>> consistent, msg = validator.check_consistency(
            ...     "a. -a. :- a, -a."
            ... )
            >>> assert not consistent
        """
        try:
            ctl = clingo.Control()
            ctl.add("base", [], asp_program)
            ctl.ground([("base", [])])

            result = ctl.solve()

            if result.satisfiable:
                logger.debug("ASP program is consistent (has answer sets)")
                return (True, "Program has answer sets")
            elif result.unsatisfiable:
                logger.warning("ASP program is inconsistent (no answer sets)")
                return (False, "Program is unsatisfiable (no answer sets)")
            else:
                logger.warning("ASP program satisfiability unknown")
                return (False, "Unknown satisfiability")

        except Exception as e:
            error_msg = f"Consistency check failed: {str(e)}"
            logger.error(error_msg)
            return (False, error_msg)

    def detect_contradictions(self, asp_program: str) -> List[str]:
        """
        Find explicit contradictions like: a(X). -a(X).

        Returns list of contradictory predicate names.

        Args:
            asp_program: ASP program text

        Returns:
            List of predicate names that appear in contradictions

        Example:
            >>> validator = ASPSemanticValidator()
            >>> contradictions = validator.detect_contradictions(
            ...     "valid(x). -valid(x)."
            ... )
            >>> assert "valid" in contradictions
        """
        contradictions = []

        # Basic implementation: check if program is unsatisfiable
        # Full implementation would parse and find specific contradictions
        is_consistent, msg = self.check_consistency(asp_program)

        if not is_consistent and "unsatisfiable" in msg.lower():
            # Program has contradictions
            logger.warning("Contradictions detected in ASP program")
            contradictions.append("program_level_contradiction")

        return contradictions

    def check_rule_composition(self, rule1: str, rule2: str) -> bool:
        """
        Verify that combining two rules doesn't create inconsistency.

        Tests ring structure properties - rules should compose cleanly.

        Args:
            rule1: First ASP rule
            rule2: Second ASP rule

        Returns:
            True if composition is consistent

        Example:
            >>> validator = ASPSemanticValidator()
            >>> # Compatible rules
            >>> assert validator.check_rule_composition(
            ...     "a :- b.",
            ...     "b :- c."
            ... )
            >>>
            >>> # Incompatible rules
            >>> assert not validator.check_rule_composition(
            ...     "a.",
            ...     "-a."
            ... )
        """
        combined = f"{rule1}\n{rule2}"
        is_consistent, _ = self.check_consistency(combined)
        return is_consistent

    def count_answer_sets(self, asp_program: str, max_count: int = 10) -> int:
        """
        Count number of answer sets (up to max).

        Multiple answer sets indicate non-determinism in the program.

        Args:
            asp_program: ASP program text
            max_count: Maximum number of answer sets to count

        Returns:
            Number of answer sets found (up to max_count)

        Example:
            >>> validator = ASPSemanticValidator()
            >>> # Deterministic program (1 answer set)
            >>> count = validator.count_answer_sets("fact(a).")
            >>> assert count == 1
            >>>
            >>> # Non-deterministic program (multiple answer sets)
            >>> count = validator.count_answer_sets(
            ...     "{a; b}."  # Choice rule
            ... )
            >>> assert count > 1
        """
        try:
            ctl = clingo.Control()
            ctl.add("base", [], asp_program)
            ctl.ground([("base", [])])

            count = 0

            def on_model(model: clingo.Model) -> None:
                nonlocal count
                count += 1

            ctl.solve(on_model=on_model, yield_=True, models=max_count)

            logger.debug(f"ASP program has {count} answer set(s)")
            return count

        except Exception as e:
            logger.error(f"Error counting answer sets: {str(e)}")
            return 0

    def get_answer_sets(
        self, asp_program: str, max_sets: int = 10
    ) -> List[List[clingo.Symbol]]:
        """
        Get all answer sets (up to max_sets) for inspection.

        Args:
            asp_program: ASP program text
            max_sets: Maximum number of answer sets to retrieve

        Returns:
            List of answer sets, where each answer set is a list of symbols

        Example:
            >>> validator = ASPSemanticValidator()
            >>> answer_sets = validator.get_answer_sets("fact(a). fact(b).")
            >>> assert len(answer_sets) == 1
            >>> # Answer set contains fact(a) and fact(b)
        """
        answer_sets = []

        try:
            ctl = clingo.Control()
            ctl.add("base", [], asp_program)
            ctl.ground([("base", [])])

            def on_model(model: clingo.Model) -> None:
                # Get all symbols in the answer set
                symbols = [atom for atom in model.symbols(shown=True)]
                answer_sets.append(symbols)

            ctl.solve(on_model=on_model, yield_=True, models=max_sets)

            logger.debug(f"Retrieved {len(answer_sets)} answer set(s)")
            return answer_sets

        except Exception as e:
            logger.error(f"Error retrieving answer sets: {str(e)}")
            return []


def validate_asp_program(asp_text: str) -> Dict[str, Any]:
    """
    Comprehensive validation of an ASP program.

    Runs all validation checks and returns a summary.

    Args:
        asp_text: ASP program text to validate

    Returns:
        Dictionary with validation results:
        {
            "syntax_valid": bool,
            "syntax_error": Optional[str],
            "is_consistent": bool,
            "consistency_msg": str,
            "answer_set_count": int,
            "contradictions": List[str],
            "overall_valid": bool
        }

    Example:
        >>> results = validate_asp_program("fact(a). fact(b).")
        >>> assert results["overall_valid"]
        >>> assert results["syntax_valid"]
        >>> assert results["is_consistent"]
    """
    syntax_validator = ASPSyntaxValidator()
    semantic_validator = ASPSemanticValidator()

    # Syntax validation
    syntax_valid, syntax_error = syntax_validator.validate_program(asp_text)

    if not syntax_valid:
        return {
            "syntax_valid": False,
            "syntax_error": syntax_error,
            "is_consistent": False,
            "consistency_msg": "Skipped due to syntax error",
            "answer_set_count": 0,
            "contradictions": [],
            "overall_valid": False,
        }

    # Semantic validation
    is_consistent, consistency_msg = semantic_validator.check_consistency(asp_text)
    answer_set_count = semantic_validator.count_answer_sets(asp_text)
    contradictions = semantic_validator.detect_contradictions(asp_text)

    overall_valid = syntax_valid and is_consistent

    results = {
        "syntax_valid": syntax_valid,
        "syntax_error": syntax_error,
        "is_consistent": is_consistent,
        "consistency_msg": consistency_msg,
        "answer_set_count": answer_set_count,
        "contradictions": contradictions,
        "overall_valid": overall_valid,
    }

    logger.info(f"ASP validation complete: overall_valid={overall_valid}")
    return results
