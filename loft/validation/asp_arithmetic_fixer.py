"""
ASP Arithmetic Syntax Fixer for LLM-generated rules.

Fixes common arithmetic syntax errors that LLMs generate when creating
ASP/Clingo rules, such as:
- Floating point literals (0.9, 0.75) -> integer percentage math
- abs() function calls -> split into positive/negative cases
- Infix multiplication with decimals -> integer multiplication

This module addresses issue #163: LLM generates invalid ASP arithmetic syntax.
"""

import re
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class ArithmeticFix:
    """Represents a fix applied to an ASP rule."""

    original_pattern: str
    replacement: str
    fix_type: str
    description: str


@dataclass
class FixResult:
    """Result of attempting to fix arithmetic issues in an ASP rule."""

    original_rule: str
    fixed_rule: str
    fixes_applied: List[ArithmeticFix]
    was_modified: bool
    could_not_fix: List[str]


class ASPArithmeticFixer:
    """
    Fixes common arithmetic syntax errors in LLM-generated ASP rules.

    The LLM often generates Python-style arithmetic that is invalid in Clingo:
    - Floating point numbers (0.9, 0.5, etc.)
    - abs() function calls
    - Infix operators without proper binding
    """

    # Common decimal percentages and their integer equivalents
    DECIMAL_TO_PERCENTAGE: Dict[str, Tuple[int, int]] = {
        "0.9": (9, 10),
        "0.90": (9, 10),
        "0.8": (4, 5),
        "0.80": (4, 5),
        "0.75": (3, 4),
        "0.7": (7, 10),
        "0.70": (7, 10),
        "0.5": (1, 2),
        "0.50": (1, 2),
        "0.25": (1, 4),
        "0.2": (1, 5),
        "0.20": (1, 5),
        "0.1": (1, 10),
        "0.10": (1, 10),
        "0.05": (1, 20),
        "0.01": (1, 100),
        "1.0": (1, 1),
        "1.5": (3, 2),
        "2.0": (2, 1),
    }

    def __init__(self):
        """Initialize the arithmetic fixer."""
        self._fix_count = 0
        self._fixes_by_type: Dict[str, int] = {}

    def fix_arithmetic_syntax(self, rule_text: str) -> FixResult:
        """
        Apply all arithmetic syntax fixes to an ASP rule.

        Args:
            rule_text: The ASP rule text to fix

        Returns:
            FixResult with the fixed rule and details of fixes applied
        """
        fixes_applied = []
        could_not_fix = []
        current_rule = rule_text

        # Apply fixes in order of complexity
        current_rule, float_fixes, float_failures = self._fix_floating_point_literals(current_rule)
        fixes_applied.extend(float_fixes)
        could_not_fix.extend(float_failures)

        current_rule, abs_fixes, abs_failures = self._fix_abs_function_calls(current_rule)
        fixes_applied.extend(abs_fixes)
        could_not_fix.extend(abs_failures)

        current_rule, infix_fixes, infix_failures = self._fix_infix_multiplication(current_rule)
        fixes_applied.extend(infix_fixes)
        could_not_fix.extend(infix_failures)

        was_modified = len(fixes_applied) > 0

        if was_modified:
            logger.debug(
                f"Fixed ASP arithmetic: {len(fixes_applied)} fixes applied, "
                f"{len(could_not_fix)} issues could not be fixed"
            )
            for fix in fixes_applied:
                self._fix_count += 1
                self._fixes_by_type[fix.fix_type] = self._fixes_by_type.get(fix.fix_type, 0) + 1

        return FixResult(
            original_rule=rule_text,
            fixed_rule=current_rule,
            fixes_applied=fixes_applied,
            was_modified=was_modified,
            could_not_fix=could_not_fix,
        )

    def _fix_floating_point_literals(
        self, rule_text: str
    ) -> Tuple[str, List[ArithmeticFix], List[str]]:
        """
        Fix floating point literals by converting to integer arithmetic.

        Examples:
            - A < 0.9 * B  ->  A * 10 < B * 9
            - A >= 0.5 * C ->  A * 2 >= C
        """
        fixes = []
        failures = []
        current_rule = rule_text

        # Pattern: VAR comparison DECIMAL * VAR  (e.g., A < 0.9 * B)
        # Pattern: VAR comparison VAR * DECIMAL  (e.g., A < B * 0.9)
        float_pattern = r"(\w+)\s*([<>=!]+)\s*(\d+\.?\d*)\s*\*\s*(\w+)"

        for match in re.finditer(float_pattern, rule_text):
            var1, op, decimal_str, var2 = match.groups()

            if decimal_str in self.DECIMAL_TO_PERCENTAGE:
                numerator, denominator = self.DECIMAL_TO_PERCENTAGE[decimal_str]

                # A < 0.9 * B  ->  A * denominator < B * numerator
                old_expr = match.group(0)
                new_expr = f"{var1} * {denominator} {op} {var2} * {numerator}"

                current_rule = current_rule.replace(old_expr, new_expr)

                fixes.append(
                    ArithmeticFix(
                        original_pattern=old_expr,
                        replacement=new_expr,
                        fix_type="floating_point_to_integer",
                        description=f"Converted {decimal_str} multiplication to "
                        f"integer ratio {numerator}/{denominator}",
                    )
                )
            else:
                failures.append(
                    f"Unknown decimal literal: {decimal_str} in expression {match.group(0)}"
                )

        # Pattern: VAR comparison (expression) * DECIMAL
        paren_float_pattern = r"(\w+)\s*([<>=!]+)\s*\(([^)]+)\)\s*\*\s*(\d+\.\d+)"

        for match in re.finditer(paren_float_pattern, current_rule):
            var1, op, expr, decimal_str = match.groups()

            if decimal_str in self.DECIMAL_TO_PERCENTAGE:
                numerator, denominator = self.DECIMAL_TO_PERCENTAGE[decimal_str]

                old_expr = match.group(0)
                new_expr = f"{var1} * {denominator} {op} ({expr}) * {numerator}"

                current_rule = current_rule.replace(old_expr, new_expr)

                fixes.append(
                    ArithmeticFix(
                        original_pattern=old_expr,
                        replacement=new_expr,
                        fix_type="floating_point_to_integer",
                        description=f"Converted {decimal_str} multiplication to "
                        f"integer ratio {numerator}/{denominator}",
                    )
                )
            else:
                failures.append(
                    f"Unknown decimal literal: {decimal_str} in expression {match.group(0)}"
                )

        return current_rule, fixes, failures

    def _fix_abs_function_calls(self, rule_text: str) -> Tuple[str, List[ArithmeticFix], List[str]]:
        """
        Fix abs() function calls by suggesting split into two rules.

        Clingo doesn't have a built-in abs() function, so we need to
        handle absolute value using two separate conditions.

        Note: This fix cannot be done inline - it returns a failure
        suggesting the user split the rule manually or use a different approach.
        """
        fixes = []
        failures = []

        # Pattern: abs(EXPR) comparison VALUE
        abs_pattern = r"abs\s*\(([^)]+)\)\s*([<>=!]+)\s*(\w+)"

        for match in re.finditer(abs_pattern, rule_text):
            expr, op, value = match.groups()

            # We can't easily fix abs() inline - it requires splitting into
            # two separate conditions or rules
            failures.append(
                f"abs() function not supported in Clingo. "
                f"Expression 'abs({expr}) {op} {value}' should be split into: "
                f"'{expr} >= 0, {expr} {op} {value}' OR "
                f"'{expr} < 0, -{expr} {op} {value}'"
            )

        return rule_text, fixes, failures

    def _fix_infix_multiplication(
        self, rule_text: str
    ) -> Tuple[str, List[ArithmeticFix], List[str]]:
        """
        Fix standalone infix multiplication that might cause parsing issues.

        Ensures that arithmetic expressions are properly formatted.
        """
        fixes = []
        failures = []
        current_rule = rule_text

        # Check for multiplication with unbound variables
        # Pattern: VAR = NUMBER * UNBOUND_VAR
        unbound_mult_pattern = r"(\w+)\s*=\s*(\d+)\s*\*\s*([A-Z][a-zA-Z0-9_]*)"

        for match in re.finditer(unbound_mult_pattern, rule_text):
            result_var, number, mult_var = match.groups()
            old_expr = match.group(0)

            # This is a potential error - the multiplied variable should be bound first
            failures.append(
                f"Potentially unbound variable in arithmetic: {old_expr}. "
                f"Ensure '{mult_var}' is bound by a predicate before this expression."
            )

        return current_rule, fixes, failures

    def detect_arithmetic_issues(self, rule_text: str) -> List[str]:
        """
        Detect potential arithmetic issues without fixing them.

        Useful for validation warnings before attempting fixes.

        Args:
            rule_text: The ASP rule text to check

        Returns:
            List of issue descriptions
        """
        issues = []

        # Check for floating point literals
        float_pattern = r"\d+\.\d+"
        floats_found = re.findall(float_pattern, rule_text)
        if floats_found:
            issues.append(
                f"Floating point literals found: {', '.join(set(floats_found))}. "
                "Clingo uses integer arithmetic."
            )

        # Check for abs() function
        if "abs(" in rule_text.lower():
            issues.append(
                "abs() function found. Clingo doesn't have built-in abs(). "
                "Split into positive/negative cases."
            )

        # Check for function-style arithmetic (Python syntax)
        func_patterns = ["min(", "max(", "pow(", "sqrt(", "round(", "floor(", "ceil("]
        for func in func_patterns:
            if func in rule_text.lower():
                issues.append(
                    f"Python-style function '{func}' found. "
                    "Use Clingo aggregate syntax or integer arithmetic."
                )

        # Check for division (Clingo uses integer division)
        if "/" in rule_text and "//" not in rule_text:
            issues.append(
                "Division operator '/' found. Clingo uses integer division. "
                "Be aware of truncation behavior."
            )

        return issues

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics on fixes applied."""
        return {
            "total_fixes_applied": self._fix_count,
            "fixes_by_type": self._fixes_by_type.copy(),
        }


def fix_asp_arithmetic(rule_text: str) -> Tuple[str, bool, List[str]]:
    """
    Convenience function to fix arithmetic in an ASP rule.

    Args:
        rule_text: The ASP rule to fix

    Returns:
        Tuple of (fixed_rule, was_modified, warnings)
    """
    fixer = ASPArithmeticFixer()
    result = fixer.fix_arithmetic_syntax(rule_text)

    warnings = []
    if result.could_not_fix:
        warnings.extend(result.could_not_fix)

    return result.fixed_rule, result.was_modified, warnings


def detect_asp_arithmetic_issues(rule_text: str) -> List[str]:
    """
    Convenience function to detect arithmetic issues in an ASP rule.

    Args:
        rule_text: The ASP rule to check

    Returns:
        List of issue descriptions
    """
    fixer = ASPArithmeticFixer()
    return fixer.detect_arithmetic_issues(rule_text)
