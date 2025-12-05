"""
Unit tests for ASP arithmetic syntax fixer.

Tests the fixes for issue #163: LLM generates invalid ASP arithmetic syntax.
"""

from loft.validation.asp_arithmetic_fixer import (
    ASPArithmeticFixer,
    ArithmeticFix,
    FixResult,
    fix_asp_arithmetic,
    detect_asp_arithmetic_issues,
)


class TestASPArithmeticFixer:
    """Tests for ASPArithmeticFixer class."""

    def test_fix_floating_point_with_0_9_multiplier(self):
        """Test that 0.9 * X is converted to integer arithmetic."""
        fixer = ASPArithmeticFixer()

        rule = "amount_check(X) :- amount(X, A), target(X, T), A < 0.9 * T."
        result = fixer.fix_arithmetic_syntax(rule)

        assert result.was_modified
        assert "0.9" not in result.fixed_rule
        # 0.9 = 9/10, so A < 0.9 * T becomes A * 10 < T * 9
        assert "A * 10 < T * 9" in result.fixed_rule
        assert len(result.fixes_applied) == 1
        assert result.fixes_applied[0].fix_type == "floating_point_to_integer"

    def test_fix_floating_point_with_0_5_multiplier(self):
        """Test that 0.5 * X is converted to integer arithmetic."""
        fixer = ASPArithmeticFixer()

        rule = "half_check(X) :- value(X, V), limit(X, L), V >= 0.5 * L."
        result = fixer.fix_arithmetic_syntax(rule)

        assert result.was_modified
        assert "0.5" not in result.fixed_rule
        # 0.5 = 1/2, so V >= 0.5 * L becomes V * 2 >= L * 1
        assert "V * 2 >= L * 1" in result.fixed_rule

    def test_fix_floating_point_with_0_75_multiplier(self):
        """Test that 0.75 * X is converted to integer arithmetic."""
        fixer = ASPArithmeticFixer()

        rule = "threshold_check(X) :- actual(X, A), expected(X, E), A > 0.75 * E."
        result = fixer.fix_arithmetic_syntax(rule)

        assert result.was_modified
        assert "0.75" not in result.fixed_rule
        # 0.75 = 3/4, so A > 0.75 * E becomes A * 4 > E * 3
        assert "A * 4 > E * 3" in result.fixed_rule

    def test_no_modification_for_valid_rule(self):
        """Test that valid rules are not modified."""
        fixer = ASPArithmeticFixer()

        rule = "enforceable(X) :- contract(X), amount(X, A), threshold(T), A > T."
        result = fixer.fix_arithmetic_syntax(rule)

        assert not result.was_modified
        assert result.fixed_rule == rule
        assert len(result.fixes_applied) == 0

    def test_detect_abs_function(self):
        """Test that abs() function calls are detected."""
        fixer = ASPArithmeticFixer()

        rule = "difference_check(X) :- val_a(X, A), val_b(X, B), abs(A - B) > 100."
        result = fixer.fix_arithmetic_syntax(rule)

        # abs() can't be fixed inline, should be in could_not_fix
        assert len(result.could_not_fix) == 1
        assert "abs()" in result.could_not_fix[0]
        assert "split" in result.could_not_fix[0].lower()

    def test_detect_arithmetic_issues_floating_point(self):
        """Test detection of floating point literals."""
        fixer = ASPArithmeticFixer()

        rule = "check(X) :- value(X, V), V < 0.9 * 50000."
        issues = fixer.detect_arithmetic_issues(rule)

        assert len(issues) >= 1
        assert any("Floating point" in issue for issue in issues)

    def test_detect_arithmetic_issues_abs_function(self):
        """Test detection of abs() function."""
        fixer = ASPArithmeticFixer()

        rule = "check(X) :- val(X, V), abs(V) > 100."
        issues = fixer.detect_arithmetic_issues(rule)

        assert len(issues) >= 1
        assert any("abs()" in issue for issue in issues)

    def test_detect_arithmetic_issues_python_functions(self):
        """Test detection of Python-style functions."""
        fixer = ASPArithmeticFixer()

        rules_with_funcs = [
            "check(X) :- value(X, V), min(V, 100) > 50.",
            "check(X) :- value(X, V), max(V, 0) < 100.",
            "check(X) :- value(X, V), pow(V, 2) < 100.",
            "check(X) :- value(X, V), sqrt(V) < 10.",
        ]

        for rule in rules_with_funcs:
            issues = fixer.detect_arithmetic_issues(rule)
            assert len(issues) >= 1
            assert any("Python-style function" in issue for issue in issues)

    def test_get_statistics(self):
        """Test statistics tracking."""
        fixer = ASPArithmeticFixer()

        # Apply a fix
        rule = "check(X) :- val(X, A), limit(X, L), A < 0.9 * L."
        fixer.fix_arithmetic_syntax(rule)

        stats = fixer.get_statistics()
        assert stats["total_fixes_applied"] == 1
        assert "floating_point_to_integer" in stats["fixes_by_type"]
        assert stats["fixes_by_type"]["floating_point_to_integer"] == 1

    def test_multiple_fixes_in_same_rule(self):
        """Test handling multiple floating point values in one rule."""
        fixer = ASPArithmeticFixer()

        rule = "range_check(X) :- low(X, L), high(X, H), value(X, V), V > 0.1 * L, V < 0.9 * H."
        result = fixer.fix_arithmetic_syntax(rule)

        assert result.was_modified
        assert "0.1" not in result.fixed_rule
        assert "0.9" not in result.fixed_rule


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_fix_asp_arithmetic_function(self):
        """Test the fix_asp_arithmetic convenience function."""
        rule = "check(X) :- val(X, A), limit(X, L), A < 0.5 * L."
        fixed, was_modified, warnings = fix_asp_arithmetic(rule)

        assert was_modified
        assert "0.5" not in fixed
        assert "A * 2 < L * 1" in fixed

    def test_detect_asp_arithmetic_issues_function(self):
        """Test the detect_asp_arithmetic_issues convenience function."""
        rule = "check(X) :- val(X, V), abs(V) > 100, V < 0.9 * limit."
        issues = detect_asp_arithmetic_issues(rule)

        assert len(issues) >= 2
        assert any("abs()" in issue for issue in issues)
        assert any("Floating point" in issue for issue in issues)


class TestArithmeticFixDataclass:
    """Tests for ArithmeticFix dataclass."""

    def test_arithmetic_fix_creation(self):
        """Test creating ArithmeticFix objects."""
        fix = ArithmeticFix(
            original_pattern="A < 0.9 * B",
            replacement="A * 10 < B * 9",
            fix_type="floating_point_to_integer",
            description="Converted 0.9 to 9/10 ratio",
        )

        assert fix.original_pattern == "A < 0.9 * B"
        assert fix.replacement == "A * 10 < B * 9"
        assert fix.fix_type == "floating_point_to_integer"


class TestFixResultDataclass:
    """Tests for FixResult dataclass."""

    def test_fix_result_creation(self):
        """Test creating FixResult objects."""
        result = FixResult(
            original_rule="A < 0.9 * B",
            fixed_rule="A * 10 < B * 9",
            fixes_applied=[],
            was_modified=True,
            could_not_fix=[],
        )

        assert result.original_rule == "A < 0.9 * B"
        assert result.fixed_rule == "A * 10 < B * 9"
        assert result.was_modified


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_rule(self):
        """Test handling of empty rule."""
        fixer = ASPArithmeticFixer()
        result = fixer.fix_arithmetic_syntax("")

        assert not result.was_modified
        assert result.fixed_rule == ""

    def test_rule_with_no_arithmetic(self):
        """Test rule with predicates but no arithmetic."""
        fixer = ASPArithmeticFixer()
        rule = "enforceable(X) :- contract(X), valid(X), not void(X)."
        result = fixer.fix_arithmetic_syntax(rule)

        assert not result.was_modified
        assert result.fixed_rule == rule

    def test_rule_with_integer_arithmetic(self):
        """Test rule with valid integer arithmetic."""
        fixer = ASPArithmeticFixer()
        rule = "check(X) :- val(X, A), limit(X, L), A * 10 < L * 9."
        result = fixer.fix_arithmetic_syntax(rule)

        assert not result.was_modified
        assert result.fixed_rule == rule

    def test_unknown_decimal_value(self):
        """Test handling of unknown decimal values."""
        fixer = ASPArithmeticFixer()
        rule = "check(X) :- val(X, A), limit(X, L), A < 0.37 * L."
        result = fixer.fix_arithmetic_syntax(rule)

        # 0.37 is not in the known mapping, should be in could_not_fix
        assert len(result.could_not_fix) == 1
        assert "0.37" in result.could_not_fix[0]

    def test_parenthesized_expression(self):
        """Test handling of parenthesized expressions with decimals."""
        fixer = ASPArithmeticFixer()
        rule = "check(X) :- val(X, A), high(X, H), low(X, L), A < (H - L) * 0.5."
        result = fixer.fix_arithmetic_syntax(rule)

        assert result.was_modified
        assert "0.5" not in result.fixed_rule
