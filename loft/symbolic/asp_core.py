"""
Core ASP engine with Clingo integration.

Provides the main interface for loading, querying, and reasoning
with ASP programs using the Clingo solver.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import clingo
from loguru import logger
from .asp_program import StratifiedASPCore
from .asp_rule import ASPRule, StratificationLevel


@dataclass
class QueryResult:
    """Result from an ASP query."""

    symbols: List[clingo.Symbol]  # Matching symbols from answer sets
    satisfiable: bool  # Whether program is satisfiable
    answer_set_count: int  # Number of answer sets found
    model_strings: List[str]  # String representations of models

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbols": [str(s) for s in self.symbols],
            "satisfiable": self.satisfiable,
            "answer_set_count": self.answer_set_count,
            "model_strings": self.model_strings,
        }


class ASPCore:
    """
    Main ASP engine with Clingo integration.

    Manages stratified ASP programs, queries, and consistency checking.
    """

    def __init__(self, stratified_programs: Optional[StratifiedASPCore] = None):
        """
        Initialize ASP core.

        Args:
            stratified_programs: Optional pre-configured stratified core
        """
        self.stratified_programs = stratified_programs or StratifiedASPCore()
        self.control: Optional[clingo.Control] = None
        self._loaded = False

    def load_rules(self, additional_facts: Optional[List[str]] = None) -> None:
        """
        Load all stratified rules into Clingo.

        Args:
            additional_facts: Optional additional facts to add
        """
        # Create new control instance
        self.control = clingo.Control()

        # Get complete program
        program = self.stratified_programs.get_full_program()

        # Add additional facts if provided
        if additional_facts:
            program += "\n\n% === Additional Facts ===\n"
            program += "\n".join(additional_facts)

        # Load into Clingo
        self.control.add("base", [], program)

        try:
            self.control.ground([("base", [])])
            self._loaded = True
            logger.info(
                f"Loaded ASP core with {self.stratified_programs.count_total_rules()} rules"
            )
        except RuntimeError as e:
            logger.error(f"Failed to ground ASP program: {e}")
            raise

    def query(
        self,
        query_predicate: Optional[str] = None,
        max_models: int = 0,
    ) -> QueryResult:
        """
        Query the ASP program.

        Args:
            query_predicate: Optional predicate name to filter results
            max_models: Maximum number of models to compute (0 = all)

        Returns:
            QueryResult with matching symbols and metadata
        """
        if not self._loaded:
            raise RuntimeError("ASP core not loaded. Call load_rules() first.")

        if self.control is None:
            raise RuntimeError("Control instance not initialized")

        results: List[clingo.Symbol] = []
        model_strings: List[str] = []
        answer_set_count = 0

        def on_model(model: clingo.Model) -> None:
            nonlocal answer_set_count
            answer_set_count += 1

            # Get symbols
            symbols = model.symbols(shown=True)

            # Filter by predicate if specified
            if query_predicate:
                filtered = [s for s in symbols if s.name == query_predicate]
                results.extend(filtered)
            else:
                results.extend(symbols)

            # Store model string
            model_strings.append(str(model))

        # Solve
        with self.control.solve(on_model=on_model, yield_=True) as handle:
            # Iterate through models
            for _ in handle:
                pass  # Models are processed by on_model callback
            solve_result = handle.get()

        return QueryResult(
            symbols=results,
            satisfiable=solve_result.satisfiable,
            answer_set_count=answer_set_count,
            model_strings=model_strings,
        )

    def check_consistency(self) -> bool:
        """
        Check if current rule set is consistent (has answer sets).

        Returns:
            True if consistent (satisfiable), False otherwise
        """
        if not self._loaded:
            raise RuntimeError("ASP core not loaded. Call load_rules() first.")

        if self.control is None:
            raise RuntimeError("Control instance not initialized")

        result = self.control.solve()
        return bool(result.satisfiable)

    def count_answer_sets(self, max_count: int = 100) -> int:
        """
        Count the number of answer sets.

        Args:
            max_count: Maximum number to count (for performance)

        Returns:
            Number of answer sets (up to max_count)
        """
        if not self._loaded:
            raise RuntimeError("ASP core not loaded. Call load_rules() first.")

        if self.control is None:
            raise RuntimeError("Control instance not initialized")

        count = 0

        with self.control.solve(yield_=True) as handle:
            for model in handle:
                count += 1
                if count >= max_count:
                    break
        return count

    def add_rule(self, rule: ASPRule) -> None:
        """
        Add a rule to the stratified core.

        Note: Requires reload to take effect.

        Args:
            rule: ASP rule to add
        """
        self.stratified_programs.add_rule(rule)
        self._loaded = False  # Mark for reload

    def add_facts(self, facts: List[str]) -> None:
        """
        Add facts to the operational layer.

        Args:
            facts: List of fact strings
        """
        for fact in facts:
            self.stratified_programs.operational.add_fact(fact)
        self._loaded = False  # Mark for reload

    def get_all_rules(self) -> List[ASPRule]:
        """Get all rules from all stratification levels."""
        return self.stratified_programs.get_all_rules()

    def get_rules_by_level(self, level: StratificationLevel) -> List[ASPRule]:
        """Get rules at a specific stratification level."""
        program = self.stratified_programs.get_program(level)
        return program.rules

    def solve_with_assumptions(
        self,
        assumptions: List[tuple[str, bool]],
    ) -> QueryResult:
        """
        Solve with specific assumptions.

        Args:
            assumptions: List of (predicate_name, value) tuples

        Returns:
            QueryResult under the given assumptions
        """
        if not self._loaded:
            raise RuntimeError("ASP core not loaded. Call load_rules() first.")

        if self.control is None:
            raise RuntimeError("Control instance not initialized")

        # Convert assumptions to clingo format
        clingo_assumptions = []
        for pred, value in assumptions:
            # Create symbolic atom
            symbol = clingo.Function(pred)
            clingo_assumptions.append((symbol, value))

        results: List[clingo.Symbol] = []
        model_strings: List[str] = []
        answer_set_count = 0

        def on_model(model: clingo.Model) -> None:
            nonlocal answer_set_count
            answer_set_count += 1
            symbols = model.symbols(shown=True)
            results.extend(symbols)
            model_strings.append(str(model))

        solve_result = self.control.solve(assumptions=clingo_assumptions, on_model=on_model)

        return QueryResult(
            symbols=results,
            satisfiable=bool(solve_result.satisfiable),
            answer_set_count=answer_set_count,
            model_strings=model_strings,
        )

    def get_program_text(self) -> str:
        """Get the complete ASP program as text."""
        return self.stratified_programs.get_full_program()

    def detect_unsat_core(self) -> List[int]:
        """
        Detect unsatisfiable core if program is inconsistent.

        Returns:
            List of rule indices in the unsatisfiable core
        """
        if not self._loaded:
            raise RuntimeError("ASP core not loaded. Call load_rules() first.")

        if self.control is None:
            raise RuntimeError("Control instance not initialized")

        # Enable enumeration of unsatisfiable cores
        self.control.configuration.solve.opt_mode = "optN"  # type: ignore[union-attr]

        # This is a simplified version - full unsat core detection
        # requires more sophisticated Clingo features
        if self.check_consistency():
            return []  # No unsat core if consistent

        # Return empty list - full implementation would analyze the core
        return []
