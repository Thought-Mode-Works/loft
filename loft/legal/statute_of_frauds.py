"""
Statute of Frauds implementation using ASP.

This module provides the infrastructure for testing the Statute of Frauds
rules implemented in ASP, including test cases, gap detection, and explanation
generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any
from pathlib import Path

try:
    import clingo
except ImportError:
    clingo = None  # type: ignore


@dataclass
class StatuteOfFraudsTestCase:
    """
    Test case for Statute of Frauds reasoning.

    Attributes:
        case_id: Unique identifier for the test case
        description: Human-readable description of the scenario
        asp_facts: ASP facts for this case
        expected_results: Expected query results (e.g., {"enforceable": True})
        reasoning_chain: Expected inference steps
        confidence_level: Expected confidence in the determination
        requires_llm_query: Whether this case needs LLM input
        llm_query_focus: What to ask the LLM about
        legal_citations: Relevant cases/statutes
    """

    case_id: str
    description: str
    asp_facts: str
    expected_results: Dict[str, bool]
    reasoning_chain: List[str] = field(default_factory=list)
    confidence_level: Literal["high", "medium", "low"] = "high"
    requires_llm_query: bool = False
    llm_query_focus: Optional[str] = None
    legal_citations: List[str] = field(default_factory=list)


class StatuteOfFraudsSystem:
    """
    System for reasoning about Statute of Frauds using ASP.

    This system loads the statute of frauds ASP program and provides
    methods for querying, gap detection, and explanation generation.
    """

    def __init__(self):
        """Initialize the system with the statute of frauds ASP program."""
        if clingo is None:
            raise ImportError("clingo is required for statute of frauds reasoning")

        # Load the statute of frauds rules
        program_path = Path(__file__).parent / "statute_of_frauds.lp"
        with open(program_path, "r") as f:
            self.base_program = f.read()

        # Initialize Clingo control object
        self._reset_control()

    def _reset_control(self):
        """Reset the Clingo control object."""
        import sys
        import os

        # Suppress Clingo warnings by passing a silent message handler
        def silent_logger(code, message):
            pass  # Ignore all messages

        # Suppress warnings by redirecting both stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            self.control = clingo.Control(["0", "--warn=none"], logger=silent_logger)
            self.control.add("base", [], self.base_program)
            self.additional_facts = ""
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def reset(self):
        """Reset the system, clearing all facts but keeping rules."""
        self._reset_control()

    def add_facts(self, facts: str):
        """
        Add facts to the ASP core.

        Args:
            facts: ASP facts as a string
        """
        import sys
        import os

        self.additional_facts += "\n" + facts

        # Suppress Clingo warnings by redirecting both stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            self.control.add("base", [], facts)
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def query(self, predicate: str) -> List[Dict[str, Any]]:
        """
        Query the ASP core for a predicate.

        Args:
            predicate: Predicate to query (e.g., "enforceable")

        Returns:
            List of bindings for the predicate
        """
        import sys
        import os

        # Suppress Clingo warnings by redirecting both stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

            # Ground and solve
            self.control.ground([("base", [])])

            # Collect results
            results = []

            def on_model(model):
                for atom in model.symbols(shown=True):
                    if atom.name == predicate:
                        # Extract arguments
                        args = {}
                        if len(atom.arguments) > 0:
                            # Use positional argument names
                            arg_names = ["C", "W", "P", "D", "Amount", "Exception"]
                            for i, arg in enumerate(atom.arguments):
                                arg_name = arg_names[i] if i < len(arg_names) else f"arg{i}"
                                args[arg_name] = str(arg)
                        results.append(args if args else {predicate: True})

            self.control.solve(on_model=on_model)

            return results

        finally:
            # Restore stdout and stderr
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def is_enforceable(self, contract_id: str) -> Optional[bool]:
        """
        Check if a contract is enforceable.

        Args:
            contract_id: Contract identifier

        Returns:
            True if enforceable, False if unenforceable, None if unknown
        """
        enforceable_results = self.query("enforceable")
        unenforceable_results = self.query("unenforceable")

        is_enforceable = any(r.get("C") == contract_id for r in enforceable_results)
        is_unenforceable = any(r.get("C") == contract_id for r in unenforceable_results)

        if is_enforceable:
            return True
        elif is_unenforceable:
            return False
        else:
            return None

    def detect_gaps(self, contract_id: str) -> List[str]:
        """
        Detect gaps in knowledge for a contract.

        Args:
            contract_id: Contract identifier

        Returns:
            List of gap descriptions
        """
        gaps = []

        # Check for missing writing information
        missing_writing = self.query("missing_writing_info")
        for result in missing_writing:
            if result.get("C") == contract_id:
                gaps.append("Contract requires writing but no writing information provided")

        # Check for uncertain writing sufficiency
        uncertain_writing = self.query("uncertain_writing_sufficiency")
        for result in uncertain_writing:
            if result.get("C") == contract_id:
                gaps.append(f"Uncertain if writing {result.get('W')} contains essential terms")

        # Check for uncertain exceptions
        uncertain_exceptions = self.query("uncertain_exception")
        for result in uncertain_exceptions:
            if result.get("C") == contract_id:
                gaps.append(f"Uncertain if exception {result.get('Exception')} applies")

        return gaps

    def explain_conclusion(self, contract_id: str) -> str:
        """
        Generate a natural language explanation of the conclusion.

        Args:
            contract_id: Contract identifier

        Returns:
            Natural language explanation
        """
        explanation = []

        # Check enforceability
        is_enf = self.is_enforceable(contract_id)
        if is_enf is None:
            return f"Cannot determine if contract {contract_id} is enforceable."

        if is_enf:
            explanation.append(f"Contract {contract_id} is ENFORCEABLE.")
        else:
            explanation.append(f"Contract {contract_id} is UNENFORCEABLE.")

        # Check if within statute
        within_statute = self.query("within_statute")
        is_within = any(r.get("C") == contract_id for r in within_statute)

        if is_within:
            explanation.append("  - The contract falls within the Statute of Frauds")

            # Check if has sufficient writing
            sufficient_writing = self.query("has_sufficient_writing")
            has_writing = any(r.get("C") == contract_id for r in sufficient_writing)

            if has_writing:
                explanation.append("  - The contract has a sufficient writing")
                explanation.append("    - A writing references the contract")
                explanation.append("    - The writing is signed by a party to the contract")
                explanation.append("    - The writing contains essential terms")
            else:
                # Check for exceptions
                exceptions = self.query("exception_applies")
                has_exception = any(r.get("C") == contract_id for r in exceptions)

                if has_exception:
                    explanation.append("  - An exception to the writing requirement applies")
                else:
                    explanation.append("  - No sufficient writing exists")
                    explanation.append("  - No exception to the writing requirement applies")
        else:
            explanation.append("  - The contract does not fall within the Statute of Frauds")
            explanation.append("  - Therefore, no writing is required")

        return "\n".join(explanation)

    def check_consistency(self) -> bool:
        """
        Check if the ASP program is consistent.

        Returns:
            True if consistent, False otherwise
        """
        # Try to get a model
        try:
            self.control.ground([("base", [])])
            result = self.control.solve()
            return result.satisfiable
        except Exception:
            return False


class StatuteOfFraudsDemo:
    """
    Demonstration system for Statute of Frauds reasoning.

    This class provides methods to run test cases and show the system's
    reasoning capabilities.
    """

    def __init__(self):
        """Initialize the demo system."""
        self.system = StatuteOfFraudsSystem()
        self.test_cases: Dict[str, StatuteOfFraudsTestCase] = {}

    def register_case(self, test_case: StatuteOfFraudsTestCase):
        """
        Register a test case.

        Args:
            test_case: Test case to register
        """
        self.test_cases[test_case.case_id] = test_case

    def run_case(self, case_id: str) -> Dict[str, Any]:
        """
        Run a specific test case.

        Args:
            case_id: ID of the test case to run

        Returns:
            Dictionary with results, explanation, and gaps
        """
        if case_id not in self.test_cases:
            raise ValueError(f"Test case {case_id} not found")

        test_case = self.test_cases[case_id]

        # Reset system
        self.system.reset()

        # Add facts
        self.system.add_facts(test_case.asp_facts)

        # Query for results
        results = {}
        for predicate in test_case.expected_results.keys():
            query_results = self.system.query(predicate)
            results[predicate] = query_results

        # Get explanation
        contract_id = self._extract_contract_id(test_case.asp_facts)
        explanation = self.system.explain_conclusion(contract_id)

        # Detect gaps
        gaps = self.system.detect_gaps(contract_id)

        return {
            "case_id": case_id,
            "description": test_case.description,
            "results": results,
            "explanation": explanation,
            "gaps": gaps,
            "confidence": test_case.confidence_level,
        }

    def _extract_contract_id(self, asp_facts: str) -> str:
        """Extract contract ID from ASP facts."""
        import re

        match = re.search(r"contract_fact\((\w+)\)", asp_facts)
        if match:
            return match.group(1)
        return "unknown"

    def run_all_cases(self) -> Dict[str, Any]:
        """
        Run all registered test cases.

        Returns:
            Summary of results including accuracy
        """
        results = []
        correct = 0
        total = 0

        for case_id, test_case in self.test_cases.items():
            result = self.run_case(case_id)

            # Check if result matches expected
            contract_id = self._extract_contract_id(test_case.asp_facts)
            is_enf = self.system.is_enforceable(contract_id)

            expected_enf = test_case.expected_results.get("enforceable")
            if expected_enf is not None:
                total += 1
                if is_enf == expected_enf:
                    correct += 1
                    result["correct"] = True
                else:
                    result["correct"] = False

            results.append(result)

        accuracy = correct / total if total > 0 else 0.0

        return {
            "results": results,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
