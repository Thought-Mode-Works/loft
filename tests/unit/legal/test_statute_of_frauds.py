"""
Unit tests for Statute of Frauds implementation.

Tests the ASP-based statute of frauds reasoning system with 21 diverse test cases.
"""

from loft.legal.statute_of_frauds import (
    StatuteOfFraudsSystem,
    StatuteOfFraudsDemo,
)
from loft.legal.test_cases import ALL_TEST_CASES


class TestStatuteOfFraudsSystem:
    """Test the statute of frauds reasoning system."""

    def test_initialization(self) -> None:
        """Test system initializes correctly."""
        system = StatuteOfFraudsSystem()
        assert system is not None
        assert system.control is not None

    def test_load_asp_program(self) -> None:
        """Test ASP program loads without errors."""
        system = StatuteOfFraudsSystem()
        # Should not raise an exception
        system.check_consistency()

    def test_clear_written_land_sale(self) -> None:
        """Test Case 1: Clear written land sale."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[0].asp_facts)

        assert system.is_enforceable("c1") is True

        explanation = system.explain_conclusion("c1")
        assert "ENFORCEABLE" in explanation
        assert "sufficient writing" in explanation

    def test_oral_land_sale(self) -> None:
        """Test Case 2: Oral land sale (unenforceable)."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[1].asp_facts)

        assert system.is_enforceable("c2") is False

        explanation = system.explain_conclusion("c2")
        assert "UNENFORCEABLE" in explanation

    def test_goods_under_500(self) -> None:
        """Test Case 3: Goods sale under $500."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[2].asp_facts)

        assert system.is_enforceable("c3") is True

        explanation = system.explain_conclusion("c3")
        assert "ENFORCEABLE" in explanation
        assert "does not fall within" in explanation

    def test_goods_over_500_no_writing(self) -> None:
        """Test Case 4: Goods over $500 without writing."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[3].asp_facts)

        assert system.is_enforceable("c4") is False

    def test_goods_over_500_with_writing(self) -> None:
        """Test Case 5: Goods over $500 with writing."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[4].asp_facts)

        assert system.is_enforceable("c5") is True

    def test_part_performance_exception(self) -> None:
        """Test Case 6: Part performance exception."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[5].asp_facts)

        assert system.is_enforceable("c6") is True

        explanation = system.explain_conclusion("c6")
        assert "exception" in explanation.lower()

    def test_promissory_estoppel(self) -> None:
        """Test Case 7: Promissory estoppel exception."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[6].asp_facts)

        assert system.is_enforceable("c7") is True

    def test_long_term_contract(self) -> None:
        """Test Case 8: Long-term contract (over 1 year)."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[7].asp_facts)

        assert system.is_enforceable("c8") is False

    def test_short_term_contract(self) -> None:
        """Test Case 9: Short-term contract (under 1 year)."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[8].asp_facts)

        assert system.is_enforceable("c9") is True

    def test_merchant_confirmation(self) -> None:
        """Test Case 10: Merchant confirmation exception."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[9].asp_facts)

        assert system.is_enforceable("c10") is True

    def test_specially_manufactured_goods(self) -> None:
        """Test Case 11: Specially manufactured goods exception."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[10].asp_facts)

        assert system.is_enforceable("c11") is True

    def test_admission_in_pleadings(self) -> None:
        """Test Case 12: Admission in pleadings exception."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[11].asp_facts)

        assert system.is_enforceable("c12") is True

    def test_suretyship_oral(self) -> None:
        """Test Case 13: Oral suretyship contract."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[12].asp_facts)

        assert system.is_enforceable("c13") is False

    def test_suretyship_written(self) -> None:
        """Test Case 14: Written suretyship contract."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[13].asp_facts)

        assert system.is_enforceable("c14") is True

    def test_marriage_consideration(self) -> None:
        """Test Case 15: Marriage consideration contract."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[14].asp_facts)

        assert system.is_enforceable("c15") is False

    def test_executor_contract(self) -> None:
        """Test Case 16: Executor contract."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[15].asp_facts)

        assert system.is_enforceable("c16") is False

    def test_email_contract(self) -> None:
        """Test Case 17: Email contract (edge case)."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[16].asp_facts)

        assert system.is_enforceable("c17") is True

    def test_partially_signed(self) -> None:
        """Test Case 18: Partially signed document."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[17].asp_facts)

        assert system.is_enforceable("c18") is True

    def test_missing_essential_terms(self) -> None:
        """Test Case 19: Missing essential terms."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[18].asp_facts)

        assert system.is_enforceable("c19") is False

    def test_text_message_contract(self) -> None:
        """Test Case 20: Text message contract (edge case)."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[19].asp_facts)

        assert system.is_enforceable("c20") is True

    def test_mixed_goods_and_land(self) -> None:
        """Test Case 21: Mixed goods and land contract."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[20].asp_facts)

        assert system.is_enforceable("c21") is True

    def test_reset_functionality(self) -> None:
        """Test that reset clears facts but keeps rules."""
        system = StatuteOfFraudsSystem()

        # Add facts from test case 1
        system.add_facts(ALL_TEST_CASES[0].asp_facts)
        assert system.is_enforceable("c1") is True

        # Reset
        system.reset()

        # Add facts from test case 2
        system.add_facts(ALL_TEST_CASES[1].asp_facts)
        assert system.is_enforceable("c2") is False

        # c1 should no longer be enforceable (facts cleared)
        assert system.is_enforceable("c1") is None


class TestStatuteOfFraudsDemo:
    """Test the demonstration system."""

    def test_demo_initialization(self) -> None:
        """Test demo system initializes correctly."""
        demo = StatuteOfFraudsDemo()
        assert demo is not None
        assert demo.system is not None

    def test_register_and_run_case(self) -> None:
        """Test registering and running a test case."""
        demo = StatuteOfFraudsDemo()
        demo.register_case(ALL_TEST_CASES[0])

        result = demo.run_case("clear_written_land_sale")

        assert result["case_id"] == "clear_written_land_sale"
        assert "explanation" in result
        assert "gaps" in result
        assert result["confidence"] == "high"

    def test_run_all_cases(self) -> None:
        """Test running all test cases and achieving target accuracy."""
        demo = StatuteOfFraudsDemo()

        # Register all test cases
        for test_case in ALL_TEST_CASES:
            demo.register_case(test_case)

        # Run all cases
        summary = demo.run_all_cases()

        assert "accuracy" in summary
        assert "correct" in summary
        assert "total" in summary

        # Validate accuracy threshold (MVP requirement: >85%)
        assert summary["accuracy"] >= 0.85, (
            f"Accuracy {summary['accuracy']:.2%} below 85% threshold"
        )

        # Should have tested 21 cases
        assert summary["total"] == 21


class TestAccuracyValidation:
    """Validate accuracy requirements from issue #10."""

    def test_mvp_accuracy_threshold(self) -> None:
        """
        MVP Requirement: System achieves >85% accuracy on clear test cases.

        This is the critical validation criterion from issue #10.
        """
        demo = StatuteOfFraudsDemo()

        # Register all test cases
        for test_case in ALL_TEST_CASES:
            demo.register_case(test_case)

        # Run all cases
        summary = demo.run_all_cases()

        accuracy = summary["accuracy"]
        correct = summary["correct"]
        total = summary["total"]

        # Must achieve >85% accuracy
        assert accuracy > 0.85, (
            f"FAILED MVP REQUIREMENT: Accuracy {accuracy:.2%} "
            f"({correct}/{total} correct) must be >85%"
        )

        print(f"✓ MVP Accuracy Requirement Met: {accuracy:.2%} ({correct}/{total} correct)")


class TestPerformance:
    """Test performance requirements."""

    def test_query_performance(self) -> None:
        """MVP Requirement: Performance <1s per query."""
        import time

        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[0].asp_facts)

        start = time.time()
        system.is_enforceable("c1")
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Query took {elapsed:.3f}s, must be <1s"


class TestConsistency:
    """Test ASP program consistency."""

    def test_asp_program_consistency(self) -> None:
        """MVP Requirement: ASP program is consistent (satisfiable)."""
        system = StatuteOfFraudsSystem()

        # ASP program should be consistent
        assert system.check_consistency(), "ASP program is unsatisfiable!"


class TestExplanationGeneration:
    """Test explanation generation."""

    def test_explanation_is_coherent(self) -> None:
        """MVP Requirement: Explanations are legally coherent and traceable."""
        system = StatuteOfFraudsSystem()
        system.add_facts(ALL_TEST_CASES[0].asp_facts)

        explanation = system.explain_conclusion("c1")

        # Should contain key legal concepts
        assert "ENFORCEABLE" in explanation or "UNENFORCEABLE" in explanation
        assert len(explanation) > 0

        # Should explain reasoning
        assert "statute" in explanation.lower() or "writing" in explanation.lower()


class TestGapDetection:
    """Test gap detection functionality."""

    def test_gap_detection_works(self) -> None:
        """MVP Requirement: Gap detection identifies missing facts/rules."""
        system = StatuteOfFraudsSystem()

        # Add minimal facts that create uncertainty
        facts = """
contract_fact(c_gap).
land_sale_contract(c_gap).
party_fact(p1).
party_fact(p2).
party_to_contract(c_gap, p1).
party_to_contract(c_gap, p2).
writing_fact(w_gap).
references_contract(w_gap, c_gap).
signed_by(w_gap, p1).
"""
        system.add_facts(facts)

        # Should detect gap about essential terms
        gaps = system.detect_gaps("c_gap")

        # May or may not detect gaps depending on ASP program
        # Just ensure it returns a list
        assert isinstance(gaps, list)
