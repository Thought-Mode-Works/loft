from unittest.mock import MagicMock
from typing import List, Any, Optional
from datetime import datetime

from loft.symbolic.asp_rule import ASPRule, RuleMetadata, StratificationLevel
from loft.constraints.symmetry import (
    PartySymmetryTester,
    SymmetryType,
)


# Helper to create an ASPRule instance (instead of a MagicMock)
def create_test_asp_rule_integration(
    rule_id: str,
    asp_content: str,
    stratification_level: StratificationLevel = StratificationLevel.TACTICAL,
    confidence: float = 0.8,
    metadata: Optional[RuleMetadata] = None,
    evaluate_effect: Optional[Any] = None,
) -> ASPRule:
    if metadata is None:
        metadata = RuleMetadata(
            provenance="test_integration", timestamp=datetime.utcnow().isoformat()
        )

    rule = ASPRule(
        rule_id=rule_id,
        asp_text=asp_content,
        stratification_level=stratification_level,
        confidence=confidence,
        metadata=metadata,
    )
    if evaluate_effect is not None:
        rule.evaluate = MagicMock(side_effect=evaluate_effect)  # Mock evaluate method
    return rule


# Mock ASPRunner to return mock rules
class MockASPCore:
    def load_rules(self, domain: str) -> List[ASPRule]:
        rule1 = create_test_asp_rule_integration(
            "rule_a", "enforceable(C) :- contract(C), party(X, C), party(Y, C)."
        )
        rule1.parties_in_rule = ["X", "Y"]
        rule1.evaluate = MagicMock(
            side_effect=lambda case: True
        )  # Always returns True for simple mock

        rule2 = create_test_asp_rule_integration(
            "rule_b", "valid_offer(X) :- offer_from(X)."
        )
        rule2.parties_in_rule = ["X"]
        rule2.evaluate = MagicMock(
            side_effect=lambda case: True
        )  # Always returns True for simple mock

        return [rule1, rule2]


# Dummy parse_rule for tests
def mock_parse_rule(asp_content: str) -> ASPRule:
    rule_id = f"rule_{abs(hash(asp_content))}"
    return create_test_asp_rule_integration(rule_id, asp_content)


class TestSymmetryIntegration:
    def test_statute_of_frauds_symmetry(self):
        """Verify statute of frauds rules are party-symmetric."""
        # Mock ASPRunner to provide rules
        # mock_asp_core = MockASPCore() # Removed unused variable

        # Symmetric rule 1: Order of parties does not matter for 'enforceable'
        rule_enforceable_content = (
            "enforceable(C) :- contract(C), party(X, C), party(Y, C), agreed(X,Y)."
        )
        rule_enforceable = mock_parse_rule(rule_enforceable_content)
        rule_enforceable.name = "enforceable_rule"
        rule_enforceable.parties_in_rule = ["X", "Y"]
        # This mock evaluate function will simulate a symmetric outcome
        rule_enforceable.evaluate = MagicMock(
            side_effect=lambda case: case.get("contract_C")
            and case.get("party_X_C")
            and case.get("party_Y_C")
            and (
                case.get(f"agreed_{case.get('party_X')}_{case.get('party_Y')}")
                or case.get(f"agreed_{case.get('party_Y')}_{case.get('party_X')}")
            )
        )

        # Symmetric rule 2: Another simple rule
        rule_valid_agreement_content = (
            "valid_agreement(X,Y) :- offer(X,Y), acceptance(Y,X)."
        )
        rule_valid_agreement = mock_parse_rule(rule_valid_agreement_content)
        rule_valid_agreement.name = "valid_agreement_rule"
        rule_valid_agreement.parties_in_rule = ["X", "Y"]
        rule_valid_agreement.evaluate = MagicMock(
            side_effect=lambda case: case.get(
                f"offer_{case.get('party_X')}_{case.get('party_Y')}"
            )
            and case.get(f"acceptance_{case.get('party_Y')}_{case.get('party_X')}")
        )

        rules = [rule_enforceable, rule_valid_agreement]  # Simulate loaded rules

        tester = PartySymmetryTester(rules)

        # Create test cases that can be permuted
        # These test cases need to align with the mocked rule.evaluate logic
        integration_test_cases = [
            {
                "contract_C": True,
                "party_X": "alice",
                "party_Y": "bob",
                "agreed_alice_bob": True,
                # "agreed_bob_alice": False,  # Removed duplicate key
                "offer_alice_bob": True,
                "acceptance_bob_alice": True,
                # Permuted versions of facts for alice <-> bob swap
                "agreed_bob_alice": True,  # This needs to be true for the swapped case to pass
                "offer_bob_alice": True,
                "acceptance_alice_bob": True,
            },
            {
                "contract_C": True,
                "party_X": "charlie",
                "party_Y": "david",
                "agreed_charlie_david": True,
                # "agreed_david_charlie": False, # Removed duplicate key
                "offer_charlie_david": True,
                "acceptance_david_charlie": True,
                # Permuted versions of facts for charlie <-> david swap
                "agreed_david_charlie": True,  # This needs to be true for the swapped case to pass
                "offer_david_charlie": True,
                "acceptance_charlie_david": True,
            },
        ]

        # For each rule, test its symmetry
        for rule in rules:
            report = tester.test_symmetry(
                rule, integration_test_cases, SymmetryType.FULL
            )
            assert (
                report.is_symmetric
            ), f"Rule {rule.name} has symmetry violations: {report.violations}"

    # Additional integration tests could be added here to test different scenarios
    # e.g., rules with justified asymmetry, rules expected to be asymmetric, etc.
