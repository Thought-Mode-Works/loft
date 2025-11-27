"""
Unit tests for Pydantic schemas for NL → ASP translation.

Tests schema creation, validation, and ASP fact generation.
"""

import pytest
from loft.translation.schemas import (
    LegalEntity,
    Party,
    Writing,
    ContractFact,
    LegalRelationship,
    ExtractedEntities,
    LegalRule,
)


class TestLegalEntity:
    """Test LegalEntity schema."""

    def test_creation(self):
        """Test creating a legal entity."""
        entity = LegalEntity(entity_id="e1", entity_type="contract")
        assert entity.entity_id == "e1"
        assert entity.entity_type == "contract"

    def test_validation(self):
        """Test schema validation."""
        # Should succeed with valid data
        entity = LegalEntity(entity_id="e1", entity_type="party")
        assert entity.entity_id == "e1"


class TestParty:
    """Test Party schema."""

    def test_creation_with_id(self):
        """Test creating party with explicit ID."""
        party = Party(name="John Doe", party_id="p1")
        assert party.name == "John Doe"
        assert party.party_id == "p1"

    def test_creation_without_id(self):
        """Test creating party without explicit ID."""
        party = Party(name="John Doe")
        assert party.name == "John Doe"
        assert party.party_id is None

    def test_to_asp_with_id(self):
        """Test ASP conversion with explicit ID."""
        party = Party(name="John Doe", party_id="john")
        asp_facts = party.to_asp()

        assert len(asp_facts) == 1
        assert asp_facts[0] == "party(john)."

    def test_to_asp_without_id(self):
        """Test ASP conversion without explicit ID."""
        party = Party(name="John Doe")
        asp_facts = party.to_asp()

        assert len(asp_facts) == 1
        assert "party(john_doe)." in asp_facts

    def test_to_asp_name_normalization(self):
        """Test that names are normalized in ASP."""
        party = Party(name="Mary Jane Smith")
        asp_facts = party.to_asp()

        assert len(asp_facts) == 1
        assert "party(mary_jane_smith)." in asp_facts


class TestWriting:
    """Test Writing schema."""

    def test_creation_minimal(self):
        """Test creating minimal writing."""
        writing = Writing(writing_id="w1")
        assert writing.writing_id == "w1"
        assert not writing.is_signed
        assert writing.signed_by is None

    def test_creation_signed(self):
        """Test creating signed writing."""
        writing = Writing(writing_id="w1", is_signed=True, signed_by=["John", "Mary"])
        assert writing.is_signed
        assert len(writing.signed_by) == 2

    def test_to_asp_unsigned(self):
        """Test ASP conversion for unsigned writing."""
        writing = Writing(writing_id="w1")
        asp_facts = writing.to_asp()

        assert len(asp_facts) == 1
        assert "writing(w1)." in asp_facts

    def test_to_asp_signed(self):
        """Test ASP conversion for signed writing."""
        writing = Writing(writing_id="w1", is_signed=True, signed_by=["John Doe", "Mary"])
        asp_facts = writing.to_asp()

        assert "writing(w1)." in asp_facts
        assert "signed_by(w1, john_doe)." in asp_facts
        assert "signed_by(w1, mary)." in asp_facts

    def test_to_asp_signed_no_signers(self):
        """Test ASP conversion when signed but no signers listed."""
        writing = Writing(writing_id="w1", is_signed=True)
        asp_facts = writing.to_asp()

        # Should only have writing fact
        assert len(asp_facts) == 1
        assert "writing(w1)." in asp_facts


class TestContractFact:
    """Test ContractFact schema."""

    def test_creation_minimal(self):
        """Test creating minimal contract."""
        contract = ContractFact(contract_id="c1")
        assert contract.contract_id == "c1"
        assert contract.contract_type is None
        assert contract.parties == []
        assert not contract.has_writing

    def test_creation_full(self):
        """Test creating full contract with all fields."""
        contract = ContractFact(
            contract_id="c1",
            contract_type="land_sale",
            parties=["John", "Mary"],
            has_writing=True,
            writing_id="w1",
            is_signed=True,
            sale_amount=500000.0,
            has_consideration=True,
            has_mutual_assent=True,
        )
        assert contract.contract_id == "c1"
        assert contract.contract_type == "land_sale"
        assert len(contract.parties) == 2
        assert contract.sale_amount == 500000.0

    def test_to_asp_minimal(self):
        """Test ASP conversion for minimal contract."""
        contract = ContractFact(contract_id="c1")
        asp_facts = contract.to_asp()

        assert "contract(c1)." in asp_facts

    def test_to_asp_with_type(self):
        """Test ASP conversion with contract type."""
        contract = ContractFact(contract_id="c1", contract_type="land_sale")
        asp_facts = contract.to_asp()

        assert "contract(c1)." in asp_facts
        assert "land_sale_contract(c1)." in asp_facts

    def test_to_asp_with_parties(self):
        """Test ASP conversion with parties."""
        contract = ContractFact(contract_id="c1", parties=["John", "Mary"])
        asp_facts = contract.to_asp()

        assert "party(john)." in asp_facts
        assert "party(mary)." in asp_facts
        assert "party_to_contract(c1, john)." in asp_facts
        assert "party_to_contract(c1, mary)." in asp_facts

    def test_to_asp_with_writing(self):
        """Test ASP conversion with writing."""
        contract = ContractFact(contract_id="c1", has_writing=True, writing_id="w1")
        asp_facts = contract.to_asp()

        assert "writing(w1)." in asp_facts
        assert "has_writing(c1, w1)." in asp_facts

    def test_to_asp_with_default_writing_id(self):
        """Test ASP conversion with default writing ID."""
        contract = ContractFact(contract_id="c1", has_writing=True)
        asp_facts = contract.to_asp()

        assert "writing(w_c1)." in asp_facts
        assert "has_writing(c1, w_c1)." in asp_facts

    def test_to_asp_with_signed_writing(self):
        """Test ASP conversion with signed writing."""
        contract = ContractFact(
            contract_id="c1",
            parties=["John"],
            has_writing=True,
            writing_id="w1",
            is_signed=True,
        )
        asp_facts = contract.to_asp()

        assert "signed_by(w1, john)." in asp_facts

    def test_to_asp_with_sale_amount(self):
        """Test ASP conversion with sale amount."""
        contract = ContractFact(contract_id="c1", sale_amount=500000.0)
        asp_facts = contract.to_asp()

        assert "sale_amount(c1, 500000)." in asp_facts

    def test_to_asp_with_float_amount(self):
        """Test ASP conversion with float sale amount."""
        contract = ContractFact(contract_id="c1", sale_amount=500000.50)
        asp_facts = contract.to_asp()

        # Should convert to int
        assert "sale_amount(c1, 500000)." in asp_facts

    def test_to_asp_with_essential_elements(self):
        """Test ASP conversion with essential elements."""
        contract = ContractFact(contract_id="c1", has_consideration=True, has_mutual_assent=True)
        asp_facts = contract.to_asp()

        assert "has_consideration(c1)." in asp_facts
        assert "has_mutual_assent(c1)." in asp_facts

    def test_to_asp_type_normalization(self):
        """Test that contract types are normalized."""
        contract = ContractFact(contract_id="c1", contract_type="Land Sale")
        asp_facts = contract.to_asp()

        assert "land_sale_contract(c1)." in asp_facts


class TestLegalRelationship:
    """Test LegalRelationship schema."""

    def test_creation(self):
        """Test creating a legal relationship."""
        rel = LegalRelationship(subject="contract c1", predicate="signed by", object="john")
        assert rel.subject == "contract c1"
        assert rel.predicate == "signed by"
        assert rel.object == "john"

    def test_to_asp(self):
        """Test ASP conversion."""
        rel = LegalRelationship(subject="contract c1", predicate="signed by", object="john")
        asp_fact = rel.to_asp()

        assert asp_fact == "signed_by(contract_c1, john)."

    def test_to_asp_normalization(self):
        """Test that identifiers are normalized."""
        rel = LegalRelationship(subject="Contract C1", predicate="Signed By", object="John Doe")
        asp_fact = rel.to_asp()

        assert asp_fact == "signed_by(contract_c1, john_doe)."


class TestExtractedEntities:
    """Test ExtractedEntities schema."""

    def test_creation_empty(self):
        """Test creating empty entity collection."""
        entities = ExtractedEntities()
        assert entities.contracts == []
        assert entities.parties == []
        assert entities.writings == []
        assert entities.relationships == []

    def test_creation_with_data(self):
        """Test creating with data."""
        contract = ContractFact(contract_id="c1")
        party = Party(name="John")
        writing = Writing(writing_id="w1")
        rel = LegalRelationship(subject="c1", predicate="has", object="w1")

        entities = ExtractedEntities(
            contracts=[contract],
            parties=[party],
            writings=[writing],
            relationships=[rel],
        )

        assert len(entities.contracts) == 1
        assert len(entities.parties) == 1
        assert len(entities.writings) == 1
        assert len(entities.relationships) == 1

    def test_to_asp_empty(self):
        """Test ASP conversion for empty entities."""
        entities = ExtractedEntities()
        asp_facts = entities.to_asp()

        assert asp_facts == []

    def test_to_asp_with_contracts(self):
        """Test ASP conversion with contracts."""
        contract = ContractFact(contract_id="c1", contract_type="land_sale")
        entities = ExtractedEntities(contracts=[contract])
        asp_facts = entities.to_asp()

        assert "contract(c1)." in asp_facts
        assert "land_sale_contract(c1)." in asp_facts

    def test_to_asp_with_parties(self):
        """Test ASP conversion with parties."""
        party1 = Party(name="John", party_id="john")
        party2 = Party(name="Mary", party_id="mary")
        entities = ExtractedEntities(parties=[party1, party2])
        asp_facts = entities.to_asp()

        assert "party(john)." in asp_facts
        assert "party(mary)." in asp_facts

    def test_to_asp_deduplicates_parties(self):
        """Test that duplicate parties are removed."""
        party1 = Party(name="John", party_id="john")
        party2 = Party(name="John", party_id="john")  # Duplicate
        entities = ExtractedEntities(parties=[party1, party2])
        asp_facts = entities.to_asp()

        # Should only have one party fact
        party_facts = [f for f in asp_facts if f.startswith("party(")]
        assert len(party_facts) == 1

    def test_to_asp_with_writings(self):
        """Test ASP conversion with writings."""
        writing = Writing(writing_id="w1", is_signed=True, signed_by=["john"])
        entities = ExtractedEntities(writings=[writing])
        asp_facts = entities.to_asp()

        assert "writing(w1)." in asp_facts
        assert "signed_by(w1, john)." in asp_facts

    def test_to_asp_with_relationships(self):
        """Test ASP conversion with relationships."""
        rel = LegalRelationship(subject="c1", predicate="involves", object="john")
        entities = ExtractedEntities(relationships=[rel])
        asp_facts = entities.to_asp()

        assert "involves(c1, john)." in asp_facts

    def test_to_asp_comprehensive(self):
        """Test ASP conversion with all entity types."""
        contract = ContractFact(contract_id="c1", parties=["john"])
        party = Party(name="Mary", party_id="mary")
        writing = Writing(writing_id="w1")
        rel = LegalRelationship(subject="c1", predicate="has_writing", object="w1")

        entities = ExtractedEntities(
            contracts=[contract],
            parties=[party],
            writings=[writing],
            relationships=[rel],
        )
        asp_facts = entities.to_asp()

        # Should have facts from all sources
        assert "contract(c1)." in asp_facts
        assert "party(john)." in asp_facts
        assert "party(mary)." in asp_facts
        assert "writing(w1)." in asp_facts
        assert "has_writing(c1, w1)." in asp_facts


class TestLegalRule:
    """Test LegalRule schema."""

    def test_creation_minimal(self):
        """Test creating minimal rule."""
        rule = LegalRule(
            rule_id="r1", head_predicate="enforceable", head_arguments=["C"], body_conditions=[]
        )
        assert rule.rule_id == "r1"
        assert rule.head_predicate == "enforceable"
        assert rule.head_arguments == ["C"]
        assert rule.confidence == 0.7  # Default

    def test_creation_full(self):
        """Test creating full rule."""
        rule = LegalRule(
            rule_id="r1",
            head_predicate="enforceable",
            head_arguments=["C"],
            body_conditions=["contract(C)", "not void(C)"],
            is_negation_as_failure=True,
            confidence=0.95,
        )
        assert rule.is_negation_as_failure
        assert rule.confidence == 0.95

    def test_to_asp_fact(self):
        """Test ASP conversion for fact (no body)."""
        rule = LegalRule(
            rule_id="r1", head_predicate="contract", head_arguments=["c1"], body_conditions=[]
        )
        asp = rule.to_asp()

        assert asp == "contract(c1)."

    def test_to_asp_simple_rule(self):
        """Test ASP conversion for simple rule."""
        rule = LegalRule(
            rule_id="r1",
            head_predicate="enforceable",
            head_arguments=["C"],
            body_conditions=["contract(C)"],
        )
        asp = rule.to_asp()

        assert asp == "enforceable(C) :- contract(C)."

    def test_to_asp_rule_with_negation(self):
        """Test ASP conversion for rule with negation."""
        rule = LegalRule(
            rule_id="r1",
            head_predicate="enforceable",
            head_arguments=["C"],
            body_conditions=["contract(C)", "not void(C)"],
            is_negation_as_failure=True,
        )
        asp = rule.to_asp()

        assert asp == "enforceable(C) :- contract(C), not void(C)."

    def test_to_asp_complex_rule(self):
        """Test ASP conversion for complex rule."""
        rule = LegalRule(
            rule_id="r1",
            head_predicate="satisfies_sof",
            head_arguments=["C"],
            body_conditions=["contract(C)", "has_writing(C, W)", "signed_by(W, P)"],
        )
        asp = rule.to_asp()

        assert asp == "satisfies_sof(C) :- contract(C), has_writing(C, W), signed_by(W, P)."

    def test_to_asp_multiple_head_args(self):
        """Test ASP conversion with multiple head arguments."""
        rule = LegalRule(
            rule_id="r1",
            head_predicate="signed_by",
            head_arguments=["W", "P"],
            body_conditions=["writing(W)", "party(P)"],
        )
        asp = rule.to_asp()

        assert asp == "signed_by(W, P) :- writing(W), party(P)."

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        rule = LegalRule(
            rule_id="r1",
            head_predicate="test",
            head_arguments=["X"],
            body_conditions=[],
            confidence=0.5,
        )
        assert rule.confidence == 0.5

        # Test boundary values
        rule_low = LegalRule(
            rule_id="r2",
            head_predicate="test",
            head_arguments=["X"],
            body_conditions=[],
            confidence=0.0,
        )
        assert rule_low.confidence == 0.0

        rule_high = LegalRule(
            rule_id="r3",
            head_predicate="test",
            head_arguments=["X"],
            body_conditions=[],
            confidence=1.0,
        )
        assert rule_high.confidence == 1.0
