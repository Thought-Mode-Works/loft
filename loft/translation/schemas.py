"""
Pydantic schemas for structured NL → ASP translation.

This module provides schemas for extracting structured legal entities
from natural language and converting them to ASP facts.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class LegalEntity(BaseModel):
    """Base schema for legal entities."""

    entity_id: str = Field(description="Unique identifier for the entity")
    entity_type: str = Field(description="Type of entity (contract, party, writing)")


class Party(BaseModel):
    """Schema for a party to a contract."""

    name: str = Field(description="Name of the party")
    party_id: Optional[str] = Field(default=None, description="Unique identifier")

    def to_asp(self) -> List[str]:
        """Convert party to ASP facts."""
        pid = self.party_id or self.name.lower().replace(" ", "_")
        return [f"party({pid})."]


class Writing(BaseModel):
    """Schema for a writing/document."""

    writing_id: str = Field(description="Unique identifier for the writing")
    is_signed: bool = Field(default=False, description="Whether the writing is signed")
    signed_by: Optional[List[str]] = Field(default=None, description="List of parties who signed")

    def to_asp(self) -> List[str]:
        """Convert writing to ASP facts."""
        facts = [f"writing({self.writing_id})."]

        if self.is_signed and self.signed_by:
            for party in self.signed_by:
                party_id = party.lower().replace(" ", "_")
                facts.append(f"signed_by({self.writing_id}, {party_id}).")

        return facts


class ContractFact(BaseModel):
    """Schema for contract-related facts."""

    contract_id: str = Field(description="Unique identifier for the contract")
    contract_type: Optional[str] = Field(
        default=None,
        description="Type of contract (land_sale, goods_sale, service, etc.)",
    )
    parties: List[str] = Field(default_factory=list, description="Parties to the contract")
    has_writing: bool = Field(default=False, description="Whether contract has a writing")
    writing_id: Optional[str] = Field(default=None, description="ID of associated writing")
    is_signed: bool = Field(default=False, description="Whether the writing is signed")
    sale_amount: Optional[float] = Field(default=None, description="Sale amount if applicable")
    has_consideration: bool = Field(default=False, description="Whether contract has consideration")
    has_mutual_assent: bool = Field(default=False, description="Whether parties have mutual assent")

    def to_asp(self) -> List[str]:
        """Convert contract to ASP facts."""
        facts = [f"contract({self.contract_id})."]

        # Add contract type
        if self.contract_type:
            type_id = self.contract_type.lower().replace(" ", "_")
            facts.append(f"{type_id}_contract({self.contract_id}).")

        # Add parties
        for party in self.parties:
            party_id = party.lower().replace(" ", "_")
            facts.append(f"party({party_id}).")
            facts.append(f"party_to_contract({self.contract_id}, {party_id}).")

        # Add writing
        if self.has_writing:
            wid = self.writing_id or f"w_{self.contract_id}"
            facts.append(f"writing({wid}).")
            facts.append(f"has_writing({self.contract_id}, {wid}).")

            if self.is_signed:
                # Sign with first party if no specific signing info
                if self.parties:
                    party_id = self.parties[0].lower().replace(" ", "_")
                    facts.append(f"signed_by({wid}, {party_id}).")

        # Add sale amount
        if self.sale_amount is not None:
            facts.append(f"sale_amount({self.contract_id}, {int(self.sale_amount)}).")

        # Add essential elements
        if self.has_consideration:
            facts.append(f"has_consideration({self.contract_id}).")

        if self.has_mutual_assent:
            facts.append(f"has_mutual_assent({self.contract_id}).")

        return facts


class LegalRelationship(BaseModel):
    """Schema for relationships between legal entities."""

    subject: str = Field(description="Subject of the relationship")
    predicate: str = Field(description="Type of relationship")
    object: str = Field(description="Object of the relationship")

    def to_asp(self) -> str:
        """Convert relationship to ASP fact."""
        subj = self.subject.lower().replace(" ", "_")
        obj = self.object.lower().replace(" ", "_")
        pred = self.predicate.lower().replace(" ", "_")
        return f"{pred}({subj}, {obj})."


class ExtractedEntities(BaseModel):
    """Schema for batch entity extraction from natural language."""

    contracts: List[ContractFact] = Field(default_factory=list, description="Extracted contracts")
    parties: List[Party] = Field(default_factory=list, description="Extracted parties")
    writings: List[Writing] = Field(default_factory=list, description="Extracted writings")
    relationships: List[LegalRelationship] = Field(
        default_factory=list, description="Extracted relationships"
    )

    def to_asp(self) -> List[str]:
        """Convert all extracted entities to ASP facts."""
        facts = []

        # Add contracts
        for contract in self.contracts:
            facts.extend(contract.to_asp())

        # Add parties (deduplicate)
        party_ids = set()
        for party in self.parties:
            pid = party.party_id or party.name.lower().replace(" ", "_")
            if pid not in party_ids:
                facts.extend(party.to_asp())
                party_ids.add(pid)

        # Add writings
        for writing in self.writings:
            facts.extend(writing.to_asp())

        # Add relationships
        for rel in self.relationships:
            facts.append(rel.to_asp())

        return facts


class LegalRule(BaseModel):
    """Schema for extracting legal rules from natural language."""

    rule_id: str = Field(description="Unique identifier for the rule")
    head_predicate: str = Field(description="Predicate in the rule head")
    head_arguments: List[str] = Field(
        description="Arguments in the head predicate (e.g., ['C'] for variables)"
    )
    body_conditions: List[str] = Field(description="List of conditions in the rule body")
    is_negation_as_failure: bool = Field(
        default=False, description="Whether rule uses negation-as-failure"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence in the extracted rule"
    )

    def to_asp(self) -> str:
        """Convert legal rule to ASP syntax."""
        # Build head
        head_args = ", ".join(self.head_arguments)
        head = f"{self.head_predicate}({head_args})"

        # Build body
        if not self.body_conditions:
            # Fact, not a rule
            return f"{head}."

        body = ", ".join(self.body_conditions)

        return f"{head} :- {body}."
