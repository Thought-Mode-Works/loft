"""
Domain configuration for legal corpus.

Defines the structure and metadata for each legal domain.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set


class LegalDomain(str, Enum):
    """Legal domains supported by the corpus."""

    CONTRACTS = "contracts"
    TORTS = "torts"
    PROPERTY = "property_law"
    PROCEDURAL = "procedural"
    STATUTE_OF_FRAUDS = "statute_of_frauds"
    ADVERSE_POSSESSION = "adverse_possession"


@dataclass
class DomainConfig:
    """Configuration for a legal domain."""

    name: str
    directory: str
    description: str
    subdomains: List[str] = field(default_factory=list)
    common_predicates: Set[str] = field(default_factory=set)
    related_domains: List[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        """Get the path to this domain's dataset directory."""
        return Path("datasets") / self.directory


DOMAIN_CONFIGS: Dict[LegalDomain, DomainConfig] = {
    LegalDomain.CONTRACTS: DomainConfig(
        name="Contract Law",
        directory="contracts",
        description="Contract formation, breach, defenses, and remedies",
        subdomains=[
            "formation",
            "consideration",
            "capacity",
            "defenses",
            "breach",
            "discharge",
            "remedies",
        ],
        common_predicates={
            "contract",
            "offer",
            "acceptance",
            "consideration",
            "capacity",
            "breach",
        },
        related_domains=["statute_of_frauds"],
    ),
    LegalDomain.TORTS: DomainConfig(
        name="Tort Law",
        directory="torts",
        description="Negligence, strict liability, and intentional torts",
        subdomains=[
            "negligence",
            "strict_liability",
            "intentional_torts",
            "defamation",
            "product_liability",
        ],
        common_predicates={
            "claim",
            "claim_type",
            "defendant",
            "plaintiff",
            "duty_owed",
            "duty_breached",
            "actual_cause",
            "proximate_cause",
            "damages",
        },
        related_domains=["procedural"],
    ),
    LegalDomain.PROPERTY: DomainConfig(
        name="Property Law",
        directory="property_law",
        description="Real property, personal property, and estates",
        subdomains=[
            "adverse_possession",
            "easements",
            "estates",
            "landlord_tenant",
            "recording_acts",
        ],
        common_predicates={
            "claim",
            "claimant",
            "property_type",
            "occupation_continuous",
            "occupation_hostile",
        },
        related_domains=["adverse_possession"],
    ),
    LegalDomain.PROCEDURAL: DomainConfig(
        name="Civil Procedure",
        directory="procedural",
        description="Jurisdiction, standing, and claim preclusion",
        subdomains=[
            "standing",
            "jurisdiction",
            "res_judicata",
            "limitations",
            "venue",
        ],
        common_predicates={
            "claim",
            "claim_type",
            "plaintiff",
            "defendant",
            "jurisdiction",
        },
        related_domains=["torts", "contracts"],
    ),
    LegalDomain.STATUTE_OF_FRAUDS: DomainConfig(
        name="Statute of Frauds",
        directory="statute_of_frauds",
        description="Writing requirements for certain contracts",
        subdomains=[
            "land_sale",
            "goods",
            "services",
            "suretyship",
            "marriage",
        ],
        common_predicates={
            "contract",
            "subject_matter",
            "has_writing",
            "has_consideration",
        },
        related_domains=["contracts"],
    ),
    LegalDomain.ADVERSE_POSSESSION: DomainConfig(
        name="Adverse Possession",
        directory="adverse_possession",
        description="Acquiring title through adverse possession",
        subdomains=[
            "successful_claim",
            "tacking",
            "tolling",
        ],
        common_predicates={
            "claim",
            "claimant",
            "property_type",
            "occupation_years",
            "occupation_continuous",
            "occupation_hostile",
            "statutory_period",
        },
        related_domains=["property_law"],
    ),
}


def get_domain_config(domain: LegalDomain) -> DomainConfig:
    """Get configuration for a domain."""
    return DOMAIN_CONFIGS[domain]


def get_all_domains() -> List[LegalDomain]:
    """Get all available domains."""
    return list(LegalDomain)


def get_domain_by_directory(directory: str) -> Optional[LegalDomain]:
    """Find domain by directory name."""
    for domain, config in DOMAIN_CONFIGS.items():
        if config.directory == directory:
            return domain
    return None
