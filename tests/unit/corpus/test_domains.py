"""Tests for corpus domain configuration."""

from loft.corpus.domains import (
    LegalDomain,
    DomainConfig,
    DOMAIN_CONFIGS,
    get_domain_config,
    get_all_domains,
    get_domain_by_directory,
)


class TestLegalDomain:
    """Tests for LegalDomain enum."""

    def test_all_domains_exist(self):
        """Test that all expected domains exist."""
        expected = [
            "contracts",
            "torts",
            "property_law",
            "procedural",
            "statute_of_frauds",
            "adverse_possession",
        ]

        for domain_value in expected:
            assert LegalDomain(domain_value) is not None

    def test_domain_values(self):
        """Test domain enum values."""
        assert LegalDomain.CONTRACTS.value == "contracts"
        assert LegalDomain.TORTS.value == "torts"
        assert LegalDomain.PROCEDURAL.value == "procedural"


class TestDomainConfig:
    """Tests for DomainConfig dataclass."""

    def test_config_creation(self):
        """Test creating a domain config."""
        config = DomainConfig(
            name="Test Domain",
            directory="test",
            description="A test domain",
            subdomains=["sub1", "sub2"],
        )

        assert config.name == "Test Domain"
        assert config.directory == "test"
        assert len(config.subdomains) == 2

    def test_path_property(self):
        """Test path property."""
        config = DomainConfig(
            name="Test",
            directory="my_domain",
            description="Test",
        )

        assert str(config.path) == "datasets/my_domain"


class TestDomainConfigs:
    """Tests for DOMAIN_CONFIGS dictionary."""

    def test_all_domains_have_configs(self):
        """Test that all domains have configurations."""
        for domain in LegalDomain:
            assert domain in DOMAIN_CONFIGS

    def test_configs_have_required_fields(self):
        """Test that all configs have required fields."""
        for domain, config in DOMAIN_CONFIGS.items():
            assert config.name, f"{domain} missing name"
            assert config.directory, f"{domain} missing directory"
            assert config.description, f"{domain} missing description"

    def test_contracts_config(self):
        """Test contracts domain configuration."""
        config = DOMAIN_CONFIGS[LegalDomain.CONTRACTS]

        assert config.name == "Contract Law"
        assert config.directory == "contracts"
        assert "formation" in config.subdomains
        assert "contract" in config.common_predicates

    def test_torts_config(self):
        """Test torts domain configuration."""
        config = DOMAIN_CONFIGS[LegalDomain.TORTS]

        assert config.name == "Tort Law"
        assert config.directory == "torts"
        assert "negligence" in config.subdomains
        assert "duty_owed" in config.common_predicates

    def test_procedural_config(self):
        """Test procedural domain configuration."""
        config = DOMAIN_CONFIGS[LegalDomain.PROCEDURAL]

        assert config.name == "Civil Procedure"
        assert "standing" in config.subdomains
        assert "jurisdiction" in config.subdomains


class TestGetDomainConfig:
    """Tests for get_domain_config function."""

    def test_get_valid_config(self):
        """Test getting a valid domain config."""
        config = get_domain_config(LegalDomain.CONTRACTS)

        assert config.name == "Contract Law"

    def test_all_domains_retrievable(self):
        """Test that all domains can be retrieved."""
        for domain in LegalDomain:
            config = get_domain_config(domain)
            assert config is not None


class TestGetAllDomains:
    """Tests for get_all_domains function."""

    def test_returns_all_domains(self):
        """Test that all domains are returned."""
        domains = get_all_domains()

        assert len(domains) == len(LegalDomain)
        assert LegalDomain.CONTRACTS in domains
        assert LegalDomain.TORTS in domains


class TestGetDomainByDirectory:
    """Tests for get_domain_by_directory function."""

    def test_find_by_directory(self):
        """Test finding domain by directory name."""
        domain = get_domain_by_directory("contracts")
        assert domain == LegalDomain.CONTRACTS

        domain = get_domain_by_directory("torts")
        assert domain == LegalDomain.TORTS

    def test_nonexistent_directory(self):
        """Test finding nonexistent directory."""
        domain = get_domain_by_directory("nonexistent")
        assert domain is None
