"""
Configuration management for legal API integrations.

Supports loading API keys and settings from environment variables,
configuration files, or direct instantiation.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from loguru import logger

from loft.external.base import LegalAPIProvider
from loft.external.courtlistener import CourtListenerClient


@dataclass
class APIConfig:
    """
    Configuration for legal API providers.

    Supports multiple ways to configure APIs:
    1. Environment variables (COURT_LISTENER_API_TOKEN)
    2. Configuration dict
    3. Direct instantiation
    """

    # CourtListener settings
    courtlistener_api_key: Optional[str] = None
    courtlistener_enabled: bool = True

    # Global settings
    timeout: int = 30
    max_retries: int = 3
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = 60

    @classmethod
    def from_env(cls) -> "APIConfig":
        """
        Load configuration from environment variables.

        Environment variables:
        - COURT_LISTENER_API_TOKEN: CourtListener API token
        - COURTLISTENER_ENABLED: Enable CourtListener (true/false)
        - LEGAL_API_TIMEOUT: Request timeout in seconds
        - LEGAL_API_MAX_RETRIES: Maximum retry attempts
        - LEGAL_API_CACHE_ENABLED: Enable caching (true/false)
        - LEGAL_API_CACHE_TTL: Cache TTL in seconds

        Returns:
            APIConfig with settings from environment
        """
        return cls(
            courtlistener_api_key=os.getenv("COURT_LISTENER_API_TOKEN"),
            courtlistener_enabled=os.getenv("COURTLISTENER_ENABLED", "true").lower() == "true",
            timeout=int(os.getenv("LEGAL_API_TIMEOUT", "30")),
            max_retries=int(os.getenv("LEGAL_API_MAX_RETRIES", "3")),
            cache_enabled=os.getenv("LEGAL_API_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("LEGAL_API_CACHE_TTL", "3600")),
            rate_limit_enabled=os.getenv("LEGAL_API_RATE_LIMIT_ENABLED", "true").lower() == "true",
            requests_per_minute=int(os.getenv("LEGAL_API_REQUESTS_PER_MINUTE", "60")),
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, any]) -> "APIConfig":
        """
        Load configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            APIConfig with settings from dict
        """
        return cls(
            courtlistener_api_key=config_dict.get("courtlistener_api_key"),
            courtlistener_enabled=config_dict.get("courtlistener_enabled", True),
            timeout=config_dict.get("timeout", 30),
            max_retries=config_dict.get("max_retries", 3),
            cache_enabled=config_dict.get("cache_enabled", True),
            cache_ttl_seconds=config_dict.get("cache_ttl_seconds", 3600),
            rate_limit_enabled=config_dict.get("rate_limit_enabled", True),
            requests_per_minute=config_dict.get("requests_per_minute", 60),
        )

    def get_enabled_providers(self) -> List[str]:
        """
        Get list of enabled provider names.

        Returns:
            List of enabled provider names
        """
        enabled = []

        if self.courtlistener_enabled:
            enabled.append("courtlistener")

        return enabled

    def get_configured_providers(self) -> List[str]:
        """
        Get list of providers with API keys configured.

        Returns:
            List of configured provider names
        """
        configured = []

        if self.courtlistener_api_key:
            configured.append("courtlistener")

        return configured


def get_configured_clients(
    config: Optional[APIConfig] = None,
) -> Dict[str, LegalAPIProvider]:
    """
    Get dictionary of configured API clients.

    Args:
        config: APIConfig to use (loads from env if None)

    Returns:
        Dict mapping provider name to client instance
    """
    if config is None:
        config = APIConfig.from_env()

    clients = {}

    # CourtListener
    if config.courtlistener_enabled:
        try:
            client = CourtListenerClient(
                api_key=config.courtlistener_api_key,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
            clients["courtlistener"] = client
            logger.info("Initialized CourtListener client")
        except Exception as e:
            logger.warning(f"Failed to initialize CourtListener: {e}")

    logger.info(f"Configured {len(clients)} legal API clients")

    return clients


def validate_config(config: APIConfig) -> List[str]:
    """
    Validate API configuration.

    Args:
        config: Configuration to validate

    Returns:
        List of validation warnings/errors
    """
    warnings = []

    # Check if any providers are enabled
    if not config.get_enabled_providers():
        warnings.append("No API providers enabled")

    # Check if enabled providers have API keys
    enabled = config.get_enabled_providers()
    configured = config.get_configured_providers()

    for provider in enabled:
        if provider not in configured:
            warnings.append(
                f"{provider} is enabled but no API key configured (limited functionality)"
            )

    # Validate timeout
    if config.timeout < 5:
        warnings.append("Timeout is very low (< 5 seconds)")

    # Validate cache TTL
    if config.cache_enabled and config.cache_ttl_seconds < 60:
        warnings.append("Cache TTL is very low (< 60 seconds)")

    return warnings
