"""
Unified manager for legal API integrations.

Provides:
- Multi-provider search with fallback
- Response caching
- Rate limiting
- Fault tolerance
- Result aggregation
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger

from loft.external.base import (
    LegalAPIProvider,
    CaseLawDocument,
    SearchQuery,
    SearchResult,
    APIError,
    RateLimitError,
)
from loft.external.config import APIConfig, get_configured_clients


class LegalAPIManager:
    """
    Unified manager for legal API access.

    Features:
    - Multi-provider search with automatic fallback
    - Response caching to reduce API calls
    - Rate limiting to respect API limits
    - Fault tolerance with retries
    - Result aggregation and deduplication
    """

    def __init__(
        self,
        config: Optional[APIConfig] = None,
        clients: Optional[Dict[str, LegalAPIProvider]] = None,
    ):
        """
        Initialize API manager.

        Args:
            config: API configuration (loads from env if None)
            clients: Pre-configured clients (creates from config if None)
        """
        self.config = config or APIConfig.from_env()

        if clients is None:
            self.clients = get_configured_clients(self.config)
        else:
            self.clients = clients

        # Cache for responses
        self.cache: Dict[str, Dict[str, Any]] = {}

        # Rate limiting state
        self.request_history: Dict[str, List[float]] = {
            provider: [] for provider in self.clients.keys()
        }

        logger.info(
            f"Initialized LegalAPIManager with {len(self.clients)} providers: "
            f"{list(self.clients.keys())}"
        )

    def search(
        self,
        query: SearchQuery,
        providers: Optional[List[str]] = None,
        use_cache: bool = True,
        aggregate_results: bool = False,
    ) -> SearchResult:
        """
        Search across legal databases.

        Args:
            query: Search parameters
            providers: List of provider names to use (all if None)
            use_cache: Whether to use cached results
            aggregate_results: Combine results from multiple providers

        Returns:
            SearchResult from first successful provider, or aggregated results

        Raises:
            APIError: If all providers fail
        """
        # Determine which providers to use
        if providers is None:
            providers = list(self.clients.keys())

        # Check cache first
        if use_cache and self.config.cache_enabled:
            cache_key = self._generate_cache_key(query, providers)
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.info(f"Returning cached results for query: {query.query_text}")
                return SearchResult(**cached)

        # Try each provider
        results = []
        errors = []

        for provider_name in providers:
            client = self.clients.get(provider_name)
            if not client:
                logger.warning(f"Provider {provider_name} not available")
                continue

            try:
                # Check rate limit
                if self.config.rate_limit_enabled:
                    self._wait_for_rate_limit(provider_name)

                # Execute search
                logger.info(f"Searching {provider_name} for: {query.query_text[:50]}...")
                result = client.search(query)

                # Record request
                self._record_request(provider_name)

                # Cache result
                if use_cache and self.config.cache_enabled:
                    cache_key = self._generate_cache_key(query, [provider_name])
                    self._store_in_cache(cache_key, result.to_dict())

                if aggregate_results:
                    results.append(result)
                else:
                    # Return first successful result
                    return result

            except RateLimitError as e:
                logger.warning(f"Rate limit hit for {provider_name}: {e}")
                errors.append((provider_name, str(e)))
                continue

            except Exception as e:
                logger.warning(f"Error with {provider_name}: {e}")
                errors.append((provider_name, str(e)))
                continue

        # Aggregate results if requested
        if aggregate_results and results:
            return self._aggregate_results(results, query)

        # Return first result if any
        if results:
            return results[0]

        # All providers failed
        error_msg = "; ".join([f"{p}: {e}" for p, e in errors])
        raise APIError(f"All providers failed: {error_msg}")

    def get_document(
        self,
        document_id: str,
        provider: str,
        use_cache: bool = True,
    ) -> CaseLawDocument:
        """
        Get specific document.

        Args:
            document_id: Document identifier
            provider: Provider name
            use_cache: Whether to use cached document

        Returns:
            CaseLawDocument

        Raises:
            APIError: If retrieval fails
        """
        client = self.clients.get(provider)
        if not client:
            raise APIError(f"Provider {provider} not available")

        # Check cache
        if use_cache and self.config.cache_enabled:
            cache_key = f"doc:{provider}:{document_id}"
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.info(f"Returning cached document {document_id}")
                return CaseLawDocument(**cached)

        # Check rate limit
        if self.config.rate_limit_enabled:
            self._wait_for_rate_limit(provider)

        # Fetch document
        document = client.get_document(document_id)

        # Record request
        self._record_request(provider)

        # Cache document
        if use_cache and self.config.cache_enabled:
            cache_key = f"doc:{provider}:{document_id}"
            self._store_in_cache(cache_key, document.to_dict())

        return document

    def get_available_providers(self) -> List[str]:
        """
        Get list of available providers.

        Returns:
            List of provider names that are configured and available
        """
        return [name for name, client in self.clients.items() if client.is_available()]

    def search_by_citation(
        self, citation: str, providers: Optional[List[str]] = None
    ) -> Optional[CaseLawDocument]:
        """
        Find case by citation across providers.

        Args:
            citation: Legal citation (e.g., "410 U.S. 113")
            providers: Providers to search (all if None)

        Returns:
            CaseLawDocument if found, None otherwise
        """
        if providers is None:
            providers = list(self.clients.keys())

        for provider_name in providers:
            client = self.clients.get(provider_name)
            if not client:
                continue

            try:
                # Try provider-specific citation search if available
                if hasattr(client, "search_by_citation"):
                    doc = client.search_by_citation(citation)
                    if doc:
                        return doc
                elif hasattr(client, "get_by_citation"):
                    doc = client.get_by_citation(citation)
                    if doc:
                        return doc
                else:
                    # Fall back to regular search
                    query = SearchQuery(query_text=citation, max_results=1)
                    result = client.search(query)
                    if result.documents:
                        return result.documents[0]

            except Exception as e:
                logger.warning(f"Error searching {provider_name} for citation '{citation}': {e}")
                continue

        return None

    def clear_cache(self):
        """Clear all cached responses."""
        self.cache.clear()
        logger.info("Cleared API response cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_entries = len(self.cache)
        total_size_bytes = sum(len(json.dumps(v)) for v in self.cache.values())

        # Count expired entries
        now = datetime.now()
        expired = sum(
            1
            for v in self.cache.values()
            if datetime.fromisoformat(v["cached_at"])
            + timedelta(seconds=self.config.cache_ttl_seconds)
            < now
        )

        return {
            "total_entries": total_entries,
            "expired_entries": expired,
            "active_entries": total_entries - expired,
            "total_size_bytes": total_size_bytes,
            "ttl_seconds": self.config.cache_ttl_seconds,
        }

    def _generate_cache_key(self, query: SearchQuery, providers: List[str]) -> str:
        """Generate cache key for query."""
        key_data = {
            "query": query.to_dict(),
            "providers": sorted(providers),
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache if not expired."""
        if key not in self.cache:
            return None

        cached = self.cache[key]
        cached_at = datetime.fromisoformat(cached["cached_at"])
        ttl = timedelta(seconds=self.config.cache_ttl_seconds)

        if datetime.now() - cached_at > ttl:
            # Expired
            del self.cache[key]
            return None

        return cached["data"]

    def _store_in_cache(self, key: str, data: Dict[str, Any]):
        """Store value in cache."""
        self.cache[key] = {
            "data": data,
            "cached_at": datetime.now().isoformat(),
        }

    def _wait_for_rate_limit(self, provider: str):
        """Wait if necessary to respect rate limits."""
        if provider not in self.request_history:
            return

        # Clean old requests
        now = time.time()
        cutoff = now - 60  # Last minute
        self.request_history[provider] = [
            ts for ts in self.request_history[provider] if ts > cutoff
        ]

        # Check if we need to wait
        recent_requests = len(self.request_history[provider])
        if recent_requests >= self.config.requests_per_minute:
            # Wait until oldest request is outside the window
            oldest = min(self.request_history[provider])
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f}s for {provider}")
                time.sleep(wait_time)

    def _record_request(self, provider: str):
        """Record a request for rate limiting."""
        if provider in self.request_history:
            self.request_history[provider].append(time.time())

    def _aggregate_results(self, results: List[SearchResult], query: SearchQuery) -> SearchResult:
        """
        Aggregate results from multiple providers.

        Deduplicates based on case name and removes duplicates.

        Args:
            results: List of SearchResults from different providers
            query: Original query

        Returns:
            Aggregated SearchResult
        """
        all_documents = []
        seen_cases = set()

        for result in results:
            for doc in result.documents:
                # Deduplicate by case name
                case_key = doc.case_name.lower().strip()
                if case_key not in seen_cases:
                    seen_cases.add(case_key)
                    all_documents.append(doc)

        # Sort by relevance score
        all_documents.sort(key=lambda d: d.relevance_score, reverse=True)

        # Limit to requested max_results
        all_documents = all_documents[: query.max_results]

        return SearchResult(
            query=query,
            documents=all_documents,
            total_results=sum(r.total_results for r in results),
            page=query.offset // query.max_results + 1,
            has_more=any(r.has_more for r in results),
            search_time_ms=sum(r.search_time_ms for r in results),
            source_api="aggregated",
        )
