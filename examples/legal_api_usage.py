"""
Example usage of the Legal API integration system.

This example demonstrates how to:
1. Initialize the API manager with CourtListener token from .env
2. Search for legal cases
3. Get specific documents
4. Search by citation
5. Use caching and configuration options
"""

import os
from dotenv import load_dotenv

from loft.external import (
    LegalAPIManager,
    SearchQuery,
    APIConfig,
    CourtListenerClient,
)

# Load environment variables from .env file
load_dotenv()


def example_basic_usage():
    """Example 1: Basic usage with environment variables."""
    print("=" * 60)
    print("Example 1: Basic Usage with Environment Variables")
    print("=" * 60)

    # Initialize manager - automatically loads COURT_LISTENER_API_TOKEN from .env
    manager = LegalAPIManager()

    # Check available providers
    providers = manager.get_available_providers()
    print(f"\nAvailable providers: {providers}")

    # Simple search
    query = SearchQuery(
        query_text="contract law",
        max_results=5
    )

    print(f"\nSearching for: '{query.query_text}'...")
    result = manager.search(query)

    print(f"Found {result.total_results} results")
    print(f"Showing {len(result.documents)} documents")

    for i, doc in enumerate(result.documents, 1):
        print(f"\n{i}. {doc.case_name}")
        print(f"   Court: {doc.court}")
        print(f"   URL: {doc.url}")


def example_advanced_search():
    """Example 2: Advanced search with filters."""
    print("\n" + "=" * 60)
    print("Example 2: Advanced Search with Filters")
    print("=" * 60)

    manager = LegalAPIManager()

    # Search with jurisdiction and court filters
    from datetime import datetime

    query = SearchQuery(
        query_text="constitutional rights",
        jurisdiction="federal",
        court="scotus",  # Supreme Court
        date_from=datetime(2020, 1, 1),
        max_results=10
    )

    print(f"\nSearching for: '{query.query_text}'")
    print(f"Jurisdiction: {query.jurisdiction}")
    print(f"Court: {query.court}")
    print(f"From date: {query.date_from}")

    result = manager.search(query)

    print(f"\nFound {result.total_results} results")
    print(f"Search time: {result.search_time_ms:.2f}ms")


def example_citation_search():
    """Example 3: Search by citation."""
    print("\n" + "=" * 60)
    print("Example 3: Search by Citation")
    print("=" * 60)

    manager = LegalAPIManager()

    # Search for a specific case by citation
    citation = "410 U.S. 113"  # Roe v. Wade
    print(f"\nSearching for citation: {citation}")

    case = manager.search_by_citation(citation)

    if case:
        print(f"\nFound: {case.case_name}")
        print(f"Court: {case.court}")
        print(f"Decision date: {case.decision_date}")
        print(f"Citations: {case.citations}")
    else:
        print(f"No case found for citation: {citation}")


def example_custom_configuration():
    """Example 4: Custom configuration."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)

    # Create custom config
    config = APIConfig(
        courtlistener_api_key=os.getenv("COURT_LISTENER_API_TOKEN"),
        courtlistener_enabled=True,
        timeout=60,  # Longer timeout
        cache_enabled=True,
        cache_ttl_seconds=7200,  # 2 hour cache
        rate_limit_enabled=True,
        requests_per_minute=30  # Conservative rate limit
    )

    manager = LegalAPIManager(config=config)

    print(f"\nConfiguration:")
    print(f"  Timeout: {config.timeout}s")
    print(f"  Cache enabled: {config.cache_enabled}")
    print(f"  Cache TTL: {config.cache_ttl_seconds}s")
    print(f"  Rate limit: {config.requests_per_minute} requests/minute")

    # Test search with custom config
    query = SearchQuery(query_text="privacy law", max_results=3)
    result = manager.search(query)

    print(f"\nSearch completed: {len(result.documents)} results")

    # Check cache stats
    stats = manager.get_cache_stats()
    print(f"\nCache stats:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Active entries: {stats['active_entries']}")


def example_direct_client():
    """Example 5: Using CourtListener client directly."""
    print("\n" + "=" * 60)
    print("Example 5: Direct Client Usage")
    print("=" * 60)

    # Use CourtListener client directly
    client = CourtListenerClient(
        api_key=os.getenv("COURT_LISTENER_API_TOKEN"),
        timeout=30,
        max_retries=3
    )

    print(f"\nClient: {client.get_provider_name()}")
    print(f"Available: {client.is_available()}")

    # Direct search
    query = SearchQuery(
        query_text="employment discrimination",
        max_results=5
    )

    result = client.search(query)

    print(f"\nResults: {len(result.documents)}")
    for doc in result.documents[:3]:
        print(f"  - {doc.case_name}")


def example_caching_demo():
    """Example 6: Demonstrating caching."""
    print("\n" + "=" * 60)
    print("Example 6: Caching Demonstration")
    print("=" * 60)

    manager = LegalAPIManager()

    query = SearchQuery(query_text="trademark", max_results=5)

    # First search (hits API)
    print("\n1. First search (hits API)...")
    result1 = manager.search(query)
    print(f"   Search time: {result1.search_time_ms:.2f}ms")

    # Second search (uses cache)
    print("\n2. Second search (uses cache)...")
    result2 = manager.search(query)
    print(f"   Search time: {result2.search_time_ms:.2f}ms")
    print(f"   Results identical: {len(result1.documents) == len(result2.documents)}")

    # Clear cache
    print("\n3. Clearing cache...")
    manager.clear_cache()
    print("   Cache cleared")

    # Third search (hits API again)
    print("\n4. Third search (hits API again)...")
    result3 = manager.search(query)
    print(f"   Search time: {result3.search_time_ms:.2f}ms")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Legal API Integration Examples")
    print("=" * 60)
    print("\nThese examples demonstrate using the CourtListener API")
    print("via the legal API integration system.")
    print("\nNote: You need COURT_LISTENER_API_TOKEN in your .env file")
    print("=" * 60)

    # Check if API token is configured
    if not os.getenv("COURT_LISTENER_API_TOKEN"):
        print("\nERROR: COURT_LISTENER_API_TOKEN not found in environment!")
        print("Please add it to your .env file")
        exit(1)

    # Run examples
    try:
        example_basic_usage()
        example_advanced_search()
        example_citation_search()
        example_custom_configuration()
        example_direct_client()
        example_caching_demo()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()
