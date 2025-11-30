"""
Corpus utilities for multi-domain legal datasets.

This module provides utilities for loading, querying, and managing
legal test case corpora across multiple domains.
"""

from loft.corpus.loader import (
    CorpusLoader,
    LegalCase,
    CorpusStats,
    get_corpus_stats,
    load_domain,
    load_all_domains,
)
from loft.corpus.domains import (
    LegalDomain,
    DOMAIN_CONFIGS,
    get_domain_config,
)

__all__ = [
    "CorpusLoader",
    "LegalCase",
    "CorpusStats",
    "get_corpus_stats",
    "load_domain",
    "load_all_domains",
    "LegalDomain",
    "DOMAIN_CONFIGS",
    "get_domain_config",
]
