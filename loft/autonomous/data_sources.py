"""
Data Source Adapters for Autonomous Test Harness.

This module provides adapters for different data sources including:
- CourtListener API for real legal case data
- Local JSON/JSONL files
- Generated test data

Each adapter converts data into the format expected by the runner.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CaseData:
    """Standardized case data for autonomous processing.

    Attributes:
        id: Unique case identifier
        domain: Legal domain (contracts, property, torts, etc.)
        facts: Case facts description
        text: Full case text (optional)
        metadata: Additional case metadata
    """

    id: str
    domain: str
    facts: str
    text: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for runner."""
        return {
            "id": self.id,
            "domain": self.domain,
            "facts": self.facts,
            "text": self.text,
            **self.metadata,
        }


class DataSourceAdapter(ABC):
    """Abstract base class for data source adapters."""

    @abstractmethod
    def get_cases(self, limit: Optional[int] = None) -> Iterator[CaseData]:
        """Yield cases from the data source.

        Args:
            limit: Maximum number of cases to yield

        Yields:
            CaseData objects
        """
        pass

    @abstractmethod
    def get_case_count(self) -> int:
        """Get total number of available cases."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Get human-readable source name."""
        pass


class CourtListenerAdapter(DataSourceAdapter):
    """Adapter for fetching cases from CourtListener API.

    Fetches real legal cases from the Free Law Project's CourtListener
    database and converts them to the format expected by the runner.

    Example usage:
        adapter = CourtListenerAdapter(
            api_key="your_api_key",
            search_queries=["statute of frauds", "adverse possession"],
            domains={"statute of frauds": "contracts", "adverse possession": "property"},
            max_per_query=50
        )
        for case in adapter.get_cases(limit=100):
            process(case)
    """

    # Default domain mappings based on search terms
    DEFAULT_DOMAIN_MAPPINGS = {
        "statute of frauds": "contracts",
        "contract breach": "contracts",
        "consideration": "contracts",
        "promissory estoppel": "contracts",
        "adverse possession": "property",
        "easement": "property",
        "property rights": "property",
        "negligence": "torts",
        "duty of care": "torts",
        "intentional tort": "torts",
        "strict liability": "torts",
        "product liability": "torts",
        "search and seizure": "constitutional",
        "fourth amendment": "constitutional",
        "due process": "constitutional",
        "equal protection": "constitutional",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        domain_mappings: Optional[Dict[str, str]] = None,
        max_per_query: int = 50,
        jurisdictions: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ):
        """Initialize CourtListener adapter.

        Args:
            api_key: CourtListener API key (optional for limited access)
            search_queries: List of search terms to fetch cases for
            domain_mappings: Map of search terms to domain names
            max_per_query: Maximum cases to fetch per query
            jurisdictions: Filter to specific jurisdictions
            date_from: Filter cases filed after this date
            date_to: Filter cases filed before this date
        """
        self._api_key = api_key or os.environ.get("COURT_LISTENER_API_TOKEN")
        self._search_queries = search_queries or list(
            self.DEFAULT_DOMAIN_MAPPINGS.keys()
        )
        self._domain_mappings = {
            **self.DEFAULT_DOMAIN_MAPPINGS,
            **(domain_mappings or {}),
        }
        self._max_per_query = max_per_query
        self._jurisdictions = jurisdictions
        self._date_from = date_from
        self._date_to = date_to
        self._client = None
        self._cached_cases: List[CaseData] = []
        self._fetched = False

    def _get_client(self):
        """Lazily initialize CourtListener client."""
        if self._client is None:
            from loft.external.courtlistener import CourtListenerClient

            self._client = CourtListenerClient(api_key=self._api_key)
        return self._client

    def _fetch_cases(self) -> None:
        """Fetch cases from CourtListener API."""
        if self._fetched:
            return

        from loft.external.base import SearchQuery

        client = self._get_client()

        for query_text in self._search_queries:
            domain = self._domain_mappings.get(query_text, "general")

            try:
                query = SearchQuery(
                    query_text=query_text,
                    max_results=self._max_per_query,
                    date_from=self._date_from,
                    date_to=self._date_to,
                )

                result = client.search(query)

                for doc in result.documents:
                    case = self._convert_document(doc, domain)
                    self._cached_cases.append(case)

                logger.info(
                    f"Fetched {len(result.documents)} cases for query '{query_text}' "
                    f"(domain: {domain})"
                )

            except Exception as e:
                logger.error(f"Error fetching cases for query '{query_text}': {e}")
                continue

        self._fetched = True
        logger.info(f"Total cases fetched: {len(self._cached_cases)}")

    def _convert_document(self, doc, domain: str) -> CaseData:
        """Convert CaseLawDocument to CaseData.

        Args:
            doc: CaseLawDocument from CourtListener
            domain: Legal domain classification

        Returns:
            CaseData instance
        """
        # Extract facts from summary or first part of text
        facts = doc.summary
        if not facts and doc.text:
            # Use first 500 chars as facts summary
            facts = doc.text[:500].strip()
            if len(doc.text) > 500:
                facts += "..."

        metadata = {
            "case_name": doc.case_name,
            "court": doc.court,
            "court_level": doc.court_level.value if doc.court_level else None,
            "docket_number": doc.docket_number,
            "decision_date": (
                doc.decision_date.isoformat() if doc.decision_date else None
            ),
            "citations": doc.citations,
            "judges": doc.judges,
            "source_api": doc.source_api,
            "url": doc.url,
            "relevance_score": doc.relevance_score,
        }

        return CaseData(
            id=doc.document_id,
            domain=domain,
            facts=facts,
            text=doc.text,
            metadata=metadata,
        )

    def get_cases(self, limit: Optional[int] = None) -> Iterator[CaseData]:
        """Yield cases from CourtListener.

        Args:
            limit: Maximum number of cases to yield

        Yields:
            CaseData objects
        """
        self._fetch_cases()

        count = 0
        for case in self._cached_cases:
            if limit and count >= limit:
                break
            yield case
            count += 1

    def get_case_count(self) -> int:
        """Get total number of available cases."""
        self._fetch_cases()
        return len(self._cached_cases)

    @property
    def source_name(self) -> str:
        """Get human-readable source name."""
        return "CourtListener API"


class LocalFileAdapter(DataSourceAdapter):
    """Adapter for loading cases from local JSON/JSONL files.

    Supports:
    - JSON files with array of cases
    - JSON files with {"cases": [...]} structure
    - JSONL files with one case per line
    - Directories containing multiple case files
    """

    def __init__(self, paths: List[str]):
        """Initialize local file adapter.

        Args:
            paths: List of file or directory paths
        """
        self._paths = [Path(p) for p in paths]
        self._cached_cases: List[CaseData] = []
        self._loaded = False

    def _load_cases(self) -> None:
        """Load cases from all specified paths."""
        if self._loaded:
            return

        for path in self._paths:
            if not path.exists():
                logger.warning(f"Path does not exist: {path}")
                continue

            if path.is_dir():
                self._load_from_directory(path)
            elif path.suffix == ".jsonl":
                self._load_jsonl(path)
            elif path.suffix == ".json":
                self._load_json(path)
            else:
                logger.warning(f"Unsupported file type: {path}")

        self._loaded = True
        logger.info(f"Loaded {len(self._cached_cases)} cases from local files")

    def _load_from_directory(self, directory: Path) -> None:
        """Load cases from all JSON/JSONL files in directory."""
        for file_path in directory.glob("**/*.json"):
            self._load_json(file_path)
        for file_path in directory.glob("**/*.jsonl"):
            self._load_jsonl(file_path)

    def _load_json(self, file_path: Path) -> None:
        """Load cases from JSON file."""
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                cases = data
            elif isinstance(data, dict) and "cases" in data:
                cases = data["cases"]
            else:
                logger.warning(f"Unexpected JSON structure in {file_path}")
                return

            for case_data in cases:
                case = self._parse_case(case_data)
                if case:
                    self._cached_cases.append(case)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    def _load_jsonl(self, file_path: Path) -> None:
        """Load cases from JSONL file."""
        try:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        case_data = json.loads(line)
                        case = self._parse_case(case_data)
                        if case:
                            self._cached_cases.append(case)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    def _parse_case(self, data: Dict[str, Any]) -> Optional[CaseData]:
        """Parse case data from dictionary."""
        try:
            # Extract required fields
            case_id = str(data.get("id", data.get("case_id", "")))
            if not case_id:
                case_id = f"case_{len(self._cached_cases)}"

            domain = data.get("domain", data.get("legal_domain", "general"))
            facts = data.get("facts", data.get("summary", data.get("text", "")))

            # All other fields go to metadata
            metadata = {
                k: v
                for k, v in data.items()
                if k not in ["id", "case_id", "domain", "facts"]
            }

            return CaseData(
                id=case_id,
                domain=domain,
                facts=facts,
                text=data.get("text", ""),
                metadata=metadata,
            )

        except Exception as e:
            logger.warning(f"Error parsing case data: {e}")
            return None

    def get_cases(self, limit: Optional[int] = None) -> Iterator[CaseData]:
        """Yield cases from local files."""
        self._load_cases()

        count = 0
        for case in self._cached_cases:
            if limit and count >= limit:
                break
            yield case
            count += 1

    def get_case_count(self) -> int:
        """Get total number of available cases."""
        self._load_cases()
        return len(self._cached_cases)

    @property
    def source_name(self) -> str:
        """Get human-readable source name."""
        return f"Local Files ({len(self._paths)} paths)"


class CompositeAdapter(DataSourceAdapter):
    """Combines multiple data source adapters."""

    def __init__(self, adapters: List[DataSourceAdapter]):
        """Initialize composite adapter.

        Args:
            adapters: List of data source adapters
        """
        self._adapters = adapters

    def get_cases(self, limit: Optional[int] = None) -> Iterator[CaseData]:
        """Yield cases from all adapters."""
        count = 0
        for adapter in self._adapters:
            for case in adapter.get_cases():
                if limit and count >= limit:
                    return
                yield case
                count += 1

    def get_case_count(self) -> int:
        """Get total cases across all adapters."""
        return sum(adapter.get_case_count() for adapter in self._adapters)

    @property
    def source_name(self) -> str:
        """Get combined source names."""
        names = [adapter.source_name for adapter in self._adapters]
        return " + ".join(names)


def create_courtlistener_adapter(
    search_queries: Optional[List[str]] = None,
    domains: Optional[Dict[str, str]] = None,
    max_cases_per_query: int = 50,
) -> CourtListenerAdapter:
    """Factory function to create a CourtListener adapter.

    Args:
        search_queries: Search queries to use (default: common legal topics)
        domains: Domain mappings for queries
        max_cases_per_query: Max cases to fetch per query

    Returns:
        Configured CourtListenerAdapter
    """
    return CourtListenerAdapter(
        search_queries=search_queries,
        domain_mappings=domains,
        max_per_query=max_cases_per_query,
    )


def create_local_adapter(paths: List[str]) -> LocalFileAdapter:
    """Factory function to create a local file adapter.

    Args:
        paths: Paths to files or directories

    Returns:
        Configured LocalFileAdapter
    """
    return LocalFileAdapter(paths)
