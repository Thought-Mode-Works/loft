"""
Corpus loader for multi-domain legal datasets.

Provides utilities for loading, filtering, and querying legal test cases.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

from loft.corpus.domains import (
    DOMAIN_CONFIGS,
    LegalDomain,
    get_domain_by_directory,
)


@dataclass
class LegalCase:
    """A single legal test case."""

    id: str
    domain: str
    subdomain: Optional[str]
    description: str
    facts: List[str]
    asp_facts: str
    question: str
    ground_truth: str
    rationale: str
    legal_citations: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    source_file: Optional[Path] = None

    @classmethod
    def from_json(cls, data: Dict, source_file: Optional[Path] = None) -> "LegalCase":
        """Create a LegalCase from JSON data."""
        return cls(
            id=data.get("id", ""),
            domain=data.get("domain", ""),
            subdomain=data.get("subdomain"),
            description=data.get("description", ""),
            facts=data.get("facts", []),
            asp_facts=data.get("asp_facts", ""),
            question=data.get("question", ""),
            ground_truth=data.get("ground_truth", ""),
            rationale=data.get("rationale", ""),
            legal_citations=data.get("legal_citations", []),
            difficulty=data.get("difficulty", "medium"),
            source_file=source_file,
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "description": self.description,
            "facts": self.facts,
            "asp_facts": self.asp_facts,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "rationale": self.rationale,
            "legal_citations": self.legal_citations,
            "difficulty": self.difficulty,
        }


@dataclass
class CorpusStats:
    """Statistics about the corpus."""

    total_cases: int
    cases_by_domain: Dict[str, int]
    cases_by_outcome: Dict[str, int]
    cases_by_difficulty: Dict[str, int]
    domains_loaded: List[str]
    unique_predicates: Set[str] = field(default_factory=set)

    def __str__(self) -> str:
        """Format stats as string."""
        lines = [
            f"Total cases: {self.total_cases}",
            f"Domains: {len(self.domains_loaded)}",
            "",
            "Cases by domain:",
        ]
        for domain, count in sorted(self.cases_by_domain.items()):
            lines.append(f"  {domain}: {count}")

        lines.append("")
        lines.append("Cases by outcome:")
        for outcome, count in sorted(self.cases_by_outcome.items()):
            pct = count / self.total_cases * 100 if self.total_cases > 0 else 0
            lines.append(f"  {outcome}: {count} ({pct:.1f}%)")

        lines.append("")
        lines.append("Cases by difficulty:")
        for difficulty, count in sorted(self.cases_by_difficulty.items()):
            lines.append(f"  {difficulty}: {count}")

        return "\n".join(lines)


class CorpusLoader:
    """Load and query legal test cases from the corpus."""

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the corpus loader.

        Args:
            base_path: Base path to datasets directory.
                      Defaults to 'datasets' in project root.
        """
        if base_path is None:
            # Try to find datasets relative to this file or cwd
            candidates = [
                Path(__file__).parent.parent.parent.parent / "datasets",
                Path.cwd() / "datasets",
            ]
            for candidate in candidates:
                if candidate.exists():
                    base_path = candidate
                    break
            else:
                base_path = Path("datasets")

        self.base_path = Path(base_path)
        self._cases: List[LegalCase] = []
        self._loaded_domains: Set[str] = set()

    def load_domain(self, domain: LegalDomain) -> List[LegalCase]:
        """
        Load all cases from a specific domain.

        Args:
            domain: The domain to load

        Returns:
            List of cases from the domain
        """
        config = DOMAIN_CONFIGS[domain]
        domain_path = self.base_path / config.directory

        if not domain_path.exists():
            return []

        cases = []
        for json_file in sorted(domain_path.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Set domain if not in file
                if "domain" not in data:
                    data["domain"] = config.directory

                case = LegalCase.from_json(data, source_file=json_file)
                cases.append(case)
                self._cases.append(case)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue

        self._loaded_domains.add(config.directory)
        return cases

    def load_all_domains(self) -> List[LegalCase]:
        """
        Load cases from all available domains.

        Returns:
            List of all cases
        """
        self._cases = []
        self._loaded_domains = set()

        for domain in LegalDomain:
            self.load_domain(domain)

        return self._cases

    def load_directory(self, directory: str) -> List[LegalCase]:
        """
        Load cases from a directory by name.

        Args:
            directory: Directory name (e.g., 'torts', 'contracts')

        Returns:
            List of cases from the directory
        """
        domain = get_domain_by_directory(directory)
        if domain:
            return self.load_domain(domain)

        # Try loading directly if not a known domain
        dir_path = self.base_path / directory
        if not dir_path.exists():
            return []

        cases = []
        for json_file in sorted(dir_path.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "domain" not in data:
                    data["domain"] = directory
                case = LegalCase.from_json(data, source_file=json_file)
                cases.append(case)
                self._cases.append(case)
            except (json.JSONDecodeError, KeyError):
                continue

        self._loaded_domains.add(directory)
        return cases

    def get_cases(
        self,
        domain: Optional[str] = None,
        subdomain: Optional[str] = None,
        outcome: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> List[LegalCase]:
        """
        Get cases matching the specified filters.

        Args:
            domain: Filter by domain
            subdomain: Filter by subdomain
            outcome: Filter by ground truth outcome
            difficulty: Filter by difficulty level

        Returns:
            List of matching cases
        """
        cases = self._cases

        if domain:
            cases = [c for c in cases if c.domain == domain]
        if subdomain:
            cases = [c for c in cases if c.subdomain == subdomain]
        if outcome:
            cases = [c for c in cases if c.ground_truth == outcome]
        if difficulty:
            cases = [c for c in cases if c.difficulty == difficulty]

        return cases

    def get_stats(self) -> CorpusStats:
        """
        Get statistics about the loaded corpus.

        Returns:
            CorpusStats with aggregate information
        """
        cases_by_domain: Dict[str, int] = {}
        cases_by_outcome: Dict[str, int] = {}
        cases_by_difficulty: Dict[str, int] = {}
        predicates: Set[str] = set()

        for case in self._cases:
            # Count by domain
            cases_by_domain[case.domain] = cases_by_domain.get(case.domain, 0) + 1

            # Count by outcome
            cases_by_outcome[case.ground_truth] = (
                cases_by_outcome.get(case.ground_truth, 0) + 1
            )

            # Count by difficulty
            cases_by_difficulty[case.difficulty] = (
                cases_by_difficulty.get(case.difficulty, 0) + 1
            )

            # Extract predicates from asp_facts
            import re

            for match in re.finditer(r"([a-z_][a-z0-9_]*)\s*\(", case.asp_facts):
                predicates.add(match.group(1))

        return CorpusStats(
            total_cases=len(self._cases),
            cases_by_domain=cases_by_domain,
            cases_by_outcome=cases_by_outcome,
            cases_by_difficulty=cases_by_difficulty,
            domains_loaded=list(self._loaded_domains),
            unique_predicates=predicates,
        )

    def iter_cases(self) -> Iterator[LegalCase]:
        """Iterate over all loaded cases."""
        yield from self._cases

    def __len__(self) -> int:
        """Return number of loaded cases."""
        return len(self._cases)

    def __iter__(self) -> Iterator[LegalCase]:
        """Iterate over cases."""
        return iter(self._cases)


# Module-level convenience functions


def load_domain(
    domain: LegalDomain, base_path: Optional[Path] = None
) -> List[LegalCase]:
    """Load cases from a single domain."""
    loader = CorpusLoader(base_path)
    return loader.load_domain(domain)


def load_all_domains(base_path: Optional[Path] = None) -> List[LegalCase]:
    """Load cases from all domains."""
    loader = CorpusLoader(base_path)
    return loader.load_all_domains()


def get_corpus_stats(base_path: Optional[Path] = None) -> CorpusStats:
    """Get statistics for the entire corpus."""
    loader = CorpusLoader(base_path)
    loader.load_all_domains()
    return loader.get_stats()
