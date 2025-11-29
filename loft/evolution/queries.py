"""
Query interface for rule evolution data.

Provides a SQL-like query builder for exploring rule evolution history,
finding rules by various criteria, and analyzing evolution patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

from .tracking import (
    RuleMetadata,
    ABTestResult,
    StratificationLayer,
    RuleStatus,
)
from .storage import RuleEvolutionStorage


class SortField(Enum):
    """Fields available for sorting query results."""

    CREATED_AT = "created_at"
    VERSION = "version"
    ACCURACY = "accuracy"
    LAYER = "layer"
    STATUS = "status"


class SortOrder(Enum):
    """Sort order for query results."""

    ASC = "asc"
    DESC = "desc"


@dataclass
class EvolutionQuery:
    """
    Query builder for rule evolution data.

    Provides a fluent interface for building complex queries
    over rule evolution history.

    Example:
        query = (
            EvolutionQuery()
            .with_status(RuleStatus.ACTIVE)
            .with_min_accuracy(0.8)
            .in_layer(StratificationLayer.TACTICAL)
            .sort_by(SortField.ACCURACY, SortOrder.DESC)
            .limit(10)
        )
        results = db.execute(query)
    """

    # Filter criteria
    status_filter: Optional[RuleStatus] = None
    layer_filter: Optional[StratificationLayer] = None
    min_accuracy: Optional[float] = None
    max_accuracy: Optional[float] = None
    min_version: Optional[int] = None
    max_version: Optional[int] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    parent_id: Optional[str] = None
    has_dialectical: Optional[bool] = None
    search_text: Optional[str] = None
    custom_filters: List[Callable[[RuleMetadata], bool]] = field(default_factory=list)

    # Sorting
    sort_field: SortField = SortField.CREATED_AT
    sort_order: SortOrder = SortOrder.DESC

    # Pagination
    limit_count: Optional[int] = None
    offset_count: int = 0

    def with_status(self, status: RuleStatus) -> "EvolutionQuery":
        """Filter by rule status."""
        self.status_filter = status
        return self

    def in_layer(self, layer: StratificationLayer) -> "EvolutionQuery":
        """Filter by stratification layer."""
        self.layer_filter = layer
        return self

    def with_min_accuracy(self, min_acc: float) -> "EvolutionQuery":
        """Filter for rules with accuracy >= min_acc."""
        self.min_accuracy = min_acc
        return self

    def with_max_accuracy(self, max_acc: float) -> "EvolutionQuery":
        """Filter for rules with accuracy <= max_acc."""
        self.max_accuracy = max_acc
        return self

    def with_version_range(
        self, min_ver: Optional[int] = None, max_ver: Optional[int] = None
    ) -> "EvolutionQuery":
        """Filter by version range."""
        self.min_version = min_ver
        self.max_version = max_ver
        return self

    def created_between(
        self, after: Optional[datetime] = None, before: Optional[datetime] = None
    ) -> "EvolutionQuery":
        """Filter by creation date range."""
        self.created_after = after
        self.created_before = before
        return self

    def with_parent(self, parent_id: str) -> "EvolutionQuery":
        """Filter for rules derived from a specific parent."""
        self.parent_id = parent_id
        return self

    def with_dialectical_history(self, has_history: bool = True) -> "EvolutionQuery":
        """Filter for rules with/without dialectical refinement."""
        self.has_dialectical = has_history
        return self

    def containing_text(self, text: str) -> "EvolutionQuery":
        """Search in rule text and natural language description."""
        self.search_text = text.lower()
        return self

    def where(self, predicate: Callable[[RuleMetadata], bool]) -> "EvolutionQuery":
        """Add custom filter predicate."""
        self.custom_filters.append(predicate)
        return self

    def sort_by(self, field: SortField, order: SortOrder = SortOrder.DESC) -> "EvolutionQuery":
        """Set sort field and order."""
        self.sort_field = field
        self.sort_order = order
        return self

    def limit(self, count: int) -> "EvolutionQuery":
        """Limit number of results."""
        self.limit_count = count
        return self

    def offset(self, count: int) -> "EvolutionQuery":
        """Skip first N results."""
        self.offset_count = count
        return self

    def matches(self, rule: RuleMetadata) -> bool:
        """Check if a rule matches all query criteria."""
        # Status filter
        if self.status_filter is not None and rule.status != self.status_filter:
            return False

        # Layer filter
        if self.layer_filter is not None and rule.current_layer != self.layer_filter:
            return False

        # Accuracy filters
        if self.min_accuracy is not None and rule.current_accuracy < self.min_accuracy:
            return False
        if self.max_accuracy is not None and rule.current_accuracy > self.max_accuracy:
            return False

        # Version filters
        rule_version = int(rule.version.split(".")[0]) if "." in rule.version else int(rule.version)
        if self.min_version is not None and rule_version < self.min_version:
            return False
        if self.max_version is not None and rule_version > self.max_version:
            return False

        # Date filters
        if self.created_after is not None and rule.created_at < self.created_after:
            return False
        if self.created_before is not None and rule.created_at > self.created_before:
            return False

        # Parent filter
        if self.parent_id is not None and rule.parent_rule_id != self.parent_id:
            return False

        # Dialectical filter
        if self.has_dialectical is not None:
            has_history = rule.dialectical.cycles_completed > 0
            if has_history != self.has_dialectical:
                return False

        # Text search
        if self.search_text is not None:
            text_to_search = (rule.rule_text + " " + rule.natural_language).lower()
            if self.search_text not in text_to_search:
                return False

        # Custom filters
        for predicate in self.custom_filters:
            if not predicate(rule):
                return False

        return True

    def get_sort_key(self, rule: RuleMetadata):
        """Get sort key for a rule."""
        if self.sort_field == SortField.CREATED_AT:
            return rule.created_at
        elif self.sort_field == SortField.VERSION:
            return int(rule.version.split(".")[0]) if "." in rule.version else int(rule.version)
        elif self.sort_field == SortField.ACCURACY:
            return rule.current_accuracy
        elif self.sort_field == SortField.LAYER:
            return rule.current_layer.value
        elif self.sort_field == SortField.STATUS:
            return rule.status.value
        else:
            return rule.created_at


@dataclass
class QueryResult:
    """Result of executing an evolution query."""

    rules: List[RuleMetadata]
    total_count: int
    query: EvolutionQuery

    @property
    def count(self) -> int:
        """Number of rules in this result set."""
        return len(self.rules)

    @property
    def has_more(self) -> bool:
        """Check if there are more results beyond this page."""
        if self.query.limit_count is None:
            return False
        return self.query.offset_count + len(self.rules) < self.total_count

    def first(self) -> Optional[RuleMetadata]:
        """Get first result or None."""
        return self.rules[0] if self.rules else None

    def ids(self) -> List[str]:
        """Get list of rule IDs."""
        return [r.rule_id for r in self.rules]


class RuleEvolutionDB:
    """
    Database interface for rule evolution queries.

    Provides high-level query methods and aggregation functions
    over rule evolution data.

    Example:
        db = RuleEvolutionDB(storage)

        # Find top performing rules
        top_rules = db.execute(
            EvolutionQuery()
            .with_status(RuleStatus.ACTIVE)
            .sort_by(SortField.ACCURACY, SortOrder.DESC)
            .limit(10)
        )

        # Get rules by lineage
        family = db.get_rule_lineage("rule_abc123")

        # Analyze evolution patterns
        stats = db.get_evolution_stats()
    """

    def __init__(self, storage: RuleEvolutionStorage):
        """
        Initialize database interface.

        Args:
            storage: Storage backend to query
        """
        self.storage = storage

    def execute(self, query: EvolutionQuery) -> QueryResult:
        """
        Execute a query and return results.

        Args:
            query: Query to execute

        Returns:
            QueryResult with matching rules
        """
        # Load all rules
        all_rules = self.storage.load_all_rules()

        # Filter
        matching = [r for r in all_rules if query.matches(r)]
        total_count = len(matching)

        # Sort
        reverse = query.sort_order == SortOrder.DESC
        matching.sort(key=query.get_sort_key, reverse=reverse)

        # Paginate
        if query.offset_count > 0:
            matching = matching[query.offset_count :]
        if query.limit_count is not None:
            matching = matching[: query.limit_count]

        return QueryResult(rules=matching, total_count=total_count, query=query)

    def get_rule_by_id(self, rule_id: str) -> Optional[RuleMetadata]:
        """Get a single rule by ID."""
        return self.storage.load_rule(rule_id)

    def get_rule_versions(self, base_rule_id: str) -> List[RuleMetadata]:
        """
        Get all versions of a rule.

        Args:
            base_rule_id: ID of any version of the rule

        Returns:
            List of all versions, ordered by version number
        """
        # Find root rule
        rule = self.storage.load_rule(base_rule_id)
        if not rule:
            return []

        # Walk up to root
        while rule.parent_rule_id:
            parent = self.storage.load_rule(rule.parent_rule_id)
            if not parent:
                break
            rule = parent

        # Collect all descendants
        versions = [rule]
        visited = {rule.rule_id}

        def collect_descendants(r: RuleMetadata):
            for child_id in r.downstream_rules:
                if child_id not in visited:
                    child = self.storage.load_rule(child_id)
                    if child:
                        visited.add(child_id)
                        versions.append(child)
                        collect_descendants(child)

        collect_descendants(rule)

        # Sort by version
        versions.sort(
            key=lambda r: int(r.version.split(".")[0]) if "." in r.version else int(r.version)
        )
        return versions

    def get_rule_lineage(self, rule_id: str) -> Dict[str, RuleMetadata]:
        """
        Get full lineage (ancestors and descendants) of a rule.

        Args:
            rule_id: ID of the rule

        Returns:
            Dictionary of rule_id -> RuleMetadata for all related rules
        """
        lineage: Dict[str, RuleMetadata] = {}

        def add_ancestors(r_id: str):
            if r_id in lineage:
                return
            rule = self.storage.load_rule(r_id)
            if not rule:
                return
            lineage[r_id] = rule
            if rule.parent_rule_id:
                add_ancestors(rule.parent_rule_id)

        def add_descendants(r_id: str):
            if r_id in lineage:
                return
            rule = self.storage.load_rule(r_id)
            if not rule:
                return
            lineage[r_id] = rule
            for child_id in rule.downstream_rules:
                add_descendants(child_id)

        # Start from the given rule
        rule = self.storage.load_rule(rule_id)
        if rule:
            lineage[rule_id] = rule
            if rule.parent_rule_id:
                add_ancestors(rule.parent_rule_id)
            for child_id in rule.downstream_rules:
                add_descendants(child_id)

        return lineage

    def get_active_ab_tests(self) -> List[ABTestResult]:
        """Get all currently running A/B tests."""
        all_tests = self.storage.load_all_ab_tests()
        return [t for t in all_tests if t.completed_at is None]

    def get_completed_ab_tests(self, winner: Optional[str] = None) -> List[ABTestResult]:
        """
        Get completed A/B tests, optionally filtered by winner.

        Args:
            winner: Filter for tests where specified variant won ("a" or "b")

        Returns:
            List of completed ABTestResult objects
        """
        all_tests = self.storage.load_all_ab_tests()
        completed = [t for t in all_tests if t.completed_at is not None]

        if winner is not None:
            completed = [t for t in completed if t.winner == winner]

        return completed

    def get_evolution_stats(self) -> Dict:
        """
        Get comprehensive statistics about rule evolution.

        Returns:
            Dictionary with evolution statistics
        """
        rules = self.storage.load_all_rules()
        tests = self.storage.load_all_ab_tests()

        if not rules:
            return {
                "total_rules": 0,
                "total_ab_tests": 0,
                "active_ab_tests": 0,
                "status_distribution": {},
                "layer_distribution": {},
                "avg_accuracy": 0.0,
                "avg_versions": 0.0,
                "rules_with_dialectical": 0,
                "avg_dialectical_cycles": 0.0,
            }

        # Status distribution
        status_dist: Dict[str, int] = {}
        for rule in rules:
            status = rule.status.value
            status_dist[status] = status_dist.get(status, 0) + 1

        # Layer distribution
        layer_dist: Dict[str, int] = {}
        for rule in rules:
            layer = rule.current_layer.value
            layer_dist[layer] = layer_dist.get(layer, 0) + 1

        # Accuracy stats
        accuracies = [r.current_accuracy for r in rules if r.current_accuracy > 0]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

        # Version stats
        versions = [
            int(r.version.split(".")[0]) if "." in r.version else int(r.version) for r in rules
        ]
        avg_versions = sum(versions) / len(versions) if versions else 0.0

        # Dialectical stats
        rules_with_dialectical = sum(1 for r in rules if r.dialectical.cycles_completed > 0)
        dialectical_cycles = [
            r.dialectical.cycles_completed for r in rules if r.dialectical.cycles_completed > 0
        ]
        avg_dialectical = (
            sum(dialectical_cycles) / len(dialectical_cycles) if dialectical_cycles else 0.0
        )

        return {
            "total_rules": len(rules),
            "total_ab_tests": len(tests),
            "active_ab_tests": sum(1 for t in tests if t.completed_at is None),
            "completed_ab_tests": sum(1 for t in tests if t.completed_at is not None),
            "status_distribution": status_dist,
            "layer_distribution": layer_dist,
            "avg_accuracy": avg_accuracy,
            "avg_versions": avg_versions,
            "rules_with_dialectical": rules_with_dialectical,
            "avg_dialectical_cycles": avg_dialectical,
        }

    def find_improvement_candidates(
        self, min_accuracy: float = 0.5, max_accuracy: float = 0.8
    ) -> List[RuleMetadata]:
        """
        Find rules that might benefit from improvement.

        Identifies rules with moderate accuracy that could be
        candidates for dialectical refinement or A/B testing.

        Args:
            min_accuracy: Minimum accuracy threshold
            max_accuracy: Maximum accuracy threshold

        Returns:
            List of candidate rules
        """
        query = (
            EvolutionQuery()
            .with_status(RuleStatus.ACTIVE)
            .with_min_accuracy(min_accuracy)
            .with_max_accuracy(max_accuracy)
            .sort_by(SortField.ACCURACY, SortOrder.ASC)
        )
        return self.execute(query).rules

    def find_stale_rules(self, max_age_days: int = 30) -> List[RuleMetadata]:
        """
        Find rules that haven't been updated recently.

        Args:
            max_age_days: Consider rules older than this stale

        Returns:
            List of stale rules
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=max_age_days)

        query = (
            EvolutionQuery()
            .with_status(RuleStatus.ACTIVE)
            .created_between(before=cutoff)
            .sort_by(SortField.CREATED_AT, SortOrder.ASC)
        )
        return self.execute(query).rules

    def get_top_performers(
        self, layer: Optional[StratificationLayer] = None, limit: int = 10
    ) -> List[RuleMetadata]:
        """
        Get top performing rules by accuracy.

        Args:
            layer: Optional layer filter
            limit: Maximum number of results

        Returns:
            List of top performing rules
        """
        query = (
            EvolutionQuery()
            .with_status(RuleStatus.ACTIVE)
            .sort_by(SortField.ACCURACY, SortOrder.DESC)
            .limit(limit)
        )
        if layer:
            query.in_layer(layer)

        return self.execute(query).rules

    def get_recently_modified(self, limit: int = 10) -> List[RuleMetadata]:
        """
        Get most recently created/modified rules.

        Args:
            limit: Maximum number of results

        Returns:
            List of recent rules
        """
        query = EvolutionQuery().sort_by(SortField.CREATED_AT, SortOrder.DESC).limit(limit)
        return self.execute(query).rules
