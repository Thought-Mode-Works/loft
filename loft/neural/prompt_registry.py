"""
Prompt template registry with versioning and performance tracking.

This module manages prompt templates, enabling A/B testing, version comparison,
and empirical performance measurement across different prompt formulations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger
import json
from pathlib import Path


@dataclass
class PromptPerformance:
    """Track performance metrics for a prompt template."""

    template_name: str
    version: str
    total_uses: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    avg_confidence: float = 0.0
    avg_syntax_validity: float = 0.0  # % of syntactically valid rules
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    last_used: Optional[str] = None  # ISO timestamp

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_uses == 0:
            return 0.0
        return self.successful_generations / self.total_uses

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template_name": self.template_name,
            "version": self.version,
            "total_uses": self.total_uses,
            "successful_generations": self.successful_generations,
            "failed_generations": self.failed_generations,
            "success_rate": self.success_rate,
            "avg_confidence": self.avg_confidence,
            "avg_syntax_validity": self.avg_syntax_validity,
            "avg_latency_ms": self.avg_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "last_used": self.last_used,
        }


@dataclass
class PromptTemplate:
    """
    A versioned prompt template with metadata.

    Represents a single version of a prompt template with tracking
    for performance measurement and A/B testing.
    """

    name: str
    version: str
    template: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: List[str] = field(default_factory=list)
    performance: PromptPerformance = field(init=False)

    def __post_init__(self):
        """Initialize performance tracking."""
        self.performance = PromptPerformance(
            template_name=self.name,
            version=self.version,
        )

    def format(self, **kwargs: Any) -> str:
        """
        Format the template with provided arguments.

        Args:
            **kwargs: Template variables

        Returns:
            Formatted prompt string
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Missing required template variable: {e}. Template: {self.name} v{self.version}"
            )

    def record_use(
        self,
        success: bool,
        confidence: float = 0.0,
        syntax_valid: bool = True,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> None:
        """
        Record usage of this prompt template.

        Args:
            success: Whether generation succeeded
            confidence: Confidence score of generated output
            syntax_valid: Whether generated rule has valid syntax
            latency_ms: Generation latency in milliseconds
            cost_usd: Cost of this generation
        """
        self.performance.total_uses += 1

        if success:
            self.performance.successful_generations += 1
        else:
            self.performance.failed_generations += 1

        # Update rolling averages
        n = self.performance.total_uses

        # Weighted average for confidence
        self.performance.avg_confidence = (
            (self.performance.avg_confidence * (n - 1)) + confidence
        ) / n

        # Weighted average for syntax validity (as 0/1)
        validity_score = 1.0 if syntax_valid else 0.0
        self.performance.avg_syntax_validity = (
            (self.performance.avg_syntax_validity * (n - 1)) + validity_score
        ) / n

        # Weighted average for latency
        self.performance.avg_latency_ms = (
            (self.performance.avg_latency_ms * (n - 1)) + latency_ms
        ) / n

        # Cumulative cost
        self.performance.total_cost_usd += cost_usd

        # Update timestamp
        self.performance.last_used = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "tags": self.tags,
            "performance": self.performance.to_dict(),
            "template_preview": self.template[:200] + "..."
            if len(self.template) > 200
            else self.template,
        }


class PromptRegistry:
    """
    Registry for managing versioned prompt templates.

    Supports:
    - Version management (add, get, list versions)
    - A/B testing (compare performance across versions)
    - Performance tracking (success rate, confidence, cost)
    - Persistence (save/load from disk)
    """

    def __init__(self, persist_path: Optional[Path] = None):
        """
        Initialize prompt registry.

        Args:
            persist_path: Optional path to persist performance metrics
        """
        self.templates: Dict[str, Dict[str, PromptTemplate]] = {}
        self.persist_path = persist_path
        logger.info(f"Initialized PromptRegistry with persist_path={persist_path}")

    def register(
        self,
        name: str,
        version: str,
        template: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """
        Register a new prompt template.

        Args:
            name: Template name (e.g., "principle_to_rule")
            version: Version string (e.g., "v1.0")
            template: The prompt template string
            description: Description of this template version
            tags: Optional tags for categorization

        Returns:
            The registered PromptTemplate

        Example:
            >>> registry = PromptRegistry()
            >>> template = registry.register(
            ...     name="principle_to_rule",
            ...     version="v1.0",
            ...     template="Convert this principle: {principle}",
            ...     description="Basic principle conversion"
            ... )
        """
        if name not in self.templates:
            self.templates[name] = {}

        if version in self.templates[name]:
            logger.warning(f"Overwriting existing template {name} v{version}")

        prompt_template = PromptTemplate(
            name=name,
            version=version,
            template=template,
            description=description,
            tags=tags or [],
        )

        self.templates[name][version] = prompt_template
        logger.info(f"Registered template {name} v{version}")

        return prompt_template

    def get(self, name: str, version: str = "latest") -> PromptTemplate:
        """
        Get a prompt template by name and version.

        Args:
            name: Template name
            version: Version string or "latest"

        Returns:
            PromptTemplate

        Raises:
            KeyError: If template or version not found
        """
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found. Available: {list(self.templates.keys())}")

        versions = self.templates[name]

        if version == "latest":
            # Get version with highest success rate
            if not versions:
                raise KeyError(f"No versions for template '{name}'")

            # Sort by version string (assumes semantic versioning)
            sorted_versions = sorted(versions.keys(), reverse=True)
            version = sorted_versions[0]

        if version not in versions:
            raise KeyError(
                f"Version '{version}' not found for '{name}'. Available: {list(versions.keys())}"
            )

        return versions[version]

    def list_templates(self) -> List[str]:
        """
        List all template names.

        Returns:
            List of template names
        """
        return list(self.templates.keys())

    def list_versions(self, name: str) -> List[str]:
        """
        List all versions for a template.

        Args:
            name: Template name

        Returns:
            List of version strings

        Raises:
            KeyError: If template not found
        """
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found")

        return list(self.templates[name].keys())

    def compare_versions(
        self, name: str, versions: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare performance across template versions.

        Args:
            name: Template name
            versions: Specific versions to compare (None = all)

        Returns:
            Dictionary mapping versions to performance metrics

        Example:
            >>> comparison = registry.compare_versions("principle_to_rule")
            >>> for version, metrics in comparison.items():
            ...     print(f"{version}: {metrics['success_rate']:.2%}")
        """
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found")

        template_versions = self.templates[name]

        if versions is None:
            versions = list(template_versions.keys())

        comparison = {}
        for version in versions:
            if version not in template_versions:
                logger.warning(f"Version {version} not found for {name}")
                continue

            template = template_versions[version]
            comparison[version] = template.performance.to_dict()

        return comparison

    def get_best_version(self, name: str, metric: str = "success_rate") -> str:
        """
        Get the best-performing version for a template.

        Args:
            name: Template name
            metric: Metric to optimize ("success_rate", "avg_confidence", etc.)

        Returns:
            Version string of best-performing template

        Raises:
            KeyError: If template not found
            ValueError: If no versions have been used
        """
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found")

        versions = self.templates[name]

        # Filter to used versions
        used_versions = {v: t for v, t in versions.items() if t.performance.total_uses > 0}

        if not used_versions:
            raise ValueError(f"No versions of '{name}' have been used yet")

        # Find best by metric
        best_version = max(
            used_versions.items(),
            key=lambda item: getattr(item[1].performance, metric, 0.0),
        )

        return best_version[0]

    def record_use(
        self,
        name: str,
        version: str,
        success: bool,
        confidence: float = 0.0,
        syntax_valid: bool = True,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> None:
        """
        Record usage of a template.

        Args:
            name: Template name
            version: Template version
            success: Whether generation succeeded
            confidence: Confidence score
            syntax_valid: Whether syntax is valid
            latency_ms: Latency in milliseconds
            cost_usd: Cost in USD
        """
        template = self.get(name, version)
        template.record_use(success, confidence, syntax_valid, latency_ms, cost_usd)

        # Persist if configured
        if self.persist_path:
            self.save()

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save registry to disk.

        Args:
            path: Optional path override
        """
        save_path = path or self.persist_path

        if not save_path:
            logger.warning("No persist_path configured, skipping save")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            "templates": {
                name: {ver: tmpl.to_dict() for ver, tmpl in versions.items()}
                for name, versions in self.templates.items()
            }
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved registry to {save_path}")

    def load(self, path: Optional[Path] = None) -> None:
        """
        Load registry from disk.

        Args:
            path: Optional path override

        Note:
            This loads performance metrics only. Templates must still be
            registered via code.
        """
        load_path = path or self.persist_path

        if not load_path:
            logger.warning("No persist_path configured, skipping load")
            return

        load_path = Path(load_path)

        if not load_path.exists():
            logger.warning(f"Registry file not found: {load_path}")
            return

        with open(load_path, "r") as f:
            data = json.load(f)

        # Restore performance metrics
        for name, versions in data.get("templates", {}).items():
            if name not in self.templates:
                continue

            for version, template_data in versions.items():
                if version not in self.templates[name]:
                    continue

                template = self.templates[name][version]
                perf_data = template_data.get("performance", {})

                # Restore performance metrics
                template.performance = PromptPerformance(**perf_data)

        logger.info(f"Loaded registry from {load_path}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all templates.

        Returns:
            Dictionary with aggregate statistics
        """
        total_templates = len(self.templates)
        total_versions = sum(len(v) for v in self.templates.values())
        total_uses = sum(
            t.performance.total_uses
            for versions in self.templates.values()
            for t in versions.values()
        )
        total_cost = sum(
            t.performance.total_cost_usd
            for versions in self.templates.values()
            for t in versions.values()
        )

        return {
            "total_templates": total_templates,
            "total_versions": total_versions,
            "total_uses": total_uses,
            "total_cost_usd": total_cost,
            "templates": list(self.templates.keys()),
        }
