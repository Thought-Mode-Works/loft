"""
Hybrid translator combining canonical ontology with LLM fallback.

Provides graceful degradation when the canonical ontology lacks explicit
mappings by using LLM to semantically match predicates across domains.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from loft.ontology.canonical_translator import CanonicalTranslator

# Optional LLM imports
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None


@dataclass
class TranslationResult:
    """Result of a predicate or rule translation."""

    translated: Optional[str]
    confidence: float
    method: str  # "canonical", "llm", "none"
    reasoning: Optional[str] = None
    translations: Dict[str, str] = field(default_factory=dict)


@dataclass
class TranslationStats:
    """Statistics for translation operations."""

    canonical_translations: int = 0
    llm_translations: int = 0
    failed_translations: int = 0
    total_predicates: int = 0

    @property
    def canonical_rate(self) -> float:
        """Fraction of translations using canonical mappings."""
        if self.total_predicates == 0:
            return 0.0
        return self.canonical_translations / self.total_predicates

    @property
    def llm_rate(self) -> float:
        """Fraction of translations using LLM fallback."""
        if self.total_predicates == 0:
            return 0.0
        return self.llm_translations / self.total_predicates

    @property
    def success_rate(self) -> float:
        """Fraction of successful translations."""
        if self.total_predicates == 0:
            return 0.0
        return (self.canonical_translations + self.llm_translations) / self.total_predicates

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "canonical_translations": self.canonical_translations,
            "llm_translations": self.llm_translations,
            "failed_translations": self.failed_translations,
            "total_predicates": self.total_predicates,
            "canonical_rate": self.canonical_rate,
            "llm_rate": self.llm_rate,
            "success_rate": self.success_rate,
        }


LLM_TRANSLATION_PROMPT = """You are translating ASP (Answer Set Programming) predicates between legal domains.

Source domain: {source_domain}
Target domain: {target_domain}

Source predicate to translate: {predicate}

Available target predicates:
{target_predicates}

Task: Find the target predicate that is semantically equivalent to the source predicate.

Rules:
1. Only select from the available target predicates listed above
2. If no good match exists, respond with "NO_MATCH"
3. Provide your confidence (0.0-1.0) in the match
4. Consider the semantic meaning, not just surface similarity

Respond in JSON format only:
{{
  "target_predicate": "predicate_name" or "NO_MATCH",
  "confidence": 0.8,
  "reasoning": "Brief explanation of why this is a good match"
}}"""


class HybridTranslator:
    """
    Combines canonical RDF mappings with LLM fallback for predicate translation.

    Uses a two-stage approach:
    1. First attempts canonical (deterministic) translation via RDF ontology
    2. Falls back to LLM for predicates without canonical mappings

    Confidence scores reflect translation certainty:
    - Canonical: 1.0 (explicit RDF mapping)
    - LLM: 0.6-0.95 (LLM-assessed confidence)
    - Failed: 0.0 (no viable translation)

    Example:
        translator = HybridTranslator()

        # Translate a rule
        result = translator.translate_rule(
            rule="enforceable(X) :- new_predicate(X, yes).",
            source_domain="adverse_possession",
            target_domain="property_law"
        )

        print(f"Translated: {result.translated}")
        print(f"Confidence: {result.confidence}")
        print(f"Method: {result.method}")
    """

    def __init__(
        self,
        canonical_translator: Optional[CanonicalTranslator] = None,
        model: str = "claude-3-5-haiku-20241022",
        min_llm_confidence: float = 0.6,
        enable_llm: bool = True,
    ):
        """
        Initialize hybrid translator.

        Args:
            canonical_translator: Canonical translator instance. If None, creates one.
            model: LLM model to use for fallback translation
            min_llm_confidence: Minimum confidence threshold for LLM translations
            enable_llm: Whether to enable LLM fallback (can be disabled for testing)
        """
        if canonical_translator is None:
            self.canonical = CanonicalTranslator()
        else:
            self.canonical = canonical_translator

        self.model = model
        self.min_llm_confidence = min_llm_confidence
        self.enable_llm = enable_llm

        # Cache for LLM translations: (predicate, source, target) -> (result, confidence)
        self._llm_cache: Dict[Tuple[str, str, str], Tuple[Optional[str], float, str]] = {}

        # Translation statistics
        self.stats = TranslationStats()

        # Validated translations that can be promoted to canonical
        self._validated_translations: List[Dict] = []

        # Initialize LLM client if available
        self._llm_client = None
        if enable_llm and ANTHROPIC_AVAILABLE:
            try:
                self._llm_client = anthropic.Anthropic()
            except Exception as e:
                logger.warning(f"Could not initialize LLM client: {e}")

    def translate_predicate(
        self,
        predicate: str,
        source_domain: str,
        target_domain: str,
        target_predicates: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], float, str]:
        """
        Translate a single predicate from source to target domain.

        Args:
            predicate: Predicate name to translate
            source_domain: Source domain name
            target_domain: Target domain name
            target_predicates: Available predicates in target domain (for LLM)

        Returns:
            Tuple of (translated_predicate, confidence, method)
            method is one of: "canonical", "llm", "none"
        """
        self.stats.total_predicates += 1

        # Step 1: Try canonical mapping
        canonical_result = self.canonical.translate_predicate(
            predicate, source_domain, target_domain
        )

        if canonical_result is not None:
            self.stats.canonical_translations += 1
            logger.debug(f"Canonical translation: {predicate} -> {canonical_result}")
            return canonical_result, 1.0, "canonical"

        # Step 2: Fall back to LLM
        if not self.enable_llm or self._llm_client is None:
            self.stats.failed_translations += 1
            return None, 0.0, "none"

        # Check cache first
        cache_key = (predicate, source_domain, target_domain)
        if cache_key in self._llm_cache:
            result, confidence, method = self._llm_cache[cache_key]
            if result is not None:
                self.stats.llm_translations += 1
            else:
                self.stats.failed_translations += 1
            return result, confidence, method

        # Get target predicates if not provided
        if target_predicates is None:
            target_predicates = self.canonical.get_domain_predicates(target_domain)

        if not target_predicates:
            self.stats.failed_translations += 1
            return None, 0.0, "none"

        # Call LLM for translation
        result, confidence, reasoning = self._llm_translate(
            predicate, source_domain, target_domain, target_predicates
        )

        # Cache the result
        method = "llm" if result is not None else "none"
        self._llm_cache[cache_key] = (result, confidence, method)

        if result is not None:
            self.stats.llm_translations += 1
            logger.debug(f"LLM translation: {predicate} -> {result} (conf: {confidence:.2f})")
        else:
            self.stats.failed_translations += 1
            logger.debug(f"No translation found for: {predicate}")

        return result, confidence, method

    def _llm_translate(
        self,
        predicate: str,
        source_domain: str,
        target_domain: str,
        target_predicates: List[str],
    ) -> Tuple[Optional[str], float, str]:
        """
        Use LLM to find best matching predicate.

        Args:
            predicate: Predicate to translate
            source_domain: Source domain name
            target_domain: Target domain name
            target_predicates: Available predicates in target domain

        Returns:
            Tuple of (translated_predicate, confidence, reasoning)
        """
        if self._llm_client is None:
            return None, 0.0, "LLM not available"

        prompt = LLM_TRANSLATION_PROMPT.format(
            source_domain=source_domain,
            target_domain=target_domain,
            predicate=predicate,
            target_predicates="\n".join(f"  - {p}" for p in target_predicates),
        )

        try:
            response = self._llm_client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text.strip()

            # Parse JSON response
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = re.sub(r"```(?:json)?\n?", "", content)
                content = content.strip()

            parsed = json.loads(content)

            target_pred = parsed.get("target_predicate")
            confidence = float(parsed.get("confidence", 0.0))
            reasoning = parsed.get("reasoning", "")

            if target_pred == "NO_MATCH":
                return None, 0.0, reasoning

            if confidence < self.min_llm_confidence:
                return (
                    None,
                    0.0,
                    f"Confidence {confidence} below threshold {self.min_llm_confidence}",
                )

            # Verify target predicate exists
            if target_pred not in target_predicates:
                logger.warning(f"LLM suggested unknown predicate: {target_pred}")
                return None, 0.0, f"Unknown predicate: {target_pred}"

            return target_pred, confidence, reasoning

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None, 0.0, f"JSON parse error: {e}"
        except Exception as e:
            logger.warning(f"LLM translation error: {e}")
            return None, 0.0, f"Error: {e}"

    def translate_rule(
        self,
        rule: str,
        source_domain: str,
        target_domain: str,
        target_predicates: Optional[List[str]] = None,
    ) -> TranslationResult:
        """
        Translate an ASP rule from source to target domain.

        Uses canonical mappings where available, falls back to LLM for unmapped
        predicates. The overall confidence is the product of individual predicate
        confidences.

        Args:
            rule: ASP rule string
            source_domain: Source domain name
            target_domain: Target domain name
            target_predicates: Available predicates in target domain

        Returns:
            TranslationResult with translated rule, confidence, and method info
        """
        # Get target predicates if not provided
        if target_predicates is None:
            target_predicates = self.canonical.get_domain_predicates(target_domain)

        # Extract predicates from rule
        predicate_pattern = r"([a-z][a-z0-9_]*)\s*\("
        predicates_found = set(re.findall(predicate_pattern, rule))

        if not predicates_found:
            return TranslationResult(
                translated=rule,
                confidence=1.0,
                method="none",
                reasoning="No predicates to translate",
            )

        translated_rule = rule
        translations = {}
        overall_confidence = 1.0
        methods_used = set()
        reasoning_parts = []

        for pred in predicates_found:
            target, confidence, method = self.translate_predicate(
                pred, source_domain, target_domain, target_predicates
            )

            methods_used.add(method)

            if target is None:
                # Cannot translate this predicate
                return TranslationResult(
                    translated=None,
                    confidence=0.0,
                    method="none",
                    reasoning=f"Cannot translate predicate: {pred}",
                    translations={},
                )

            if target != pred:
                translations[pred] = target
                # Replace predicate in rule
                pattern = rf"\b{re.escape(pred)}\s*\("
                replacement = f"{target}("
                translated_rule = re.sub(pattern, replacement, translated_rule)
                reasoning_parts.append(f"{pred}->{target}({method})")

            overall_confidence *= confidence

        # Determine overall method
        if "llm" in methods_used:
            overall_method = "hybrid" if "canonical" in methods_used else "llm"
        elif "canonical" in methods_used:
            overall_method = "canonical"
        else:
            overall_method = "none"

        return TranslationResult(
            translated=translated_rule,
            confidence=overall_confidence,
            method=overall_method,
            reasoning=", ".join(reasoning_parts) if reasoning_parts else "No changes needed",
            translations=translations,
        )

    def validate_translation(
        self,
        source_pred: str,
        target_pred: str,
        source_domain: str,
        target_domain: str,
        success: bool,
    ) -> None:
        """
        Record validation result for a translation.

        Used to track which LLM translations are successful, enabling
        future promotion to canonical mappings.

        Args:
            source_pred: Source predicate name
            target_pred: Target predicate name
            source_domain: Source domain
            target_domain: Target domain
            success: Whether the translation was successful
        """
        self._validated_translations.append(
            {
                "source_pred": source_pred,
                "target_pred": target_pred,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "success": success,
            }
        )

    def get_validated_for_promotion(self, min_success_count: int = 3) -> List[Dict]:
        """
        Get validated translations that can be promoted to canonical.

        Args:
            min_success_count: Minimum successful validations needed

        Returns:
            List of translations suitable for promotion
        """
        # Count successes per translation
        success_counts: Dict[Tuple[str, str, str, str], int] = {}

        for v in self._validated_translations:
            if v["success"]:
                key = (
                    v["source_pred"],
                    v["target_pred"],
                    v["source_domain"],
                    v["target_domain"],
                )
                success_counts[key] = success_counts.get(key, 0) + 1

        # Return translations with sufficient validation
        promotable = []
        for key, count in success_counts.items():
            if count >= min_success_count:
                promotable.append(
                    {
                        "source_pred": key[0],
                        "target_pred": key[1],
                        "source_domain": key[2],
                        "target_domain": key[3],
                        "success_count": count,
                    }
                )

        return promotable

    def promote_to_canonical(
        self,
        source_pred: str,
        target_pred: str,
        source_domain: str,
        target_domain: str,
        canonical_name: Optional[str] = None,
    ) -> bool:
        """
        Add validated LLM translation to canonical ontology.

        Creates a new canonical concept linking the source and target predicates.

        Args:
            source_pred: Source predicate name
            target_pred: Target predicate name
            source_domain: Source domain
            target_domain: Target domain
            canonical_name: Name for canonical concept (auto-generated if None)

        Returns:
            True if promotion succeeded
        """
        try:
            from rdflib import RDF, Literal, Namespace, URIRef

            CANON = Namespace("https://loft.legal/canonical/")
            LOFT = Namespace("https://loft.legal/ontology/")

            # Generate canonical name if not provided
            if canonical_name is None:
                canonical_name = f"auto_{source_pred}_{target_pred}"

            canonical_uri = URIRef(f"https://loft.legal/canonical/{canonical_name}")

            # Add canonical predicate
            self.canonical.graph.add((canonical_uri, RDF.type, CANON.CanonicalPredicate))

            # Add source domain mapping
            source_uri = URIRef(f"https://loft.legal/domains/{source_domain}/{source_pred}")
            self.canonical.graph.add((source_uri, RDF.type, CANON.DomainPredicate))
            self.canonical.graph.add((source_uri, CANON.mapsTo, canonical_uri))
            self.canonical.graph.add((source_uri, LOFT.domain, Literal(source_domain)))

            # Add target domain mapping
            target_uri = URIRef(f"https://loft.legal/domains/{target_domain}/{target_pred}")
            self.canonical.graph.add((target_uri, RDF.type, CANON.DomainPredicate))
            self.canonical.graph.add((target_uri, CANON.mapsTo, canonical_uri))
            self.canonical.graph.add((target_uri, LOFT.domain, Literal(target_domain)))

            # Rebuild mapping tables
            self.canonical._build_mappings()

            # Clear LLM cache for this translation
            cache_key = (source_pred, source_domain, target_domain)
            if cache_key in self._llm_cache:
                del self._llm_cache[cache_key]

            logger.info(
                f"Promoted LLM translation to canonical: "
                f"{source_pred} ({source_domain}) <-> {target_pred} ({target_domain})"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to promote translation: {e}")
            return False

    def save_ontology(self, path: Optional[Path] = None) -> None:
        """
        Save the ontology with any promoted translations.

        Args:
            path: Output path. If None, overwrites original ontology file.
        """
        if path is None:
            path = self.canonical.ontology_path

        self.canonical.graph.serialize(destination=str(path), format="turtle")
        logger.info(f"Saved ontology to: {path}")

    def get_stats(self) -> TranslationStats:
        """Get translation statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset translation statistics."""
        self.stats = TranslationStats()

    def clear_cache(self) -> None:
        """Clear LLM translation cache."""
        self._llm_cache.clear()

    def __repr__(self) -> str:
        return (
            f"HybridTranslator(model={self.model}, "
            f"min_confidence={self.min_llm_confidence}, "
            f"llm_enabled={self.enable_llm})"
        )
