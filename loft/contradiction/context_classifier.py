"""
Context classification system for rule selection (Phase 4.4).

Classifies case contexts to determine which rules/interpretations apply.
"""

import uuid
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

from loft.contradiction.contradiction_schemas import ContextClassification


class ContextClassifier:
    """
    Classifies case contexts for context-dependent rule selection.

    Uses feature extraction and matching to determine which rules and
    interpretations are applicable given specific case facts.
    """

    def __init__(self):
        """Initialize context classifier."""
        # Context type patterns (extensible)
        self.context_patterns = {
            "contract_formation": [
                "offer",
                "acceptance",
                "consideration",
                "mutual_assent",
            ],
            "contract_breach": ["breach", "performance", "default", "non_performance"],
            "contract_remedy": ["damages", "specific_performance", "rescission"],
            "property_transfer": ["conveyance", "deed", "title", "transfer"],
            "property_recording": ["recording", "notice", "registry", "file"],
            "criminal_intent": ["mens_rea", "intent", "knowledge", "recklessness"],
            "criminal_liability": ["actus_reus", "liability", "causation"],
        }

        logger.info("Initialized ContextClassifier")

    def classify_context(
        self, case_facts: Dict[str, Any], domain: str = "general"
    ) -> ContextClassification:
        """
        Classify case context from facts.

        Args:
            case_facts: Dictionary of case facts and predicates
            domain: Domain/area of law (contract, property, criminal, etc.)

        Returns:
            ContextClassification with type and confidence
        """
        # Extract key features
        key_features = self._extract_features(case_facts)

        # Determine context type
        context_type, confidence = self._match_context_type(key_features)

        # Create classification
        classification = ContextClassification(
            context_id=str(uuid.uuid4()),
            context_type=context_type,
            confidence=confidence,
            key_features=key_features,
            domain=domain,
            jurisdiction=case_facts.get("jurisdiction"),
        )

        logger.debug(f"Classified context as '{context_type}' with confidence {confidence:.2f}")

        return classification

    def get_applicable_rules(
        self,
        context: ContextClassification,
        candidate_rules: List[Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        """
        Get rules applicable to a context with confidence scores.

        Args:
            context: The classified context
            candidate_rules: List of candidate rules with metadata
                           Each dict should have: rule_id, rule_text, context_tags

        Returns:
            List of (rule_id, confidence) tuples sorted by confidence
        """
        applicable = []

        for rule in candidate_rules:
            rule_id = rule.get("rule_id", "unknown")
            context_tags = rule.get("context_tags", [])

            # Calculate applicability score
            confidence = self._calculate_applicability(context, context_tags)

            if confidence > 0.3:  # Threshold for consideration
                applicable.append((rule_id, confidence))

        # Sort by confidence descending
        applicable.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Found {len(applicable)} applicable rules for context '{context.context_type}'"
        )

        return applicable

    def select_interpretation(
        self,
        context: ContextClassification,
        interpretations: List[Dict[str, Any]],
    ) -> Optional[Tuple[str, float]]:
        """
        Select the best interpretation for a context.

        Args:
            context: The classified context
            interpretations: List of competing interpretations

        Returns:
            (interpretation_id, confidence) tuple or None
        """
        best_match = None
        best_score = 0.0

        for interp in interpretations:
            interp_id = interp.get("interpretation_id", "unknown")
            applicable_contexts = interp.get("applicable_contexts", [])
            exclusion_contexts = interp.get("exclusion_contexts", [])

            # Check exclusions first
            if context.context_type in exclusion_contexts:
                continue

            # Calculate match score
            score = self._calculate_interpretation_match(context, applicable_contexts)

            if score > best_score:
                best_score = score
                best_match = (interp_id, score)

        logger.debug(f"Selected interpretation with confidence {best_score:.2f}")

        return best_match

    def _extract_features(self, case_facts: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key features from case facts."""
        features = {}

        # Extract predicates (ASP-style facts)
        if "predicates" in case_facts:
            features["predicates"] = case_facts["predicates"]

        # Extract fact types
        if "facts" in case_facts:
            features["fact_types"] = list(case_facts["facts"].keys())

        # Extract entities
        if "entities" in case_facts:
            features["entities"] = case_facts["entities"]

        # Extract keywords from text if present
        if "text" in case_facts:
            features["keywords"] = self._extract_keywords(case_facts["text"])

        return features

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Simple keyword extraction (could be enhanced with NLP)
        keywords = []

        for pattern_keywords in self.context_patterns.values():
            for keyword in pattern_keywords:
                if keyword.lower() in text.lower():
                    keywords.append(keyword)

        return list(set(keywords))

    def _match_context_type(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Match features to context type.

        Returns:
            (context_type, confidence) tuple
        """
        # Get keywords from features
        keywords = features.get("keywords", [])
        predicates = features.get("predicates", [])

        # Combine for matching
        all_terms = keywords + predicates

        # Score each context type
        scores = {}
        for context_type, pattern_keywords in self.context_patterns.items():
            matches = sum(1 for kw in pattern_keywords if kw in all_terms)
            if len(pattern_keywords) > 0:
                scores[context_type] = matches / len(pattern_keywords)

        # Find best match
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]
            return best_type, confidence
        else:
            # Default to generic
            return "general", 0.5

    def _calculate_applicability(
        self, context: ContextClassification, context_tags: List[str]
    ) -> float:
        """Calculate how applicable a rule is to a context."""
        if not context_tags:
            return 0.5  # Neutral if no tags

        # Check for direct match
        if context.context_type in context_tags:
            return 0.9

        # Check for feature overlap
        context_keywords = set(context.key_features.get("keywords", []))
        tag_set = set(context_tags)

        overlap = len(context_keywords & tag_set)
        if overlap > 0:
            return 0.6 + (overlap * 0.1)

        return 0.3  # Low default applicability

    def _calculate_interpretation_match(
        self, context: ContextClassification, applicable_contexts: List[str]
    ) -> float:
        """Calculate how well an interpretation matches a context."""
        if not applicable_contexts:
            return 0.5  # Neutral if no explicit applicability

        # Direct match
        if context.context_type in applicable_contexts:
            return 0.95

        # Partial match on features
        context_keywords = set(context.key_features.get("keywords", []))
        applicable_set = set(applicable_contexts)

        overlap = len(context_keywords & applicable_set)
        if overlap > 0:
            return 0.6 + (overlap * 0.1)

        return 0.4  # Low match
