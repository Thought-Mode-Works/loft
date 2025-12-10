from typing import Dict, List, Optional, Tuple

from loft.translation.context import TranslationContext
from loft.translation.nl_to_asp import NLToASPTranslator
from loft.translation.asp_to_nl import (
    ASPToNLTranslator,
    extract_predicates,
    asp_to_nl_statement,
)


class ContextPreservingTranslator:
    """Translator that preserves context for improved back-translation."""

    def __init__(self, llm_interface, max_cache_size: int = 1000):
        self.llm = llm_interface
        self.context_cache: Dict[str, TranslationContext] = {}
        self.max_cache_size = max_cache_size
        self.nl_to_asp_translator = NLToASPTranslator(llm_interface)
        self.asp_to_nl_translator = ASPToNLTranslator()

    def translate_nl_to_asp(self, nl_text: str) -> Tuple[str, str]:
        """Translate NL to ASP and store context."""
        # Extract key information before translation
        key_terms = self._extract_key_terms(nl_text)
        key_entities = self._extract_entities(nl_text)

        # Perform translation
        asp_result = self.nl_to_asp_translator.translate(nl_text)
        asp_code = asp_result.asp_code
        predicates = extract_predicates(asp_code)

        # Store context
        context = TranslationContext(
            original_nl=nl_text,
            asp_code=asp_code,
            predicates_used=predicates,
            key_entities=key_entities,
            key_terms=key_terms,
        )
        self._store_context(context)

        return asp_code, context.context_id

    def translate_asp_to_nl(
        self, asp_code: str, context_id: Optional[str] = None
    ) -> str:
        """Translate ASP to NL, using context if available."""
        if context_id and context_id in self.context_cache:
            return self._reconstruct_with_context(
                asp_code, self.context_cache[context_id]
            )
        return asp_to_nl_statement(asp_code)

    def _reconstruct_with_context(
        self, asp_code: str, context: TranslationContext
    ) -> str:
        """Use original context to guide reconstruction."""
        prompt = f"""
        Reconstruct a natural language statement from this ASP code.

        Original statement (for reference): {context.original_nl}
        Key terms to preserve: {', '.join(context.key_terms)}
        Entity names: {context.key_entities}

        ASP code: {asp_code}

        Generate a statement that:
        1. Preserves the original meaning
        2. Uses the key terms where appropriate
        3. Maintains the declarative form
        """
        return self.llm.query(prompt)

    def _store_context(self, context: TranslationContext):
        """Store context with LRU eviction."""
        if len(self.context_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_id = min(
                self.context_cache, key=lambda k: self.context_cache[k].timestamp
            )
            del self.context_cache[oldest_id]

        self.context_cache[context.context_id] = context

    def _extract_key_terms(self, nl_text: str) -> List[str]:
        """Placeholder for key term extraction."""
        return []

    def _extract_entities(self, nl_text: str) -> Dict[str, str]:
        """Placeholder for entity extraction."""
        return {}
