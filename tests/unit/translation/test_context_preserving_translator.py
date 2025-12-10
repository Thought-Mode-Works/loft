import unittest
from unittest.mock import MagicMock

from loft.translation.context_preserving_translator import ContextPreservingTranslator
from loft.translation.schemas import LegalRule
from loft.translation.nl_to_asp import NLToASPResult


class MockLLM:
    def query(self, prompt: str = "", **kwargs) -> object:
        class MockResponse:
            def __init__(self, content):
                self.content = content

        if "Extract legal entities" in prompt:
            return MockResponse(LegalRule(rule_id="test_rule", head="", body=[]))
        elif prompt:
            return MockResponse(f"LLM translation of: {prompt}")
        return MockResponse("LLM response")


class MockNLToASPTranslator:
    def translate(self, nl_text: str) -> NLToASPResult:
        if "contract is valid" in nl_text:
            return self.translate_to_rule(nl_text)
        return NLToASPResult(
            asp_facts=[f"% Translated: {nl_text}"],
            source_nl=nl_text
        )

    def translate_to_rule(self, nl_text: str) -> NLToASPResult:
        rule = "contract_valid(C) :- has_offer(C), has_acceptance(C), has_consideration(C)."
        return NLToASPResult(
            asp_facts=[rule],
            source_nl=nl_text,
            confidence=0.9,
            extraction_method="llm_constrained",
        )

class TestContextPreservingTranslator(unittest.TestCase):
    def setUp(self):
        self.llm = MockLLM()
        self.translator = ContextPreservingTranslator(self.llm, max_cache_size=2)
        self.translator.nl_to_asp_translator = MockNLToASPTranslator()

    def test_translate_nl_to_asp_with_context_storage(self):
        nl_text = "A contract is valid if it has offer, acceptance, and consideration."
        asp_code, context_id = self.translator.translate_nl_to_asp(nl_text)

        self.assertIn(context_id, self.translator.context_cache)
        context = self.translator.context_cache[context_id]
        self.assertEqual(context.original_nl, nl_text)
        self.assertIn("contract_valid(C)", asp_code)

    def test_translate_asp_to_nl_with_context_retrieval(self):
        nl_text = "A contract is valid if it has offer, acceptance, and consideration."
        asp_code, context_id = self.translator.translate_nl_to_asp(nl_text)

        # Mock the LLM query method to check the prompt
        self.llm.query = MagicMock(return_value="reconstructed text")

        reconstructed_nl = self.translator.translate_asp_to_nl(asp_code, context_id)

        self.llm.query.assert_called_once()
        prompt = self.llm.query.call_args[0][0]
        self.assertIn(nl_text, prompt)
        self.assertEqual(reconstructed_nl, "reconstructed text")

    def test_lru_cache_eviction(self):
        # Add two items to the cache
        _, context_id1 = self.translator.translate_nl_to_asp("First statement")
        _, context_id2 = self.translator.translate_nl_to_asp("Second statement")

        # The cache should be full
        self.assertEqual(len(self.translator.context_cache), 2)

        # Add a third item, which should evict the first one
        _, context_id3 = self.translator.translate_nl_to_asp("Third statement")

        self.assertEqual(len(self.translator.context_cache), 2)
        self.assertIn(context_id2, self.translator.context_cache)
        self.assertIn(context_id3, self.translator.context_cache)
        self.assertNotIn(context_id1, self.translator.context_cache)


if __name__ == "__main__":
    unittest.main()
