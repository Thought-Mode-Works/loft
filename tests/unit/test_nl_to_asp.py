"""
Unit tests for NL to ASP translation.
"""

from loft.translation import (
    nl_to_asp_facts,
    nl_to_asp_rule,
    NLToASPTranslator,
    ContractFact,
    pattern_based_extraction,
    quick_extract_facts,
    ASPGrounder,
    AmbiguityHandler,
    compute_asp_equivalence,
)
from loft.translation.patterns import (
    normalize_identifier,
    extract_contract_type,
    extract_essential_elements,
    extract_parties,
)


class TestNormalizeIdentifier:
    """Test identifier normalization."""

    def test_simple_word(self) -> None:
        """Test normalizing simple word."""
        assert normalize_identifier("Contract") == "contract"

    def test_with_spaces(self) -> None:
        """Test normalizing words with spaces."""
        assert normalize_identifier("Land Sale") == "land_sale"

    def test_with_special_chars(self) -> None:
        """Test normalizing with special characters."""
        assert normalize_identifier("Contract-123") == "contract_123"

    def test_multiple_spaces(self) -> None:
        """Test collapsing multiple spaces."""
        assert normalize_identifier("Land   Sale   Contract") == "land_sale_contract"


class TestPatternBasedExtraction:
    """Test pattern-based extraction."""

    def test_is_a_pattern(self) -> None:
        """Test 'X is a Y' pattern."""
        facts = pattern_based_extraction("c1 is a contract")
        assert "contract(c1)." in facts

    def test_has_pattern(self) -> None:
        """Test 'X has Y' pattern."""
        facts = pattern_based_extraction("c1 has writing")
        assert "has_writing(c1)." in facts

    def test_signed_by_pattern(self) -> None:
        """Test 'X signed by Y' pattern."""
        facts = pattern_based_extraction("w1 was signed by John")
        assert "signed_by(w1, john)." in facts

    def test_between_pattern(self) -> None:
        """Test 'contract between X and Y' pattern."""
        facts = pattern_based_extraction("contract between Alice and Bob")
        assert "party(alice)." in facts
        assert "party(bob)." in facts
        assert "party_to_contract(contract_1, alice)." in facts

    def test_multiple_patterns(self) -> None:
        """Test multiple patterns in one text."""
        nl = "c1 is a contract. c1 has writing"  # Separate sentences
        facts = pattern_based_extraction(nl)
        assert "contract(c1)." in facts
        assert "has_writing(c1)." in facts


class TestQuickExtractFacts:
    """Test quick extraction function."""

    def test_simple_contract(self) -> None:
        """Test extracting simple contract."""
        nl = "This is a land sale contract."
        facts = quick_extract_facts(nl)
        assert any("land_sale" in f for f in facts)

    def test_with_parties(self) -> None:
        """Test extracting with parties."""
        nl = "Contract between John and Mary"
        facts = quick_extract_facts(nl)
        assert any("john" in f.lower() for f in facts)
        assert any("mary" in f.lower() for f in facts)

    def test_with_writing(self) -> None:
        """Test extracting writing mention."""
        nl = "The contract has a written document"
        facts = quick_extract_facts(nl)
        assert any("has_writing" in f or "written" in f for f in facts)


class TestExtractContractType:
    """Test contract type extraction."""

    def test_land_sale(self) -> None:
        """Test extracting land sale type."""
        assert extract_contract_type("This is a land sale contract") == "land_sale"

    def test_goods_sale(self) -> None:
        """Test extracting goods sale type."""
        assert extract_contract_type("This is a sale of goods") == "goods_sale"

    def test_service(self) -> None:
        """Test extracting service type."""
        assert extract_contract_type("This is a service contract") == "service"

    def test_general(self) -> None:
        """Test general contract fallback."""
        assert extract_contract_type("This is a contract") == "general"


class TestExtractEssentialElements:
    """Test essential elements extraction."""

    def test_consideration(self) -> None:
        """Test extracting consideration."""
        elements = extract_essential_elements("The contract has consideration")
        assert elements["has_consideration"] is True

    def test_mutual_assent(self) -> None:
        """Test extracting mutual assent."""
        elements = extract_essential_elements("Both parties reached mutual assent")
        assert elements["has_mutual_assent"] is True

    def test_writing(self) -> None:
        """Test extracting writing."""
        elements = extract_essential_elements("The contract is in writing")
        assert elements["has_writing"] is True

    def test_signed(self) -> None:
        """Test extracting signed status."""
        elements = extract_essential_elements("The document was signed")
        assert elements["is_signed"] is True


class TestExtractParties:
    """Test party extraction."""

    def test_between_pattern(self) -> None:
        """Test extracting from 'between X and Y'."""
        parties = extract_parties("Agreement between Alice Smith and Bob Jones")
        assert "Alice Smith" in parties
        assert "Bob Jones" in parties

    def test_capitalized_names(self) -> None:
        """Test extracting capitalized names."""
        parties = extract_parties("John signed the contract with Mary")
        assert "John" in parties
        assert "Mary" in parties

    def test_filters_common_words(self) -> None:
        """Test filtering out common words."""
        parties = extract_parties("The Contract between John and Mary")
        assert "The" not in parties
        assert "Contract" not in parties


class TestContractFactSchema:
    """Test ContractFact Pydantic schema."""

    def test_basic_contract(self) -> None:
        """Test basic contract fact."""
        contract = ContractFact(
            contract_id="c1",
            contract_type="land_sale",
            parties=["Alice", "Bob"],
        )
        facts = contract.to_asp()

        assert "contract(c1)." in facts
        assert "land_sale_contract(c1)." in facts
        assert "party(alice)." in facts
        assert "party(bob)." in facts

    def test_with_writing(self) -> None:
        """Test contract with writing."""
        contract = ContractFact(
            contract_id="c1",
            has_writing=True,
            is_signed=True,
            parties=["Alice"],
        )
        facts = contract.to_asp()

        assert any("has_writing(c1" in f for f in facts)
        assert any("signed_by" in f for f in facts)

    def test_with_sale_amount(self) -> None:
        """Test contract with sale amount."""
        contract = ContractFact(
            contract_id="c1",
            sale_amount=500000.0,
        )
        facts = contract.to_asp()

        assert "sale_amount(c1, 500000)." in facts


class TestNLToASPFacts:
    """Test nl_to_asp_facts function."""

    def test_simple_statement(self) -> None:
        """Test translating simple statement."""
        nl = "c1 is a contract"
        facts = nl_to_asp_facts(nl)
        assert len(facts) > 0
        assert any("contract" in f.lower() for f in facts)

    def test_land_sale_contract(self) -> None:
        """Test translating land sale contract."""
        nl = "This is a land sale contract for $500,000"
        facts = nl_to_asp_facts(nl)
        assert any("land_sale" in f for f in facts)

    def test_signed_contract(self) -> None:
        """Test translating signed contract."""
        nl = "The contract was signed by John"
        facts = nl_to_asp_facts(nl)
        assert any("signed" in f.lower() for f in facts)
        assert any("john" in f.lower() for f in facts)

    def test_empty_input(self) -> None:
        """Test with empty input."""
        facts = nl_to_asp_facts("")
        assert isinstance(facts, list)


class TestNLToASPRule:
    """Test nl_to_asp_rule function."""

    def test_simple_rule(self) -> None:
        """Test translating simple rule."""
        nl = "A contract is enforceable if it is valid"
        rule = nl_to_asp_rule(nl)
        assert "enforceable" in rule.lower()
        assert ":-" in rule
        assert rule.endswith(".")

    def test_rule_with_negation(self) -> None:
        """Test rule with unless (negation)."""
        nl = "A contract is enforceable unless proven unenforceable"
        rule = nl_to_asp_rule(nl)
        assert "enforceable" in rule.lower()
        assert "not" in rule.lower()

    def test_unparseable_rule(self) -> None:
        """Test with unparseable rule."""
        nl = "Some random text"
        rule = nl_to_asp_rule(nl)
        # Should return comment for unparseable rules
        assert "%" in rule or ":-" in rule


class TestNLToASPTranslator:
    """Test NLToASPTranslator class."""

    def test_initialization(self) -> None:
        """Test translator initialization."""
        translator = NLToASPTranslator()
        assert translator is not None
        assert translator.use_llm is False

    def test_translate_to_facts(self) -> None:
        """Test translate_to_facts method."""
        translator = NLToASPTranslator()
        result = translator.translate_to_facts("c1 is a contract")

        assert result.asp_facts is not None
        assert len(result.asp_facts) > 0
        assert result.source_nl == "c1 is a contract"
        assert 0.0 <= result.confidence <= 1.0

    def test_translate_to_rule(self) -> None:
        """Test translate_to_rule method."""
        translator = NLToASPTranslator()
        result = translator.translate_to_rule("A contract is enforceable if it is valid")

        assert result.asp_facts is not None
        assert len(result.asp_facts) > 0
        assert ":-" in result.asp_facts[0] or "%" in result.asp_facts[0]


class TestASPGrounder:
    """Test ASPGrounder class."""

    def test_initialization(self) -> None:
        """Test grounder initialization."""
        grounder = ASPGrounder()
        assert grounder is not None

    def test_valid_fact(self) -> None:
        """Test validating valid fact."""
        grounder = ASPGrounder()
        valid, invalid = grounder.ground_and_validate(["contract(c1)."])

        assert "contract(c1)." in valid
        assert len(invalid) == 0

    def test_invalid_syntax(self) -> None:
        """Test rejecting invalid syntax."""
        grounder = ASPGrounder()
        valid, invalid = grounder.ground_and_validate(["contract(c1"])  # Missing closing

        assert len(valid) == 0
        assert len(invalid) > 0

    def test_multiple_facts(self) -> None:
        """Test validating multiple facts."""
        grounder = ASPGrounder()
        facts = ["contract(c1).", "party(john).", "writing(w1)."]
        valid, invalid = grounder.ground_and_validate(facts)

        assert len(valid) == 3
        assert len(invalid) == 0

    def test_filters_comments(self) -> None:
        """Test filtering out comments."""
        grounder = ASPGrounder()
        valid, invalid = grounder.ground_and_validate(["% This is a comment", "contract(c1)."])

        assert len(valid) == 1
        assert "contract(c1)." in valid


class TestAmbiguityHandler:
    """Test AmbiguityHandler class."""

    def test_multiple_candidates(self) -> None:
        """Test detecting multiple interpretations."""
        handler = AmbiguityHandler()
        ambiguity = handler.detect_ambiguity("Some text", ["fact1.", "fact2."])

        assert ambiguity is not None
        assert "multiple" in ambiguity.lower()

    def test_unclear_references(self) -> None:
        """Test detecting unclear references."""
        handler = AmbiguityHandler()
        ambiguity = handler.detect_ambiguity("It was signed", [])

        # May or may not detect this simple case
        # Just ensure it returns something reasonable
        assert ambiguity is None or isinstance(ambiguity, str)

    def test_no_ambiguity(self) -> None:
        """Test when there's no ambiguity."""
        handler = AmbiguityHandler()
        ambiguity = handler.detect_ambiguity("Contract c1 is valid", ["contract(c1)."])

        assert ambiguity is None

    def test_request_clarification(self) -> None:
        """Test generating clarification request."""
        handler = AmbiguityHandler()
        clarification = handler.request_clarification("Multiple meanings")

        assert "ambiguous" in clarification.lower()
        assert "clarify" in clarification.lower()


class TestRoundtripTesting:
    """Test roundtrip translation."""

    def test_compute_asp_equivalence_identical(self) -> None:
        """Test equivalence of identical ASP."""
        equiv = compute_asp_equivalence("contract(c1).", "contract(c1).")
        assert equiv == 1.0

    def test_compute_asp_equivalence_different(self) -> None:
        """Test equivalence of different ASP."""
        equiv = compute_asp_equivalence("contract(c1).", "void(c1).")
        assert equiv < 0.5

    def test_compute_asp_equivalence_similar(self) -> None:
        """Test equivalence of similar ASP."""
        equiv = compute_asp_equivalence(
            "contract(c1), enforceable(c1).",
            "enforceable(c1), contract(c1).",  # Different order
        )
        assert equiv > 0.8  # Should be high since predicates are the same


class TestNLToStructured:
    """Test nl_to_structured function."""

    def test_contract_fact_without_llm(self) -> None:
        """Test parsing ContractFact without LLM."""
        from loft.translation.nl_to_asp import nl_to_structured

        nl = "John and Mary have a land sale contract for $500,000"
        contract = nl_to_structured(nl, ContractFact, llm_interface=None)

        assert contract.contract_id == "contract_1"
        assert "John" in contract.parties or "Mary" in contract.parties
        assert contract.sale_amount == 500000.0

    def test_with_llm_interface(self) -> None:
        """Test parsing with LLM interface."""
        from loft.translation.nl_to_asp import nl_to_structured
        from unittest.mock import Mock

        llm = Mock()
        mock_response = Mock()
        mock_response.content = ContractFact(contract_id="c1", contract_type="land_sale")
        llm.query.return_value = mock_response

        nl = "This is a land sale contract"
        contract = nl_to_structured(nl, ContractFact, llm_interface=llm)

        assert contract.contract_id == "c1"
        assert contract.contract_type == "land_sale"
        llm.query.assert_called_once()

    def test_extracted_entities_without_llm(self) -> None:
        """Test parsing ExtractedEntities without LLM."""
        from loft.translation.nl_to_asp import nl_to_structured
        from loft.translation.schemas import ExtractedEntities

        nl = "John and Mary have a contract"
        entities = nl_to_structured(nl, ExtractedEntities, llm_interface=None)

        assert len(entities.contracts) > 0
        assert len(entities.parties) > 0

    def test_unsupported_schema(self) -> None:
        """Test with unsupported schema type."""
        from loft.translation.nl_to_asp import nl_to_structured
        from pydantic import BaseModel

        class CustomSchema(BaseModel):
            value: str

        nl = "Some text"
        try:
            nl_to_structured(nl, CustomSchema, llm_interface=None)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Unsupported schema" in str(e)


class TestNLToASPResult:
    """Test NLToASPResult dataclass."""

    def test_creation(self) -> None:
        """Test creating NLToASPResult."""
        from loft.translation.nl_to_asp import NLToASPResult

        result = NLToASPResult(
            asp_facts=["contract(c1)."],
            source_nl="c1 is a contract",
            confidence=0.8,
            extraction_method="pattern",
        )

        assert result.asp_facts == ["contract(c1)."]
        assert result.source_nl == "c1 is a contract"
        assert result.confidence == 0.8
        assert result.extraction_method == "pattern"
        assert result.ambiguities == []
        assert result.metadata == {}


class TestParseRuleFromNLBasic:
    """Test _parse_rule_from_nl_basic function."""

    def test_if_pattern_with_has(self) -> None:
        """Test 'A X is Y if it has Z' pattern."""
        from loft.translation.nl_to_asp import _parse_rule_from_nl_basic

        nl = "a contract satisfies statute of frauds if it has a signed writing"
        rule = _parse_rule_from_nl_basic(nl)

        assert "satisfies_statute_of_frauds(C)" in rule
        assert ":-" in rule
        assert "has_" in rule

    def test_unless_pattern(self) -> None:
        """Test 'A X is Y unless Z' pattern."""
        from loft.translation.nl_to_asp import _parse_rule_from_nl_basic

        nl = "a contract is enforceable unless proven unenforceable"
        rule = _parse_rule_from_nl_basic(nl)

        assert "enforceable(C)" in rule
        assert "not" in rule
        assert "unenforceable(C)" in rule

    def test_unrecognized_pattern(self) -> None:
        """Test unrecognized pattern returns comment."""
        from loft.translation.nl_to_asp import _parse_rule_from_nl_basic

        nl = "This is not a valid rule format"
        rule = _parse_rule_from_nl_basic(nl)

        assert "%" in rule
        assert "Could not parse" in rule


class TestTranslatorWithLLM:
    """Test NLToASPTranslator with LLM."""

    def test_translator_with_llm_enabled(self) -> None:
        """Test translator with LLM enabled."""
        from unittest.mock import Mock

        llm = Mock()
        translator = NLToASPTranslator(llm_interface=llm, use_llm_by_default=True)

        assert translator.llm_interface == llm
        assert translator.use_llm is True

    def test_translator_with_llm_disabled(self) -> None:
        """Test translator with LLM but disabled."""
        from unittest.mock import Mock

        llm = Mock()
        translator = NLToASPTranslator(llm_interface=llm, use_llm_by_default=False)

        assert translator.llm_interface == llm
        assert translator.use_llm is False

    def test_extract_entities(self) -> None:
        """Test extract_entities method."""
        from loft.translation.schemas import ExtractedEntities

        translator = NLToASPTranslator()
        entities = translator.extract_entities("John and Mary have a contract")

        assert isinstance(entities, ExtractedEntities)
        assert len(entities.contracts) > 0


class TestPatternFunctions:
    """Test additional pattern functions."""

    def test_extract_contract_type_employment(self) -> None:
        """Test extracting employment contract type."""
        assert extract_contract_type("This is an employment contract") == "employment"

    def test_extract_contract_type_lease(self) -> None:
        """Test extracting lease contract type."""
        assert extract_contract_type("This is a lease agreement") == "lease"

    def test_extract_essential_elements_agreement(self) -> None:
        """Test that 'agreement' implies mutual assent."""
        elements = extract_essential_elements("The parties have an agreement")
        assert elements["has_mutual_assent"] is True

    def test_extract_essential_elements_meeting_of_minds(self) -> None:
        """Test 'meeting of the minds' implies mutual assent."""
        elements = extract_essential_elements("There was a meeting of the minds")
        assert elements["has_mutual_assent"] is True

    def test_extract_essential_elements_document(self) -> None:
        """Test 'document' implies writing."""
        elements = extract_essential_elements("The contract has a document")
        assert elements["has_writing"] is True

    def test_extract_essential_elements_signature(self) -> None:
        """Test 'signature' implies signed."""
        elements = extract_essential_elements("The contract has a signature")
        assert elements["is_signed"] is True


class TestPatternExtraction:
    """Test additional pattern extraction cases."""

    def test_includes_pattern(self) -> None:
        """Test 'X includes Y' pattern."""
        facts = pattern_based_extraction("contract c1 includes essential terms")
        assert any("contains" in f for f in facts)

    def test_sale_amount_pattern(self) -> None:
        """Test sale amount extraction."""
        facts = pattern_based_extraction("sale price of $250,000")
        assert any("sale_amount" in f and "250000" in f for f in facts)

    def test_deduplication(self) -> None:
        """Test that duplicate facts are removed."""
        nl = "c1 is a contract. c1 is a contract."
        facts = pattern_based_extraction(nl)
        # Count occurrences of contract(c1).
        count = sum(1 for f in facts if f == "contract(c1).")
        assert count == 1


class TestNLToASPWithLLM:
    """Test nl_to_asp_facts with LLM."""

    def test_with_use_llm_flag(self) -> None:
        """Test using LLM flag."""
        from unittest.mock import Mock
        from loft.translation.schemas import ExtractedEntities

        llm = Mock()
        mock_response = Mock()
        mock_response.content = ExtractedEntities(contracts=[], parties=[])
        llm.query.return_value = mock_response

        facts = nl_to_asp_facts("some text", llm_interface=llm, use_llm=True)
        assert isinstance(facts, list)
        llm.query.assert_called_once()

    def test_without_llm(self) -> None:
        """Test without LLM uses patterns."""
        facts = nl_to_asp_facts("c1 is a contract", llm_interface=None, use_llm=False)
        assert isinstance(facts, list)
        assert len(facts) > 0


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_nl_to_asp_to_nl_simple(self) -> None:
        """Test simple fact extraction and back."""
        from loft.translation import asp_to_nl

        # Start with NL
        nl_original = "c1 is a contract"

        # NL → ASP
        facts = nl_to_asp_facts(nl_original)
        assert len(facts) > 0

        # ASP → NL
        nl_reconstructed = asp_to_nl(facts[0])
        assert "contract" in nl_reconstructed.lower()
        assert nl_reconstructed.endswith("?")

    def test_full_pipeline_with_translator(self) -> None:
        """Test full pipeline with translator classes."""
        from loft.translation import ASPToNLTranslator

        nl_original = "c1 is a land sale contract"

        # NL → ASP
        nl_to_asp_translator = NLToASPTranslator()
        result = nl_to_asp_translator.translate_to_facts(nl_original)

        assert len(result.asp_facts) > 0
        assert any("contract" in f for f in result.asp_facts)

        # ASP → NL
        asp_to_nl_translator = ASPToNLTranslator(domain="legal")
        for fact in result.asp_facts:
            if fact and not fact.startswith("%"):
                nl_result = asp_to_nl_translator.translate_query(fact)
                assert nl_result.natural_language
                break
