"""
Test cases for Statute of Frauds reasoning.

This module contains 20+ diverse test cases covering:
- Clear written contracts
- Oral contracts
- Contracts below/above thresholds
- Edge cases (email, text message)
- Exception scenarios
- Ambiguous cases requiring LLM input
"""

from loft.legal.statute_of_frauds import StatuteOfFraudsTestCase


# Test Case 1: Clear Written Land Sale
TEST_CASE_1 = StatuteOfFraudsTestCase(
    case_id="clear_written_land_sale",
    description="Land sale contract with proper written agreement",
    asp_facts="""
contract_fact(c1).
land_sale_contract(c1).
party_fact(john).
party_fact(mary).
party_to_contract(c1, john).
party_to_contract(c1, mary).
writing_fact(w1).
references_contract(w1, c1).
signed_by(w1, john).
signed_by(w1, mary).
identifies_parties(w1).
describes_subject_matter(w1).
states_consideration(w1).
""",
    expected_results={"enforceable": True},
    reasoning_chain=[
        "Contract is a land sale",
        "Land sales are within statute",
        "Has sufficient writing",
        "Therefore satisfies statute of frauds",
        "Therefore enforceable",
    ],
    confidence_level="high",
    legal_citations=["Statute of Frauds", "Restatement (Second) § 125"],
)

# Test Case 2: Oral Land Sale (No Writing)
TEST_CASE_2 = StatuteOfFraudsTestCase(
    case_id="oral_land_sale",
    description="Oral land sale contract without any writing",
    asp_facts="""
contract_fact(c2).
land_sale_contract(c2).
party_fact(alice).
party_fact(bob).
party_to_contract(c2, alice).
party_to_contract(c2, bob).
""",
    expected_results={"enforceable": False},
    reasoning_chain=[
        "Contract is a land sale",
        "Land sales are within statute",
        "No writing exists",
        "No exception applies",
        "Therefore unenforceable",
    ],
    confidence_level="high",
    legal_citations=["Statute of Frauds", "Restatement (Second) § 125"],
)

# Test Case 3: Goods Sale Under $500
TEST_CASE_3 = StatuteOfFraudsTestCase(
    case_id="goods_under_500",
    description="Oral goods sale under $500 threshold",
    asp_facts="""
contract_fact(c3).
goods_sale_contract(c3).
sale_amount(c3, 300).
party_fact(seller1).
party_fact(buyer1).
party_to_contract(c3, seller1).
party_to_contract(c3, buyer1).
""",
    expected_results={"enforceable": True},
    reasoning_chain=[
        "Contract is for goods sale",
        "Sale amount is $300 (under $500)",
        "Not within statute",
        "No writing required",
        "Therefore enforceable",
    ],
    confidence_level="high",
    legal_citations=["UCC § 2-201"],
)

# Test Case 4: Goods Sale Over $500 Without Writing
TEST_CASE_4 = StatuteOfFraudsTestCase(
    case_id="goods_over_500_no_writing",
    description="Oral goods sale over $500 without writing",
    asp_facts="""
contract_fact(c4).
goods_sale_contract(c4).
sale_amount(c4, 750).
party_fact(seller2).
party_fact(buyer2).
party_to_contract(c4, seller2).
party_to_contract(c4, buyer2).
""",
    expected_results={"enforceable": False},
    reasoning_chain=[
        "Contract is for goods sale",
        "Sale amount is $750 (over $500)",
        "Within statute (UCC)",
        "No writing exists",
        "Therefore unenforceable",
    ],
    confidence_level="high",
    legal_citations=["UCC § 2-201"],
)

# Test Case 5: Goods Sale Over $500 With Writing
TEST_CASE_5 = StatuteOfFraudsTestCase(
    case_id="goods_over_500_with_writing",
    description="Written goods sale over $500",
    asp_facts="""
contract_fact(c5).
goods_sale_contract(c5).
sale_amount(c5, 750).
party_fact(seller3).
party_fact(buyer3).
party_to_contract(c5, seller3).
party_to_contract(c5, buyer3).
writing_fact(w5).
references_contract(w5, c5).
signed_by(w5, seller3).
identifies_parties(w5).
describes_subject_matter(w5).
states_consideration(w5).
""",
    expected_results={"enforceable": True},
    confidence_level="high",
    legal_citations=["UCC § 2-201"],
)

# Test Case 6: Part Performance Exception
TEST_CASE_6 = StatuteOfFraudsTestCase(
    case_id="part_performance_exception",
    description="Oral land sale with part performance exception",
    asp_facts="""
contract_fact(c6).
land_sale_contract(c6).
party_fact(buyer4).
party_fact(seller4).
party_to_contract(c6, buyer4).
party_to_contract(c6, seller4).
part_performance(c6).
substantial_actions_taken(c6).
detrimental_reliance(c6).
""",
    expected_results={"enforceable": True},
    reasoning_chain=[
        "Contract is a land sale",
        "Within statute",
        "No writing, but part performance exception applies",
        "Substantial actions taken",
        "Detrimental reliance",
        "Therefore satisfies statute via exception",
        "Therefore enforceable",
    ],
    confidence_level="high",
    legal_citations=["Shaughnessy v. Eidsmo", "Restatement (Second) § 129"],
)

# Test Case 7: Promissory Estoppel Exception
TEST_CASE_7 = StatuteOfFraudsTestCase(
    case_id="promissory_estoppel",
    description="Contract with promissory estoppel exception",
    asp_facts="""
contract_fact(c7).
land_sale_contract(c7).
party_fact(promisor).
party_fact(promisee).
party_to_contract(c7, promisor).
party_to_contract(c7, promisee).
clear_promise(c7).
reasonable_reliance(c7).
substantial_detriment(c7).
injustice_without_enforcement(c7).
""",
    expected_results={"enforceable": True},
    reasoning_chain=[
        "Clear promise made",
        "Promisee reasonably relied",
        "Suffered substantial detriment",
        "Injustice without enforcement",
        "Promissory estoppel exception applies",
        "Therefore enforceable",
    ],
    confidence_level="high",
    legal_citations=["Restatement (Second) § 139"],
)

# Test Case 8: Long-term Contract (Over 1 Year)
TEST_CASE_8 = StatuteOfFraudsTestCase(
    case_id="long_term_contract_oral",
    description="Oral contract that cannot be performed within one year",
    asp_facts="""
contract_fact(c8).
contract_fact(c8).
duration_months(c8, 24).
party_fact(employer).
party_fact(employee).
party_to_contract(c8, employer).
party_to_contract(c8, employee).
""",
    expected_results={"enforceable": False},
    reasoning_chain=[
        "Contract duration is 24 months",
        "Cannot be performed within 1 year",
        "Within statute",
        "No writing",
        "Therefore unenforceable",
    ],
    confidence_level="high",
    legal_citations=["Restatement (Second) § 130"],
)

# Test Case 9: Short-term Contract (Under 1 Year)
TEST_CASE_9 = StatuteOfFraudsTestCase(
    case_id="short_term_contract_oral",
    description="Oral contract performable within one year",
    asp_facts="""
contract_fact(c9).
contract_fact(c9).
duration_months(c9, 6).
party_fact(company).
party_fact(contractor).
party_to_contract(c9, company).
party_to_contract(c9, contractor).
""",
    expected_results={"enforceable": True},
    reasoning_chain=[
        "Contract duration is 6 months",
        "Can be performed within 1 year",
        "Not within statute",
        "No writing required",
        "Therefore enforceable",
    ],
    confidence_level="high",
    legal_citations=["Restatement (Second) § 130"],
)

# Test Case 10: Merchant Confirmation
TEST_CASE_10 = StatuteOfFraudsTestCase(
    case_id="merchant_confirmation",
    description="Goods sale between merchants with written confirmation",
    asp_facts="""
contract_fact(c10).
goods_sale_contract(c10).
sale_amount(c10, 1000).
party_fact(merchant1).
party_fact(merchant2).
party_to_contract(c10, merchant1).
party_to_contract(c10, merchant2).
both_parties_merchants(c10).
written_confirmation_sent(c10).
""",
    expected_results={"enforceable": True},
    reasoning_chain=[
        "Goods sale over $500",
        "Within statute",
        "Both parties are merchants",
        "Written confirmation sent",
        "No objection within 10 days",
        "Merchant confirmation exception applies",
        "Therefore enforceable",
    ],
    confidence_level="high",
    legal_citations=["UCC § 2-201(2)"],
)

# Test Case 11: Specially Manufactured Goods
TEST_CASE_11 = StatuteOfFraudsTestCase(
    case_id="specially_manufactured_goods",
    description="Specially manufactured goods exception",
    asp_facts="""
contract_fact(c11).
goods_sale_contract(c11).
sale_amount(c11, 2000).
party_fact(manufacturer).
party_fact(buyer5).
party_to_contract(c11, manufacturer).
party_to_contract(c11, buyer5).
specially_manufactured(c11).
not_suitable_for_others(c11).
substantial_beginning_of_manufacture(c11).
""",
    expected_results={"enforceable": True},
    reasoning_chain=[
        "Goods sale over $500",
        "Within statute",
        "Goods are specially manufactured",
        "Not suitable for sale to others",
        "Substantial beginning of manufacture",
        "Exception applies",
        "Therefore enforceable",
    ],
    confidence_level="high",
    legal_citations=["UCC § 2-201(3)(a)"],
)

# Test Case 12: Admission in Pleadings
TEST_CASE_12 = StatuteOfFraudsTestCase(
    case_id="admission_in_pleadings",
    description="Contract admitted in court pleadings",
    asp_facts="""
contract_fact(c12).
land_sale_contract(c12).
party_fact(defendant).
party_fact(plaintiff).
party_to_contract(c12, defendant).
party_to_contract(c12, plaintiff).
admitted_in_pleadings(c12, defendant).
""",
    expected_results={"enforceable": True},
    reasoning_chain=[
        "Land sale within statute",
        "No writing",
        "But defendant admitted contract in pleadings",
        "Admission exception applies",
        "Therefore enforceable",
    ],
    confidence_level="high",
    legal_citations=["UCC § 2-201(3)(b)"],
)

# Test Case 13: Suretyship Contract Without Writing
TEST_CASE_13 = StatuteOfFraudsTestCase(
    case_id="suretyship_oral",
    description="Oral suretyship contract",
    asp_facts="""
contract_fact(c13).
suretyship_contract(c13).
party_fact(surety).
party_fact(creditor).
party_to_contract(c13, surety).
party_to_contract(c13, creditor).
""",
    expected_results={"enforceable": False},
    reasoning_chain=[
        "Suretyship contract",
        "Within statute",
        "No writing",
        "No exception applies",
        "Therefore unenforceable",
    ],
    confidence_level="high",
    legal_citations=["Restatement (Second) § 112"],
)

# Test Case 14: Suretyship Contract With Writing
TEST_CASE_14 = StatuteOfFraudsTestCase(
    case_id="suretyship_written",
    description="Written suretyship contract",
    asp_facts="""
contract_fact(c14).
suretyship_contract(c14).
party_fact(surety2).
party_fact(creditor2).
party_to_contract(c14, surety2).
party_to_contract(c14, creditor2).
writing_fact(w14).
references_contract(w14, c14).
signed_by(w14, surety2).
identifies_parties(w14).
describes_subject_matter(w14).
states_consideration(w14).
""",
    expected_results={"enforceable": True},
    confidence_level="high",
    legal_citations=["Restatement (Second) § 112"],
)

# Test Case 15: Marriage Consideration Contract
TEST_CASE_15 = StatuteOfFraudsTestCase(
    case_id="marriage_consideration_oral",
    description="Oral contract in consideration of marriage",
    asp_facts="""
contract_fact(c15).
marriage_consideration_contract(c15).
party_fact(party1).
party_fact(party2).
party_to_contract(c15, party1).
party_to_contract(c15, party2).
""",
    expected_results={"enforceable": False},
    confidence_level="high",
    legal_citations=["Restatement (Second) § 124"],
)

# Test Case 16: Executor Contract
TEST_CASE_16 = StatuteOfFraudsTestCase(
    case_id="executor_contract_oral",
    description="Oral executor/administrator contract",
    asp_facts="""
contract_fact(c16).
executor_contract(c16).
party_fact(executor).
party_fact(beneficiary).
party_to_contract(c16, executor).
party_to_contract(c16, beneficiary).
""",
    expected_results={"enforceable": False},
    confidence_level="high",
    legal_citations=["Restatement (Second) § 111"],
)

# Test Case 17: Email Contract (Edge Case)
TEST_CASE_17 = StatuteOfFraudsTestCase(
    case_id="email_contract",
    description="Land sale contract via email with electronic signature",
    asp_facts="""
contract_fact(c17).
land_sale_contract(c17).
party_fact(p1).
party_fact(p2).
party_to_contract(c17, p1).
party_to_contract(c17, p2).
writing_fact(email1).
references_contract(email1, c17).
signed_by(email1, p1).
identifies_parties(email1).
describes_subject_matter(email1).
states_consideration(email1).
""",
    expected_results={"enforceable": True},
    confidence_level="medium",
    requires_llm_query=True,
    llm_query_focus="Is an email with electronic signature a valid 'writing' and 'signature' under statute of frauds?",
    legal_citations=["ESIGN Act", "UETA"],
)

# Test Case 18: Partially Signed Document
TEST_CASE_18 = StatuteOfFraudsTestCase(
    case_id="partially_signed",
    description="Contract signed by only one party",
    asp_facts="""
contract_fact(c18).
land_sale_contract(c18).
party_fact(p3).
party_fact(p4).
party_to_contract(c18, p3).
party_to_contract(c18, p4).
writing_fact(w18).
references_contract(w18, c18).
signed_by(w18, p3).
identifies_parties(w18).
describes_subject_matter(w18).
states_consideration(w18).
""",
    expected_results={"enforceable": True},
    reasoning_chain=[
        "Contract within statute",
        "Writing exists signed by p3",
        "Only party to be charged needs to sign",
        "p3 can be enforced against",
        "Therefore enforceable (against p3)",
    ],
    confidence_level="high",
    legal_citations=["Restatement (Second) § 134"],
)

# Test Case 19: Missing Essential Terms
TEST_CASE_19 = StatuteOfFraudsTestCase(
    case_id="missing_essential_terms",
    description="Written contract missing essential terms",
    asp_facts="""
contract_fact(c19).
land_sale_contract(c19).
party_fact(p5).
party_fact(p6).
party_to_contract(c19, p5).
party_to_contract(c19, p6).
writing_fact(w19).
references_contract(w19, c19).
signed_by(w19, p5).
signed_by(w19, p6).
identifies_parties(w19).
""",
    expected_results={"enforceable": False},
    reasoning_chain=[
        "Contract within statute",
        "Writing exists and is signed",
        "But writing lacks essential terms",
        "Does not contain subject matter or consideration",
        "No sufficient writing",
        "Therefore unenforceable",
    ],
    confidence_level="high",
    legal_citations=["Crabtree v. Elizabeth Arden Sales Corp."],
)

# Test Case 20: Text Message Contract (Edge Case)
TEST_CASE_20 = StatuteOfFraudsTestCase(
    case_id="text_message_contract",
    description="Goods sale via text message",
    asp_facts="""
contract_fact(c20).
goods_sale_contract(c20).
sale_amount(c20, 600).
party_fact(seller5).
party_fact(buyer6).
party_to_contract(c20, seller5).
party_to_contract(c20, buyer6).
writing_fact(text1).
references_contract(text1, c20).
signed_by(text1, seller5).
identifies_parties(text1).
describes_subject_matter(text1).
states_consideration(text1).
""",
    expected_results={"enforceable": True},
    confidence_level="medium",
    requires_llm_query=True,
    llm_query_focus="Are text messages valid writings under statute of frauds? Can sender name constitute a signature?",
    legal_citations=["ESIGN Act", "St. John's Holdings, LLC v. Two Electronics, LLC"],
)

# Test Case 21: Mixed Scenario - Goods and Land
TEST_CASE_21 = StatuteOfFraudsTestCase(
    case_id="mixed_goods_and_land",
    description="Contract involving both goods and land sale",
    asp_facts="""
contract_fact(c21).
land_sale_contract(c21).
goods_sale_contract(c21).
sale_amount(c21, 300).
party_fact(p7).
party_fact(p8).
party_to_contract(c21, p7).
party_to_contract(c21, p8).
writing_fact(w21).
references_contract(w21, c21).
signed_by(w21, p7).
signed_by(w21, p8).
identifies_parties(w21).
describes_subject_matter(w21).
states_consideration(w21).
""",
    expected_results={"enforceable": True},
    reasoning_chain=[
        "Contract involves land sale (within statute)",
        "Also involves goods under $500 (also within statute as land component)",
        "Has sufficient writing",
        "Therefore enforceable",
    ],
    confidence_level="high",
)

# All test cases
ALL_TEST_CASES = [
    TEST_CASE_1,
    TEST_CASE_2,
    TEST_CASE_3,
    TEST_CASE_4,
    TEST_CASE_5,
    TEST_CASE_6,
    TEST_CASE_7,
    TEST_CASE_8,
    TEST_CASE_9,
    TEST_CASE_10,
    TEST_CASE_11,
    TEST_CASE_12,
    TEST_CASE_13,
    TEST_CASE_14,
    TEST_CASE_15,
    TEST_CASE_16,
    TEST_CASE_17,
    TEST_CASE_18,
    TEST_CASE_19,
    TEST_CASE_20,
    TEST_CASE_21,
]
