"""
Generate test scenario JSON files for statute of frauds testing.

Creates 20 diverse test scenarios covering different aspects of statute of frauds.
"""

import json
from pathlib import Path

# Define 20 test scenarios
SCENARIOS = [
    {
        "id": "sof_001",
        "description": "Oral promise to sell land",
        "facts": [
            "Alice orally promised to sell her house to Bob",
            "Bob agreed to pay $500,000",
            "No written contract was signed",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "unenforceable",
        "rationale": "Land sale contracts must be in writing under statute of frauds",
        "asp_facts": "contract_fact(c1). land_sale_contract(c1). not in_writing(c1).",
        "legal_citations": ["Statute of Frauds"],
        "difficulty": "easy",
    },
    {
        "id": "sof_002",
        "description": "Written land sale contract",
        "facts": [
            "Charlie signed a written contract to buy land from Diana",
            "The contract includes all material terms",
            "Both parties signed the document",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "enforceable",
        "rationale": "Written and signed land sale contract satisfies statute of frauds",
        "asp_facts": "contract_fact(c2). land_sale_contract(c2). in_writing(c2). signed_by_party(c2, charlie). signed_by_party(c2, diana).",
        "legal_citations": ["Statute of Frauds"],
        "difficulty": "easy",
    },
    {
        "id": "sof_003",
        "description": "Sale of goods over $500",
        "facts": [
            "Eve orally agreed to buy 100 widgets from Frank for $600",
            "No written contract exists",
            "The widgets have not been delivered",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "unenforceable",
        "rationale": "UCC 2-201 requires writing for goods over $500",
        "asp_facts": "contract_fact(c3). goods_sale_contract(c3). value(c3, 600). not in_writing(c3).",
        "legal_citations": ["UCC 2-201"],
        "difficulty": "easy",
    },
    {
        "id": "sof_004",
        "description": "Sale of goods under $500",
        "facts": [
            "Grace orally agreed to buy a lamp from Henry for $300",
            "No written contract exists",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "enforceable",
        "rationale": "Oral contracts for goods under $500 are enforceable",
        "asp_facts": "contract_fact(c4). goods_sale_contract(c4). value(c4, 300). not in_writing(c4).",
        "legal_citations": ["UCC 2-201"],
        "difficulty": "easy",
    },
    {
        "id": "sof_005",
        "description": "Merchant confirmation exception",
        "facts": [
            "Two merchants had oral agreement for goods worth $1000",
            "One merchant sent written confirmation within reasonable time",
            "Other merchant received it but did not object within 10 days",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "enforceable",
        "rationale": "UCC 2-201(2) merchant confirmation exception applies",
        "asp_facts": "contract_fact(c5). goods_sale_contract(c5). between_merchants(c5). confirmation_sent(c5). not objection_within_10_days(c5).",
        "legal_citations": ["UCC 2-201(2)"],
        "difficulty": "medium",
    },
    {
        "id": "sof_006",
        "description": "Specially manufactured goods",
        "facts": [
            "Iris ordered custom widgets from Jack for $2000",
            "Widgets are specially manufactured and not suitable for others",
            "Jack has begun manufacturing",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "enforceable",
        "rationale": "UCC 2-201(3)(a) specially manufactured goods exception",
        "asp_facts": "contract_fact(c6). goods_sale_contract(c6). specially_manufactured(c6). substantial_beginning_made(c6).",
        "legal_citations": ["UCC 2-201(3)(a)"],
        "difficulty": "medium",
    },
    {
        "id": "sof_007",
        "description": "Part performance - land with possession",
        "facts": [
            "Kim orally agreed to buy land from Leo",
            "Kim took possession of the land",
            "Kim made substantial improvements to the property",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "enforceable",
        "rationale": "Part performance exception with possession and improvements",
        "asp_facts": "contract_fact(c7). land_sale_contract(c7). possession_taken(c7). improvements_made(c7).",
        "legal_citations": ["Part Performance Doctrine"],
        "difficulty": "medium",
    },
    {
        "id": "sof_008",
        "description": "Admission in court",
        "facts": [
            "Mike sued Nora for breach of oral contract for goods worth $800",
            "Nora admitted in court testimony that contract existed",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "enforceable",
        "rationale": "UCC 2-201(3)(b) admission exception",
        "asp_facts": "contract_fact(c8). goods_sale_contract(c8). value(c8, 800). admission_in_court(c8, nora).",
        "legal_citations": ["UCC 2-201(3)(b)"],
        "difficulty": "medium",
    },
    {
        "id": "sof_009",
        "description": "Services contract for one year",
        "facts": [
            "Oscar hired Pam as consultant for 6 months",
            "Contract was oral only",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "enforceable",
        "rationale": "Statute of frauds only applies to contracts that cannot be performed within one year",
        "asp_facts": "contract_fact(c9). service_contract(c9). duration_months(c9, 6). not in_writing(c9).",
        "legal_citations": ["Statute of Frauds - One Year Rule"],
        "difficulty": "medium",
    },
    {
        "id": "sof_010",
        "description": "Services contract over one year",
        "facts": [
            "Quinn orally hired Rachel as employee for 18 months",
            "No written contract exists",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "unenforceable",
        "rationale": "Contracts that cannot be performed within one year must be in writing",
        "asp_facts": "contract_fact(c10). service_contract(c10). duration_months(c10, 18). not in_writing(c10).",
        "legal_citations": ["Statute of Frauds - One Year Rule"],
        "difficulty": "medium",
    },
    {
        "id": "sof_011",
        "description": "Payment and acceptance exception",
        "facts": [
            "Sam orally agreed to buy goods from Tina for $1500",
            "Sam paid in full",
            "Tina accepted payment and delivered goods",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "enforceable",
        "rationale": "UCC 2-201(3)(c) payment and acceptance exception",
        "asp_facts": "contract_fact(c11). goods_sale_contract(c11). payment_made(c11). goods_accepted(c11).",
        "legal_citations": ["UCC 2-201(3)(c)"],
        "difficulty": "medium",
    },
    {
        "id": "sof_012",
        "description": "Suretyship agreement",
        "facts": [
            "Uma orally promised to pay Victor's debt if Victor defaulted",
            "No written agreement",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "unenforceable",
        "rationale": "Suretyship agreements must be in writing",
        "asp_facts": "contract_fact(c12). suretyship_agreement(c12). not in_writing(c12).",
        "legal_citations": ["Statute of Frauds - Suretyship"],
        "difficulty": "hard",
    },
    {
        "id": "sof_013",
        "description": "Main purpose exception to suretyship",
        "facts": [
            "Wendy orally guaranteed Xavier's debt to protect her own business interest",
            "The main purpose was Wendy's economic advantage",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "enforceable",
        "rationale": "Main purpose exception to suretyship rule",
        "asp_facts": "contract_fact(c13). suretyship_agreement(c13). main_purpose_own_benefit(c13, wendy).",
        "legal_citations": ["Main Purpose Exception"],
        "difficulty": "hard",
    },
    {
        "id": "sof_014",
        "description": "Marriage contract",
        "facts": [
            "Yuri orally promised Zelda $10,000 if she married him",
            "No written agreement",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "unenforceable",
        "rationale": "Contracts in consideration of marriage must be in writing",
        "asp_facts": "contract_fact(c14). marriage_consideration(c14). not in_writing(c14).",
        "legal_citations": ["Statute of Frauds - Marriage"],
        "difficulty": "medium",
    },
    {
        "id": "sof_015",
        "description": "Electronic signature on land sale",
        "facts": [
            "Alice and Bob signed land sale contract via DocuSign",
            "Contract includes all material terms",
            "Electronic signatures are valid under ESIGN Act",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "enforceable",
        "rationale": "Electronic signatures satisfy writing requirement",
        "asp_facts": "contract_fact(c15). land_sale_contract(c15). electronic_signature(c15). esign_compliant(c15).",
        "legal_citations": ["ESIGN Act", "Statute of Frauds"],
        "difficulty": "medium",
    },
    {
        "id": "sof_016",
        "description": "Insufficient memorandum",
        "facts": [
            "Carol wrote note on napkin about selling land to Dave",
            "Note lacks material terms like price",
            "Carol signed it",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "unenforceable",
        "rationale": "Writing must contain material terms to satisfy statute of frauds",
        "asp_facts": "contract_fact(c16). land_sale_contract(c16). in_writing(c16). lacks_material_terms(c16).",
        "legal_citations": ["Statute of Frauds - Writing Requirements"],
        "difficulty": "hard",
    },
    {
        "id": "sof_017",
        "description": "Partial written confirmation",
        "facts": [
            "Eve and Frank had oral agreement for $5000 in goods",
            "Eve sent written confirmation covering only $3000",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "unenforceable",
        "rationale": "Contract only enforceable to extent admitted in writing",
        "asp_facts": "contract_fact(c17). goods_sale_contract(c17). value(c17, 5000). partial_writing_amount(c17, 3000).",
        "legal_citations": ["UCC 2-201"],
        "difficulty": "hard",
    },
    {
        "id": "sof_018",
        "description": "Promissory estoppel exception",
        "facts": [
            "Grace orally promised land to Henry",
            "Henry relied on promise and built house on the land",
            "Grace knew Henry would rely on the promise",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "enforceable",
        "rationale": "Promissory estoppel can overcome statute of frauds in some jurisdictions",
        "asp_facts": "contract_fact(c18). land_sale_contract(c18). detrimental_reliance(c18, henry). foreseeable_reliance(c18).",
        "legal_citations": ["Promissory Estoppel"],
        "difficulty": "hard",
    },
    {
        "id": "sof_019",
        "description": "Sale of growing crops",
        "facts": [
            "Iris sold unharvested corn to Jack for $600",
            "Oral agreement only",
            "Corn still in ground",
        ],
        "question": "Is the contract enforceable?",
        "ground_truth": "unenforceable",
        "rationale": "Growing crops are considered goods requiring writing if over $500",
        "asp_facts": "contract_fact(c19). goods_sale_contract(c19). growing_crops(c19). value(c19, 600). not in_writing(c19).",
        "legal_citations": ["UCC 2-201", "UCC 2-107"],
        "difficulty": "hard",
    },
    {
        "id": "sof_020",
        "description": "Lease with option to purchase",
        "facts": [
            "Kim leased property from Leo with option to purchase",
            "Lease is written, but option to purchase is only oral",
            "Kim wants to exercise option",
        ],
        "question": "Is the purchase option enforceable?",
        "ground_truth": "unenforceable",
        "rationale": "Option to purchase land must be in writing",
        "asp_facts": "contract_fact(c20). lease_contract(c20). in_writing(c20). option_to_purchase(c20). option_not_in_writing(c20).",
        "legal_citations": ["Statute of Frauds"],
        "difficulty": "hard",
    },
]


def generate_test_files(output_dir: Path):
    """Generate test scenario JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario in SCENARIOS:
        filename = f"{scenario['id']}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(scenario, f, indent=2)

        print(f"Created: {filepath}")

    print(f"\nGenerated {len(SCENARIOS)} test scenarios")


if __name__ == "__main__":
    output_dir = (
        Path(__file__).parent.parent
        / "experiments"
        / "data"
        / "contracts"
        / "statute_of_frauds"
    )
    generate_test_files(output_dir)
