"""
LLM prompts for legal question parsing.

Provides prompts for converting natural language legal questions
into structured ASP queries.

Issue #272: Legal Question Answering Interface
"""

QUESTION_PARSING_PROMPT = """You are a legal reasoning expert that converts natural language legal questions into Answer Set Programming (ASP) queries.

Your task: Convert the given legal question into:
1. A list of ASP facts representing the given information
2. An ASP query goal to answer the question
3. The legal domain (contracts, torts, property, etc.)

## ASP Syntax Rules

- Facts end with a period: `offer(contract1).`
- Negation uses `not`: `not consideration(contract1).`
- Variables use uppercase: `X`, `Contract`
- Constants use lowercase: `contract1`, `party_a`
- Queries start with `?-`: `?- valid_contract(contract1).`

## Examples

Example 1:
Question: "Is a contract valid if there is offer and acceptance but no consideration?"
Domain: contracts
Facts:
- offer(contract1).
- acceptance(contract1).
- not consideration(contract1).
Query: valid_contract(contract1)

Example 2:
Question: "Can a plaintiff recover damages for negligence if they contributed to their own injury?"
Domain: torts
Facts:
- negligence(defendant).
- contributory_negligence(plaintiff).
Query: can_recover_damages(plaintiff)

Example 3:
Question: "Is a minor's contract voidable?"
Domain: contracts
Facts:
- party(contract1, minor1).
- minor(minor1).
Query: voidable(contract1)

Example 4:
Question: "What are the elements required to establish adverse possession?"
Domain: property
Facts:
- claim(adverse_possession).
Query: elements(adverse_possession, X)

Example 5:
Question: "If someone enters land without permission and stays for 15 years, can they claim ownership?"
Domain: property
Facts:
- enters_land(person1, land1).
- not permission(person1, land1).
- occupation_years(person1, land1, 15).
- occupation_continuous(person1, land1, yes).
Query: can_claim_ownership(person1, land1)

## Guidelines

1. **Extract all given information as facts**: Every piece of information in the question should become a fact.
2. **Use specific entities**: Create specific entities (contract1, person1, land1) rather than variables in facts.
3. **Identify what's being asked**: The query should directly answer the question.
4. **Infer the legal domain**: Common domains are contracts, torts, property, criminal, constitutional.
5. **Use negation correctly**: "without X" or "no X" becomes `not X(...)`.
6. **Keep predicates simple**: Use clear predicate names like `offer`, `acceptance`, `consideration`.

Now convert this question:

Question: "{question}"

Respond in this exact format:

Domain: <legal_domain>
Facts:
- <fact1>.
- <fact2>.
- ...
Query: <query_goal>
Confidence: <0.0-1.0>

Confidence should reflect how well you could parse the question (1.0 = very clear, 0.5 = ambiguous)."""


QUESTION_PARSING_SYSTEM_PROMPT = """You are an expert in formal legal reasoning and Answer Set Programming (ASP).
Your role is to convert natural language legal questions into precise ASP queries that can be answered
by an automated reasoning system. Focus on accuracy and precision in translation."""


def build_parsing_prompt(question: str) -> str:
    """
    Build the complete parsing prompt for a given question.

    Args:
        question: Natural language legal question

    Returns:
        Formatted prompt string
    """
    return QUESTION_PARSING_PROMPT.format(question=question)


# Domain-specific parsing hints
DOMAIN_HINTS = {
    "contracts": {
        "common_predicates": [
            "offer",
            "acceptance",
            "consideration",
            "valid_contract",
            "enforceable",
            "voidable",
            "breach",
            "damages",
            "minor",
            "capacity",
            "mutual_assent",
        ],
        "keywords": [
            "contract",
            "agreement",
            "offer",
            "acceptance",
            "consideration",
            "breach",
            "enforceable",
        ],
    },
    "torts": {
        "common_predicates": [
            "duty",
            "breach",
            "causation",
            "damages",
            "negligence",
            "intentional",
            "strict_liability",
            "contributory_negligence",
            "comparative_negligence",
            "proximate_cause",
        ],
        "keywords": [
            "negligence",
            "tort",
            "duty",
            "damages",
            "injury",
            "liability",
            "fault",
        ],
    },
    "property": {
        "common_predicates": [
            "ownership",
            "possession",
            "adverse_possession",
            "easement",
            "title",
            "transfer",
            "landlocked",
            "occupation_continuous",
            "occupation_years",
        ],
        "keywords": [
            "property",
            "ownership",
            "possession",
            "land",
            "title",
            "deed",
            "adverse possession",
        ],
    },
    "criminal": {
        "common_predicates": [
            "actus_reus",
            "mens_rea",
            "guilty",
            "intent",
            "knowledge",
            "recklessness",
            "negligence",
            "defense",
        ],
        "keywords": [
            "crime",
            "criminal",
            "guilty",
            "intent",
            "actus reus",
            "mens rea",
        ],
    },
}


def get_domain_hints(domain: str) -> dict:
    """
    Get parsing hints for a specific legal domain.

    Args:
        domain: Legal domain (contracts, torts, etc.)

    Returns:
        Dictionary with common predicates and keywords
    """
    return DOMAIN_HINTS.get(domain.lower(), {})
