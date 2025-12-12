"""
Legal domain primitives for contract law.

Provides example ASP rules for common legal concepts,
particularly focused on the statute of frauds.
"""

from .asp_rule import ASPRule, StratificationLevel, RuleMetadata, create_rule_id
from .asp_program import ASPProgram
from datetime import datetime


def create_statute_of_frauds_rules() -> ASPProgram:
    """
    Create ASP rules for the statute of frauds.

    Returns:
        ASPProgram with statute of frauds rules
    """
    program = ASPProgram(
        name="statute_of_frauds",
        description="Rules for determining if a contract satisfies the statute of frauds",
    )

    # Rule 1: Contracts within statute require writing
    rule1 = ASPRule(
        rule_id=create_rule_id(),
        asp_text=(
            "satisfies_statute_of_frauds(C) :- "
            "contract(C), "
            "within_statute(C), "
            "has_signed_writing(C, W), "
            "essential_terms_present(W)."
        ),
        stratification_level=StratificationLevel.STRATEGIC,
        confidence=0.95,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=1.0,
            tags=["statute_of_frauds", "contracts"],
            notes="Main rule for statute of frauds compliance",
        ),
    )
    program.add_rule(rule1)

    # Rule 2: Exception - part performance
    rule2 = ASPRule(
        rule_id=create_rule_id(),
        asp_text=(
            "satisfies_statute_of_frauds(C) :- "
            "contract(C), "
            "within_statute(C), "
            "exception_applies(C)."
        ),
        stratification_level=StratificationLevel.STRATEGIC,
        confidence=0.92,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=1.0,
            tags=["statute_of_frauds", "exceptions"],
            notes="Exception rule for statute of frauds",
        ),
    )
    program.add_rule(rule2)

    # Rule 3: Contracts within statute - land sales
    rule3 = ASPRule(
        rule_id=create_rule_id(),
        asp_text="within_statute(C) :- land_sale(C).",
        stratification_level=StratificationLevel.TACTICAL,
        confidence=0.98,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=1.0,
            tags=["statute_of_frauds", "land"],
        ),
    )
    program.add_rule(rule3)

    # Rule 4: Contracts within statute - long-term contracts
    rule4 = ASPRule(
        rule_id=create_rule_id(),
        asp_text="within_statute(C) :- long_term_contract(C).",
        stratification_level=StratificationLevel.TACTICAL,
        confidence=0.97,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=1.0,
            tags=["statute_of_frauds", "duration"],
        ),
    )
    program.add_rule(rule4)

    # Rule 5: Contracts within statute - goods over $500
    rule5 = ASPRule(
        rule_id=create_rule_id(),
        asp_text="within_statute(C) :- goods_sale(C, Amount), Amount > 500.",
        stratification_level=StratificationLevel.TACTICAL,
        confidence=0.99,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=1.0,
            tags=["statute_of_frauds", "goods", "ucc"],
        ),
    )
    program.add_rule(rule5)

    # Rule 6: Exception - part performance
    rule6 = ASPRule(
        rule_id=create_rule_id(),
        asp_text="exception_applies(C) :- part_performance(C).",
        stratification_level=StratificationLevel.TACTICAL,
        confidence=0.90,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=0.95,
            tags=["statute_of_frauds", "exceptions", "part_performance"],
        ),
    )
    program.add_rule(rule6)

    # Rule 7: Exception - promissory estoppel
    rule7 = ASPRule(
        rule_id=create_rule_id(),
        asp_text="exception_applies(C) :- promissory_estoppel_applies(C).",
        stratification_level=StratificationLevel.TACTICAL,
        confidence=0.88,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=0.92,
            tags=["statute_of_frauds", "exceptions", "estoppel"],
        ),
    )
    program.add_rule(rule7)

    # Rule 8: Has signed writing requirement
    rule8 = ASPRule(
        rule_id=create_rule_id(),
        asp_text=(
            "has_signed_writing(C, W) :- "
            "contract(C), "
            "writing(W), "
            "references(W, C), "
            "signed(W, P), "
            "party_to_contract(C, P)."
        ),
        stratification_level=StratificationLevel.TACTICAL,
        confidence=0.96,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=1.0,
            tags=["statute_of_frauds", "writing"],
        ),
    )
    program.add_rule(rule8)

    # Rule 9: Essential terms present
    rule9 = ASPRule(
        rule_id=create_rule_id(),
        asp_text=(
            "essential_terms_present(W) :- "
            "writing(W), "
            "has_term(W, parties), "
            "has_term(W, subject_matter), "
            "has_term(W, consideration)."
        ),
        stratification_level=StratificationLevel.TACTICAL,
        confidence=0.94,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=0.98,
            tags=["statute_of_frauds", "essential_terms"],
        ),
    )
    program.add_rule(rule9)

    # Rule 10: Default - enforceable if not proven otherwise
    rule10 = ASPRule(
        rule_id=create_rule_id(),
        asp_text="enforceable(C) :- contract(C), not unenforceable(C).",
        stratification_level=StratificationLevel.STRATEGIC,
        confidence=0.93,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=1.0,
            tags=["enforceability", "default_reasoning"],
            notes="Non-monotonic default rule",
        ),
    )
    program.add_rule(rule10)

    # Rule 11: Unenforceable if within statute and doesn't satisfy
    rule11 = ASPRule(
        rule_id=create_rule_id(),
        asp_text=(
            "unenforceable(C) :- within_statute(C), not satisfies_statute_of_frauds(C)."
        ),
        stratification_level=StratificationLevel.STRATEGIC,
        confidence=0.96,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=1.0,
            tags=["enforceability", "statute_of_frauds"],
        ),
    )
    program.add_rule(rule11)

    return program


def create_contract_basics_rules() -> ASPProgram:
    """
    Create basic contract law rules.

    Returns:
        ASPProgram with fundamental contract principles
    """
    program = ASPProgram(
        name="contract_basics",
        description="Fundamental principles of contract law",
    )

    # Valid contract requires mutual assent
    rule1 = ASPRule(
        rule_id=create_rule_id(),
        asp_text=(
            "valid_contract(C) :- "
            "contract(C), "
            "mutual_assent(C), "
            "consideration(C), "
            "legal_capacity(C), "
            "legal_purpose(C)."
        ),
        stratification_level=StratificationLevel.CONSTITUTIONAL,
        confidence=1.0,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=1.0,
            tags=["contracts", "formation"],
            notes="Core elements of valid contract",
        ),
    )
    program.add_rule(rule1)

    # Mutual assent from offer and acceptance
    rule2 = ASPRule(
        rule_id=create_rule_id(),
        asp_text="mutual_assent(C) :- offer(C, O), acceptance(C, A), match(O, A).",
        stratification_level=StratificationLevel.STRATEGIC,
        confidence=0.95,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=1.0,
            tags=["contracts", "mutual_assent"],
        ),
    )
    program.add_rule(rule2)

    # Consideration requires bargained-for exchange
    rule3 = ASPRule(
        rule_id=create_rule_id(),
        asp_text=(
            "consideration(C) :- "
            "contract(C), "
            "promise(C, P1), "
            "promise(C, P2), "
            "P1 != P2, "
            "bargained_for(C)."
        ),
        stratification_level=StratificationLevel.STRATEGIC,
        confidence=0.92,
        metadata=RuleMetadata(
            provenance="human",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=0.95,
            tags=["contracts", "consideration"],
        ),
    )
    program.add_rule(rule3)

    return program


def create_meta_reasoning_rules() -> ASPProgram:
    """
    Create meta-reasoning rules (rules about rules).

    Returns:
        ASPProgram with meta-level reasoning capabilities
    """
    program = ASPProgram(
        name="meta_reasoning",
        description="Rules for reasoning about the rule base itself",
    )

    # Identify gaps in knowledge
    rule1 = ASPRule(
        rule_id=create_rule_id(),
        asp_text="missing_rule(R) :- required_rule(R), not defined_rule(R).",
        stratification_level=StratificationLevel.OPERATIONAL,
        confidence=0.85,
        metadata=RuleMetadata(
            provenance="system",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=0.90,
            tags=["meta", "knowledge_gaps"],
            notes="Identifies missing required rules",
        ),
    )
    program.add_rule(rule1)

    # Track low confidence rules
    rule2 = ASPRule(
        rule_id=create_rule_id(),
        asp_text="low_confidence_rule(R) :- rule(R), confidence(R, C), C < 70.",
        stratification_level=StratificationLevel.OPERATIONAL,
        confidence=0.90,
        metadata=RuleMetadata(
            provenance="system",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=0.95,
            tags=["meta", "confidence"],
            notes="Flags rules with low confidence",
        ),
    )
    program.add_rule(rule2)

    # Identify conflicting rules
    rule3 = ASPRule(
        rule_id=create_rule_id(),
        asp_text=(
            "potential_conflict(R1, R2) :- "
            "rule(R1), "
            "rule(R2), "
            "R1 != R2, "
            "concludes(R1, C), "
            "concludes(R2, -C)."
        ),
        stratification_level=StratificationLevel.OPERATIONAL,
        confidence=0.80,
        metadata=RuleMetadata(
            provenance="system",
            timestamp=datetime.utcnow().isoformat(),
            validation_score=0.85,
            tags=["meta", "conflicts"],
            notes="Detects potentially conflicting rules",
        ),
    )
    program.add_rule(rule3)

    return program
