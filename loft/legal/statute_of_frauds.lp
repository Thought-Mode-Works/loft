%%% ============================================
%%% STATUTE OF FRAUDS - ASP IMPLEMENTATION
%%% ============================================
%%%
%%% This ASP program implements the Statute of Frauds rules for contract law.
%%% The Statute of Frauds requires certain contracts to be in writing to be
%%% enforceable.
%%%
%%% References:
%%% - UCC § 2-201 (goods)
%%% - Restatement (Second) of Contracts § 110-150
%%% - ESIGN Act, UETA (electronic signatures)
%%%

%%% --- Type Definitions ---

contract(C) :- contract_fact(C).
party(P) :- party_fact(P).
writing(W) :- writing_fact(W).

%%% --- Contract Categories (within statute) ---

% Land sale contracts must be in writing
within_statute(C) :- land_sale_contract(C).

% Long-term contracts (cannot be performed within 1 year)
within_statute(C) :-
    contract(C),
    duration_months(C, D),
    D > 12.

% Goods sales over $500 (UCC § 2-201)
within_statute(C) :-
    goods_sale_contract(C),
    sale_amount(C, Amount),
    Amount > 500.

% Suretyship (answering for debt of another)
within_statute(C) :- suretyship_contract(C).

% Marriage consideration
within_statute(C) :- marriage_consideration_contract(C).

% Executor/administrator contracts
within_statute(C) :- executor_contract(C).

%%% --- Writing Requirements ---

% A contract has a sufficient writing if:
% 1. There exists a writing referencing the contract
% 2. The writing is signed by the party to be charged
% 3. The writing contains essential terms

has_sufficient_writing(C) :-
    contract(C),
    writing(W),
    references_contract(W, C),
    signed_by(W, P),
    party_to_contract(C, P),
    contains_essential_terms(W).

% Essential terms are present when writing includes:
% - Identity of parties
% - Subject matter
% - Consideration/price
contains_essential_terms(W) :-
    writing(W),
    identifies_parties(W),
    describes_subject_matter(W),
    states_consideration(W).

%%% --- Exceptions to Writing Requirement ---

% Part Performance Exception
% (typically for land sales: substantial actions + detrimental reliance)
exception_applies(C) :-
    contract(C),
    part_performance(C),
    substantial_actions_taken(C),
    detrimental_reliance(C).

% Promissory Estoppel Exception
% (promise + reasonable reliance + injustice without enforcement)
exception_applies(C) :-
    contract(C),
    clear_promise(C),
    reasonable_reliance(C),
    substantial_detriment(C),
    injustice_without_enforcement(C).

% Merchant Confirmation (UCC § 2-201)
% (between merchants, written confirmation, no objection within 10 days)
exception_applies(C) :-
    goods_sale_contract(C),
    both_parties_merchants(C),
    written_confirmation_sent(C),
    not objection_within_10_days(C).

% Admission in Court
exception_applies(C) :-
    contract(C),
    admitted_in_pleadings(C, P),
    party_to_contract(C, P).

% Specially Manufactured Goods (UCC § 2-201(3)(a))
exception_applies(C) :-
    goods_sale_contract(C),
    specially_manufactured(C),
    not_suitable_for_others(C),
    substantial_beginning_of_manufacture(C).

%%% --- Main Enforceability Rules ---

% A contract satisfies statute of frauds if:
% Option 1: It has sufficient writing
satisfies_statute_of_frauds(C) :-
    contract(C),
    within_statute(C),
    has_sufficient_writing(C).

% Option 2: An exception applies
satisfies_statute_of_frauds(C) :-
    contract(C),
    within_statute(C),
    exception_applies(C).

% Option 3: Contract is not within statute (doesn't require writing)
satisfies_statute_of_frauds(C) :-
    contract(C),
    not within_statute(C).

% Enforceability determination
enforceable(C) :-
    contract(C),
    satisfies_statute_of_frauds(C),
    not void_for_other_reasons(C).

unenforceable(C) :-
    contract(C),
    within_statute(C),
    not satisfies_statute_of_frauds(C).

% Default reasoning: enforceable unless proven otherwise
enforceable(C) :-
    contract(C),
    not within_statute(C),
    not void_for_other_reasons(C).

%%% --- Meta-Reasoning Predicates ---

% Identify gaps in knowledge
uncertain_writing_sufficiency(C, W) :-
    contract(C),
    writing(W),
    references_contract(W, C),
    not contains_essential_terms(W),
    not -contains_essential_terms(W).

uncertain_exception(C, Exception) :-
    contract(C),
    potential_exception(C, Exception),
    not exception_applies(C).

% Detect when a contract within statute has no writing information at all
missing_writing_info(C) :-
    contract(C),
    within_statute(C),
    not has_sufficient_writing(C),
    not exception_applies(C),
    not writing_referenced(C).

% Helper: check if any writing references this contract
writing_referenced(C) :-
    writing(W),
    references_contract(W, C).

% Track confidence based on clarity of facts
high_confidence_determination(C) :-
    contract(C),
    all_facts_clear(C).

low_confidence_determination(C) :-
    contract(C),
    ambiguous_facts(C).

%%% --- Constraints ---

% Cannot be both enforceable and unenforceable
:- contract(C), enforceable(C), unenforceable(C).

% If within statute and has writing, writing must be valid
:- contract(C),
   within_statute(C),
   has_sufficient_writing(C),
   writing(W),
   references_contract(W, C),
   -signed_by(W, _).

%%% --- Display Rules ---

#show enforceable/1.
#show unenforceable/1.
#show satisfies_statute_of_frauds/1.
#show within_statute/1.
#show has_sufficient_writing/1.
#show exception_applies/1.
#show uncertain_writing_sufficiency/2.
#show uncertain_exception/2.
#show missing_writing_info/1.
