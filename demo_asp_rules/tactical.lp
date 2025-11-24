% Added: 2025-11-24T08:23:42.200219, Cycle: 1, Confidence: 0.8
enforceable(C) :- contract(C), has_sufficient_writing(C), within_statute(C), not unenforceable(C).

