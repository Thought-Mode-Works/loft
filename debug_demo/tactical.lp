% Added: 2025-11-24T08:17:10.293103, Cycle: 1, Confidence: 0.85
enforceable(C) :- contract(C), written(C), signed(C).

% Added: 2025-11-24T08:17:10.296459, Cycle: 1, Confidence: 0.88
valid_consideration(C) :- contract(C), payment(C,P), P > 0.

