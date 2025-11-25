% Added: 2025-11-24T08:24:08.114398, Cycle: 1, Confidence: 0.85
enforceable(C) :- contract(C), written(C), signed(C).

% Added: 2025-11-24T08:24:08.115127, Cycle: 1, Confidence: 0.88
valid_consideration(C) :- contract(C), payment(C,P), P > 0.

% Added: 2025-11-24T08:24:08.115746, Cycle: 1, Confidence: 0.86
valid_acceptance(C) :- offer(C), acceptance(C), same_terms(C).

% Added: 2025-11-24T08:24:08.116327, Cycle: 1, Confidence: 0.9
has_capacity(P) :- party(P), adult(P), sound_mind(P).

% Added: 2025-11-24T08:24:08.116905, Cycle: 1, Confidence: 0.87
mutual_assent(C) :- contract(C), meeting_of_minds(C), no_fraud(C).

