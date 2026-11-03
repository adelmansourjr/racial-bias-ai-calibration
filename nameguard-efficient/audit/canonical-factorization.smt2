; Conditional canonicalization theorem. Q1=Q2 is the role/evidence premise.
; Tokenization, embedding, transformer and readout are arbitrary fixed functions.
; This checks congruence, NOT the correctness of role extraction or MLX code.
(set-logic QF_UF)
(declare-sort Evidence 0)
(declare-sort Tokens 0)
(declare-sort State 0)
(declare-sort Score 0)
(declare-fun tokenize (Evidence) Tokens)
(declare-fun embed (Tokens) State)
(declare-fun transform (State) State)
(declare-fun readout (State) Score)
(declare-const Q1 Evidence)
(declare-const Q2 Evidence)
(assert (= Q1 Q2))
(assert (not (= (readout (transform (embed (tokenize Q1))))
                (readout (transform (embed (tokenize Q2)))))))
(check-sat)
