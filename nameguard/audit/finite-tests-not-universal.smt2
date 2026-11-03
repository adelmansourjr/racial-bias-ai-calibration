; A counterexample to inferring universal invariance from finite name tests.
(set-logic QF_UFLIA)
(declare-fun score (Int Int) Int)
; The first argument is fixed evidence, the second a symbolic name identity.
(assert (= (score 7 0) (score 7 1)))
(assert (= (score 7 1) (score 7 2)))
; A model can satisfy every observed equality yet differ on another name.
(assert (not (= (score 7 3) (score 7 0))))
(check-sat)
(get-model)
