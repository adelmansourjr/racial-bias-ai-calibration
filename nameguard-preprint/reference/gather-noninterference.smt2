; A symbolic one-identity-field activation noninterference lemma.
; Arbitrary nonnegative identity lengths, arbitrary name/evidence token arrays,
; arbitrary pointwise embedding. No concrete name values or embedding weights.
; Extending the lemma to finitely many identity fields is by induction.
(set-option :produce-proofs true)
(set-logic AUFLIA)
(declare-const evidence (Array Int Int))
(declare-const name1 (Array Int Int))
(declare-const name2 (Array Int Int))
(declare-fun embedding (Int) Int)
(declare-const public_length Int)
(declare-const insertion_position Int)
(declare-const name1_length Int)
(declare-const name2_length Int)
(declare-const public_index Int)
(assert (>= public_length 1))
(assert (and (>= insertion_position 0) (<= insertion_position public_length)))
(assert (>= name1_length 0))
(assert (>= name2_length 0))
(assert (and (>= public_index 0) (< public_index public_length)))
(define-fun raw1 ((j Int)) Int
 (embedding (ite (< j insertion_position)
   (select evidence j)
   (ite (< j (+ insertion_position name1_length))
     (select name1 (- j insertion_position))
     (select evidence (- j name1_length))))))
(define-fun raw2 ((j Int)) Int
 (embedding (ite (< j insertion_position)
   (select evidence j)
   (ite (< j (+ insertion_position name2_length))
     (select name2 (- j insertion_position))
     (select evidence (- j name2_length))))))
(define-fun index1 () Int
 (ite (< public_index insertion_position) public_index (+ public_index name1_length)))
(define-fun index2 () Int
 (ite (< public_index insertion_position) public_index (+ public_index name2_length)))
; Look for ANY public coordinate differing across names, or differing from
; direct evidence embedding. UNSAT proves that no such coordinate exists.
(assert (or (not (= (raw1 index1) (raw2 index2)))
            (not (= (raw1 index1) (embedding (select evidence public_index))))))
(check-sat)
(get-proof)
