import threading
import unittest
from nameguard.contracts import ReviewAuthority, sign_review, ContractError
from nameguard.encoding import ScalarEncoder
from nameguard.runtime import ScoringService, PublicScore, RuntimeRejected, rank_scores
from test_contracts import KEY, NODES, FakeTokenizer


class Backend:
    def __init__(self): self.calls = 0; self.fail = False
    def infer(self, encoded):
        self.calls += 1
        if self.fail: raise RuntimeError('PRIVATE name and GPU details')
        return PublicScore(1., .7310585786300049)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.service = ScoringService(ReviewAuthority({'reviewer':KEY}), ScalarEncoder(FakeTokenizer()), self.backend)
        self.wire = sign_review(NODES, reviewer='reviewer', key=KEY)
    def test_public_result_has_no_identity_audit(self):
        result = self.service.score(self.wire)
        self.assertEqual(set(result.as_dict()), {'margin','conditional_A_probability'})
        with self.assertRaises(AttributeError): result.raw_embedding_hash
    def test_audit_flag_is_not_an_api_option(self):
        with self.assertRaises(TypeError): self.service.score(self.wire, audit=True)
    def test_invalid_review_never_reaches_model(self):
        with self.assertRaises(ContractError): self.service.score(b'{}')
        self.assertEqual(self.backend.calls,0)
        self.service.score(self.wire)
        self.assertEqual(self.backend.calls,1)
    def test_backend_failure_poisoned_and_message_sanitized(self):
        self.backend.fail = True
        with self.assertRaisesRegex(RuntimeRejected, 'Runtime computation failed') as error: self.service.score(self.wire)
        self.assertNotIn('PRIVATE',str(error.exception))
        self.backend.fail = False
        with self.assertRaisesRegex(RuntimeRejected,'unavailable'): self.service.score(self.wire)
        self.assertEqual(self.backend.calls,1)
    def test_closed_engine_rejected(self):
        self.service.close()
        with self.assertRaises(RuntimeRejected): self.service.score(self.wire)
    def test_nonfinite_scores_rejected(self):
        for margin,p in [(float('nan'),.5),(0.,float('inf')),(1.,2.)]:
            with self.assertRaises(RuntimeRejected): PublicScore(margin,p)
    def test_concurrent_request_rejected(self):
        entered = threading.Event(); release = threading.Event()
        class BlockingBackend(Backend):
            def infer(inner, encoded):
                entered.set(); release.wait(timeout=5)
                return super().infer(encoded)
        self.service._backend = BlockingBackend(); errors = []
        def run():
            try: self.service.score(self.wire)
            except BaseException as error: errors.append(str(error))
        thread=threading.Thread(target=run);thread.start()
        try:
            self.assertTrue(entered.wait(timeout=5))
            with self.assertRaisesRegex(RuntimeRejected,'busy'): self.service.score(self.wire)
            with self.assertRaises(RuntimeRejected): self.service.close()
        finally: release.set();thread.join(timeout=5)
        self.assertEqual(errors,[])
    def test_tie_break_uses_opaque_slots(self):
        p=PublicScore(1.,.7)
        self.assertEqual(rank_scores([(8,p),(3,p)]),[3,8])
    def test_duplicate_slots_rejected(self):
        with self.assertRaises(ValueError):rank_scores([(1,PublicScore(0.,.5)),(1,PublicScore(0.,.5))])
