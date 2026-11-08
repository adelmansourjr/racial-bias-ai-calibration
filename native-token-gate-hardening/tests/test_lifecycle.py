import sys
import threading
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gate
from gate import SlotScorer, ContractError, RuntimeRejected
from nameguard.runtime import PublicScore
from nameguard.annotations import ConsensusDocument
from nameguard.contracts import Node


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class Authority:
    def verify(self, wire):
        if wire != b'valid':
            raise ContractError('Invalid test packet')
        return ConsensusDocument((Node('evidence','Evidence'),),('a','b'),'digest',('Name',))


class Instrumented(SlotScorer):
    def __init__(self):
        self.authority = Authority()
        self._tokenizer = Tokenizer()
        self._request_lock = threading.Lock()
        self._closed = self._poisoned = False
        self.calls = 0
        self.fail = False

    def _infer_layout(self, layout, mode, observe):
        self.calls += 1
        if self.fail:
            raise RuntimeError('PRIVATE applicant identity detail')
        return PublicScore(1., .7310585786300049)


class Lifecycle(unittest.TestCase):
    def test_bad_request_is_recoverable_and_never_enters_backend(self):
        scorer = Instrumented()
        with self.assertRaises(ContractError):
            scorer.score(b'bad')
        self.assertEqual(scorer.calls,0)
        self.assertEqual(scorer.score(b'valid').margin,1.)

    def test_runtime_failure_sanitizes_and_poisoned_service_cannot_retry(self):
        scorer = Instrumented()
        scorer.fail = True
        with self.assertRaises(RuntimeRejected) as error:
            scorer.score(b'valid')
        self.assertNotIn('PRIVATE',str(error.exception))
        scorer.fail = False
        with self.assertRaisesRegex(RuntimeRejected,'unavailable'):
            scorer.score(b'valid')
        self.assertEqual(scorer.calls,1)
        fresh = Instrumented()
        self.assertEqual(fresh.score(b'valid').margin,1.)

    def test_global_busy_is_recoverable(self):
        scorer = Instrumented()
        self.assertTrue(gate._MODEL_GATE.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(RuntimeRejected,'busy'):
                scorer.score(b'valid')
        finally:
            gate._MODEL_GATE.release()
        self.assertEqual(scorer.score(b'valid').margin,1.)

    def test_close_is_idempotent_and_blocks_future_scoring(self):
        scorer = Instrumented()
        scorer.close()
        scorer.close()
        with self.assertRaisesRegex(RuntimeRejected,'unavailable'):
            scorer.score(b'valid')
        self.assertEqual(scorer.calls,0)

    def test_overlapping_request_and_close_are_rejected(self):
        scorer = Instrumented()
        scorer._request_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeRejected,'busy'):
                scorer.score(b'valid')
            with self.assertRaisesRegex(RuntimeRejected,'busy'):
                scorer.close()
        finally:
            scorer._request_lock.release()
        self.assertEqual(scorer.score(b'valid').margin,1.)

    def test_public_score_has_no_trace_switch(self):
        with self.assertRaises(TypeError):
            Instrumented().score(b'valid',observe=lambda *_: None)
