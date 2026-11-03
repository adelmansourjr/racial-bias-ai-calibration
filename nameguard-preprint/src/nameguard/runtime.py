"""Serialized immutable-request gate and narrowly scoped public scoring API."""
from dataclasses import dataclass
import math
import threading


class RuntimeRejected(RuntimeError): pass


@dataclass(frozen=True)
class PublicScore:
    margin: float
    conditional_A_probability: float
    def __post_init__(self):
        if not math.isfinite(self.margin) or not math.isfinite(self.conditional_A_probability) or not 0 <= self.conditional_A_probability <= 1:
            raise RuntimeRejected('Invalid model score')
    def as_dict(self):
        return {'margin': self.margin, 'conditional_A_probability': self.conditional_A_probability}


class ScoringService:
    """For untrusted request data, not malicious code in the hosting process.

    authority and backend are operator configuration. Construct them through
    the trusted loader. Python is not a process-isolation security boundary.
    """
    def __init__(self, authority, encoder, backend):
        self._authority = authority; self._encoder = encoder; self._backend = backend
        self._gate = threading.Lock(); self._closed = False; self._poisoned = False
    def score(self, wire):
        if not self._gate.acquire(blocking=False): raise RuntimeRejected('Runtime busy')
        try:
            if self._closed or self._poisoned: raise RuntimeRejected('Runtime unavailable')
            document = self._authority.verify(wire)
            encoded = self._encoder.encode(document)
            # Contract failures above do not poison the engine. No model work
            # has happened yet. A backend failure requires rebuilding it.
            try:
                result = self._backend.infer(encoded)
                if type(result) is not PublicScore: raise RuntimeRejected('Unexpected public-result type')
                return result
            except BaseException:
                self._poisoned = True
                raise RuntimeRejected('Runtime computation failed; rebuild required') from None
        finally:
            self._gate.release()
    def close(self):
        if not self._gate.acquire(blocking=False): raise RuntimeRejected('Runtime busy')
        try: self._closed = True
        finally: self._gate.release()


def rank_scores(slots_and_scores):
    """Tie order uses operator-assigned stable integer slots, never names.

    Independence of the slot assignment itself remains an input premise.
    """
    rows = list(slots_and_scores); slots = [r[0] for r in rows]
    if any(type(s) is not int or s < 0 for s in slots) or len(set(slots)) != len(slots):
        raise ValueError('Distinct nonnegative name-independent integer slots required')
    if any(type(score) is not PublicScore for _, score in rows): raise ValueError('Public scores required')
    return [slot for slot, score in sorted(rows, key=lambda row: (-row[1].margin, row[0]))]
