"""Bind interchangeable reviewed variants to one approved evidence-token sequence."""
from dataclasses import dataclass
import hashlib
import json

from gate import native_layout, ContractError


def evidence_digest(token_ids):
    values = tuple(token_ids)
    if not values or any(type(token) is not int or token < 0 for token in values):
        raise ContractError('Nonempty native evidence-token sequence required')
    return hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class EvidenceContract:
    digest: str

    def __post_init__(self):
        if type(self.digest) is not str or len(self.digest)!=64 or any(c not in '0123456789abcdef' for c in self.digest):
            raise ContractError('Expected approved SHA-256 evidence digest')

    @classmethod
    def from_reviewed(cls, wire, authority, tokenizer):
        layout = native_layout(authority.verify(wire),tokenizer)
        return cls(evidence_digest(layout.public_ids))

    def verify(self, layout):
        if evidence_digest(layout.public_ids)!=self.digest:
            raise ContractError('Retained evidence differs from the approved equivalence contract')


class BoundEvidenceScorer:
    """Operator-supplied CV contract. Incoming requests cannot replace its digest."""

    def __init__(self, scorer, contract):
        if type(contract) is not EvidenceContract:
            raise ContractError('An approved evidence contract is required')
        self._scorer = scorer
        self._contract = contract

    def score(self, wire):
        document = self._scorer.authority.verify(wire)
        layout = native_layout(document,self._scorer._tokenizer)
        self._contract.verify(layout)
        return self._scorer.score(wire)

    def close(self):
        self._scorer.close()
