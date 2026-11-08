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
    token_ids: tuple[int, ...]

    def __post_init__(self):
        if type(self.token_ids) is not tuple:
            raise ContractError('Immutable approved evidence-token tuple required')
        evidence_digest(self.token_ids)

    @property
    def digest(self):
        """A reporting identifier; scoring admission compares the actual IDs."""
        return evidence_digest(self.token_ids)

    @classmethod
    def from_reviewed(cls, wire, authority, tokenizer):
        layout = native_layout(authority.verify(wire),tokenizer)
        return cls(layout.public_ids)

    def verify(self, layout):
        if layout.public_ids != self.token_ids:
            raise ContractError('Retained evidence differs from the approved equivalence contract')


class BoundEvidenceScorer:
    """Operator-supplied CV contract. Requests cannot replace its approved IDs."""

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
