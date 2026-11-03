"""Segmentation-invariant scalarwise encoding of ALL original Unicode text.

Each original Unicode scalar is separately encoded by the original tokenizer.
This is a different input convention from native whole-string BPE. It costs
more tokens and need not preserve native grading accuracy.
"""
from dataclasses import dataclass
from functools import lru_cache
from .contracts import ContractError, ReviewedDocument

MAX_PUBLIC_TOKENS = 2048
MAX_TOTAL_TOKENS = 16384


@dataclass(frozen=True)
class EncodedDocument:
    all_ids: tuple
    keep: tuple


class ScalarEncoder:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._encode_scalar = lru_cache(maxsize=4096)(self._scalar)
    def _scalar(self, scalar):
        ids = self._tokenizer.encode(scalar, add_special_tokens=False)
        if not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ContractError('Tokenizer must encode each scalar without dropping it')
        return tuple(ids)
    def encode(self, document):
        if not isinstance(document, ReviewedDocument): raise ContractError('Verified immutable document required')
        all_ids = []; keep = []
        for node in document.nodes:
            # No identity string is normalized, deleted, redacted or replaced.
            for scalar in node.text:
                ids = self._encode_scalar(scalar)
                if node.role == 'evidence': keep.extend(range(len(all_ids), len(all_ids) + len(ids)))
                all_ids.extend(ids)
                if len(keep) > MAX_PUBLIC_TOKENS or len(all_ids) > MAX_TOTAL_TOKENS:
                    raise ContractError('Supported token limit exceeded; no truncation')
        if not keep: raise ContractError('No retained tokens')
        return EncodedDocument(tuple(all_ids), tuple(keep))
