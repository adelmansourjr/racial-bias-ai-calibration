"""Prepare only the complete original source; no evidence tokenization here."""
from dataclasses import dataclass
from .annotations import ConsensusDocument
from .contracts import ContractError
from .encoding import MAX_TOTAL_TOKENS


@dataclass(frozen=True)
class PreparedSource:
    all_ids: tuple
    document: ConsensusDocument


class NativeSourceEncoder:
    def __init__(self, tokenizer): self._tokenizer = tokenizer
    def encode(self, document):
        if type(document) is not ConsensusDocument: raise ContractError('Verified consensus document required')
        source = ''.join(node.text for node in document.nodes)
        ids = self._tokenizer.encode(source, add_special_tokens=False)
        if not 0 < len(ids) <= MAX_TOTAL_TOKENS or any(type(i) is not int or i < 0 for i in ids):
            raise ContractError('Unsupported complete-source token sequence; no truncation')
        return PreparedSource(tuple(ids), document)
