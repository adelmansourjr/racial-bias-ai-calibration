"""Source-exact boundary proposals for an explicitly approved whitespace policy."""
from dataclasses import dataclass

from gate import native_layout, ContractError, ConsensusDocument
from nameguard.annotations import proposal_from_spans
from nameguard.contracts import Node


@dataclass(frozen=True)
class BoundaryProposal:
    payload: dict
    formatting_expansions: tuple[tuple[int, int], ...]


def propose_whitespace_repair(source, spans, registry, tokenizer, *, allow_formatting=False):
    """Return an unresolved proposal. Explicitly reclassify only adjacent whitespace.

    A caller must review the expanded spans and resolve the payload before signing.
    The original source string and its full tokenizer output are preserved.
    """
    original = proposal_from_spans(source, spans, registry)
    intervals = [(span['start'], span['end']) for span in spans]
    if len(intervals) != 1:
        raise ContractError('This repair policy supports one identity interval')
    start, end = intervals[0]
    candidates = [(start, end)]
    if allow_formatting:
        left, right = start, end
        while left > 0 and source[left - 1].isspace():
            left -= 1
        while right < len(source) and source[right].isspace():
            right += 1
        if start-left > 16 or right-end > 16:
            raise ContractError('Adjacent formatting exceeds the review policy limit')
        candidates = sorted(
            ((a, b) for a in range(left, start + 1) for b in range(end, right + 1)),
            key=lambda pair: ((start-pair[0])+(pair[1]-end), start-pair[0], pair[1]-end),
        )
    for left, right in candidates:
        payload = proposal_from_spans(
            source, [{'start': left, 'end': right, 'text': source[left:right]}], registry)
        nodes = tuple(Node(node['role'], node['text']) for node in payload['nodes'])
        # Structural preflight only. It grants no semantic approval or credential.
        document = ConsensusDocument(nodes, ('proposal', 'proposal'), payload['source_sha256'], tuple(registry))
        try:
            native_layout(document, tokenizer)
        except ContractError:
            continue
        return BoundaryProposal(payload, ((left, right),) if (left, right) != (start, end) else ())
    raise ContractError('No token-compatible role boundary under the approved formatting policy')
