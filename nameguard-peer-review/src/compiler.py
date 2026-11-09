"""Exhaustive, score-blind evidence-contract compilation under reviewed whitespace policy."""
from dataclasses import dataclass
from collections import defaultdict
import bootstrap
from gate import native_layout, ContractError, ConsensusDocument
from nameguard.annotations import proposal_from_spans
from nameguard.contracts import Node


@dataclass(frozen=True)
class Candidate:
    evidence: tuple[int,...]
    start: int
    end: int
    formatting_cost: int


@dataclass(frozen=True)
class CompiledContract:
    evidence: tuple[int,...]
    panel_size: int
    supported: int
    total_formatting_cost: int


def enumerate_candidates(source,start,end,tokenizer,*,budget=16):
    """Propose all native-compatible expansions; never grant semantic approval.

    Only adjacent whitespace may be reclassified, at most budget per side.
    The original source and tokenizer IDs are never rewritten. Empty support
    means infeasible under this policy, not a fallback to deleting evidence.
    """
    if type(source) is not str or type(start) is not int or type(end) is not int:
        raise ContractError('Exact source and character offsets required')
    if not 0<=start<end<=len(source) or type(budget) is not int or not 0<=budget<=16:
        raise ContractError('Invalid span or bounded formatting policy')
    left=start
    while left>0 and start-left<budget and source[left-1].isspace(): left-=1
    right=end
    while right<len(source) and right-end<budget and source[right].isspace(): right+=1
    output=[]
    for a in range(left,start+1):
        for b in range(end,right+1):
            nodes=(Node('evidence',source[:a]),Node('identity',source[a:b]),Node('evidence',source[b:]))
            document=ConsensusDocument(nodes,('proposal_a','proposal_b'),'structural-preflight',('fixture',))
            try:
                layout=native_layout(document,tokenizer)
            except ContractError:
                continue
            output.append(Candidate(layout.public_ids,a,b,start-a+b-end))
    return tuple(sorted(output,key=lambda c:(c.formatting_cost,c.start,c.end)))


def select_contract(candidate_sets):
    """Maximum panel support; minimum total formatting; lexicographic IDs.

    Every panel member has weight one, including empty sets. Duplicate candidates
    from one member count once. No model outputs or suitability labels are used.
    """
    panel=tuple(candidate_sets)
    if not panel:
        raise ContractError('Nonempty declared panel required')
    support=defaultdict(list)
    for candidates in panel:
        minimum={}
        for c in candidates:
            minimum[c.evidence]=min(minimum.get(c.evidence,c.formatting_cost),c.formatting_cost)
        for evidence,cost in minimum.items(): support[evidence].append(cost)
    if not support:
        raise ContractError('No admissible evidence contract exists')
    evidence=min(support,key=lambda e:(-len(support[e]),sum(support[e]),e))
    return CompiledContract(evidence,len(panel),len(support[evidence]),sum(support[evidence]))


def match_contract(candidates,contract):
    """Return least-cost unresolved witness; an external reviewer must approve."""
    matches=[c for c in candidates if c.evidence==contract.evidence]
    if not matches: raise ContractError('No witness for the fixed evidence contract')
    return min(matches,key=lambda c:(c.formatting_cost,c.start,c.end))


def unresolved_payload(source,candidate,registry):
    return proposal_from_spans(source,[{'start':candidate.start,'end':candidate.end,
                                       'text':source[candidate.start:candidate.end]}],registry)
