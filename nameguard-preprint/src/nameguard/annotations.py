"""Source-bound two-credential review plus explicit registered-name coverage.

Mechanical completeness over registered textual matches is not universal
semantic completeness. Two correct-looking signatures can approve a bad review.
"""
from dataclasses import dataclass
import hashlib
import hmac
import re
from types import MappingProxyType
import unicodedata
from .contracts import ContractError, Node, canonical, strict_loads, validate_nodes, source_hash, MAX_WIRE_BYTES

POLICY = 'nameguard-canonical-bpe-v2'
RETAINED_ROLES = {'skill', 'third_party', 'organization', 'job_fact'}


@dataclass(frozen=True)
class ConsensusDocument:
    nodes: tuple
    reviewers: tuple
    source_sha256: str
    registry: tuple


def normalized(text):
    chars = []; offsets = []
    for index, char in enumerate(text):
        if unicodedata.category(char) == 'Cf': continue
        for part in unicodedata.normalize('NFKD', char.casefold()):
            if unicodedata.category(part).startswith('M'): continue
            if part.isalnum(): chars.append(part); offsets.append(index)
    return ''.join(chars), offsets


def occurrences(source, value):
    haystack, positions = normalized(source); needle, _ = normalized(value)
    if not needle: raise ContractError('Registry value must contain supported name characters')
    def significant(i, direction):
        while 0 <= i < len(source):
            if unicodedata.category(source[i]) not in {'Cf', 'Mn', 'Mc', 'Me'}: return source[i]
            i += direction
        return ''
    output = []; cursor = 0
    while True:
        found = haystack.find(needle, cursor)
        if found < 0: break
        start, end = positions[found], positions[found + len(needle) - 1] + 1
        if not significant(start-1, -1).isalnum() and not significant(end, 1).isalnum(): output.append((start, end))
        cursor = found + 1
    return sorted(set(output))


def validate_registry(registry):
    if type(registry) is not list or not 1 <= len(registry) <= 128: raise ContractError('Explicit bounded identity registry required')
    if any(type(value) is not str or not 1 <= len(value) <= 16384 for value in registry): raise ContractError('Invalid registry value')
    try:
        if sum(len(value.encode('utf-8',errors='strict'))for value in registry)>32768:raise ContractError('Registry byte limit exceeded')
    except UnicodeError as error:raise ContractError('Invalid registry Unicode')from error
    for value in registry: occurrences('', value)
    return tuple(registry)


def validate_payload(payload):
    if type(payload) is not dict or set(payload) != {'policy', 'nodes', 'source_sha256', 'registry', 'retained', 'resolved'}:
        raise ContractError('Exact consensus payload required')
    if payload['policy'] != POLICY or payload['resolved'] is not True: raise ContractError('Unresolved or wrong-policy review')
    nodes = validate_nodes(payload['nodes']); source = ''.join(n.text for n in nodes)
    if payload['source_sha256'] != source_hash(nodes): raise ContractError('Source binding mismatch')
    registry = validate_registry(payload['registry'])
    identities = []; offset = 0
    for node in nodes:
        if node.role == 'identity' and node.text: identities.append((offset, offset + len(node.text)))
        offset += len(node.text)
    kept = payload['retained']
    if type(kept) is not list or len(kept) > 128: raise ContractError('Bounded retained-role list required')
    for entry in kept:
        if type(entry) is not dict or set(entry) != {'start', 'end', 'text', 'role', 'reason'}: raise ContractError('Exact source-bound retained role required')
        a, b = entry['start'], entry['end']
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(source) or source[a:b] != entry['text']:
            raise ContractError('Stale retained-role offsets/text')
        if type(entry['role'])is not str or entry['role'] not in RETAINED_ROLES or type(entry['reason']) is not str or not 1 <= len(entry['reason'].strip()) <= 1024:
            raise ContractError('Explicit retained meaning and reason required')
        if any(a < end and b > start for start, end in identities): raise ContractError('Retained role overlaps identity')
    for value in registry:
        for start, end in occurrences(source, value):
            covered = any(a <= start and end <= b for a, b in identities)
            retained = any(e['start'] <= start and end <= e['end'] for e in kept)
            if not covered and not retained: raise ContractError('Uncovered registered identity reference; adjudication required')
    return nodes, registry


def proposal_from_spans(source, identities, registry, retained=(), *, resolved=False):
    """Build a source-exact annotation; unresolved is the default.

    identities consists of sorted exact (start,end,text) records. No input
    characters can be deleted or rewritten by this helper.
    """
    if type(source) is not str: raise ContractError('Original source string required')
    if type(identities) is not list: raise ContractError('Identity span list required')
    nodes = []; cursor = 0
    for span in identities:
        if type(span) is not dict or set(span) != {'start', 'end', 'text'}: raise ContractError('Exact identity span required')
        a, b = span['start'], span['end']
        if type(a) is not int or type(b) is not int or not cursor <= a < b <= len(source) or source[a:b] != span['text']:
            raise ContractError('Unsorted, overlapping or stale identity span')
        nodes.append({'role': 'evidence', 'text': source[cursor:a]}); nodes.append({'role': 'identity', 'text': source[a:b]}); cursor = b
    nodes.append({'role': 'evidence', 'text': source[cursor:]})
    validate_nodes(nodes)
    payload = {'policy': POLICY, 'nodes': nodes, 'source_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
               'registry': list(registry), 'retained': list(retained), 'resolved': resolved}
    if resolved: validate_payload(payload)
    else:
        validate_nodes(nodes); validate_registry(list(registry))
    return payload


def propose_candidates(source, registry):
    """Return unresolved lexical candidates; never auto-assign semantic roles."""
    validate_registry(registry)
    return [{'start': a, 'end': b, 'text': source[a:b], 'registry_value': value, 'status': 'needs_semantic_review'}
            for value in registry for a, b in occurrences(source, value)]


def endorse(payload, *, reviewer, key):
    validate_payload(payload)
    if type(key) is not bytes or len(key) < 32: raise ContractError('Strong configured reviewer key required')
    if type(reviewer) is not str or not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', reviewer): raise ContractError('Invalid reviewer ID')
    message = canonical({'reviewer': reviewer, 'payload': payload})
    return {'reviewer': reviewer, 'signature': hmac.digest(key, message, 'sha256').hex()}


def packet(payload, endorsements):
    # The authority validates signatures and payload equality; this serializer
    # merely snapshots caller objects into immutable request bytes.
    wire = canonical({'payload': payload, 'endorsements': endorsements})
    if len(wire) > MAX_WIRE_BYTES: raise ContractError('Request too large')
    return wire


class DualReviewAuthority:
    def __init__(self, reviewer_keys):
        if type(reviewer_keys) is not dict or len(reviewer_keys) < 2: raise ContractError('Two configured reviewer credentials required')
        if any(type(n) is not str or not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', n) or type(k) is not bytes or len(k) < 32 for n, k in reviewer_keys.items()):
            raise ContractError('Invalid reviewer key configuration')
        if len(set(reviewer_keys.values())) != len(reviewer_keys): raise ContractError('Reviewers must not share the same key')
        self._keys = MappingProxyType(dict(reviewer_keys))
    def verify(self, wire):
        request = strict_loads(wire)
        if type(request) is not dict or set(request) != {'payload', 'endorsements'}: raise ContractError('Dual review envelope required')
        endorsements = request['endorsements']
        if type(endorsements) is not list or len(endorsements) != 2: raise ContractError('Exactly two reviewer endorsements required')
        reviewers = []
        for approval in endorsements:
            if type(approval) is not dict or set(approval) != {'reviewer', 'signature'}: raise ContractError('Exact endorsement schema required')
            name, signature = approval['reviewer'], approval['signature']
            if type(name) is not str or name not in self._keys or name in reviewers: raise ContractError('Two distinct known reviewer credentials required')
            if type(signature) is not str or not re.fullmatch(r'[0-9a-f]{64}', signature): raise ContractError('Invalid endorsement encoding')
            expected = hmac.digest(self._keys[name], canonical({'reviewer': name, 'payload': request['payload']}), 'sha256').hex()
            if not hmac.compare_digest(signature, expected): raise ContractError('Review disagreement, tampering or wrong key')
            reviewers.append(name)
        nodes, registry = validate_payload(request['payload'])
        return ConsensusDocument(nodes, tuple(reviewers), source_hash(nodes), registry)
