"""Strict immutable document contracts and source-bound review authentication.

An HMAC authenticates possession of a configured reviewer key. It does not
prove that a role judgment is correct, or provide asymmetric non-repudiation.
"""
from dataclasses import dataclass
import hashlib
import hmac
import json
import re
from types import MappingProxyType

POLICY = 'nameguard-scalar-v1'
MAX_NODES = 128
MAX_SOURCE_BYTES = 131072
MAX_IDENTITY_BYTES = 16384
MAX_WIRE_BYTES = 262144


class ContractError(ValueError):
    """Request rejected before model computation."""


def canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8', errors='strict')
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ContractError('Invalid bounded JSON request') from error


def strict_loads(data):
    if type(data) is not bytes or len(data) > MAX_WIRE_BYTES:
        raise ContractError('Bounded immutable request bytes required')
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result: raise ContractError('Duplicate JSON keys are forbidden')
            result[key] = value
        return result
    try:
        return json.loads(data.decode('utf-8', errors='strict'), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ContractError('Nonfinite JSON number')))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ContractError('Invalid JSON request') from error


@dataclass(frozen=True)
class Node:
    role: str
    text: str


@dataclass(frozen=True)
class ReviewedDocument:
    nodes: tuple
    reviewer: str
    source_sha256: str


def validate_nodes(raw):
    if type(raw) is not list or not 1 <= len(raw) <= MAX_NODES:
        raise ContractError('Bounded ordered role/text nodes required')
    nodes = []; total = 0; identity = 0
    for row in raw:
        if type(row) is not dict or set(row) != {'role', 'text'}:
            raise ContractError('Exact role/text schema required')
        if type(row['role']) is not str or row['role'] not in {'identity', 'evidence'} or type(row['text']) is not str:
            raise ContractError('Unsupported field role/text')
        try: size = len(row['text'].encode('utf-8', errors='strict'))
        except UnicodeError as error: raise ContractError('Invalid Unicode') from error
        total += size
        if row['role'] == 'identity': identity += size
        if total > MAX_SOURCE_BYTES or identity > MAX_IDENTITY_BYTES:
            raise ContractError('Source/identity byte limit exceeded')
        nodes.append(Node(row['role'], row['text']))
    if not any(n.role == 'evidence' and n.text for n in nodes):
        raise ContractError('Nonempty retained evidence required')
    return tuple(nodes)


def source_hash(nodes):
    return hashlib.sha256(''.join(n.text for n in nodes).encode('utf-8')).hexdigest()


def sign_review(raw_nodes, *, reviewer, key):
    """Trusted review step, not an endpoint an untrusted applicant may call.

    The caller must review roles. This function validates representation only.
    Returned immutable bytes cannot be changed by later mutation of raw_nodes.
    """
    if type(key) is not bytes or len(key) < 32:
        raise ContractError('At least 256 bits of reviewer key material required')
    if type(reviewer) is not str or not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', reviewer):
        raise ContractError('Bounded configured reviewer identifier required')
    nodes = validate_nodes(raw_nodes)
    payload = {'policy': POLICY, 'reviewer': reviewer, 'source_sha256': source_hash(nodes),
               'nodes': [{'role': n.role, 'text': n.text} for n in nodes]}
    signature = hmac.digest(key, canonical(payload), 'sha256').hex()
    wire = canonical({'payload': payload, 'signature': signature})
    if len(wire) > MAX_WIRE_BYTES: raise ContractError('Serialized request exceeds limit')
    return wire


class ReviewAuthority:
    def __init__(self, reviewer_keys):
        if type(reviewer_keys) is not dict or not reviewer_keys:
            raise ContractError('Configured reviewer keys required')
        for name, key in reviewer_keys.items():
            if type(name) is not str or not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', name) or type(key) is not bytes or len(key) < 32:
                raise ContractError('Invalid configured reviewer key')
        self._keys = MappingProxyType(dict(reviewer_keys))
    def verify(self, wire):
        request = strict_loads(wire)
        if type(request) is not dict or set(request) != {'payload', 'signature'}:
            raise ContractError('Signed review envelope required')
        payload = request['payload']; signature = request['signature']
        if type(payload) is not dict or set(payload) != {'policy', 'reviewer', 'source_sha256', 'nodes'}:
            raise ContractError('Exact review payload required')
        if payload['policy'] != POLICY or type(payload['reviewer']) is not str or payload['reviewer'] not in self._keys:
            raise ContractError('Unknown policy or reviewer')
        if type(signature) is not str or not re.fullmatch(r'[0-9a-f]{64}', signature):
            raise ContractError('Invalid review signature encoding')
        expected = hmac.digest(self._keys[payload['reviewer']], canonical(payload), 'sha256').hex()
        if not hmac.compare_digest(signature, expected): raise ContractError('Review authentication failed')
        nodes = validate_nodes(payload['nodes'])
        if type(payload['source_sha256']) is not str or payload['source_sha256'] != source_hash(nodes):
            raise ContractError('Reviewed source binding failed')
        return ReviewedDocument(nodes, payload['reviewer'], payload['source_sha256'])
