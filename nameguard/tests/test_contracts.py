import copy
import dataclasses
import json
import unittest
from nameguard.contracts import ContractError, ReviewAuthority, sign_review, canonical
from nameguard.encoding import ScalarEncoder

KEY = bytes(range(32))  # Synthetic test fixture, NEVER a production reviewer key.
NODES = [{'role': 'evidence', 'text': 'Applicant: '}, {'role': 'identity', 'text': 'John Miller'},
         {'role': 'evidence', 'text': '\nUsed Python.'}]


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        # A special whole-string merge deliberately makes ordinary fieldwise
        # tokenization depend on segmentation; scalarwise encoding does not.
        if text == 'Used Python.': return [999999]
        return [ord(c) for c in text]


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.authority = ReviewAuthority({'reviewer': KEY})
        self.nodes = copy.deepcopy(NODES)
    def wire(self): return sign_review(self.nodes, reviewer='reviewer', key=KEY)
    def test_authentic_document_is_immutable(self):
        doc = self.authority.verify(self.wire())
        with self.assertRaises(dataclasses.FrozenInstanceError): doc.nodes[0].role = 'identity'
    def test_mutating_original_nodes_cannot_mutate_signed_snapshot(self):
        wire = self.wire(); self.nodes[1]['text'] = 'Changed'
        self.assertEqual(self.authority.verify(wire).nodes[1].text, 'John Miller')
    def test_tampered_identity_role_rejected(self):
        envelope = json.loads(self.wire()); envelope['payload']['nodes'][1]['role'] = 'evidence'
        with self.assertRaises(ContractError): self.authority.verify(canonical(envelope))
    def test_tampered_text_rejected(self):
        envelope = json.loads(self.wire()); envelope['payload']['nodes'][1]['text'] = 'Other'
        with self.assertRaises(ContractError): self.authority.verify(canonical(envelope))
    def test_wrong_key_rejected(self):
        with self.assertRaises(ContractError): ReviewAuthority({'reviewer': b'x'*32}).verify(self.wire())
    def test_wrong_policy_rejected(self):
        envelope = json.loads(self.wire()); envelope['payload']['policy'] = 'bypass'
        with self.assertRaises(ContractError): self.authority.verify(canonical(envelope))
    def test_unknown_reviewer_rejected(self):
        with self.assertRaises(ContractError): ReviewAuthority({'other': KEY}).verify(self.wire())
    def test_unsigned_request_rejected(self):
        with self.assertRaises(ContractError): self.authority.verify(canonical(self.nodes))
    def test_extra_claims_rejected(self):
        envelope = json.loads(self.wire()); envelope['grade'] = 100
        with self.assertRaises(ContractError): self.authority.verify(canonical(envelope))
    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ContractError): self.authority.verify(b'{"payload":{},"payload":{},"signature":"x"}')
    def test_invalid_unicode_rejected(self):
        self.nodes[1]['text'] = '\ud800'
        with self.assertRaises(ContractError): self.wire()
    def test_empty_retained_evidence_rejected(self):
        self.nodes = [{'role': 'identity', 'text': 'Name'}]
        with self.assertRaises(ContractError): self.wire()
    def test_oversize_identity_rejected(self):
        self.nodes[1]['text'] = 'x'*16385
        with self.assertRaises(ContractError): self.wire()
    def test_short_key_rejected(self):
        with self.assertRaises(ContractError): sign_review(self.nodes, reviewer='reviewer', key=b'short')
    def test_non_bytes_request_rejected(self):
        with self.assertRaises(ContractError): self.authority.verify(bytearray(self.wire()))
    def test_source_binding_is_explicit(self):
        doc = self.authority.verify(self.wire())
        import hashlib
        self.assertEqual(doc.source_sha256, hashlib.sha256(''.join(n['text'] for n in self.nodes).encode()).hexdigest())
    def test_scalar_encoding_keeps_actual_name_tokens(self):
        encoded = ScalarEncoder(FakeTokenizer()).encode(self.authority.verify(self.wire()))
        self.assertEqual(''.join(chr(i) for i in encoded.all_ids), ''.join(n['text'] for n in self.nodes))
    def test_all_evidence_resegmentations_match(self):
        encoder = ScalarEncoder(FakeTokenizer()); text = 'Used Python.'
        reference = None
        for cut in range(len(text)+1):
            nodes = [{'role':'evidence','text':text[:cut]}, {'role':'identity','text':'Different Ω'},
                     {'role':'evidence','text':text[cut:]}]
            encoded = encoder.encode(self.authority.verify(sign_review(nodes, reviewer='reviewer', key=KEY)))
            public = tuple(encoded.all_ids[i] for i in encoded.keep)
            if reference is None: reference = public
            self.assertEqual(public, reference)
    def test_same_spelled_skill_retained(self):
        self.nodes[1]['text'] = 'Python'
        e = ScalarEncoder(FakeTokenizer()).encode(self.authority.verify(self.wire()))
        self.assertIn('Used Python.', ''.join(chr(e.all_ids[i]) for i in e.keep))
    def test_bad_review_still_not_semantically_corrected(self):
        # Honest negative control: even an authentic reviewer can mislabel a name.
        self.nodes[1]['role'] = 'evidence'
        e = ScalarEncoder(FakeTokenizer()).encode(self.authority.verify(self.wire()))
        self.assertIn('John Miller', ''.join(chr(e.all_ids[i]) for i in e.keep))
