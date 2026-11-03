import copy
import json
import unittest
from nameguard.annotations import proposal_from_spans, endorse, packet, DualReviewAuthority, propose_candidates
from nameguard.contracts import ContractError
from nameguard.native_encoding import NativeSourceEncoder

KEYS={'one':b'1'*32,'two':b'2'*32}  # Public synthetic test fixtures only.


def signed(payload,keys=KEYS):
    return packet(payload,[endorse(payload,reviewer=name,key=key)for name,key in keys.items()])


def proposal(source='Name: John Miller\nJohn Miller used Python.', omit=False):
    found=[];start=0
    while True:
        at=source.find('John Miller',start)
        if at<0:break
        found.append({'start':at,'end':at+11,'text':'John Miller'});start=at+11
    return proposal_from_spans(source,found[:1]if omit else found,['John Miller'],resolved=True)


class ReviewTests(unittest.TestCase):
    def test_two_distinct_credentials_required(self):
        with self.assertRaises(ContractError):DualReviewAuthority({'one':KEYS['one']})
        with self.assertRaises(ContractError):DualReviewAuthority({'one':KEYS['one'],'two':KEYS['one']})
    def test_consensus_accepts_source_exact_roles(self):
        p=proposal();doc=DualReviewAuthority(KEYS).verify(signed(p))
        self.assertEqual(''.join(n.text for n in doc.nodes),'Name: John Miller\nJohn Miller used Python.')
        self.assertEqual(set(doc.reviewers),set(KEYS))
    def test_missing_body_reference_rejected(self):
        with self.assertRaisesRegex(ContractError,'Uncovered'):proposal(omit=True)
    def test_one_endorsement_rejected(self):
        p=proposal()
        with self.assertRaises(ContractError):DualReviewAuthority(KEYS).verify(packet(p,[endorse(p,reviewer='one',key=KEYS['one'])]))
    def test_duplicate_reviewer_rejected(self):
        p=proposal();e=endorse(p,reviewer='one',key=KEYS['one'])
        with self.assertRaises(ContractError):DualReviewAuthority(KEYS).verify(packet(p,[e,e]))
    def test_disagreeing_review_rejected(self):
        p=proposal();other=copy.deepcopy(p);other['registry'].append('John')
        with self.assertRaises(ContractError):DualReviewAuthority(KEYS).verify(packet(p,[endorse(p,reviewer='one',key=KEYS['one']),endorse(other,reviewer='two',key=KEYS['two'])]))
    def test_wrong_signature_rejected(self):
        wire=json.loads(signed(proposal()));wire['endorsements'][1]['signature']='0'*64
        from nameguard.contracts import canonical
        with self.assertRaises(ContractError):DualReviewAuthority(KEYS).verify(canonical(wire))
    def test_unresolved_is_default_and_cannot_be_endorsed(self):
        p=proposal_from_spans('John used Python.',[{'start':0,'end':4,'text':'John'}],['John'])
        with self.assertRaisesRegex(ContractError,'Unresolved'):endorse(p,reviewer='one',key=KEYS['one'])
    def test_known_alias_omission_rejected(self):
        with self.assertRaises(ContractError):proposal_from_spans('Name: John\nAlias Leaf completed the project.',[{'start':6,'end':10,'text':'John'}],['John','Leaf'],resolved=True)
    def test_normalized_registered_variants_must_be_covered(self):
        for variant in ['Ｊｏｈｎ Ｍｉｌｌｅｒ','Jóhn Míller','J\u200bohn Miller','J.o.h.n M.i.l.l.e.r']:
            with self.assertRaises(ContractError):proposal_from_spans('Name: John Miller\nReference: '+variant,[{'start':6,'end':17,'text':'John Miller'}],['John Miller'],resolved=True)
    def test_same_spelled_skill_requires_explicit_retained_role(self):
        source='Name: Python\nUsed Python in work.'
        spans=[{'start':6,'end':12,'text':'Python'}]
        with self.assertRaises(ContractError):proposal_from_spans(source,spans,['Python'],resolved=True)
        at=source.index('Python in')
        p=proposal_from_spans(source,spans,['Python'],[{'start':at,'end':at+6,'text':'Python','role':'skill','reason':'Programming language in work evidence'}],resolved=True)
        self.assertTrue(DualReviewAuthority(KEYS).verify(signed(p)))
    def test_retained_role_cannot_overlap_identity(self):
        with self.assertRaises(ContractError):proposal_from_spans('John used Python.',[{'start':0,'end':4,'text':'John'}],['John'],[{'start':0,'end':4,'text':'John','role':'third_party','reason':'Contradictory'}],resolved=True)
    def test_stale_identity_offsets_rejected(self):
        with self.assertRaises(ContractError):proposal_from_spans('John used Python.',[{'start':0,'end':4,'text':'Jane'}],['John'],resolved=True)
    def test_unknown_alias_remains_an_explicit_negative_control(self):
        p=proposal_from_spans('John, also called Leaf, used Python.',[{'start':0,'end':4,'text':'John'}],['John'],resolved=True)
        doc=DualReviewAuthority(KEYS).verify(signed(p))
        self.assertIn('Leaf',''.join(n.text for n in doc.nodes if n.role=='evidence'))
    def test_candidates_never_automatically_assign_identity(self):
        candidates=propose_candidates('Jordan delivered work for Jordan.', ['Jordan'])
        self.assertEqual(len(candidates),2)
        self.assertTrue(all(c['status']=='needs_semantic_review'for c in candidates))
    def test_native_source_encoder_does_not_tokenize_evidence_separately(self):
        calls=[]
        class Tok:
            def encode(self,text,add_special_tokens=False):calls.append(text);return[ord(c)for c in text]
        p=proposal();doc=DualReviewAuthority(KEYS).verify(signed(p))
        source=NativeSourceEncoder(Tok()).encode(doc)
        self.assertEqual(calls,[''.join(n.text for n in doc.nodes)])
        self.assertIn('John Miller',calls[0]);self.assertIs(source.document,doc)
