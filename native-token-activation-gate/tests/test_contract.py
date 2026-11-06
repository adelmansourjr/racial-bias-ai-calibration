"""Protect native-token geometry and reject role-boundary failures."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gate import native_layout, ContractError, ConsensusDocument
from nameguard.contracts import Node


class TableTokenizer:
    def __init__(self, encodings):
        self.encodings = encodings
        self.calls = []

    def encode(self, value, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError('This contract uses the original serialized prompt')
        self.calls.append(value)
        return self.encodings[value]


def document(*runs):
    return ConsensusDocument(tuple(Node(role,text) for role,text in runs), ('a','b'),'source',('fixture',))


class TokenContract(unittest.TestCase):
    def test_full_native_tokens_and_physical_slots_survive_different_span_lengths(self):
        tokenizer = TableTokenizer({'left':[10], 'right':[20], 'short':[30], 'long':[40,41,42],
                                   'leftshortright':[10,30,20], 'leftlongright':[10,40,41,42,20]})
        a = native_layout(document(('evidence','left'),('identity','short'),('evidence','right')),tokenizer)
        b = native_layout(document(('evidence','left'),('identity','long'),('evidence','right')),tokenizer)
        self.assertEqual(a.ids,(10,30,20))
        self.assertEqual(b.ids,(10,40,41,42,20))
        self.assertEqual(a.public,(0,2))
        self.assertEqual(b.public,(0,4))
        self.assertEqual(a.public_ids,b.public_ids)

    def test_cross_role_merge_rejects(self):
        tokenizer = TableTokenizer({'aNb':[90,20], 'a':[10], 'N':[30], 'b':[20]})
        with self.assertRaisesRegex(ContractError,'crosses'):
            native_layout(document(('evidence','a'),('identity','N'),('evidence','b')),tokenizer)

    def test_same_role_segmentation_keeps_native_merge(self):
        tokenizer = TableTokenizer({'abNtail':[12,30,50], 'ab':[12], 'N':[30], 'tail':[50]})
        layout = native_layout(document(('evidence','a'),('evidence','b'),
                                        ('identity','N'),('evidence','tail')),tokenizer)
        self.assertEqual(layout.public_ids,(12,50))
        self.assertNotIn('a',tokenizer.calls)

    def test_unreviewed_input_and_identity_tail_reject(self):
        tokenizer = TableTokenizer({'aN':[10,30], 'a':[10], 'N':[30]})
        with self.assertRaisesRegex(ContractError,'Verified'):
            native_layout('aN',tokenizer)
        with self.assertRaisesRegex(ContractError,'readout'):
            native_layout(document(('evidence','a'),('identity','N')),tokenizer)

    def test_length_limit_rejects_instead_of_truncating(self):
        tokenizer = TableTokenizer({'aNtail':[10,30,50]})
        with self.assertRaisesRegex(ContractError,'length'):
            native_layout(document(('evidence','a'),('identity','N'),('evidence','tail')),tokenizer,limit=2)
