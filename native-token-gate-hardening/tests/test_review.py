import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review import propose_whitespace_repair
from gate import native_layout, ConsensusDocument, ContractError
from nameguard.annotations import endorse
from nameguard.contracts import Node


class MergingTokenizer:
    def __init__(self, piece):
        self.piece = piece

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            if text.startswith(self.piece):
                ids.append(900)
                text = text[len(self.piece):]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ids


class ReviewPolicy(unittest.TestCase):
    def test_explicit_whitespace_permission_yields_unresolved_source_exact_proposal(self):
        source = 'Name: Ada\nTail'
        span = {'start':6,'end':9,'text':'Ada'}
        tokenizer = MergingTokenizer(' Ada')
        result = propose_whitespace_repair(source,[span],['Ada'],tokenizer,allow_formatting=True)
        self.assertEqual(result.formatting_expansions,((5,9),))
        self.assertFalse(result.payload['resolved'])
        self.assertEqual(''.join(node['text'] for node in result.payload['nodes']),source)
        with self.assertRaisesRegex(ContractError,'Unresolved'):
            endorse(result.payload,reviewer='a',key=b'x'*32)

    def test_no_permission_does_not_reclassify_evidence_whitespace(self):
        with self.assertRaises(ContractError):
            propose_whitespace_repair('Name: Ada\nTail',[{'start':6,'end':9,'text':'Ada'}],
                                      ['Ada'],MergingTokenizer(' Ada'))

    def test_substantive_merged_fact_cannot_be_absorbed(self):
        with self.assertRaises(ContractError):
            propose_whitespace_repair('ToolAda\nTail',[{'start':4,'end':7,'text':'Ada'}],
                                      ['Ada'],MergingTokenizer('ToolAda'),allow_formatting=True)
