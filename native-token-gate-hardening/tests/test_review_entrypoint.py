import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_review import propose_whitespace_repair
from test_review import MergingTokenizer


class ReviewEntrypoint(unittest.TestCase):
    def test_already_valid_long_formatting_is_preserved(self):
        source = 'Name:'+(' '*20)+'Ada\nTail'
        start = source.index('Ada')
        span = {'start':start,'end':start+3,'text':'Ada'}
        proposal = propose_whitespace_repair(
            source,[span],['Ada'],MergingTokenizer('NEVER_PRESENT'),allow_formatting=True)
        self.assertEqual(proposal.formatting_expansions,())
        self.assertFalse(proposal.payload['resolved'])

    def test_actual_space_name_merge_can_still_be_proposed(self):
        proposal = propose_whitespace_repair(
            'Name: Ada\nTail',[{'start':6,'end':9,'text':'Ada'}],['Ada'],
            MergingTokenizer(' Ada'),allow_formatting=True)
        self.assertEqual(proposal.formatting_expansions,((5,9),))
