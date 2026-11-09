import sys
from pathlib import Path
import unittest
import itertools
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compiler import Candidate,select_contract,match_contract,enumerate_candidates,unresolved_payload,ContractError


def c(e,cost=0): return Candidate(tuple(e),1,2,cost)


class CompilerMath(unittest.TestCase):
    def test_frequency_of_support_not_candidate_duplicates(self):
        contract=select_contract([[c([1]),c([1]),c([2])],[c([2])],[]])
        self.assertEqual(contract.evidence,(2,))
        self.assertEqual(contract.supported,2)
        self.assertEqual(contract.panel_size,3)

    def test_minimum_formatting_and_deterministic_tie_break(self):
        contract=select_contract([[c([1],2),c([2],0)],[c([1],2),c([2],1)]])
        self.assertEqual((contract.evidence,contract.total_formatting_cost),((2,),1))
        self.assertEqual(select_contract([[c([2]),c([1])]]).evidence,(1,))

    def test_exhaustive_small_support_optimality(self):
        options=[[],[c([1])],[c([2])],[c([1]),c([2])]]
        for panel in itertools.product(options,repeat=4):
            best=max(sum(any(x.evidence==(e,) for x in member) for member in panel) for e in (1,2))
            if best==0:
                with self.assertRaises(ContractError): select_contract(panel)
            else: self.assertEqual(select_contract(panel).supported,best)

    def test_infeasible_and_unseen_mismatch_fail_closed(self):
        for panel in ([],[[],[]]):
            with self.assertRaises(ContractError): select_contract(panel)
        with self.assertRaises(ContractError): match_contract([c([2])],select_contract([[c([1])]]))

    def test_witness_minimizes_cost(self):
        contract=select_contract([[c([1],2),c([1],0)]])
        self.assertEqual(match_contract([c([1],2),c([1],0)],contract).formatting_cost,0)

    def test_no_substantive_expansion_and_unresolved_review(self):
        class Tokenizer:
            def encode(self,text,add_special_tokens=False):return list(map(ord,text))
        text='Name:  Ada \nEvidence'
        self.assertEqual(text[7:10],'Ada')
        candidates=enumerate_candidates(text,7,10,Tokenizer(),budget=2)
        self.assertTrue(candidates)
        self.assertEqual({(c.start,c.end) for c in candidates},
                         {(a,b) for a in (5,6,7) for b in (10,11,12)})
        for cand in candidates:
            self.assertTrue(text[cand.start:7].isspace() or cand.start==7)
            self.assertTrue(text[10:cand.end].isspace() or cand.end==10)
            self.assertFalse(unresolved_payload(text,cand,['Ada'])['resolved'])

    def test_wide_whitespace_keeps_original_valid_candidate(self):
        class Tokenizer:
            def encode(self,text,add_special_tokens=False):return list(map(ord,text))
        text='Name:'+20*' '+'Ada\nEvidence'
        self.assertTrue(any(c.formatting_cost==0 for c in enumerate_candidates(text,25,28,Tokenizer())))

    def test_cross_boundary_merge_requires_authorized_space_expansion(self):
        class Merging:
            def encode(self,text,add_special_tokens=False):
                out=[]
                while text:
                    if text.startswith(' Ada'):out.append(999);text=text[4:]
                    else:out.append(ord(text[0]));text=text[1:]
                return out
        source='Name: Ada\nTail'
        self.assertEqual(enumerate_candidates(source,6,9,Merging(),budget=0),())
        candidates=enumerate_candidates(source,6,9,Merging(),budget=1)
        self.assertTrue(candidates)
        self.assertTrue(all(c.start==5 for c in candidates))
