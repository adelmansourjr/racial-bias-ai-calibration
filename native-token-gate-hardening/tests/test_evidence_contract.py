import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evidence_contract import EvidenceContract, BoundEvidenceScorer, evidence_digest
from gate import Layout, ContractError
from test_lifecycle import Instrumented


class EvidenceBinding(unittest.TestCase):
    def test_name_tokens_change_while_approved_evidence_stays_fixed(self):
        contract = EvidenceContract((10,20))
        contract.verify(Layout((10,99,20),(0,2)))
        contract.verify(Layout((10,55,56,20),(0,3)))

    def test_substantive_or_formatting_token_change_rejects(self):
        contract = EvidenceContract((10,20))
        with self.assertRaisesRegex(ContractError,'Retained evidence'):
            contract.verify(Layout((10,99,21),(0,2)))

    def test_mismatch_never_reaches_model_and_remains_recoverable(self):
        raw = Instrumented()
        rejected = BoundEvidenceScorer(raw,EvidenceContract((0,)))
        with self.assertRaises(ContractError):
            rejected.score(b'valid')
        self.assertEqual(raw.calls,0)
        accepted = BoundEvidenceScorer(raw,EvidenceContract(tuple(map(ord,'Evidence'))))
        self.assertEqual(accepted.score(b'valid').margin,1.)
        self.assertEqual(raw.calls,1)

    def test_malformed_contract_rejects(self):
        for value in ['','x'*64,1,[],(),(True,),(-1,)]:
            with self.assertRaises(ContractError):
                EvidenceContract(value)

    def test_digest_collision_cannot_admit_changed_evidence(self):
        with patch('evidence_contract.evidence_digest',return_value='0'*64):
            contract = EvidenceContract((10,20))
            self.assertEqual(contract.digest,'0'*64)
            with self.assertRaises(ContractError):
                contract.verify(Layout((10,99,21),(0,2)))

    def test_approved_sequence_cannot_mutate(self):
        from dataclasses import FrozenInstanceError
        contract = EvidenceContract((10,20))
        with self.assertRaises(FrozenInstanceError):
            contract.token_ids = (10,21)
