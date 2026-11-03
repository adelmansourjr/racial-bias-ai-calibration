import copy
import unittest
from nameguard.annotations import DualReviewAuthority
from nameguard.native_encoding import NativeSourceEncoder
from nameguard.runtime import ScoringService,PublicScore
from nameguard.contracts import ContractError
from test_annotations import KEYS,proposal,signed


class EndToEndBoundary(unittest.TestCase):
    def test_malformed_review_never_reaches_original_embedding_backend(self):
        calls=[]
        class Tokenizer:
            def encode(self,text,add_special_tokens=False):calls.append(text);return[ord(c)for c in text]
        class Backend:
            def infer(self,source):calls.append('BACKEND');return PublicScore(0.,.5)
        service=ScoringService(DualReviewAuthority(KEYS),NativeSourceEncoder(Tokenizer()),Backend())
        with self.assertRaises(ContractError):service.score(b'{}')
        self.assertEqual(calls,[])
        service.score(signed(proposal()))
        self.assertEqual(calls[-1],'BACKEND')
        self.assertIn('John Miller',calls[0])
