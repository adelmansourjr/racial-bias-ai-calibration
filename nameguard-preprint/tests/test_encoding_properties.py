import random
import unittest
from nameguard.contracts import ReviewAuthority, sign_review
from nameguard.encoding import ScalarEncoder
from test_contracts import KEY, FakeTokenizer


class EncodingProperties(unittest.TestCase):
    def test_2000_random_unicode_identity_insertions_and_resegmentations(self):
        rng=random.Random(3112026);authority=ReviewAuthority({'test':KEY});encoder=ScalarEncoder(FakeTokenizer())
        alphabet='ab c\n\tΩé中محمد\u200b\x00<|>'
        for _ in range(2000):
            public=''.join(rng.choice(alphabet)for _ in range(rng.randint(1,50)))
            points=sorted([0,len(public)]+[rng.randrange(len(public)+1)for _ in range(rng.randint(0,6))])
            nodes=[]
            for a,b in zip(points,points[1:]):
                nodes.append({'role':'evidence','text':public[a:b]})
                nodes.append({'role':'identity','text':''.join(rng.choice(alphabet)for _ in range(rng.randint(0,15)))})
            encoded=encoder.encode(authority.verify(sign_review(nodes,reviewer='test',key=KEY)))
            actual=tuple(encoded.all_ids[i]for i in encoded.keep)
            expected=tuple(ord(c)for c in public)
            self.assertEqual(actual,expected)
