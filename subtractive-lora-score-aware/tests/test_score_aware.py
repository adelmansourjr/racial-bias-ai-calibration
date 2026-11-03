from fractions import Fraction as F
import unittest


def dot(a,b):return sum(x*y for x,y in zip(a,b))


class ScoreAware(unittest.TestCase):
    def test_rank_one_adapter_applies_exact_margin_correction(self):
        w=(F(2),F(1));v=(F(1,3),F(-1,7));mu=(F(1),F(2));h=(F(3),F(5))
        a=tuple(x/dot(w,w)for x in w);c=dot(v,tuple(x-y for x,y in zip(h,mu)))
        edited=tuple(x-y*c for x,y in zip(h,a))
        self.assertEqual(dot(w,edited),dot(w,h)-c)

    def test_exact_name_score_cancellation_need_not_remove_name_state(self):
        # Margin reads coordinate 1; name survives in coordinate 2.
        w=(F(1),F(0));v=(F(1),F(0));a=(F(1),F(0))
        h0=(F(1),F(2));h1=(F(2),F(3))
        edited=[tuple(x-y*dot(v,h)for x,y in zip(h,a))for h in [h0,h1]]
        self.assertEqual(dot(w,edited[0]),dot(w,edited[1]))
        self.assertNotEqual(edited[0],edited[1])

    def test_minimum_norm_readout_direction(self):
        w=(F(2),F(1));a=tuple(x/dot(w,w)for x in w)
        self.assertEqual(dot(w,a),1)
        for t in range(-4,5):
            candidate=(a[0]+t,a[1]-2*t)
            self.assertEqual(dot(w,candidate),1)
            self.assertGreaterEqual(dot(candidate,candidate),dot(a,a))
