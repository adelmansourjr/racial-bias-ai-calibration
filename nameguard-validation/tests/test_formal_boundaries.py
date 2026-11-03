from fractions import Fraction as F
import unittest


class FormalBoundaries(unittest.TestCase):
    def test_unchanged_unequal_scores_cannot_become_equal(self):
        a,b=F(1,5),F(4,5)
        for k in range(-10,21):
            c=F(k,10)
            self.assertGreaterEqual(max(abs(c-a),abs(c-b)),abs(a-b)/2)

    def test_summed_nonorthogonal_direction_edits_miss_target(self):
        # Unit vectors u=(1,0), v=(3/5,4/5), h=(1,1), b=0.
        u=(F(1),F(0));v=(F(3,5),F(4,5));h=(F(1),F(1))
        dot=lambda a,b:sum(x*y for x,y in zip(a,b))
        edited=tuple(h[i]-dot(u,h)*u[i]-dot(v,h)*v[i]for i in range(2))
        self.assertNotEqual((dot(u,edited),dot(v,edited)),(0,0))

    def test_joint_gram_inverse_imposes_both_targets(self):
        # Gram [[1,3/5],[3/5,1]]; h projections [1,7/5].
        rho=F(3,5);det=1-rho*rho
        a=(1-rho*F(7,5))/det;b=(-rho+F(7,5))/det
        edited=(1-a-b*F(3,5),1-b*F(4,5))
        self.assertEqual(edited,(0,0))
