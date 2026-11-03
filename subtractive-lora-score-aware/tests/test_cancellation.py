"""Exact arithmetic witnesses for subtractive low-rank adapter claims."""
from fractions import Fraction as F
import unittest


def dot(a,b):return sum(x*y for x,y in zip(a,b))
def subtract(h,u,alpha=1,mu=(0,0)):
    c=alpha*dot(u,[x-y for x,y in zip(h,mu)])
    return tuple(x-c*y for x,y in zip(h,u))


class Cancellation(unittest.TestCase):
    def test_unit_direction_exactly_cancelled(self):
        u=(F(3,5),F(4,5));h=(F(2),F(9))
        self.assertEqual(dot(u,subtract(h,u)),0)

    def test_only_full_strength_annihilates(self):
        u=(F(1),F(0));h=(F(3),F(2))
        self.assertNotEqual(dot(u,subtract(h,u,F(1,2))),0)

    def test_unseen_orthogonal_name_channel_survives(self):
        u=(F(1),F(0))
        self.assertEqual(subtract((1,0),u),subtract((2,0),u))
        self.assertNotEqual(subtract((1,0),u),subtract((1,1),u))

    def test_task_overlap_cannot_be_preserved(self):
        u=(F(1),F(0))
        self.assertNotEqual(dot(u,(1,2)),dot(u,subtract((1,2),u)))

    def test_orthogonal_task_direction_is_preserved(self):
        u=(F(1),F(0));v=(F(0),F(1));h=(F(9),F(2))
        self.assertEqual(dot(v,h),dot(v,subtract(h,u)))

    def test_affine_head_factorization(self):
        u=(F(3,5),F(4,5));w=(F(2),F(7));h=(F(5),F(3));mu=(F(1),F(4))
        actual=dot(w,subtract(h,u,mu=mu))
        factored=dot(w,h)-dot(w,u)*dot(u,h)+dot(w,u)*dot(u,mu)
        self.assertEqual(actual,factored)

    def test_negative_lora_is_already_positive_lora_with_signed_factor(self):
        h=(F(4),F(2));a=(F(1),F(-1));b=F(3)
        self.assertEqual(-b*dot(a,h),b*dot(tuple(-x for x in a),h))
