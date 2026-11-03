import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze import stratified_cluster_interval,quantile,probability
from prepare import select,CATEGORIES


class StatisticsAndSampling(unittest.TestCase):
    def test_resampling_keeps_paired_null_difference_zero(self):
        self.assertEqual(stratified_cluster_interval({'a':0.,'b':0.},{'a':'x','b':'x'},100),[0.,0.])

    def test_sampling_excludes_previous_and_keeps_four_each(self):
        rows=[{'ID':f'{c}-{i}','Category':c}for c in CATEGORIES for i in range(8)]
        excluded={f'{c}-0'for c in CATEGORIES};chosen=select(rows,excluded)
        self.assertEqual(len(chosen),32)
        self.assertFalse(any(x['ID']in excluded for x in chosen))
        self.assertEqual(chosen,select(list(reversed(rows)),excluded))

    def test_quantile_interpolation(self):
        self.assertEqual(quantile([1,3],.5),2)

    def test_cluster_bootstrap_is_seed_reproducible(self):
        args=({'a':.1,'b':.8,'c':.4,'d':.6},{'a':'x','b':'x','c':'y','d':'y'})
        self.assertEqual(stratified_cluster_interval(*args,draws=100),stratified_cluster_interval(*args,draws=100))

    def test_probability_limits(self):
        self.assertEqual(probability(0),.5)
        self.assertLess(probability(-1000),1e-100)
        self.assertEqual(probability(1000),1.)
