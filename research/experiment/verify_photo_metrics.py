"""Executable validation of the equations transcribed from the thesis photos."""
import json
from pathlib import Path
import numpy as np


def presence(a,b):
    numerator=np.linalg.norm(a.mean(axis=0)-b.mean(axis=0))
    denominator=.5*(np.linalg.norm(a,axis=-1).mean()+np.linalg.norm(b,axis=-1).mean())
    return numerator/denominator if denominator>0 else None


def retention(after,baseline,injected):
    denominator=injected-baseline
    return (after-baseline)/denominator if denominator else None


def main():
    a=np.array([[1.,0.],[1.,0.]])
    b=-a
    assert presence(a,b)==2
    assert np.linalg.norm(np.concatenate([a,b]).mean(axis=0))==0
    a=np.array([[-1.],[1.]])
    b=np.array([[-2.],[2.]])
    assert presence(a,b)==0
    assert np.all(np.abs(a)<1.5) and np.all(np.abs(b)>1.5)
    assert presence(10*a,10*b)==presence(a,b)
    # Protected signal survives a large shared background even as Presence falls.
    a=np.array([[-1.,1000.]])
    b=np.array([[1.,1000.]])
    assert 0<presence(a,b)<.003
    assert retention(.2,.2,.8)==0  # Return to nonzero baseline is not full erasure.
    assert retention(0,.2,.8)<0
    assert retention(.9,.2,.8)>1
    assert retention(.1,.2,.2) is None
    assert presence(np.zeros((2,2)),np.zeros((2,2))) is None
    rng=np.random.default_rng(7)
    for _ in range(100):
        a=rng.normal(size=(20,10)); b=rng.normal(size=(20,10))
        numerator=np.linalg.norm(a.mean(0)-b.mean(0))
        paired=np.linalg.norm(a-b,axis=-1).mean()
        denominator=.5*(np.linalg.norm(a,axis=-1).mean()+np.linalg.norm(b,axis=-1).mean())
        assert numerator<=paired+1e-12
        assert paired<=2*denominator+1e-12
    result={"all_assertions_passed":True,"norm":"Euclidean",
            "balanced_presence_range":[0,2],"zero_denominator":"undefined",
            "retention_requires_three_conditions":True,
            "finite_checks":100,"scope":"Algebra and counterexamples, not empirical model fairness"}
    Path(__file__).with_name('photo-metric-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
