"""Prespecified paired statistics, with contexts rather than names as units."""
import math
import random


def zero_failure_upper(n,alpha=.05):
    if type(n) is not int or n<=0 or not 0<alpha<1: raise ValueError('Invalid sample size or alpha')
    return -math.expm1(math.log(alpha)/n)


def paired_bootstrap(values,*,seed=20261109,repetitions=4000):
    """Percentile interval for equally weighted independent context-level values."""
    values=tuple(values)
    if not values or any(not math.isfinite(v) for v in values): raise ValueError('Finite nonempty values required')
    rng=random.Random(seed)
    means=sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(repetitions))
    return {'mean':sum(values)/len(values),'lower_95':means[int(.025*repetitions)],
            'upper_95':means[min(repetitions-1,int(.975*repetitions))],
            'units':len(values),'bootstrap_repetitions':repetitions,'seed':seed}


def total_outcome_disagreement(outcomes):
    """None denotes rejection; scalars denote published margins."""
    values=tuple(outcomes)
    if len(values)<2: raise ValueError('At least two planned variants required')
    accepted=[v for v in values if v is not None]
    return {'any_rejected':len(accepted)!=len(values),
            'admission_disagreement':0<len(accepted)<len(values),
            'conditional_score_disagreement':len(set(accepted))>1,
            'total_outcome_disagreement':len(set(values))>1}
