"""Exploratory CV-clustered uncertainty for mean naming-cue range change."""
from collections import defaultdict
import json
import math
from pathlib import Path
import random
ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=json.loads((ROOT/'results/final.json').read_text())['rows'];groups=defaultdict(list)
    for r in rows:groups[r['cv_cluster']].append(r)
    p=lambda x:1/(1+math.exp(-max(-700,min(700,x))))
    differences=[]
    for group in groups.values():
        ds=[]
        for cid in {r['context_id']for r in group}:
            selected=[r for r in group if r['context_id']==cid]
            gap=lambda arm:max(p(r[arm+'_margin'])for r in selected)-min(p(r[arm+'_margin'])for r in selected)
            ds.append(gap('adapted')-gap('raw'))
        differences.append(sum(ds)/len(ds))
    rng=random.Random(20261103)
    draws=sorted(sum(rng.choice(differences)for _ in differences)/len(differences)for _ in range(5000))
    result={'CV_clusters':len(differences),'bootstrap_draws':5000,'seed':20261103,
      'mean_adapted_minus_raw_probability_range':sum(differences)/len(differences),
      'CV_clustered_percentile_95_interval':[draws[124],draws[4874]],
      'scope':'Small exploratory convenience sample; not a universal fairness or racial-group outcome interval.'}
    target=ROOT/'results/clustered-range-analysis.json'
    if target.exists():
        old=json.loads(target.read_text())
        if any(not math.isclose(x,y,abs_tol=1e-14)for x,y in zip(old['CV_clustered_percentile_95_interval'],result['CV_clustered_percentile_95_interval'])):
            raise ValueError('Stored uncertainty analysis mismatch')
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
