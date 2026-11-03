"""Validate frozen sources, public coverage, summary arithmetic and adapter bytes."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import math
ROOT=Path(__file__).resolve().parents[1]


def metrics(rows,arm):
    contexts=defaultdict(list);cvs=defaultdict(dict)
    for r in rows:
        value=r[arm+'_margin'];contexts[r['context_id']].append(value)
        cvs[r['cv_cluster']].setdefault(r['job_kind'],[]).append(value)
    avg=lambda a:sum(a)/len(a)
    p=lambda x:1/(1+math.exp(-max(-700,min(700,x))))
    return {'mean_margin_range':avg([max(v)-min(v)for v in contexts.values()]),
      'worst_margin_range':max(max(v)-min(v)for v in contexts.values()),
      'mean_probability_range':avg([max(map(p,v))-min(map(p,v))for v in contexts.values()]),
      'flip_contexts':sum(min(v)<0<=max(v)for v in contexts.values()),
      'category_pair_relevance':avg([float(avg(v['matched'])>avg(v['mismatched']))for v in cvs.values()])}


def main():
    plan=json.loads((ROOT/'audit/plan.json').read_text());result=json.loads((ROOT/'results/final.json').read_text());summary=result['summary'];rows=result['rows']
    for rel,h in plan['source_hashes'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('Frozen source drift: '+rel)
    if len(rows)!=256 or len({r['cv_cluster']for r in rows})!=16:raise ValueError('Wrong confirmation size')
    prior=set(json.loads((ROOT/'audit/previous-cv-clusters.json').read_text()))
    if prior.intersection(r['cv_cluster']for r in rows):raise ValueError('Confirmation CV overlap')
    groups=defaultdict(set)
    for r in rows:groups[r['context_id']].add(r['name'])
    if len(groups)!=32 or any(names!=set(plan['confirmation_names'])for names in groups.values()):raise ValueError('Incomplete name coverage')
    for arm in ['raw','adapted']:
        for k,v in metrics(rows,arm).items():
            if not math.isclose(summary[arm][k],v,abs_tol=1e-12,rel_tol=1e-12):raise ValueError('Metric mismatch: '+arm+':'+k)
    if len(result['capability'])!=64:raise ValueError('Missing capability rows')
    for arm in ['raw','adapted']:
        accuracy=sum(r[arm+'_prediction']==r['label']for r in result['capability'])/64
        if accuracy!=summary['capability'][arm+'_accuracy']:raise ValueError('Capability mismatch')
    expected=json.loads((ROOT/'audit/checkpoint-manifest.json').read_text())['parameter_digest']
    if expected!=summary['base_parameter_digest_after']or summary['base_weights_modified']:raise ValueError('Base integrity failed')
    print(json.dumps({'verified_rows':len(rows),'actual_base_forwards':320,'frozen_sources':len(plan['source_hashes']),
      'scope':'Saved-record consistency, not independent GPU reinference, universal fairness or probe replication.'},indent=2))


if __name__=='__main__':main()
