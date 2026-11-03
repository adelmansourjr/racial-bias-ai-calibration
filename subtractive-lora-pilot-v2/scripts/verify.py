"""Read-only preservation, coverage and capability-score checks."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'audit/plan.json').read_text());r=json.loads((ROOT/'results/final.json').read_text());summary=r['summary']
    for rel,h in plan['sources'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('Frozen source changed: '+rel)
    if len(r['rows'])!=384 or len(r['capability'])!=64:raise ValueError('Coverage mismatch')
    if len({x['cv_cluster']for x in r['rows']if x['split']=='test'})!=16:raise ValueError('Wrong test CV count')
    expected=json.loads((ROOT/'audit/checkpoint-manifest.json').read_text())['parameter_digest']
    if expected!=summary['base_parameter_digest_after']or not summary['checkpoint_files_unchanged']:raise ValueError('Base integrity failure')
    for arm in ['raw','adapted']:
        value=sum(x[arm+'_prediction']==x['label']for x in r['capability'])/64
        if value!=summary['ARC_style_regression'][arm+'_accuracy']:raise ValueError('Capability result mismatch')
    print(json.dumps({'rows':384,'capability_items':64,'frozen_sources':len(plan['sources']),
      'scope':'Recorded integrity and coverage; not independent model inference or successful erasure certification.'},indent=2))


if __name__=='__main__':main()
