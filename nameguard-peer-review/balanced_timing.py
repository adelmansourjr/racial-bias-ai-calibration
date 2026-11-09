"""Follow-up timing balanced across all eight public credential families."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import secrets
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from bootstrap import ROOT
from compiler import enumerate_candidates,select_contract,match_contract,unresolved_payload
from fixtures import FIT_NAMES,GPU_NAMES,contexts,rendered
from paired_stats import paired_bootstrap
from comparators import ExternalEvidenceScorer
from gate import SlotScorer
from nameguard.annotations import DualReviewAuthority,endorse,packet
from nameguard.mlx_backend import load_pinned


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);args=p.parse_args()
    target=ROOT/'results/balanced-timing.json'
    if target.exists() or (ROOT/'audit/balanced-timing-plan.json').exists():raise FileExistsError('Preserve completed timing')
    plan={'before_inference':True,'adaptive_followup':True,
          'reason':'Original timing selected the first eight cases and overrepresented one task family.',
          'design':'One direct positive context per each of eight families, two fixed names, five shuffled paired repeats. Two identical-policy authenticated score APIs.',
          'families':8,'names':[GPU_NAMES[0],GPU_NAMES[4]],'pair_repetitions':5,
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'limitation':'Other user-approved training jobs may continue. Timing measures this host under concurrent load; no universal latency claim.'}
    (ROOT/'audit/balanced-timing-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    model,tok,verify_files=load_pinned(args.checkpoint,ROOT/'audit/checkpoint-manifest.json')
    keys={k:secrets.token_bytes(32) for k in ('a','b')};authority=DualReviewAuthority(keys)
    internal=SlotScorer(model,tok,authority);external=ExternalEvidenceScorer(model,tok,authority)
    rng=random.Random(20261110);records=[]
    for context in [c for c in contexts() if c['label'] and c['style']=='direct']:
        prefix='Applicant name: ';suffix='\n'+context['evidence']
        contract=select_contract([enumerate_candidates(*rendered(tok,name,prefix,suffix),tok) for name in FIT_NAMES])
        for name in (GPU_NAMES[0],GPU_NAMES[4]):
            source,a,b=rendered(tok,name,prefix,suffix)
            witness=match_contract(enumerate_candidates(source,a,b,tok),contract)
            payload=unresolved_payload(source,witness,[name]);payload['resolved']=True
            wire=packet(payload,[endorse(payload,reviewer=k,key=v) for k,v in keys.items()])
            if internal.score(wire)!=external.score(wire):raise AssertionError('Compared services differ')
            for repeat in range(5):
                order=['internal','external'];rng.shuffle(order);row={'family':context['task_cluster'],'name':name,'repeat':repeat}
                for arm in order:
                    start=time.perf_counter();(internal if arm=='internal' else external).score(wire)
                    row[arm+'_seconds']=time.perf_counter()-start
                records.append(row)
    family_ratios=[]
    for family in range(8):
        block=[r for r in records if r['family']==family]
        family_ratios.append(statistics.median(r['internal_seconds']/r['external_seconds'] for r in block))
    result={'paired_records':records,'family_median_ratios':family_ratios,
            'paired_family_bootstrap':paired_bootstrap(family_ratios),
            'median_family_ratio':statistics.median(family_ratios),
            'parameter_digest':internal.verify_parameters(),'source_sha256':plan['source_sha256']}
    verify_files();internal.close();external.close()
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='paired_records'},indent=2))


if __name__=='__main__':main()
