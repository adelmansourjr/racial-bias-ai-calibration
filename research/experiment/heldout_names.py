"""New naming cues on final probe-test applicants; local GPU models run separately.

No correction fitting or tuning. Unequal token lengths are allowed for behavioral
testing; Clef full-tensor projection is deliberately not applied to these inputs.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = [{"name":"Brett Murray", "cue_group":"white_associated"},
         {"name":"Kareem Murray", "cue_group":"black_associated"}]
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['qwen','clef'],required=True)
    p.add_argument('--model')
    p.add_argument('--out',type=Path,default=ROOT/'heldout_names')
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    all_cases=json.loads((ROOT/'cases.json').read_text())['cases']
    split=json.loads((ROOT/'results/qwen/probe-results.json').read_text())['fresh_probe_split']
    ids=set(split['probe_test_case_ids'])
    cases=[c for c in all_cases if c['case_id'] in ids]
    assert len(cases)==8 and all(c['split']=='test' for c in cases)
    assert not ids.intersection(split['correction_fit_case_ids'])
    assert not ids.intersection(split['probe_train_case_ids'])
    import mlx.core as mx
    from mlx_lm import load
    from mlx.utils import tree_flatten
    mx.set_cache_limit(128*1024*1024)
    started=time.perf_counter()
    if args.kind=='qwen':
        model_path=args.model or '/Users/adelmansour/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
    else:
        model_path=args.model or '/Users/adelmansour/Desktop/jev like/models/clef-flash-mlx-8bit'
    model,tok=load(model_path,lazy=True)
    for _,v in tree_flatten(model.parameters()): mx.eval(v)
    mx.clear_cache()
    records=[]; captures=[]
    if args.kind=='qwen':
        from qwen_qlora_demo import prompt_for
        from qwen_audit import probability
        from independent_qwen_validation import fit_basis, project
        old=np.load(ROOT/'results/qwen/activations.npz')
        h=old['hidden'].astype(np.float64); contrast=old['contrast'].astype(np.float64)
        fit=np.array([c['split']=='calibration' for c in all_cases])
        u=fit_basis(h,fit); mu=h[fit].mean((0,2))
        tasks=['shortlist','evidence']
        for case in cases:
            states=[]
            ci=next(i for i,c in enumerate(all_cases) if c['case_id']==case['case_id'])
            for ti,task in enumerate(tasks):
                task_states=[]
                for variant in NAMES:
                    variant={**variant,'text':f"NAME: {variant['name']}\n{case['resume_text']}"}
                    token_ids=tok.encode(prompt_for(case,variant,task,tok),add_special_tokens=False)
                    assert len(token_ids)<1000
                    final=model.model(mx.array(token_ids)[None])[0,-1].astype(mx.float32)
                    mx.eval(final); x=np.asarray(final).astype(np.float64); task_states.append(x)
                    z=project(x,u); centered=mu[ti]+project(x-mu[ti],u)
                    # Same four-name bank/weights, independent of received new cue.
                    orbit_h=h[ci,ti].mean(0)
                    records.append({'case_id':case['case_id'],'job_id':case['job_id'],'task':task,
                        'name':variant['name'],'cue_group':variant['cue_group'],'tokens':len(token_ids),
                        'raw_score':float(probability(x,contrast)),
                        'projection_score':float(probability(z,contrast)),
                        'centered_projection_score_exploratory':float(probability(centered,contrast)),
                        'fixed_bank_score_mean':float(probability(h[ci,ti],contrast).mean()),
                        'fixed_bank_activation_mean_score':float(probability(orbit_h,contrast))})
                    del final; mx.clear_cache()
                states.append(np.stack(task_states))
            captures.append(np.stack(states))
            print(f"Qwen new-name case {len(captures)}/8",flush=True)
        np.savez_compressed(args.out/'qwen-activations.npz',hidden=np.stack(captures).astype(np.float32),basis=u.astype(np.float32),contrast=contrast.astype(np.float32))
    else:
        support = Path(model_path) if (Path(model_path)/'clef_mlx.py').exists() else ROOT.parent/'model_support'
        sys.path.insert(0,str(support))
        from clef_mlx import JointSchemaHead,lexical_fn,encode_record
        from clef_audit import record_for
        head=JointSchemaHead(model_path); mx.eval(head.W); lex=lexical_fn(model)
        old=json.loads((ROOT/'clef_results/results.json').read_text())
        old_by_id={r['case_id']:r for r in old['cases']}
        for case in cases:
            case_h=[]
            for variant in NAMES:
                variant={**variant,'text':f"NAME: {variant['name']}\n{case['resume_text']}"}
                token_ids,qs=encode_record(tok,record_for(case,variant),max_length=10**9)
                assert len(token_ids)<2048
                hmx=model.language_model.model(mx.array(token_ids)[None])[0]
                logits=head(hmx,token_ids,qs,lex); mx.eval(hmx,*logits)
                hs=np.asarray(hmx.astype(mx.float32))
                case_h.append(np.stack([hs[-1],hs.mean(0)]))
                reference=old_by_id[case['case_id']]
                for task,q,z in zip(['shortlist','evidence'],qs,logits):
                    probs=np.asarray(mx.softmax(z).astype(mx.float32))
                    answers=dict(zip(q['oids'],map(float,probs)))
                    records.append({'case_id':case['case_id'],'job_id':case['job_id'],'task':task,
                        'name':variant['name'],'cue_group':variant['cue_group'],'tokens':len(token_ids),
                        'raw_score':answers['true'],'projection_score':None,
                        'projection_reason':'Unequal name token lengths; full-tensor alignment not assumed; fitted projector not persisted',
                        'fixed_bank_score_mean':float(np.mean([s[task]['true'] for s in reference['raw']])),
                        'fixed_bank_activation_mean_score':reference['orbit_mean'][0][task]['true']})
                del hmx,logits,hs; mx.clear_cache()
            captures.append(np.stack(case_h))
            print(f"Clef new-name case {len(captures)}/8",flush=True)
        np.savez_compressed(args.out/'clef-activation-summaries.npz',hidden=np.stack(captures).astype(np.float32))
    aggregates={}
    fields=['raw_score','projection_score','centered_projection_score_exploratory','fixed_bank_score_mean','fixed_bank_activation_mean_score']
    for task in ['shortlist','evidence']:
        aggregates[task]={}
        for field in fields:
            if not all(field in r and r[field] is not None for r in records if r['task']==task): continue
            ps=np.array([[next(r[field] for r in records if r['task']==task and r['case_id']==c['case_id'] and r['name']==v['name']) for v in NAMES] for c in cases])
            aggregates[task][field]={'mean_paired_score_range':float(np.ptp(ps,axis=1).mean()),
                'max_paired_score_range':float(np.ptp(ps,axis=1).max()),
                'any_threshold_flip_count':int((np.ptp((ps>=.5).astype(int),axis=1)>0).sum()),
                'case_count':8,'signed_white_minus_black_score_gap':float((ps[:,0]-ps[:,1]).mean())}
    output={'model':model_path,'kind':args.kind,'names':NAMES,'cases':[c['case_id'] for c in cases],
        'elapsed_seconds':time.perf_counter()-started,'records':records,'metrics':aggregates,
        'name_source':'Brett and Kareem first-name treatments in Bertrand & Mullainathan 2004; common surname Murray held constant. Historical US naming associations require contemporary validation.',
        'status':'Exploratory heldout-name extension, not preregistered population study',
        'scope':'Only inserted NAME header; same preserved resume and job text. Canonical fixed-bank output ignores received header and uses original four-name bank. No upstream erasure or hiring accuracy claim.'}
    (args.out/f'{args.kind}-results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(aggregates,indent=2),flush=True)


if __name__=='__main__': main()
