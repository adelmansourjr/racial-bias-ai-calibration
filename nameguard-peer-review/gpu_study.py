"""Prospectively frozen, public synthetic comparisons and official ARC subset checks."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import secrets
import sys
import time
import platform
from importlib.metadata import version
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from bootstrap import ROOT,WORKSPACE
from compiler import enumerate_candidates,select_contract,match_contract,unresolved_payload,ContractError
from fixtures import FIT_NAMES,GPU_NAMES,contexts,rendered
from paired_stats import paired_bootstrap,total_outcome_disagreement
from comparators import ExternalEvidenceScorer
from gate import SlotScorer,native_layout
from nameguard.annotations import DualReviewAuthority,endorse,packet,proposal_from_spans
from nameguard.mlx_backend import load_pinned


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/gpu')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'plan.json').exists():raise FileExistsError('Use a fresh output directory')
    arc_path=ROOT/'local-private/arc-challenge-200-299.json'
    arc=json.loads(arc_path.read_text())['rows']
    assert len(arc)==100
    cohort=contexts()
    sources=list((ROOT/'src').glob('*.py'))+[Path(__file__)]
    deps=[WORKSPACE/'native-token-gate-hardening/gate.py']+list((WORKSPACE/'nameguard-validation/src/nameguard').glob('*.py'))
    plan={'before_inference':True,'date':'2026-11-09','sources':{str(p.relative_to(ROOT)):sha(p) for p in sources},
          'dependencies':{str(p.relative_to(WORKSPACE)):sha(p) for p in deps},
          'arc_snapshot_sha256':sha(arc_path),'arc_item_ids':[r['row']['id'] for r in arc],
          'contexts':32,'task_clusters':8,'names':GPU_NAMES,'fit_names':FIT_NAMES,
          'primary_layout':'space','test_contract':'Fit solely on fitting names, apply exact evidence tuple to disjoint test names; no score-based choices.',
          'endpoints':['planned admission','total outcome disagreement including rejection','exact score and stage equality','external same-ID equality','external retokenization differences','task accuracy by task cluster','full-vocabulary clean preservation','paired timing'],
          'calibration':'None. Original margin threshold zero; do not retune based on these labels.',
          'timing':'Eight accepted named cases, five randomized paired repetitions; report matched score-only paths and separate tokenizer/review setup.',
          'dense_control':'First eight accepted requests with physical dense and logical dense masks; descriptive numerical control.',
          'role_truth':'Synthetic generator-defined spans. No independent human annotation is claimed.',
          'clean':'100 official ARC-Challenge items test rows 200..299; compare original full vocab with all-evidence gate, A-D letter-logit readout.',
          'inference_gates':'Exact equals use array equality. Any violation is reported; no threshold relaxation or checkpoint selection.',
          'independence':'Eight synthetic task families. Names and wording variants are repeated measures. No population hiring claim.'}
    save(args.output_dir/'plan.json',plan)
    import mlx.core as mx
    import numpy as np
    model,tok,verify_files=load_pinned(args.checkpoint,ROOT/'audit/checkpoint-manifest.json')
    keys={k:secrets.token_bytes(32) for k in ('generator_a','generator_b')}
    authority=DualReviewAuthority(keys)
    scorer=SlotScorer(model,tok,authority)
    external=ExternalEvidenceScorer(model,tok,authority)
    letters=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABCD']
    def sign(payload):
        payload['resolved']=True # explicit generator truth, not independent review
        return packet(payload,[endorse(payload,reviewer=k,key=v) for k,v in keys.items()])
    def native(ids,trace=False):
        state=model.model.embed_tokens(mx.array(ids)[None])
        mask=None
        from mlx_lm.models.base import create_attention_mask
        mask=create_attention_mask(state,None)
        stages=[]
        if trace:mx.eval(state);stages.append(np.array(state.astype(mx.float32)))
        for layer in model.layers:
            state=layer(state,mask=mask,cache=None)
            if trace:mx.eval(state);stages.append(np.array(state.astype(mx.float32)))
        state=model.model.norm(state)
        if trace:mx.eval(state);stages.append(np.array(state.astype(mx.float32)))
        logits=model.model.embed_tokens.as_linear(state[:,-1:].astype(mx.float32)).astype(mx.float32)
        mx.eval(logits)
        z=np.array(logits)
        if trace:stages.append(z)
        return z,stages
    def margin(z):return float(z[0,0,letters[0]]-z[0,0,letters[1]])
    def arrays_for(layout,wire):
        values=[]
        zero=0
        ids_ok=False
        def observe(stage,array):
            nonlocal zero,ids_ok
            mx.eval(array)
            if stage=='native_ids':
                ids_ok=np.array_equal(np.array(array),np.array(layout.ids)[None]);return
            if stage=='raw_embedding':return
            data=np.array(array.astype(mx.float32))
            if stage!='vocabulary':
                private=np.ones(len(layout.ids),dtype=bool);private[list(layout.public)]=False
                if data.shape[1]!=len(layout.ids) or np.any(data[:,private,:].view(np.uint32)):
                    raise AssertionError('Nonzero identity or original slot count lost')
                zero+=1
                data=data[:,layout.public,:]
            values.append(np.ascontiguousarray(data))
        score=scorer._run(wire,'packed',observe)
        if not ids_ok:raise AssertionError('Original IDs changed')
        return score,values,zero
    rows=[]
    times=[]
    timing_cases=[]
    stage_comparisons=zero_checks=0
    dense_rows=[]
    total_start=time.perf_counter()
    for index,context in enumerate(cohort):
        prefix='Applicant name: '
        suffix='\n'+context['evidence']
        compilation_start=time.perf_counter()
        fit=[enumerate_candidates(*rendered(tok,name,prefix,suffix),tok) for name in FIT_NAMES]
        contract=select_contract(fit)
        compilation_seconds=time.perf_counter()-compilation_start
        first=next((v[0] for v in fit if v),None)
        anchor=first.evidence if first else None
        reference=None
        for ni,name in enumerate(GPU_NAMES):
            source,a,b=rendered(tok,name,prefix,suffix)
            ids=tuple(tok.encode(source,add_special_tokens=False))
            candidates=enumerate_candidates(source,a,b,tok)
            strict=next((c for c in candidates if c.formatting_cost==0),None)
            greedy=candidates[0] if candidates else None
            row={'context':context['id'],'cluster':context['task_cluster'],'name':name,'label':context['label'],
                 'context_compilation_seconds':compilation_seconds,
                 'style':context['style'],'original_tokens':len(ids),'strict_admitted':strict is not None,
                 'greedy_anchor_admitted':bool(greedy and greedy.evidence==anchor),
                 'compiled_admitted':False,'compiled_margin':None}
            z,_=native(ids)
            row['native_margin']=margin(z)
            try:chosen=match_contract(candidates,contract)
            except ContractError:
                rows.append(row);continue
            wire=sign(unresolved_payload(source,chosen,[name]))
            layout=native_layout(authority.verify(wire),tok)
            score,states,zero=arrays_for(layout,wire)
            zero_checks+=zero
            row.update(compiled_admitted=True,compiled_margin=score.margin,evidence_tokens=len(layout.public),
                       formatting_cost=chosen.formatting_cost,
                       stage_fingerprints=[hashlib.sha256(x.tobytes()).hexdigest() for x in states])
            if reference is None:reference=states
            else:
                exact=[np.array_equal(x,y) for x,y in zip(states,reference)]
                row['counterfactual_stages_exact']=all(exact)
                stage_comparisons+=len(exact)
            ext_z,ext_states=native(layout.public_ids,trace=True)
            exact=[np.array_equal(x,y) for x,y in zip(states,ext_states)]
            row['external_same_ids_stages_exact']=all(exact)
            row['external_same_ids_margin']=margin(ext_z)
            stage_comparisons+=len(exact)
            deleted=source[:chosen.start]+source[chosen.end:]
            text_ids=tuple(tok.encode(deleted,add_special_tokens=False))
            text_z,_=native(text_ids)
            row['external_text_ids_equal']=text_ids==layout.public_ids
            row['external_text_margin']=margin(text_z)
            if len(dense_rows)<8:
                dense={key:scorer._run(wire,key,None).margin for key in ('dense-logical','dense-physical')}
                dense_rows.append({'context':context['id'],'name':name,'packed':score.margin,**dense})
            if len(timing_cases)<8:
                timing_cases.append((context['id'],name,wire,layout,text_ids))
            rows.append(row)
        print(f'Context {index+1}/32 complete',flush=True)
    save(args.output_dir/'synthetic-rows.json',rows)
    save(args.output_dir/'dense-control.json',dense_rows)
    # Paired score-only timings exclude authenticated review preparation for all
    # paths. Native baselines receive already prepared IDs. Internal uses score().
    rng=random.Random(20261109)
    for ci,(context,name,wire,layout,text_ids) in enumerate(timing_cases):
        arms={'internal_score':lambda:scorer.score(wire),
              'external_same_ids_matched_service':lambda:external.score(wire),
              'external_text':lambda:native(text_ids),
              'native_named':lambda:native(layout.ids)}
        for fn in arms.values():fn()
        for rep in range(5):
            order=list(arms);rng.shuffle(order)
            for arm in order:
                begin=time.perf_counter();arms[arm]();elapsed=time.perf_counter()-begin
                times.append({'case':ci,'context':context,'name':name,'repeat':rep,'arm':arm,'seconds':elapsed})
        print(f'Timing block {ci+1}/8 complete',flush=True)
    save(args.output_dir/'timing.json',times)
    clean=[]
    for index,record in enumerate(arc):
        item=record['row'];choices=item['choices']
        texts=choices['text'];original_labels=choices['label']
        if len(texts)!=4:raise ValueError('ARC subset contains a non-four-choice item')
        target=original_labels.index(item['answerKey'])
        question=item['question']+'\n'+'\n'.join(f'{chr(65+i)}. {text}' for i,text in enumerate(texts))+'\nAnswer only A, B, C, or D.'
        source=tok.apply_chat_template([{'role':'user','content':question}],tokenize=False,add_generation_prompt=True)
        ids=tuple(tok.encode(source,add_special_tokens=False))
        z,_=native(ids)
        payload=proposal_from_spans(source,[],['fixture_name_absent'],resolved=True)
        wire=sign(payload)
        observed=[]
        def vocab(stage,value):
            if stage=='vocabulary':mx.eval(value);observed.append(np.array(value))
        scorer._run(wire,'packed',vocab)
        zg=observed[0]
        clean.append({'id':item['id'],'target':target,'native_prediction':int(np.argmax(z[0,0,letters])),
                      'gated_prediction':int(np.argmax(zg[0,0,letters])),
                      'full_vocabulary_exact':np.array_equal(z,zg),
                      'native_vocab_sha256':hashlib.sha256(z.tobytes()).hexdigest(),
                      'gated_vocab_sha256':hashlib.sha256(zg.tobytes()).hexdigest()})
        if (index+1)%20==0:print(f'ARC {index+1}/100 complete',flush=True)
    save(args.output_dir/'arc-rows.json',clean)
    digest=scorer.verify_parameters();verify_files();scorer.close();external.close()
    save(args.output_dir/'runtime.json',{'parameter_digest':digest,'checkpoint_files_verified':True,
         'python':platform.python_version(),'platform':platform.platform(),'elapsed_seconds':time.perf_counter()-total_start,
         'zero_identity_stage_checks':zero_checks,'direct_stage_comparisons':stage_comparisons,
         'mlx':version('mlx')})
    print('GPU study complete; run analyze.py for all planned endpoints.',flush=True)


if __name__=='__main__':main()
