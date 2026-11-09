"""Follow-up: matched external service outputs and explicit complete-stage metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from bootstrap import ROOT
from compiler import enumerate_candidates,select_contract,match_contract,unresolved_payload
from fixtures import FIT_NAMES,GPU_NAMES,contexts,rendered
from comparators import ExternalEvidenceScorer
from gate import SlotScorer,native_layout
from nameguard.annotations import DualReviewAuthority,endorse,packet
from nameguard.mlx_backend import load_pinned


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);args=p.parse_args()
    target=ROOT/'results/service-conformance.json'
    if target.exists() or (ROOT/'audit/service-conformance-plan.json').exists():raise FileExistsError('Preserve completed checks')
    plan={'before_inference':True,'adaptive_followup':True,
          'reason':'Review requested a bridge between timed service and direct reference, plus explicit trace completeness.',
          'rows':192,'comparison':'Normal score() for internal instrumented service and the matched external service; preserve original scores/fingerprints; explicit ordered 31 stages, shapes, finite values.',
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'audit/service-conformance-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    import mlx.core as mx
    import numpy as np
    model,tok,verify_files=load_pinned(args.checkpoint,ROOT/'audit/checkpoint-manifest.json')
    keys={k:secrets.token_bytes(32) for k in ('a','b')};authority=DualReviewAuthority(keys)
    expected_names=['boundary']+[f'block_{i+1}' for i in range(len(model.layers))]+['final_hidden','vocabulary']
    class Observed(SlotScorer):
        def _infer_layout(self,layout,mode,observe):
            self.stages=[]
            def record(stage,value):
                if stage in ('native_ids','raw_embedding'):return
                mx.eval(value);array=np.array(value.astype(mx.float32))
                if stage!='vocabulary':array=array[:,layout.public,:]
                if not np.isfinite(array).all():raise AssertionError('Nonfinite scoring state')
                self.stages.append({'stage':stage,'shape':list(array.shape),'dtype':str(array.dtype),
                                    'sha256':hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()})
            result=super()._infer_layout(layout,mode,record)
            if [r['stage'] for r in self.stages]!=expected_names:raise AssertionError('Missing or misordered scoring stages')
            return result
    internal=Observed(model,tok,authority);external=ExternalEvidenceScorer(model,tok,authority)
    prior={(r['context'],r['name']):r for r in json.loads((ROOT/'results/gpu/synthetic-rows.json').read_text())}
    records=[]
    for ci,context in enumerate(contexts()):
        prefix='Applicant name: ';suffix='\n'+context['evidence']
        contract=select_contract([enumerate_candidates(*rendered(tok,n,prefix,suffix),tok) for n in FIT_NAMES])
        for name in GPU_NAMES:
            source,a,b=rendered(tok,name,prefix,suffix)
            chosen=match_contract(enumerate_candidates(source,a,b,tok),contract)
            payload=unresolved_payload(source,chosen,[name]);payload['resolved']=True
            wire=packet(payload,[endorse(payload,reviewer=k,key=v) for k,v in keys.items()])
            score=internal.score(wire);ref=external.score(wire);old=prior[context['id'],name]
            if score!=ref or score.margin!=old['compiled_margin'] or score.margin!=old['external_same_ids_margin']:
                raise AssertionError('Matched service output mismatch')
            if [r['sha256'] for r in internal.stages]!=old['stage_fingerprints']:
                raise AssertionError('Recorded stages drifted')
            layout=native_layout(authority.verify(wire),tok)
            for meta in internal.stages[:-1]:
                if meta['shape']!=[1,len(layout.public),model.args.hidden_size]:raise AssertionError('Stage shape mismatch')
            if internal.stages[-1]['shape']!=[1,1,model.args.vocab_size]:raise AssertionError('Vocabulary shape mismatch')
            records.append({'context':context['id'],'name':name,'margin':score.margin,'stages':internal.stages})
        if (ci+1)%8==0:print(f'Conformance {ci+1}/32 complete',flush=True)
    result={'rows':len(records),'stages_per_row':len(expected_names),'all_original_fingerprints_preserved':True,
            'matched_external_service_scores_exact':len(records),'all_stages_finite_and_expected_shape':True,
            'records':records,'parameter_digest':internal.verify_parameters(),
            'source_sha256':plan['source_sha256']}
    verify_files();internal.close();external.close()
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
