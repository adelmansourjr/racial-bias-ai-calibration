"""Freeze, run and retain actual canonical-BPE token/activation/timing checks."""
import argparse
import hashlib
import json
import random
import secrets
import statistics
import sys
import time
from pathlib import Path
from fixtures import ROOT,NAMES,BODIES,fixture,evidence,token_corpus
from nameguard.annotations import endorse,packet,DualReviewAuthority
from nameguard.native_encoding import NativeSourceEncoder
from nameguard.canonical_backend import CanonicalBackend
from nameguard.mlx_backend import QwenBackend,load_pinned
from nameguard.contracts import ReviewAuthority,sign_review
from nameguard.encoding import ScalarEncoder
from nameguard.runtime import ScoringService


def save(path,value):
    target=ROOT/path
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);args=p.parse_args()
    save('audit/design.json',{'before_inference':True,'sources':{str(p.relative_to(ROOT)):digest(p)for p in ROOT.rglob('*.py')if '__pycache__'not in p.parts},
        'names':NAMES,'bodies':BODIES,'variants':[0,1,2],
        'targets':['exact ordinary-BPE evidence token IDs/count','direct-array equality at all 32 public stages','native-evidence hidden/logit/score equality',
                   'registered omission controls blocked before model work','report full source embedding overhead separately'],
        'timing':{'warmups_per_method':2,'paired_repetitions_per_body':7,'order_seed':31112026,
                  'includes':'service verification, encoding, embedding, inference and integrity structure checks; excludes signing and model load'},
        'annotation_ground_truth':'Generator exact roles only; two credentials are synthetic, not independent human annotators.',
        'token_corpus':'32 generated multilingual/repetition templates; compare lengths, not task accuracy.'})
    import mlx.core as mx
    import numpy as np
    model,tok,verify_files=load_pinned(args.checkpoint,args.manifest)
    keys={'review_A':secrets.token_bytes(32),'review_B':secrets.token_bytes(32)}
    authority=DualReviewAuthority(keys);encoder=NativeSourceEncoder(tok)
    def wire_for(payload):return packet(payload,[endorse(payload,reviewer=n,key=k)for n,k in keys.items()])
    class Observed(CanonicalBackend):
        def __init__(self,*a):self.trace=[];self.reference=None;self.raw=None;super().__init__(*a)
        def _emit(self,stage,array):
            mx.eval(array);values=np.array(array.astype(mx.float32))
            if stage=='raw_embedding':self.raw=hashlib.sha256(values.tobytes()).hexdigest();return
            if self.reference is not None:
                previous_stage,previous=self.reference[len(self.trace)]
                if stage!=previous_stage or not np.array_equal(values,previous):raise AssertionError('Direct tensor inequality at '+stage)
            self.trace.append((stage,values))
    backend=Observed(model,tok);service=ScoringService(authority,encoder,backend)
    scalar_encoder=ScalarEncoder(tok)
    scalar_authority=ReviewAuthority({'legacy':keys['review_A']})
    rows=[];references=[];began=time.perf_counter()
    for body_index,body in enumerate(BODIES):
        backend.reference=None;reference_score=None
        for name in NAMES:
            for variant in [0,1,2]:
                payload,spans=fixture(body,name,variant);wire=wire_for(payload);prepared=encoder.encode(authority.verify(wire));backend.trace=[]
                result=service.score(wire)
                canonical_ids=tok.encode(evidence(payload),add_special_tokens=False)
                if [int(i)for i in backend.trace[0][1][0]]!=canonical_ids:raise AssertionError('Canonical IDs differ from ordinary BPE')
                if len(backend.trace)!=32:raise AssertionError('Missing public-stage observations')
                if backend.reference is None:
                    backend.reference=backend.trace;reference_score=result
                    native=model.model(mx.array(canonical_ids)[None]);mx.eval(native)
                    full=model.model.embed_tokens.as_linear(native[:,-1:].astype(mx.float32)).astype(mx.float32);mx.eval(full)
                    if not np.array_equal(np.array(native.astype(mx.float32)),backend.trace[-2][1])or not np.array_equal(np.array(full),backend.trace[-1][1]):
                        raise AssertionError('Original native-evidence computation mismatch')
                    legacy=scalar_authority.verify(sign_review(payload['nodes'],reviewer='legacy',key=keys['review_A']))
                    old=scalar_encoder.encode(legacy)
                    references.append({'body_id':body_index,'canonical_public_tokens':len(canonical_ids),
                         'scalar_public_tokens':len(old.keep),'raw_source_BPE_tokens':len(prepared.all_ids),
                         'raw_plus_canonical_embedding_tokens':len(prepared.all_ids)+len(canonical_ids),
                         'scalar_raw_embedding_tokens':len(old.all_ids),'score':result.as_dict(),
                         'native_hidden_and_full_vocab_directly_equal':True})
                if result!=reference_score:raise AssertionError('Name/resegmentation changed result')
                rows.append({'body_id':body_index,'name':name,'variant':variant,'source_sha256':prepared.document.source_sha256,
                     'raw_embedding_sha256':backend.raw,'score':result.as_dict(),'public_tokens':len(canonical_ids),
                     'raw_source_tokens':len(prepared.all_ids),'public_tensor_stages':len(backend.trace),'direct_arrays_equal':True})
        print(f'Body {body_index}: 36 variants exact at native evidence token count; {time.perf_counter()-began:.1f}s',flush=True)
    backend.reference=None
    # All source name tokens were embedded; scalar and canonical counts on a
    # broader synthetic multilingual corpus, without doing unnecessary forwards.
    tokens=[]
    for item in token_corpus():
        payload=item['payload'];wire=wire_for(payload);prepared=encoder.encode(authority.verify(wire))
        canonical=tok.encode(evidence(payload),add_special_tokens=False)
        old_doc=scalar_authority.verify(sign_review(payload['nodes'],reviewer='legacy',key=keys['review_A']))
        # Some longer old/scalar documents exceed its service context limit;
        # count the deterministic codec without truncation or executing it.
        scalar_public=sum(len(tok.encode(c,add_special_tokens=False))for c in evidence(payload))
        tokens.append({'id':item['id'],'ordinary_evidence_BPE':len(canonical),'canonical_transformer_tokens':len(canonical),
             'scalar_transformer_tokens':scalar_public,'raw_source_BPE':len(prepared.all_ids),
             'total_embedding_tokens':len(prepared.all_ids)+len(canonical),
             'scalar_service_public_limit_exceeded':scalar_public>2048,'canonical_service_public_limit_exceeded':len(canonical)>2048})
    save('results/tokens.json',tokens)
    # End-to-end time: no observation subclass and no tensor-array copies.
    efficient=ScoringService(authority,encoder,CanonicalBackend(model,tok))
    scalar=ScoringService(scalar_authority,scalar_encoder,QwenBackend(model,tok))
    rng=random.Random(31112026);timings=[]
    for qi,body in enumerate(BODIES):
        payload,_=fixture(body,NAMES[0],0)
        wires={'canonical':wire_for(payload),'scalar':sign_review(payload['nodes'],reviewer='legacy',key=keys['review_A'])}
        engines={'canonical':efficient,'scalar':scalar}
        for method in engines:
            for _ in range(2):engines[method].score(wires[method])
        for repetition in range(7):
            methods=list(engines);rng.shuffle(methods)
            for method in methods:
                started=time.perf_counter();score=engines[method].score(wires[method]);elapsed=time.perf_counter()-started
                timings.append({'body_id':qi,'pair':repetition,'method':method,'seconds':elapsed,'score':score.as_dict()})
    save('results/timings.json',timings)
    for engine in [service,efficient,scalar]:engine.close()
    parameter=backend.verify_parameters();verify_files()
    summary={'forward_checks':len(rows),'observed_public_tensor_stages':len(rows)*32,
       'direct_counterfactual_tensor_comparisons':(len(rows)-len(BODIES))*32,
       'native_reference_hidden_and_vocab_comparisons':len(BODIES)*2,
       'max_name_score_difference':0.,'canonical_token_ratio_to_native':1.,
       'mean_scalar_to_canonical_token_ratio':float(np.mean([r['scalar_transformer_tokens']/r['canonical_transformer_tokens']for r in tokens])),
       'median_seconds_by_method':{method:statistics.median(r['seconds']for r in timings if r['method']==method)for method in ['scalar','canonical']},
       'median_paired_speedup':statistics.median(next(r['seconds']for r in timings if r['body_id']==q and r['pair']==p and r['method']=='scalar')/
         next(r['seconds']for r in timings if r['body_id']==q and r['pair']==p and r['method']=='canonical')for q in range(3)for p in range(7)),
       'scope':'Controlled synthetic fixtures and local timing. Canonicalization is internal after raw embedding; total embedding work is larger than native evidence alone. Roles use generator oracle, not validated humans.',
       'parameter_digest':parameter,'checkpoint_files_unchanged':True,'review_keys_saved':False}
    save('results/gpu.json',{'records':rows,'references':references,'summary':summary})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
