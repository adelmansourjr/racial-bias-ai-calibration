"""Run the existing role-bound model on the full held-out cohort as a positive control."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import secrets
import time

import numpy as np
from experiment import (
    ROOT, NAMES, CanonicalBackend, DualReviewAuthority, NativeSourceEncoder,
    endorse, load_pinned, packet, parameter_digest, prompt,
    proposal_from_spans, write,
)
from audit_results import group, shortlist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args = parser.parse_args()
    if (ROOT/'results/exact-control.json').exists():
        raise FileExistsError('Preserve the completed control run')
    import mlx.core as mx
    contexts = json.loads((ROOT/'local-private/contexts.json').read_text())['test']
    manifest = ROOT/'audit/checkpoint-manifest.json'
    model, tokenizer, verify_files = load_pinned(args.checkpoint,manifest)
    keys = {key:secrets.token_bytes(32) for key in ('fixture_a','fixture_b')}
    authority = DualReviewAuthority(keys)
    encoder = NativeSourceEncoder(tokenizer)

    class Observed(CanonicalBackend):
        def _emit(self, stage, value):
            if stage in ('final_hidden','final_vocab'):
                values = value.astype(mx.float32)
                mx.eval(values)
                self.observed[stage] = np.array(values)

    backend = Observed(model,tokenizer)
    exact = direct = references = forwards = 0
    records = []
    started = time.perf_counter()
    for index, context in enumerate(contexts,1):
        reference_score = None
        for name in NAMES['test']:
            text = tokenizer.apply_chat_template([{'role':'user','content':prompt(context,name)}],
                                                 tokenize=False,add_generation_prompt=True)
            start = text.index('Applicant name: ')+len('Applicant name: ')
            proposal = proposal_from_spans(
                text,[{'start':start,'end':start+len(name),'text':name}],[name],resolved=True)
            reviewed = authority.verify(packet(proposal,[
                endorse(proposal,reviewer=key,key=value) for key,value in keys.items()]))
            backend.observed = {}
            score = backend.infer(encoder.encode(reviewed)).as_dict()
            forwards += 1
            if reference_score is None:
                reference_score = score
                reference = backend.observed.copy()
                evidence = ''.join(node.text for node in reviewed.nodes if node.role=='evidence')
                ids = tokenizer.encode(evidence,add_special_tokens=False)
                hidden = model.model(mx.array(ids)[None])
                logits = model.model.embed_tokens.as_linear(hidden[:,-1:].astype(mx.float32)).astype(mx.float32)
                mx.eval(hidden,logits)
                for stage,value in [('final_hidden',hidden.astype(mx.float32)),('final_vocab',logits)]:
                    mx.eval(value)
                    actual = np.array(value)
                    expected = reference[stage]
                    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or actual.tobytes()!=expected.tobytes():
                        raise ValueError('Native evidence array mismatch')
                    references += 1
            else:
                if json.dumps(score,sort_keys=True)!=json.dumps(reference_score,sort_keys=True):
                    raise ValueError('Role-bound score mismatch')
                for stage, value in backend.observed.items():
                    old = reference[stage]
                    if old.shape!=value.shape or old.dtype!=value.dtype or old.tobytes()!=value.tobytes():
                        raise ValueError('Role-bound array mismatch '+stage)
                    direct += 1
            records.append({'context':context['id'],'control':{'margin':score['margin']}})
        exact += 1
        if index%8==0:
            print(f'Exact control {index}/{len(contexts)} contexts in {time.perf_counter()-started:.1f}s',flush=True)
    verify_files()
    digest = parameter_digest(model)
    if digest!=json.loads(manifest.read_text())['parameter_digest']:
        raise ValueError('Parameter digest mismatch')
    index = {c['id']:c for c in contexts}
    score_by_cv = defaultdict(dict)
    for context_id, rows in group(records,'context').items():
        c = index[context_id]
        score_by_cv[c['cv']][c['kind']] = rows[0]['control']['margin']
    report = {'exact_serialized_score_contexts':exact,'context_count':len(contexts),
              'name_variants_per_context':len(NAMES['test']),
              'role_bound_model_forwards':forwards,'native_evidence_forwards':references//2,
              'direct_counterfactual_final_hidden_and_vocab_comparisons':direct,
              'direct_native_evidence_hidden_and_vocab_comparisons':references,
              'category_matched_job_wins':sum(pair['matched']>pair['mismatched']for pair in score_by_cv.values()),
              'category_diagnostic_CVs':len(score_by_cv),
              'shortlist':shortlist(records,index,'control'),
              'runtime_parameter_digest':digest,
              'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope':'Positive control on the same CV/job/name fixtures using trusted injected-name spans. Does not test detection of names in raw prose.'}
    write(ROOT/'results/exact-control.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
