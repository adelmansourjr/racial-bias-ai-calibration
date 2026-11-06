"""Check full physical state equality for a fixed-length identity substitution."""
import argparse
import json
from pathlib import Path
import secrets

import numpy as np

from experiment import ROOT, NAMES, split_prompt, write
from gate import SlotScorer, native_layout
from nameguard.annotations import DualReviewAuthority, endorse, packet, proposal_from_spans
from nameguard.mlx_backend import load_pinned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args = parser.parse_args()
    if (ROOT/'results/equal-geometry.json').exists():
        raise FileExistsError('Preserve the completed geometry control')
    import mlx.core as mx
    model, tokenizer, verify_files = load_pinned(args.checkpoint,ROOT/'audit/checkpoint-manifest.json')
    keys = {label:secrets.token_bytes(32) for label in ('a','b')}
    authority = DualReviewAuthority(keys)
    gate = SlotScorer(model,tokenizer,authority)
    cases = json.loads((ROOT/'local-private/contexts.json').read_text())[::4]
    report = {'contexts':len(cases),'full_array_equalities':0,'different_named_raw_embeddings':0,
              'exact_score_contexts':0,'fresh_process':True}
    for c in cases:
        prefix,suffix = split_prompt(c)
        outputs = []
        for name in NAMES[:2]:
            text = tokenizer.apply_chat_template([{'role':'user','content':prefix+name+suffix}],
                                                 tokenize=False,add_generation_prompt=True)
            start = text.index(prefix)+len(prefix)
            span = {'start':start,'end':start+len(name),'text':name}
            payload = proposal_from_spans(text,[span],[name],resolved=True)
            request = packet(payload,[endorse(payload,reviewer=k,key=v) for k,v in keys.items()])
            layout = native_layout(authority.verify(request),tokenizer)
            arrays = {}
            def observe(stage,value):
                if stage!='native_ids':
                    value = value.astype(mx.float32)
                    mx.eval(value)
                    arrays[stage] = np.array(value)
            result = gate._run(request,'packed',observe).as_dict()
            outputs.append((layout,arrays,result))
        (left,stages,score),(right,other,other_score) = outputs
        if left.public != right.public or len(left.ids)!=len(right.ids) or left.ids==right.ids:
            raise AssertionError('Expected different identity values at fixed geometry')
        if stages.pop('raw_embedding').tobytes()==other.pop('raw_embedding').tobytes():
            raise AssertionError('Raw identity embeddings did not differ')
        report['different_named_raw_embeddings'] += 1
        for stage,state in stages.items():
            paired = other[stage]
            if state.shape!=paired.shape or state.dtype!=paired.dtype or state.tobytes()!=paired.tobytes():
                raise AssertionError('Full physical array mismatch: '+stage)
            report['full_array_equalities'] += 1
        if json.dumps(score,sort_keys=True)!=json.dumps(other_score,sort_keys=True):
            raise AssertionError('Full geometry score mismatch')
        report['exact_score_contexts'] += 1
    report['parameter_digest'] = gate.verify_parameters()
    verify_files()
    write('results/equal-geometry.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
