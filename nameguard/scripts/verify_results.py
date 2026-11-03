"""Read-only replay of coverage, pinned sources and aggregate measurements.

Direct GPU-array comparisons happened in the recorded experiment. This verifier
does not reconstruct omitted tensor arrays or re-run inference.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    read=lambda path:json.loads((ROOT/path).read_text())
    plan=read('audit/gpu-plan.json');result=read('results/gpu.json')
    for rel,digest in plan['sources'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen experiment source drift: '+rel)
    expected={(b,n,v)for b in range(len(plan['bodies']))for n in plan['names']for v in plan['variants']}
    rows=result['records']
    if len(rows)!=len(expected)or{(r['body_id'],r['name'],r['variant'])for r in rows}!=expected:raise ValueError('Incomplete experimental coverage')
    for body in range(len(plan['bodies'])):
        group=[r for r in rows if r['body_id']==body]
        if any(r['score']!=group[0]['score']or not r['all_arrays_directly_equal']or r['stages_compared']!=31 for r in group):
            raise ValueError('Name/segmentation noninterference failure')
    if len({r['score']['margin']for r in rows})<2:raise ValueError('Only constant scores')
    if result['direct_tensor_stages']!=31*len(rows):raise ValueError('Layer comparison count mismatch')
    if set(result['tampering_blocked'])!={'role','text','reviewer','policy','source_sha256'}:raise ValueError('Authentication coverage incomplete')
    if not result['overlay_blocked']or not result['poisoned_reuse_blocked']or not result['checkpoint_files_unchanged']:
        raise ValueError('Runtime integrity check missing')
    regressions=result['segmentation_regressions']
    if len(regressions)!=4 or not all(r['legacy_token_sequences_differ']and r['new_token_sequences_equal']for r in regressions):
        raise ValueError('Actual-tokenizer segmentation regression missing')
    references=result['references']
    summary={'actual_GPU_forwards':len(rows),'reference_forwards':len(references),
         'observed_public_tensor_stages':31*len(rows),
         'direct_counterfactual_tensor_comparisons':31*(len(rows)-len(references)),
         'additional_native_hidden_tensor_comparisons':len(references),
         'maximum_recorded_name_or_segmentation_score_gap':0.,'auth_tamper_cases_rejected':len(result['tampering_blocked']),
         'ordinary_BPE_tokens':[r['ordinary_BPE_public_tokens']for r in references],
         'scalarwise_public_tokens':[r['public_tokens']for r in references],
         'scope':'Record/source/coverage reconstruction. Direct tensor comparisons ran during GPU study; this is not independent re-inference or proof of role semantics.'}
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
