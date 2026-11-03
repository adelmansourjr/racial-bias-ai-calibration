"""Read-only CPU verification of coverage, exact token counts and recorded tests."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    read=lambda p:json.loads((ROOT/p).read_text())
    plan=read('audit/design.json');r=read('results/gpu.json');tokens=read('results/tokens.json')
    for rel,digest in plan['sources'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source drift: '+rel)
    expected={(b,n,v)for b in range(len(plan['bodies']))for n in plan['names']for v in plan['variants']}
    rows=r['records']
    if len(rows)!=len(expected)or{(x['body_id'],x['name'],x['variant'])for x in rows}!=expected:raise ValueError('Missing conformance rows')
    for body in range(len(plan['bodies'])):
        group=[x for x in rows if x['body_id']==body]
        if any(x['score']!=group[0]['score']or x['public_tokens']!=group[0]['public_tokens']or x['public_tensor_stages']!=32 or not x['direct_arrays_equal']for x in group):
            raise ValueError('Name/resegmentation inconsistency')
    if len({x['score']['margin']for x in rows})<2:raise ValueError('Constant scorer only')
    if len(tokens)!=32 or len({x['id']for x in tokens})!=32:raise ValueError('Incomplete token corpus')
    if any(x['canonical_transformer_tokens']!=x['ordinary_evidence_BPE']for x in tokens):raise ValueError('Token-count target failed')
    if any(x['total_embedding_tokens']!=x['raw_source_BPE']+x['canonical_transformer_tokens']for x in tokens):raise ValueError('Embedding overhead accounting failed')
    annotations=read('results/annotations.json')
    if len(annotations['positives'])!=36 or len(annotations['known_omission_controls'])!=108 or len(annotations['review_disagreements'])!=36:
        raise ValueError('Annotation experiment coverage mismatch')
    if not annotations['unknown_unregistered_alias_can_survive_even_two_signatures']or annotations['human_annotation_accuracy_measured']:
        raise ValueError('Semantic limitation misrepresented')
    timings=read('results/timings.json')
    expected_times={(body,pair,method)for body in range(3)for pair in range(7)for method in ['scalar','canonical']}
    if len(timings)!=42 or{(x['body_id'],x['pair'],x['method'])for x in timings}!=expected_times:raise ValueError('Timing coverage mismatch')
    medians={method:statistics.median(x['seconds']for x in timings if x['method']==method)for method in ['scalar','canonical']}
    for method,value in medians.items():
        if value!=r['summary']['median_seconds_by_method'][method]:raise ValueError('Timing summary mismatch')
    if not r['summary']['checkpoint_files_unchanged']or r['summary']['review_keys_saved']:raise ValueError('Integrity/private-key failure')
    result={'matched_GPU_forwards':len(rows),'direct_counterfactual_stage_comparisons':32*(len(rows)-3),
       'all_canonical_transformer_lengths_exact_native':True,'token_corpus_examples':len(tokens),
       'registered_omissions_blocked':len(annotations['known_omission_controls']),
       'different_review_payloads_blocked':len(annotations['review_disagreements']),
       'median_seconds':medians,'scope':'Source/result/coverage reconstruction, not GPU re-inference or automatic semantic-role proof.'}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
