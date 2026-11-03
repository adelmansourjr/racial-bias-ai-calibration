"""Read-only public conformance/results verification; no private CV data needed."""
import hashlib
import json
from pathlib import Path
from analyze import summarize
ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'audit/plan.json').read_text());r=json.loads((ROOT/'results/gpu.json').read_text())
    for rel,h in plan['source_hashes'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('Source drift: '+rel)
    index=json.loads((ROOT/'audit/case-index.json').read_text())
    expected={(x['id'],n)for x in index for n in plan['names']}
    if {(x['context_id'],x['name'])for x in r['rows']}!=expected or len(r['rows'])!=len(expected):raise ValueError('Coverage mismatch')
    if len({x['cv_cluster']for x in r['rows']})!=32:raise ValueError('Cluster mismatch')
    if any(x['public_stages']!=32 or not x['direct_public_arrays_equal']for x in r['rows']):raise ValueError('Tensor checks failed')
    if len(r['native_evidence'])!=64 or any(not x['hidden_equal']or not x['full_vocab_equal']for x in r['native_evidence']):raise ValueError('Native comparison failed')
    if len(r['timings'])!=64 or len({(x['cv_cluster'],x['method'])for x in r['timings']})!=64:raise ValueError('Timing mismatch')
    computed=summarize(r);saved=json.loads((ROOT/'results/summary.json').read_text())
    if computed!=saved:raise ValueError('Summary mismatch')
    if computed['internal_max_probability_range']!=0 or not r['checkpoint_files_unchanged']or r['model_weights_changed']or r['keys_saved']:raise ValueError('Required property failed')
    print(json.dumps({'verified_name_rows':len(r['rows']),'source_hashes_checked':len(plan['source_hashes']),'scope':'Recorded coverage and deterministic reanalysis; not independent GPU re-inference.'},indent=2))


if __name__=='__main__':main()
