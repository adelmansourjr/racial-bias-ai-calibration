"""Compare the fresh-process reconstructed-checkpoint study with the original."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def main():
    a=ROOT/'results/gpu';b=ROOT/'results/gpu-reconstructed'
    ar=json.loads((a/'synthetic-rows.json').read_text());br=json.loads((b/'synthetic-rows.json').read_text())
    if len(ar)!=192 or len(br)!=192:raise ValueError('Unexpected cohort size')
    fields=('context','cluster','name','label','style','original_tokens','strict_admitted',
            'greedy_anchor_admitted','compiled_admitted','compiled_margin','native_margin',
            'evidence_tokens','formatting_cost','stage_fingerprints','counterfactual_stages_exact',
            'external_same_ids_stages_exact','external_same_ids_margin','external_text_ids_equal','external_text_margin')
    exact=sum(all(x.get(k)==y.get(k) for k in fields) for x,y in zip(ar,br))
    ac=json.loads((a/'arc-rows.json').read_text());bc=json.loads((b/'arc-rows.json').read_text())
    result={'synthetic_rows_identical_except_setup_timing':exact,
            'synthetic_stage_fingerprints_identical':sum(x['stage_fingerprints']==y['stage_fingerprints'] for x,y in zip(ar,br)),
            'arc_records_identical':ac==bc,'arc_count':len(ac),
            'checkpoint_files_recreated_exactly':json.loads((ROOT/'audit/checkpoint-reconstruction-v2.json').read_text())['matches_every_pinned_checkpoint_file'],
            'scope':'Same-host fresh-process reconstruction and repeat; not independent-lab replication.'}
    if exact!=192 or len(ac)!=100 or ac!=bc:raise ValueError('Reconstruction repeat mismatch')
    target=ROOT/'results/reconstruction-comparison.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
