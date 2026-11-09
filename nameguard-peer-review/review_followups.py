"""Post-review endpoints reconstructed from existing public records; no model fitting."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from paired_stats import paired_bootstrap
from bootstrap import ROOT


def main():
    rows=json.loads((ROOT/'results/gpu/synthetic-rows.json').read_text())
    families=defaultdict(list)
    for r in rows:families[r['cluster']].append(r)
    changes=[]
    for block in families.values():
        changes.append(sum(((r['external_text_margin']>=0)==r['label'])-((r['native_margin']>=0)==r['label']) for r in block)/len(block))
    result={'post_hoc_analysis':True,
            'character_deletion_correct':sum((r['external_text_margin']>=0)==r['label'] for r in rows),
            'native_correct':sum((r['native_margin']>=0)==r['label'] for r in rows),
            'compiled_correct':sum((r['compiled_margin']>=0)==r['label'] for r in rows),
            'deletion_vs_native_prediction_changes':sum((r['external_text_margin']>=0)!=(r['native_margin']>=0) for r in rows),
            'deletion_vs_compiled_prediction_changes':sum((r['external_text_margin']>=0)!=(r['compiled_margin']>=0) for r in rows),
            'deletion_accuracy_change_family_bootstrap':paired_bootstrap(changes),
            'boundary_ablation':json.loads((ROOT/'results/boundary-ablation.json').read_text())['totals'],
            'balanced_timing':{k:v for k,v in json.loads((ROOT/'results/balanced-timing.json').read_text()).items() if k!='paired_records'},
            'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target=ROOT/'results/review-followups.json'
    rendered=json.dumps(result,indent=2)+'\n'
    if target.exists() and target.read_text()!=rendered:raise ValueError('Preserve prior follow-up analysis')
    target.write_text(rendered);print(rendered)


if __name__=='__main__':main()
