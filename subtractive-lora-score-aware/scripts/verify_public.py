"""Verify the narrowed public source/aggregate package, without private artifacts."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[2]


def main():
    for study in ['subtractive-lora-pilot-v2','subtractive-lora-score-aware']:
        base=ROOT/study;plan=json.loads((base/'audit/plan.json').read_text())
        hashes=plan.get('sources',plan.get('source_hashes'))
        for rel,h in hashes.items():
            if hashlib.sha256((base/rel).read_bytes()).hexdigest()!=h:raise ValueError('Frozen source drift: '+study+'/'+rel)
        summary=json.loads((base/'results/summary.json').read_text())
        expected=json.loads((base/'audit/checkpoint-manifest.json').read_text())['parameter_digest']
        if summary['base_parameter_digest_after']!=expected or not summary['checkpoint_files_unchanged']:raise ValueError('Reported integrity mismatch')
        if summary['whole_model_erasure_proven']:raise ValueError('Unjustified erasure claim')
    manifest=json.loads((ROOT/'subtractive-lora-score-aware/audit/public-manifest.json').read_text())
    for rel,h in manifest['files'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('Public artifact drift: '+rel)
    print('Verified frozen source and public artifact hashes. Aggregate outcomes only; private per-case data and adapters are excluded, so this is not independent empirical reproduction.')


if __name__=='__main__':main()
