"""Verify published source and aggregates using only the standard library."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(relative):
    return json.loads((ROOT/relative).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = load('audit/public-manifest.json')
    for path, expected in manifest['sha256'].items():
        if digest(ROOT/path)!=expected:
            raise ValueError('Public artifact drift '+path)
    plan = load('audit/plan.json')
    for path, expected in plan['source_sha256'].items():
        if digest(ROOT/path)!=expected:
            raise ValueError('Frozen source drift '+path)
    for path, expected in plan['dependency_sha256'].items():
        if digest(ROOT.parent/path)!=expected:
            raise ValueError('Runtime dependency drift '+path)
    study = load('results/verified-summary.json')
    control = load('results/exact-control.json')
    if digest(ROOT/'exact_control.py')!=control['experiment_sha256']:
        raise ValueError('Exact control source drift')
    if study['acceptance']!=all(study['gates'].values()):
        raise ValueError('Inconsistent acceptance result')
    if control['direct_counterfactual_final_hidden_and_vocab_comparisons']!=(
        2 * control['context_count'] * (control['name_variants_per_context']-1)
    ):
        raise ValueError('Control comparison count mismatch')
    if study['base_parameter_digest']!=control['runtime_parameter_digest']:
        raise ValueError('Checkpoint digest disagreement')
    print('Public source and aggregate checks pass. GPU replay requires the locally retained inputs and checkpoint.')


if __name__=='__main__':
    main()
