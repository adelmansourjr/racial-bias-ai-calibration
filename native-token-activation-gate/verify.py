"""Verify source integrity and reconstruct finite-cohort equality from private records."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads((ROOT/path).read_text())


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def main():
    plan = load('audit/plan.json')
    summary = load('results/summary.json')
    for rel, expected in plan['sources'].items():
        check(digest(ROOT/rel)==expected,'Frozen source mismatch '+rel)
    for rel, expected in plan['dependencies'].items():
        check(digest(ROOT.parent/rel)==expected,'Runtime source mismatch '+rel)
    checkpoint = load('audit/checkpoint-manifest.json')['parameter_digest']
    check(checkpoint==summary['model_parameter_digest'] and summary['model_file_digests_verified'],
          'Checkpoint digest mismatch')
    check(summary['planned_name_inputs']==plan['contexts']*len(plan['names']), 'Wrong planned coverage')
    check(summary['accepted_name_inputs']+summary['rejected_name_inputs']==summary['planned_name_inputs'],
          'Incomplete accounting')
    if (ROOT/'audit/public-manifest.json').exists():
        for rel, expected in load('audit/public-manifest.json')['sha256'].items():
            check(digest(ROOT/rel)==expected,'Publication hash mismatch '+rel)
    if not (ROOT/'local-private/records.json').exists():
        print('Source and aggregate checks pass. Private records are needed to reconstruct the experiment.')
        return
    check(digest(ROOT/'local-private/contexts.json')==plan['cohort_sha256'],'Cohort drift')
    contexts = load('local-private/contexts.json')
    raw = load('local-private/records.json')
    cohort = raw['cohort']
    check(len(cohort)==summary['accepted_name_inputs'], 'Record count mismatch')
    check(len(raw['rejected'])==summary['rejected_name_inputs'], 'Rejection record count mismatch')
    check(Counter(str(r['identity_tokens']) for r in cohort)==summary['identity_token_length_counts'],
          'Token length counts mismatch')
    check(len({(r['context'],r['name']) for r in cohort})==len(cohort),'Duplicate fixture rows')
    check(len({r['cv'] for r in cohort})==plan['CVs'],'Wrong CV count')
    expected = {(row['id'],name) for row in contexts for name in plan['names']}
    observed = {(row['context'],row['name']) for row in cohort}
    rejected = {(row['context'],row['name']) for row in raw['rejected']}
    check(observed.isdisjoint(rejected) and observed|rejected==expected,'Fixture accounting mismatch')
    lookup = {row['id']:row for row in contexts}
    for row in cohort:
        context = lookup[row['context']]
        check(all(row[field]==context[field] for field in ('cv','job','kind')), 'Bad context binding')
    size = len(plan['names'])
    for arm, groups in raw['groups'].items():
        complete = [block for block in groups.values() if len(block)==size]
        result = summary['metrics'][arm]
        ranges = [max(r['margin'] for r in b)-min(r['margin'] for r in b) for b in complete]
        check(len(complete)==result['complete_contexts'],'Context coverage mismatch '+arm)
        check(sum(len({r['score'] for r in b})==1 for b in complete)==result['exact_score_contexts'],
              'Score exactness mismatch '+arm)
        check(sum(len({tuple(r['stages'].items()) for r in b})==1 for b in complete)==result['exact_stage_contexts'],
              'Stage exactness mismatch '+arm)
        check(sum(min(r['margin'] for r in b)<0<=max(r['margin'] for r in b) for b in complete)
              ==result['flip_contexts'],'Decision mismatch '+arm)
        if ranges:
            check(abs(statistics.mean(ranges)-result['mean_logit_range'])<1e-12,'Mean range mismatch '+arm)
        else:
            check(result['mean_logit_range'] is None,'Missing mean range status '+arm)
    public_margins = [row['margin'] for block in raw['groups']['packed'].values() for row in block]
    check(len(set(public_margins))>1,'Scorer collapsed to a constant')
    packed = summary['metrics']['packed']
    expected_gate = (len(cohort)==plan['contexts']*size and packed['exact_score_contexts']==plan['contexts']
                     and packed['exact_stage_contexts']==plan['contexts'])
    check(expected_gate==summary['packed_exact_gate'],'Gate summary mismatch')
    if expected_gate:
        stages = 1+28+1+1  # boundary, blocks, normalized state, vocabulary
        check(summary['checks']['direct_packed_counterfactual_array_checks']
              ==stages*(plan['contexts']*(size-1)+1), 'Counterfactual array count mismatch')
        check(summary['checks']['native_reference_exact_arrays']==2*plan['contexts'],
              'Native reference count mismatch')
        check(summary['multi_span_exact'] and summary['boundary_rejection'] and
              summary['tampered_signature_rejection'],'Missing controls')
    print('Frozen source, model digests, cohort coverage and all arm metrics verified.')


if __name__=='__main__':
    main()
