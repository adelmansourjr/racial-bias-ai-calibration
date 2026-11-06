"""Independently reconstruct exact-match gates and audit name-driven shortlist changes."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import unicodedata

ROOT = Path(__file__).resolve().parent
ARMS = ('native', 'previous', 'adapted')


def read(path):
    return json.loads((ROOT/path).read_text())


def require(condition, description):
    if not condition:
        raise ValueError(description)


def group(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def top_k_status(intervals, candidate, k):
    """Return possible and guaranteed inclusion, with opaque-ID tie breaking.

    Independently varying each candidate's name attains both extremal scenarios.
    """
    low, high = intervals[candidate]
    best_rank = 1 + sum((-lo, other) < (-high, candidate)
                        for other, (lo, _) in intervals.items() if other != candidate)
    worst_rank = 1 + sum((-hi, other) < (-low, candidate)
                         for other, (_, hi) in intervals.items() if other != candidate)
    return best_rank <= k, worst_rank <= k


def shortlist(rows, contexts, arm):
    by_job = defaultdict(dict)
    for context_id, block in group(rows, 'context').items():
        context = contexts[context_id]
        scores = [r[arm]['margin'] for r in block]
        by_job[context['job']][context['cv']] = (min(scores), max(scores))
    variable = []
    for job, candidates in sorted(by_job.items()):
        require(len(candidates) == 6, 'Expected six distinct CVs per job pool')
        status = [top_k_status(candidates, key, 3) for key in candidates]
        variable.append(sum(possible and not guaranteed for possible, guaranteed in status))
    return {'pools':len(by_job),'pool_size':6,'k':3,
            'variable_membership_pools':sum(count>0 for count in variable),
            'candidate_job_memberships_that_can_change':sum(variable)}


def stratified_range_change(rows, arm):
    by_category = defaultdict(list)
    for cv in group(rows, 'cv').values():
        diffs = []
        for block in group(cv, 'context').values():
            gap = lambda key: max(r[key]['margin'] for r in block)-min(r[key]['margin'] for r in block)
            diffs.append(gap(arm)-gap('native'))
        by_category[cv[0]['category']].append(statistics.mean(diffs))
    rng = random.Random(20261106)
    samples = sorted(statistics.mean(
        value for category in sorted(by_category)
        for value in rng.choices(by_category[category],k=len(by_category[category])))
        for _ in range(5000))
    return {'mean_difference':statistics.mean(x for values in by_category.values() for x in values),
            'percentile_95_interval':[samples[124],samples[4874]],
            'unit':'A-minus-B logit range','resampling':'CV clusters within fixed occupation strata'}


def main():
    plan = read('audit/plan.json')
    selection = read('audit/selection.json')
    summary = read('results/summary.json')
    contexts = read('local-private/contexts.json')
    rows = read('local-private/test-rows.json')
    rubric = read('local-private/rubric-rows.json')
    arc = read('local-private/capability-rows.json')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for rel, digest in plan['source_sha256'].items():
        require(sha(ROOT/rel)==digest,'Frozen source drift '+rel)
    for rel, digest in plan['dependency_sha256'].items():
        require(sha(ROOT.parent/rel)==digest,'Dependency drift '+rel)
    require(sha(ROOT/'local-private/contexts.json')==plan['cohort_sha256'],'Cohort drift')
    require(sha(ROOT/'local-private/adapter.npz')==selection['adapter_sha256'],'Adapter drift')
    require(selection['before_test_inference'],'Missing selection ordering')
    normalization = lambda x: ' '.join(unicodedata.normalize('NFKC',x).casefold().split())
    all_names = [normalization(name) for names in plan['names'].values() for name in names]
    require(len(set(all_names))==len(all_names),'Overlapping name sets')
    cv_ids = []
    unique_excerpt = {}
    for split in ('fit','dev','test'):
        subset = group(contexts[split],'cv')
        cv_ids.extend(subset)
        for cv, block in subset.items():
            require(len(block)==2 and {x['kind'] for x in block}=={'matched','mismatched'},
                    'Malformed CV cluster')
            excerpt = normalization(block[0]['text'])
            require(excerpt not in unique_excerpt,'Exact duplicate excerpt')
            unique_excerpt[excerpt] = cv
    require(len(set(cv_ids))==len(cv_ids)==56,'Overlapping CV sets')
    indexed = {c['id']:c for c in contexts['test']}
    require(len(rows)==288 and len(indexed)==48,'Unexpected test size')
    for cid, block in group(rows,'context').items():
        require(cid in indexed and sorted(r['name'] for r in block)==sorted(plan['names']['test']),
                'Incomplete counterfactual set')
        require(len({r['input'] for r in block})==6,'Name substitution did not yield distinct token inputs')
        for row in block:
            require(row['cv']==indexed[cid]['cv'] and row['kind']==indexed[cid]['kind'],'Bad context join')
    reconstructed = {}
    corrected_drift = {}
    for arm in ARMS:
        blocks = list(group(rows,'context').values())
        actual = summary['metrics'][arm]
        gap = [max(r[arm]['margin'] for r in b)-min(r[arm]['margin'] for r in b) for b in blocks]
        require(math.isclose(statistics.mean(gap),actual['mean_range'],abs_tol=1e-12),'Gap mismatch')
        require(sum(x>plan['tolerance_logit'] for x in gap)==actual['above_tolerance'],'Tolerance mismatch')
        flips = sum(len({r[arm]['decision'] for r in b})>1 for b in blocks)
        require(flips==actual['flips'],'Flip mismatch')
        for key in ('margin_bits','output','decision_logits','vocabulary','activation'):
            require(sum(len({r[arm][key] for r in b})==1 for b in blocks)
                    == actual['exact_'+key+'_contexts'],'Exact equality mismatch '+key)
        for row in rows:
            score = row[arm]
            require(math.isfinite(score['margin']), 'Invalid score')
            require(struct.pack('>f',score['margin']).hex()==score['margin_bits'],'Float bits mismatch')
            payload = {key:score[key] for key in ('margin','probability','decision')}
            require(hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
                    ==score['output'],'Serialized output mismatch')
        require(sum(r[arm]==r['label'] for r in arc)==summary['ARC_style'][arm],'ARC mismatch')
        require(sum((r[arm]>=0)==r['present'] for r in rubric)
                ==summary['rubric'][arm]['binary_correct'],'Rubric mismatch')
        by_cv = defaultdict(dict)
        for block in blocks:
            first = block[0]
            by_cv[first['cv']][first['kind']] = statistics.mean(r[arm]['margin'] for r in block)
        relevance = statistics.mean(int(pair['matched'] > pair['mismatched']) for pair in by_cv.values())
        require(relevance==actual['category_diagnostic'],'Category diagnostic mismatch')
        reconstructed[arm] = shortlist(rows,indexed,arm)
        corrected_drift[arm] = statistics.mean(abs(statistics.mean(
            r[arm]['margin'] - r['native']['margin'] for r in b)) for b in blocks)
        actual['mean_context_drift'] = corrected_drift[arm]
    adapted = summary['metrics']['adapted']
    baseline = summary['metrics']['native']
    gates = {
        'all_serialized_outputs_exact':adapted['exact_output_contexts']==48,
        'all_margins_exact_fp32':adapted['exact_margin_bits_contexts']==48,
        'zero_name_flips':adapted['flips']==0,
        'category_diagnostic_at_least_native':adapted['category_diagnostic']>=baseline['category_diagnostic'],
        'rubric_accuracy_at_least_native':summary['rubric']['adapted']['binary_correct']>=summary['rubric']['native']['binary_correct'],
        'arc_accuracy_at_least_native':summary['ARC_style']['adapted']>=summary['ARC_style']['native'],
    }
    require(gates==summary['gates'] and all(gates.values())==summary['acceptance'],'Acceptance gate mismatch')
    report = {'cohort_cv_count':len(cv_ids),'fresh_test_cv_count':24,
              'disjoint_normalized_name_sets':True,'test_records_reconstructed':len(rows),
              'shortlist_extremes':reconstructed,
              'stratified_paired_range_change':{arm:stratified_range_change(rows,arm)
                                                for arm in ('previous','adapted')},
              'scope':'Record and source consistency. Shortlists allow independent substitutions from the fixed six-name pool.'}
    path = ROOT/'results/secondary-audit.json'
    if path.exists():
        require(json.loads(path.read_text())==report,'Secondary aggregate drift')
    else:
        path.write_text(json.dumps(report,indent=2)+'\n')
    # The frozen runner passed each arm to its own drift reference on the test path.
    # Preserve the original result and publish a corrected derivative.
    summary['reporting_correction'] = (
        'Test mean_context_drift recomputed against native margins. The frozen runner used '
        'each arm as its own reference in that field. Development selection and test gates '
        'do not use the erroneous field.')
    summary['ARC_choice_changes_from_native'] = {
        arm:sum(r[arm]!=r['native'] for r in arc) for arm in ARMS}
    summary['rubric_changes_from_native'] = {
        arm:sum((r[arm]>=0)!=(r['native']>=0) for r in rubric) for arm in ARMS}
    (ROOT/'results/verified-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
