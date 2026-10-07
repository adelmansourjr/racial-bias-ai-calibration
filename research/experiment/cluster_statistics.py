"""CPU exploratory six-pool statistics; no new model inference.

Finite-bank enumeration is descriptive and exact. Sign-flip diagnostics require
independent pool clusters and symmetry of signed effects under a zero-effect
null. This convenience sample does not establish those assumptions. Bootstrap
intervals resample the six observed pools, not a population of hiring outcomes.
"""
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / 'ranking'
FAMILY = 'all_finite_family_assignments'


def bootstrap_interval(values):
    values = np.asarray(values, dtype=float)
    indices = np.asarray(list(itertools.product(range(len(values)), repeat=len(values))))
    means = values[indices].mean(axis=1)
    return {'empirical_pool_resampling_percentile_95_interval': np.quantile(means, [.025, .975]).tolist(),
            'ordered_resamples_enumerated': len(means),
            'interpretation': 'Exploratory conditional six-pool resampling interval; not a population confidence interval.'}


def sign_flip(values):
    values = np.asarray(values, dtype=float)
    signs = np.asarray(list(itertools.product([-1, 1], repeat=len(values))))
    observed = abs(float(values.mean()))
    randomized = abs((signs*values).mean(axis=1))
    extreme = int(np.sum(randomized >= observed - 1e-14))
    return {'statistic_abs_equal_pool_mean_signed_gap': observed,
            'extreme_sign_assignments': extreme, 'sign_assignments': len(signs),
            'exact_conditional_two_sided_p': extreme/len(signs),
            'assumptions': 'Under the zero-effect null, independent pool effects are jointly invariant to independent sign reversals (symmetric exchangeability). These assumptions are not established for the six convenience jobs; this is a diagnostic, not a confirmatory test.'}


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [None]*len(pvalues)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(pvalues)-rank)*pvalues[i]))
        adjusted[i] = running
    return adjusted


def main():
    results = json.loads((ROOT/'results.json').read_text())
    rows = list(csv.DictReader((ROOT/'measured-and-corrected-scores.csv').open()))
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        if row['task'] == 'shortlist':
            grouped[(row['model'],row['method'])][row['job_id']][row['candidate_id']][row['name']] = float(row['score'])
    output = {'design': {
        'pool_count': 6, 'applicant_count': 16, 'top_k': 1,
        'independent_hiring_outcomes_observed': 0,
        'analysis_status': 'Post-hoc exploratory analysis of convenience jobs and excerpts.',
        'finite_bank_estimand': 'Within each fixed applicant/job, mean score for Greg/Todd minus mean score for Jamal/Malik. Average over applicants within job, then equally over six jobs.',
        'ranking_estimand': 'For each pool, proportion of the 4^n assignments whose top-1 opaque-ID set differs from its fixed baseline assignment; average equally over six jobs.',
        'source_of_uncertainty': 'Only sensitivity to resampling the six observed job pools; no new cases, no random applicant sample, no validated qualification labels.',
        'pvalue_warning': 'The two sign-flip diagnostics do not demonstrate absence of bias or fairness. Signed mean effects can cancel despite shortlist changes.',
        'holm_family': 'Two original-model natural-shortlist signed-gap diagnostics, chosen post hoc; Holm controls this specified family only if the sign-flip p-values are valid. Other methods are descriptive.'},
        'methods': {}, 'diagnostics': {}}
    lines = ['# Exploratory statistics for six fixed hiring pools', '',
        'The strongest numerical result is an exact statement about the measured four-name family. It is not a population estimate: 288 assignments reuse 16 applicants in six convenience job pools. The following intervals describe resampling those six pools only.', '',
        '| Model / method | Equal-pool signed cue score gap | Equal-pool assignment top-1 change rate | Exploratory 95% pool-resampling interval for change rate |',
        '|---|---:|---:|---|']
    perjob = []
    for (model, method), jobs in grouped.items():
        detail=[]
        for job_id, applicants in sorted(jobs.items()):
            gaps = [np.mean([s['Greg Murray'],s['Todd Murray']])-np.mean([s['Jamal Murray'],s['Malik Murray']]) for s in applicants.values()]
            pool = next(p for p in results['models'][model][method]['shortlist']['pools'] if p['job_id']==job_id)
            detail.append({'job_id':job_id,'applicant_count':len(applicants),
                'signed_cue_score_gap':float(np.mean(gaps)),
                'mean_absolute_applicant_signed_gap':float(np.mean(np.abs(gaps))),
                'mean_applicant_four_name_range':float(np.mean([max(s.values())-min(s.values()) for s in applicants.values()])),
                'assignment_top1_change_rate':pool['families'][FAMILY]['selected_set_change_fraction'],
                'assignments_enumerated':pool['families'][FAMILY]['assignment_count'],
                'baseline_selected_ids':pool['baseline_selected_ids']})
        gaps=[d['signed_cue_score_gap'] for d in detail]
        rates=[d['assignment_top1_change_rate'] for d in detail]
        metric = {'pools':detail,'equal_pool_mean_signed_gap':float(np.mean(gaps)),
                  'equal_pool_mean_assignment_top1_change_rate':float(np.mean(rates)),
                  'signed_gap_resampling':bootstrap_interval(gaps),
                  'top1_change_resampling':bootstrap_interval(rates)}
        output['methods'].setdefault(model,{})[method]=metric
        interval=metric['top1_change_resampling']['empirical_pool_resampling_percentile_95_interval']
        lines.append(f"| {model} / {method} | {metric['equal_pool_mean_signed_gap']:.7f} | {np.mean(rates):.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] |")
        if method=='original':
            output['diagnostics'][model]=sign_flip(gaps)
        for d in detail:
            perjob.append({'model':model,'method':method,**d})
    models=list(output['diagnostics'])
    adjusted=holm([output['diagnostics'][m]['exact_conditional_two_sided_p'] for m in models])
    lines += ['', 'The sign-flip statistic is |mean of the six signed within-job cue gaps|. All 64 sign reversals are enumerated. Validity requires independence of job clusters and symmetry of their null distribution under sign reversal. The convenience sample and nonrandom choice of name cues do not establish those assumptions. Holm correction applies to these two post-hoc original-model diagnostics only.', '',
              '| Original model | Conditional sign-flip p | Holm-adjusted p |', '|---|---:|---:|']
    for model, value in zip(models,adjusted):
        output['diagnostics'][model]['holm_adjusted_p']=value
        lines.append(f"| {model} | {output['diagnostics'][model]['exact_conditional_two_sided_p']:.5f} | {value:.5f} |")
    lines += ['', 'A large p-value is not proof of no cue effect. Opposite signed effects can cancel, and score shifts can change winners even when their overall signed mean is small. Ranking invariance of the corrected fixed-bank scores follows from construction and was numerically checked; its zero resampling interval conveys that construction, not evidence that all future jobs or racial proxies are unbiased.', '',
              'No population fairness, actual hiring accuracy, equal opportunity, or groupwise calibration estimate is reported because demographic attributes and validated applicant suitability/outcome labels are absent. The 6^6 = 46,656 ordered empirical bootstrap samples are fully enumerated. Larger predeclared job samples with independent qualification ratings are needed for inferential evaluation.']
    (ROOT/'cluster-statistics.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    (ROOT/'cluster-statistics.md').write_text('\n'.join(lines)+'\n')
    with (ROOT/'per-job-statistics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(perjob[0]));writer.writeheader();writer.writerows(perjob)
    # Independent consistency check from raw records rather than summary rates.
    observed=defaultdict(lambda: {'selected_sets':set(),'changed':0,'count':0})
    examples={}
    for line in (ROOT/'assignment-ranks.jsonl').open():
        record=json.loads(line)
        if record['task']!='shortlist':
            continue
        ids=sorted(record['scores'],key=lambda cid:(-record['scores'][cid],cid))
        assert ids==record['ranking']
        assert sorted(ids[:1])==record['selected_ids']
        key=(record['model'],record['method'],record['job_id'])
        if record['family']==FAMILY:
            value=observed[key]
            value['selected_sets'].add(tuple(record['selected_ids']))
            value['count']+=1
            value['changed']+=int(record['selected_set_changed'])
        if record['method']=='original' and record['family']=='one_at_a_time' and record['selected_set_changed']:
            examples.setdefault((record['model'],record['job_id']),record)
    selections=[]
    for (model,method,job_id), value in observed.items():
        pool=next(p for p in results['models'][model][method]['shortlist']['pools'] if p['job_id']==job_id)
        assert value['count']==pool['families'][FAMILY]['assignment_count']
        assert abs(value['changed']/value['count']-pool['families'][FAMILY]['selected_set_change_fraction'])<1e-12
        selections.append({'model':model,'method':method,'job_id':job_id,
            'baseline_selected_ids':pool['baseline_selected_ids'],
            'all_observed_selected_id_sets':[list(s) for s in sorted(value['selected_sets'])],
            'changed_assignment_count':value['changed'],'assignments':value['count']})
    (ROOT/'per-job-selected-sets.json').write_text(json.dumps(selections,indent=2)+'\n')
    (ROOT/'original-one-applicant-counterexamples.json').write_text(json.dumps(list(examples.values()),indent=2)+'\n')
    (ROOT/'numerical-checks.json').write_text(json.dumps({'descending_score_opaque_id_ties_reproduced':True,
        'raw_record_counts_match_summary':True,'shortlist_records_checked':sum(v['count'] for v in observed.values()),
        'bootstrap_constant_values_identity':bootstrap_interval([0.]*6)['empirical_pool_resampling_percentile_95_interval']==[0.,0.],
        'signflip_constant_zero_p_one':sign_flip([0.]*6)['exact_conditional_two_sided_p']==1.,
        'holm_reference_check':holm([.125,.59375])==[.25,.59375]},indent=2)+'\n')
    print(json.dumps(output['diagnostics'],indent=2))


if __name__ == '__main__':
    main()
