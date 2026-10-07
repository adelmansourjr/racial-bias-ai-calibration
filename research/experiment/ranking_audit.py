"""CPU fixed-pool ranking audit reusing measured independent CV/job scores.

No GPU, model loading, or new inference. Enumerates controlled assignments in six
fixed pools. Joint-pool contextual ranking is a different, unmeasured experiment.
"""
import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ['Greg Murray', 'Todd Murray', 'Jamal Murray', 'Malik Murray']
GROUPS = ['white_associated', 'white_associated', 'black_associated', 'black_associated']
TASKS = ['shortlist', 'evidence']
SEED = 'fixed-pool-20261007-v1'


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -700, 700)))


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def load_scores(cases):
    tables = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    qwen_rows = json.loads((ROOT/'results/qwen/scores.json').read_text())
    qwen_fields = {'original': 'raw_p_A', 'uncentered_projection': 'projected_p_A',
                   'activation_orbit_mean': 'activation_average_p_A', 'score_orbit_mean': 'score_average_p_A'}
    for row in qwen_rows:
        for method, field in qwen_fields.items():
            tables['Qwen1.5B'][method][row['task']].setdefault(row['case_id'], {})[row['name']] = row[field]
    arrays = np.load(ROOT/'results/qwen/activations.npz')
    centered = np.load(ROOT/'results/qwen/centered-repair.npz')['hidden']
    p = sigmoid(centered.astype(np.float64) @ arrays['contrast'].astype(np.float64))
    for ci, case in enumerate(cases):
        for ti, task in enumerate(TASKS):
            tables['Qwen1.5B']['centered_projection_exploratory'][task][case['case_id']] = dict(zip(NAMES, p[ci, ti].tolist()))
    clef = json.loads((ROOT/'clef_results/results.json').read_text())
    for row in clef['cases']:
        for task in TASKS:
            for method, field in [('original','raw'), ('uncentered_projection','projected'),
                                  ('activation_orbit_mean','orbit_mean')]:
                tables['Clef9B'][method][task][row['case_id']] = {
                    name: score[task]['true'] for name, score in zip(row['names'], row[field])}
            mean = float(np.mean([score[task]['true'] for score in row['raw']]))
            tables['Clef9B']['score_orbit_mean'][task][row['case_id']] = {name: mean for name in NAMES}
    # Optional exact per-variant scores exported by the quantized-LoRA run.
    qlora_path = ROOT/'results/qwen_qlora/ranking-scores.json'
    if qlora_path.exists():
        qlora = json.loads(qlora_path.read_text())
        if isinstance(qlora, dict):
            qlora = qlora['scores']
        for row in qlora:
            for method, field in [('affine4bit_before_adapter','before_p_A'), ('affine4bit_lora','after_p_A')]:
                tables['Qwen1.5B'][method][row['task']].setdefault(row['case_id'], {})[row['name']] = row[field]
    else:
        qlora_activations = ROOT/'results/qwen_qlora/activations.npz'
        z = np.load(qlora_activations)
        contrast_path = ROOT/'results/qwen_qlora/contrast.npy'
        if 'contrast' in z.files or contrast_path.exists():
            contrast = z['contrast'] if 'contrast' in z.files else np.load(contrast_path)
            for method, field in [('affine4bit_before_adapter','before'), ('affine4bit_lora','after')]:
                pp = sigmoid(z[field].astype(np.float64) @ contrast.astype(np.float64))
                for ci, case in enumerate(cases):
                    for ti, task in enumerate(TASKS):
                        tables['Qwen1.5B'][method][task][case['case_id']] = dict(zip(NAMES, pp[ci,ti].tolist()))
    for model, methods in tables.items():
        for method, tasks in methods.items():
            for task, table in tasks.items():
                assert set(table) == {c['case_id'] for c in cases}, (model, method, task, 'incomplete cases')
                assert all(set(s) == set(NAMES) for s in table.values())
                assert all(np.isfinite(p) and 0 <= p <= 1 for s in table.values() for p in s.values())
    return tables


def candidate_id(case_id):
    return 'cand-'+hashlib.sha256((SEED+':'+case_id).encode()).hexdigest()[:10]


def ranking(scores, ids, k):
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    ranks = {ids[index]: rank+1 for rank, index in enumerate(order)}
    return [ids[i] for i in order], ranks, frozenset(ids[i] for i in order[:k])


def compare(scores, ids, assignment, k, reference):
    order, ranks, selected = ranking(scores, ids, k)
    ref_order, ref_ranks, ref_selected = reference
    reversals = sum((ranks[a]-ranks[b])*(ref_ranks[a]-ref_ranks[b]) < 0
                    for a, b in itertools.combinations(ids, 2))
    white = [cid for cid, ni in zip(ids, assignment) if GROUPS[ni] == 'white_associated']
    black = [cid for cid, ni in zip(ids, assignment) if GROUPS[ni] == 'black_associated']
    gap = (sum(cid in selected for cid in white)/len(white)-sum(cid in selected for cid in black)/len(black)) if white and black else None
    return {'ranking': order, 'ranks': ranks, 'selected_ids': sorted(selected),
            'selected_set_changed': selected != ref_selected,
            'top_k_entries': sorted(selected-ref_selected), 'top_k_exits': sorted(ref_selected-selected),
            'pairwise_order_reversals': reversals,
            'rank_displacements': {cid: ranks[cid]-ref_ranks[cid] for cid in ids},
            'within_job_white_minus_black_selection_rate': gap}


def assignment_families(n):
    baseline = {2:(0,2), 3:(0,2,1), 4:(0,1,2,3)}[n]
    oat = []
    for changed in range(n):
        for name_index in range(4):
            if name_index == baseline[changed]:
                continue
            a = list(baseline); a[changed] = name_index
            oat.append((tuple(a), {'changed_indices': [changed]}))
    swaps = []
    for i, j in itertools.combinations(range(n), 2):
        a = list(baseline); a[i], a[j] = a[j], a[i]
        swaps.append((tuple(a), {'changed_indices': [i,j]}))
    balanced = [(a, {'changed_indices': [i for i in range(n) if a[i] != baseline[i]]})
                for a in itertools.permutations(range(4), n)
                if abs(sum(GROUPS[i] == 'white_associated' for i in a)-n*.5) <= .5]
    exhaustive = [(a, {'changed_indices': [i for i in range(n) if a[i] != baseline[i]]})
                  for a in itertools.product(range(4), repeat=n)]
    return baseline, {'one_at_a_time': oat, 'name_swaps': swaps,
                      'balanced_assignments': balanced, 'all_finite_family_assignments': exhaustive}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'ranking')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--include-calibration', action='store_true', help='Secondary full four-candidate pools; normally pair with --k 2')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_payload = json.loads((ROOT/'cases.json').read_text())
    cases = cases_payload['cases']
    pools = defaultdict(list)
    for case in cases:
        if args.include_calibration or case['split']=='test':
            pools[str(case['job_id'])].append(case)
    assert len(pools) == 6 and all(2 <= len(p) <= 4 and args.k < len(p) for p in pools.values())
    manifest = {'candidate_id_rule': 'SHA256(fixed seed + case ID), opaque prefix; no name or model score input',
                'seed': SEED, 'top_k': args.k, 'tie_breaker': 'ascending opaque candidate ID for equal scores',
                'ranking_model': 'independently computed CV-job scores, descending',
                'names': [{'name': n, 'cue_group': g} for n,g in zip(NAMES,GROUPS)],
                'analysis_status': 'Exploratory reuse: pool manifest created after original scoring; no independent confirmatory pools',
                'scope': 'secondary full pools including calibration applicants' if args.include_calibration else 'primary correction-held-out applicants only',
                'joint_pool_contextual_ranking_measured': False, 'candidate_order_sensitivity_measured': False,
                'hiring_suitability_labels': None, 'pools': []}
    for job_id, pool in sorted(pools.items()):
        pool.sort(key=lambda c: candidate_id(c['case_id']))
        manifest['pools'].append({'job_id': job_id, 'job_title': pool[0]['job_title'],
            'candidates': [{'candidate_id':candidate_id(c['case_id']), 'case_id':c['case_id'],
                'resume_id':c['resume_id'], 'split':c['split'], 'evidence_label':c['evidence_label'],
                'qualification_label':None} for c in pool]})
    json_write(args.output_dir/'pool-manifest.json', manifest)
    tables = load_scores(cases)
    records_path = args.output_dir/'assignment-ranks.jsonl'
    summary = {'design': manifest, 'models': {}, 'checks': {}}
    score_rows = []
    with records_path.open('w') as out:
        for model, methods in tables.items():
            summary['models'][model] = {}
            for method, tasks in methods.items():
                summary['models'][model][method] = {}
                for task, table in tasks.items():
                    analyzed_pools = []
                    all_family = defaultdict(list)
                    for job_id, pool in sorted(pools.items()):
                        ids = [candidate_id(c['case_id']) for c in pool]
                        probabilities = [[table[c['case_id']][name] for name in NAMES] for c in pool]
                        for c, cid, ps in zip(pool, ids, probabilities):
                            for ni,p in enumerate(ps):
                                score_rows.append({'model':model,'method':method,'task':task,'job_id':job_id,
                                    'candidate_id':cid,'case_id':c['case_id'],'split':c['split'],
                                    'name':NAMES[ni],'cue_group':GROUPS[ni],'score':p})
                        baseline, families = assignment_families(len(pool))
                        baseline_scores = [ps[a] for ps,a in zip(probabilities, baseline)]
                        ref = ranking(baseline_scores, ids, args.k)
                        base = {'job_id':job_id,'candidate_ids':ids,'baseline_assignment':list(baseline),
                                'baseline_scores':baseline_scores,'baseline_ranking':ref[0],
                                'baseline_selected_ids':sorted(ref[2]),'candidate_splits':[c['split'] for c in pool],
                                'per_candidate_score_ranges':[max(ps)-min(ps) for ps in probabilities],
                                'baseline_cutoff_margin': sorted(baseline_scores,reverse=True)[args.k-1]-sorted(baseline_scores,reverse=True)[args.k],
                                'distinct_candidate_scores_after_name_mean':len(set(float(np.mean(ps)) for ps in probabilities)),
                                'hiring_ranking_quality':None,'families':{}}
                        for family, assignments in families.items():
                            items = []
                            for ai,(assignment, metadata) in enumerate(assignments):
                                scores = [ps[a] for ps,a in zip(probabilities, assignment)]
                                item = compare(scores,ids,assignment,args.k,ref)
                                changed = metadata['changed_indices']
                                item['changed_applicant_max_abs_rank_displacement'] = max((abs(item['rank_displacements'][ids[i]]) for i in changed),default=0)
                                item['unchanged_applicant_max_abs_rank_displacement'] = max((abs(item['rank_displacements'][ids[i]]) for i in range(len(pool)) if i not in changed),default=0)
                                selected = set(item['selected_ids'])
                                labels = {cid:bool(c['evidence_label']) for cid,c in zip(ids,pool)}
                                item['literal_evidence_precision_at_k'] = sum(labels[cid] for cid in selected)/args.k if task == 'evidence' else None
                                item['literal_evidence_recall_at_k'] = sum(labels[cid] for cid in selected)/sum(labels.values()) if task == 'evidence' else None
                                out.write(json.dumps({'model':model,'method':method,'task':task,'job_id':job_id,
                                    'family':family,'assignment_index':ai,'name_assignment':dict(zip(ids,[NAMES[i] for i in assignment])),
                                    'scores':dict(zip(ids,scores)),**item})+'\n')
                                items.append(item)
                                all_family[family].append(item)
                            base['families'][family] = {'assignment_count':len(items),
                                'selected_set_change_fraction':float(np.mean([i['selected_set_changed'] for i in items])),
                                'rank_change_fraction':float(np.mean([any(i['rank_displacements'].values()) for i in items])),
                                'max_pairwise_order_reversals':max(i['pairwise_order_reversals'] for i in items),
                                'unique_selected_sets':len({tuple(i['selected_ids']) for i in items}),
                                'max_changed_applicant_rank_displacement':max(i['changed_applicant_max_abs_rank_displacement'] for i in items),
                                'max_unchanged_applicant_rank_displacement':max(i['unchanged_applicant_max_abs_rank_displacement'] for i in items),
                                'max_abs_group_selection_rate_gap':max(abs(i['within_job_white_minus_black_selection_rate']) for i in items if i['within_job_white_minus_black_selection_rate'] is not None),
                                'literal_evidence_precision_at_k_mean':float(np.mean([i['literal_evidence_precision_at_k'] for i in items])) if task=='evidence' else None}
                        analyzed_pools.append(base)
                    totals = {}
                    for family, items in all_family.items():
                        totals[family] = {'reused_assignment_count':len(items),
                            'selected_set_change_fraction':float(np.mean([i['selected_set_changed'] for i in items])),
                            'pools_with_any_selected_set_change':sum(p['families'][family]['selected_set_change_fraction']>0 for p in analyzed_pools),
                            'pools_with_any_rank_change':sum(p['families'][family]['rank_change_fraction']>0 for p in analyzed_pools),
                            'max_changed_applicant_rank_displacement':max(i['changed_applicant_max_abs_rank_displacement'] for i in items),
                            'max_unchanged_applicant_rank_displacement':max(i['unchanged_applicant_max_abs_rank_displacement'] for i in items),
                            'max_pairwise_order_reversals':max(i['pairwise_order_reversals'] for i in items),
                            'literal_evidence_precision_at_k_mean':float(np.mean([i['literal_evidence_precision_at_k'] for i in items])) if task=='evidence' else None}
                    summary['models'][model][method][task] = {'pool_count':len(analyzed_pools),
                        'maximum_applicant_score_range':max(max(p['per_candidate_score_ranges']) for p in analyzed_pools),
                        'pools':analyzed_pools,'totals':totals}
    with (args.output_dir/'measured-and-corrected-scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(score_rows[0]));writer.writeheader();writer.writerows(score_rows)
    # Validate the invariant-score => invariant-ranking theorem over this full finite family.
    for model, methods in summary['models'].items():
        for method in ('score_orbit_mean','activation_orbit_mean'):
            for task in TASKS:
                result=methods[method][task]
                assert result['maximum_applicant_score_range'] < 1e-12
                assert result['totals']['all_finite_family_assignments']['pools_with_any_rank_change']==0
                assert result['totals']['all_finite_family_assignments']['pools_with_any_selected_set_change']==0
    summary['checks']['invariant_scores_imply_identical_all_finite_family_assignments_rankings_and_top_k_sets']=True
    summary['checks']['qualification_accuracy_unmeasured']=True
    summary['checks']['joint_pool_ranking_and_order_permutations_unmeasured']=True
    json_write(args.output_dir/'results.json',summary)
    lines=['# Fixed-pool ranking audit','',
        f"Six fixed pools contain {[len(p) for _,p in sorted(pools.items())]} distinct real resume excerpts paired with real jobs. Top {args.k} are selected by descending independently measured score; ties use name-independent opaque IDs. Scope: {manifest['scope']}. This is exploratory reuse of six convenience pools; assignment counts are not independent sample sizes.",'',
        f'| Model / method | Natural-shortlist pools with any top-{args.k} change | Pools with any rank change | Max applicant score range | Literal-evidence precision@{args.k} |',
        '|---|---:|---:|---:|---:|']
    for model,methods in summary['models'].items():
        for method,tasks in methods.items():
            short=tasks['shortlist'];total=short['totals']['all_finite_family_assignments']
            evidence=tasks['evidence']['totals']['all_finite_family_assignments']['literal_evidence_precision_at_k_mean']
            lines.append(f"| {model} / {method} | {total['pools_with_any_selected_set_change']}/6 | {total['pools_with_any_rank_change']}/6 | {short['maximum_applicant_score_range']:.6g} | {evidence:.3f} |")
    lines += ['', '**What is proved.** For a fixed pool, each applicant has a name-invariant corrected score under the declared four-name family. Therefore the score vector is invariant. Sorting that vector with the same opaque-ID tie breaker produces the same ranking and top-k ID set. All finite-family assignments (4 raised to the pool size per pool) passed the numerical check for both score and activation averaging.','',
        '**What is not established.** Actual hiring usefulness, untested names/proxies, joint-pool context interactions, order sensitivity, or population fairness. Literal evidence precision measures keyword retrieval only. Protected-group selection-rate gaps can remain nonzero in a particular balanced assignment even when every candidate score is invariant: treatment assignment can associate the selected candidate IDs with one cue group by chance.','',
        'The raw assignment records retain every score, rank, displacement, entry/exit, selected set and pairwise reversal. The pool manifest fixes candidate IDs, datasets, calibration/test labels and tie breaking. Original, failed uncentered and exploratory centered repairs remain separate.']
    (args.output_dir/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(args.output_dir),'checks':summary['checks'],'models':{m:list(v) for m,v in summary['models'].items()}}))


if __name__=='__main__':
    main()
