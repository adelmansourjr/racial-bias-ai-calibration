"""Reconstruct all study endpoints from public per-case synthetic records."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from paired_stats import paired_bootstrap,total_outcome_disagreement
from bootstrap import ROOT


def main():
    out=ROOT/'results/gpu'
    rows=json.loads((out/'synthetic-rows.json').read_text())
    assert len(rows)==32*6
    by_context=defaultdict(list)
    for row in rows:by_context[row['context']].append(row)
    context_results=[]
    for context,block in by_context.items():
        accepted=[r for r in block if r['compiled_admitted']]
        summary={'context':context,'cluster':block[0]['cluster'],
                 **total_outcome_disagreement([r['compiled_margin'] for r in block]),
                 'native_score_range':max(r['native_margin'] for r in block)-min(r['native_margin'] for r in block),
                 'native_decision_flip':len({r['native_margin']>=0 for r in block})>1,
                 'native_accuracy':sum((r['native_margin']>=0)==r['label'] for r in block)/len(block),
                 'compiled_accuracy_intent_to_score':sum(r['compiled_admitted'] and (r['compiled_margin']>=0)==r['label'] for r in block)/len(block),
                 'compiled_coverage':len(accepted)/len(block),
                 'accepted_exact_score':bool(accepted) and len({r['compiled_margin'] for r in accepted})==1,
                 'accepted_exact_stages':bool(accepted) and len({tuple(r['stage_fingerprints']) for r in accepted})==1}
        context_results.append(summary)
    accepted=[r for r in rows if r['compiled_admitted']]
    clusters=defaultdict(list)
    for row in context_results:clusters[row['cluster']].append(row)
    paired=[sum(r['compiled_accuracy_intent_to_score']-r['native_accuracy'] for r in block)/len(block) for block in clusters.values()]
    timing=json.loads((out/'timing.json').read_text())
    by_timing=defaultdict(lambda:defaultdict(list))
    for row in timing:by_timing[row['case']][row['arm']].append(row['seconds'])
    ratios=[statistics.median(arms['internal_score'])/statistics.median(arms['external_same_ids_matched_service'])
            for arms in by_timing.values()]
    arc=json.loads((out/'arc-rows.json').read_text())
    boundary=json.loads((ROOT/'results/boundary.json').read_text())
    result={'planned_rows':len(rows),'contexts':len(by_context),'task_clusters':len(clusters),
            'admission':{key:sum(r[key] for r in rows) for key in ('strict_admitted','greedy_anchor_admitted','compiled_admitted')},
            'context_checks':{key:sum(r[key] for r in context_results) for key in (
                'accepted_exact_score','accepted_exact_stages','admission_disagreement','conditional_score_disagreement','total_outcome_disagreement','native_decision_flip')},
            'all_same_id_reference_stage_matches':all(r['external_same_ids_stages_exact'] for r in accepted),
            'same_id_reference_rows':len(accepted),
            'external_text_identical_ids':sum(r['external_text_ids_equal'] for r in accepted),
            'external_text_exact_scores':sum(r['external_text_margin']==r['compiled_margin'] for r in accepted),
            'native_mean_name_logit_range':statistics.mean(r['native_score_range'] for r in context_results),
            'task_native_correct':sum((r['native_margin']>=0)==r['label'] for r in rows),
            'task_compiled_correct_intent_to_score':sum(r['compiled_admitted'] and (r['compiled_margin']>=0)==r['label'] for r in rows),
            'paired_task_accuracy_change_by_family':paired_bootstrap(paired),
            'timing_internal_over_matched_external':paired_bootstrap(ratios),
            'timing_ratio_median_cases':statistics.median(ratios),'timing_pairs':len(timing)//4,
            'arc_challenge':{'items':len(arc),'full_vocabulary_exact':sum(r['full_vocabulary_exact'] for r in arc),
                             'native_correct':sum(r['native_prediction']==r['target'] for r in arc),
                             'gated_correct':sum(r['gated_prediction']==r['target'] for r in arc)},
            'boundary_held_out':{key:sum(v[key] for v in boundary['layouts'].values()) for key in (
                'test_panel','test_strict','test_token_compatible','test_greedy_anchor','test_compiled','test_oracle_support_descriptive_only')},
            'runtime':json.loads((out/'runtime.json').read_text()),
            'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Public synthetic counterfactual conformance and official ARC subset. Generator roles, eight task families, one model/runtime. No real hiring or independent human review validation.'}
    target=ROOT/'results/summary.json'
    rendered=json.dumps(result,indent=2)+'\n'
    if target.exists() and target.read_text()!=rendered:raise ValueError('Analysis drift; preserve completed outputs')
    target.write_text(rendered)
    records=ROOT/'results/context-analysis.json'
    rendered_context=json.dumps(context_results,indent=2)+'\n'
    if records.exists() and records.read_text()!=rendered_context:raise ValueError('Context analysis drift')
    records.write_text(rendered_context)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
