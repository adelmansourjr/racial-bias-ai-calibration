"""Read-only review-artifact checks; no model, private CV data or MLX needed."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))
from fixtures import contexts,GPU_NAMES
from fidelity_audit import validate_cohort,audit


def load(rel):return json.loads((ROOT/rel).read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def check(value,detail):
    if not value:raise ValueError(detail)


def main():
    summary=load('results/summary.json')
    for folder in ('gpu','gpu-reconstructed'):
        plan=load(f'results/{folder}/plan.json')
        for path,expected in plan['sources'].items():check(digest(ROOT/path)==expected,'Inference source drift '+path)
        for path,expected in plan['dependencies'].items():check(digest(ROOT.parent/path)==expected,'Dependency drift '+path)
        rows=load(f'results/{folder}/synthetic-rows.json');validate_cohort(rows)
        check(len(plan['arc_item_ids'])==len(set(plan['arc_item_ids']))==100,'ARC plan coverage')
        arc=load(f'results/{folder}/arc-rows.json')
        check([r['id'] for r in arc]==plan['arc_item_ids'],'ARC order/coverage mismatch')
        check(all(r['full_vocabulary_exact'] and r['native_vocab_sha256']==r['gated_vocab_sha256'] for r in arc),'ARC vocab mismatch')
        check(sum(r['native_prediction']==r['target'] for r in arc)==68 and sum(r['gated_prediction']==r['target'] for r in arc)==68,'ARC accuracy mismatch')
        grouped=defaultdict(list)
        for row in rows:
            check(row['compiled_admitted'] and len(row['stage_fingerprints'])==31,'Missing score or scoring stage')
            check(row['external_same_ids_stages_exact'] and row['compiled_margin']==row['external_same_ids_margin'],'Reference mismatch')
            grouped[row['context']].append(row)
        check(len(grouped)==32 and all(len(block)==6 and len({tuple(r['stage_fingerprints']) for r in block})==1 and len({r['compiled_margin'] for r in block})==1 for block in grouped.values()),'Counterfactual mismatch')
    original=load('results/gpu/synthetic-rows.json');repeat=load('results/gpu-reconstructed/synthetic-rows.json')
    groups=defaultdict(list)
    for row in original:groups[row['context']].append(row)
    fidelity_records={r['context']:r for r in load('results/fidelity.json')['records']}
    check(set(fidelity_records)==set(groups),'Fidelity context coverage mismatch')
    for context,block in groups.items():
        expected={'context':context,**audit([r['native_margin'] for r in block],block[0]['compiled_margin'])}
        check(expected==fidelity_records[context],'Fidelity arithmetic mismatch')
    check(all(a['stage_fingerprints']==b['stage_fingerprints'] and a['native_margin']==b['native_margin'] and a['compiled_margin']==b['compiled_margin'] for a,b in zip(original,repeat)),'Reconstructed outputs drift')
    check(digest(ROOT/'analyze.py')==summary['analysis_source_sha256'],'Analysis source drift')
    check(summary['task_native_correct']==179 and summary['task_compiled_correct_intent_to_score']==168,'Task summary mismatch')
    check(sum((r['native_margin']>=0)==r['label'] for r in original)==179 and sum((r['compiled_margin']>=0)==r['label'] for r in original)==168,'Task counts mismatch')
    for result,script in (('results/boundary-ablation.json','boundary_ablation.py'),
                          ('results/balanced-timing.json','balanced_timing.py'),
                          ('results/service-conformance.json','service_conformance.py')):
        check(load(result)['source_sha256']==digest(ROOT/script),'Follow-up source drift '+script)
    boundary=load('results/boundary.json')
    for path,expected in boundary['sources'].items():check(digest(ROOT/path)==expected,'Boundary source drift '+path)
    counts=load('results/boundary-ablation.json')['totals']
    check([counts[k] for k in ('greedy_contract_greedy_witness','greedy_contract_all_witnesses','compiled_contract_greedy_witness','compiled_contract_all_witnesses')]==[158,160,154,170],'Ablation mismatch')
    service=load('results/service-conformance.json')
    check(service['rows']==service['matched_external_service_scores_exact']==192 and service['stages_per_row']==31,'Service coverage mismatch')
    check(len({(r['context'],r['name']) for r in service['records']})==192,'Duplicate service case')
    prior={(r['context'],r['name']):r for r in original}
    stage_names=['boundary']+[f'block_{i+1}' for i in range(28)]+['final_hidden','vocabulary']
    for row in service['records']:
        check([r['stage'] for r in row['stages']]==stage_names,'Missing/misordered stage metadata')
        check([r['sha256'] for r in row['stages']]==prior[row['context'],row['name']]['stage_fingerprints'],'Service fingerprint mismatch')
        evidence=prior[row['context'],row['name']]['evidence_tokens']
        check(all(r['shape']==[1,evidence,1536] for r in row['stages'][:-1]) and row['stages'][-1]['shape']==[1,1,151936],'Service shapes mismatch')
    fidelity=load('audit/fidelity-followup-validation.json')
    check(fidelity['source_sha256']==digest(ROOT/'fidelity_audit.py') and fidelity['source_rows_sha256']==digest(ROOT/'results/gpu/synthetic-rows.json') and fidelity['prospective_plan_sha256']==digest(ROOT/'results/gpu/plan.json') and fidelity['fixtures_sha256']==digest(ROOT/'src/fixtures.py'),'Fidelity provenance mismatch')
    reconstruction=load('audit/checkpoint-reconstruction-v2.json')
    check(reconstruction['source_script_sha256']==digest(ROOT/'rebuild_checkpoint.py') and reconstruction['matches_pinned_parameters'] and reconstruction['matches_every_pinned_checkpoint_file'] and reconstruction['source_files_unchanged'],'Reconstruction mismatch')
    check(load('results/reconstruction-comparison.json')['arc_records_identical'],'Reconstruction ARC mismatch')
    pinned=load('audit/arc-pinned-source.json')
    check(pinned['source_script_sha256']==digest(ROOT/'pin_arc.py') and pinned['exactly_matches_measured_rows'],'Pinned ARC source mismatch')
    timing=load('results/balanced-timing.json')
    check(len(timing['paired_records'])==80 and len({(r['family'],r['name'],r['repeat']) for r in timing['paired_records']})==80,'Timing coverage mismatch')
    check(len(timing['family_median_ratios'])==8,'Timing family coverage mismatch')
    ai=load('review-packet/ai-review-summary.json')
    check(ai['cases']==24 and ai['uncertain_per_reviewer']==[6,6],'AI review mismatch')
    for path,expected in ai['source_hashes'].items():check(digest(ROOT/'review-packet'/path)==expected,'Sealed AI review drift')
    release=ROOT/'audit/public-manifest.json'
    if release.exists():
        for path,expected in load('audit/public-manifest.json')['sha256'].items():check(digest(ROOT/path)==expected,'Release artifact drift '+path)
    print('Verified frozen inference/dependencies, complete synthetic cohorts, 31-stage records, same-ID comparisons, reconstruction, ARC 68/68, ablations and AI-review provenance. GPU inference is not rerun by this command.')


if __name__=='__main__':main()
