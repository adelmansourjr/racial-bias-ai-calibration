"""Read-only frozen-source and result verification for the hardening revision."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads((ROOT/path).read_text())


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def main():
    plan = load('audit/plan.json')
    summary = load('results/summary.json')
    for rel,expected in plan['sources'].items():
        check(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,'Frozen source drift '+rel)
    for rel,expected in plan['runtime_dependencies'].items():
        check(hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==expected,'Dependency drift '+rel)
    checkpoint = load('audit/checkpoint-manifest.json')['parameter_digest']
    check(summary['checkpoint_digest']==checkpoint and summary['checkpoint_files_verified'],
          'Model integrity result mismatch')
    regression = summary['prior_regression']
    check(regression['prior_scores_exact']==192 and regression['prior_stage_digests_exact']==192*31,
          'Regression coverage mismatch')
    check(regression['zeroed_slot_checks']==192*30,'Zeroed slot coverage mismatch')
    check(summary['all_original_results_preserved'],'Original conformance failed')
    check(summary['clean']['exact_native_vocabulary']==32,'Clean reference coverage mismatch')
    check(all(summary['lifecycle'].values()),'Lifecycle check failed')
    fitted = load('audit/calibration.json')
    check(fitted['profile']=='synthetic-active-credential-v1' and fitted['frozen_before_test'] is True,
          'Invalid task calibration')
    check(fitted['threshold']==summary['threshold'],'Calibration threshold mismatch')
    for result in summary['boundary_stress'].values():
        for arm in ('strict','formatting_allowed'):
            check(result.get(arm+'_accepted',0)+result.get(arm+'_rejected',0)==24,
                  'Incomplete tokenizer stress coverage')
    boundary = load('results/boundary-contract-exact-audit.json')
    for field,rel in (('source_sha256','boundary_audit.py'),
                      ('contract_source_sha256','evidence_contract.py'),
                      ('review_entry_source_sha256','prepare_review.py')):
        check(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==boundary[field],
              'Supplemental audit source drift '+rel)
    layouts = boundary['layouts'].values()
    compatible = sum(row['token_compatible_variants'] for row in layouts)
    matched = sum(row['matches_approved_evidence'] for row in layouts)
    changed = sum(row.get('changed_evidence_rejected',0) for row in layouts)
    check(boundary['synthetic_variants']==120 and compatible==112 and matched==103
          and changed==9 and matched+changed==compatible,'Boundary contract counts mismatch')
    integrated = load('results/service-exact-smoke.json')
    for field,rel in (('source_sha256','integration_smoke.py'),
                      ('contract_source_sha256','evidence_contract.py')):
        check(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==integrated[field],
              'Integrated service source drift '+rel)
    check(integrated['exact_short_and_long_name_pairs']==2 and
          integrated['correct_qualification_decisions']==4 and
          integrated['changed_evidence_contracts_rejected']==2 and
          integrated['raw_backend_forwards']==4 and
          integrated['parameter_digest']==checkpoint,'Integrated service mismatch')
    initial = ROOT/'audit/initial-supplemental'
    for result,source in (('results/boundary-contract-audit.json','boundary_audit.py'),
                          ('results/service-smoke.json','integration_smoke.py')):
        check(hashlib.sha256((initial/source).read_bytes()).hexdigest()==load(result)['source_sha256'],
              'Initial supplemental source snapshot mismatch')
    short,long = load('results/demo-short.json'),load('results/demo-long.json')
    check(short['score']==long['score'] and short['evidence_ids_sha256']==long['evidence_ids_sha256']
          and short['evidence_tokens']==long['evidence_tokens']==75
          and (short['original_native_tokens'],long['original_native_tokens'])==(76,82)
          and short['original_input_ids_sha256']!=long['original_input_ids_sha256'],
          'Fresh-process CLI equality mismatch')
    manifest = ROOT/'audit/public-manifest.json'
    if manifest.exists():
        for rel,expected in load('audit/public-manifest.json')['sha256'].items():
            check(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,'Release drift '+rel)
    rows_path = ROOT/'local-private/credential-rows.json'
    if rows_path.exists():
        rows = load('local-private/credential-rows.json')
        check(len(rows)==96,'Credential record count mismatch')
        check(len({row['credential'] for row in rows})==6,'Wrong held-out credential count')
        groups = defaultdict(list)
        pairs = defaultdict(dict)
        for row in rows:
            check(row['calibrated_margin']==row['raw_margin']-fitted['threshold'],
                  'Readout transformation mismatch')
            groups[row['credential'],row['style'],row['present']].append(row)
            pairs[row['credential'],row['style'],row['name']][row['present']] = row
        credential = summary['credential']
        check(sum((row['calibrated_margin']>=0)==row['present'] for row in rows)
              ==credential['calibrated_correct'],'Calibrated accuracy mismatch')
        check(sum((row['raw_margin']>=0)==row['present'] for row in rows)
              ==credential['raw_correct'],'Original rubric accuracy mismatch')
        check(len(groups)==24 and all(len(block)==4 for block in groups.values()),
              'Wrong counterfactual grouping')
        check(sum(len({r['calibrated_margin'] for r in block})==1 for block in groups.values())
              ==credential['name_equal_conditions'],'Name equality mismatch')
        check(sum(pair[True]['calibrated_margin']>pair[False]['calibrated_margin']
                  for pair in pairs.values())==credential['correct_qualification_directions'],
              'Qualification direction mismatch')
        check(set(plan['test_names'])=={row['name'] for row in rows},'Name coverage mismatch')
    print('Frozen sources, prior conformance, lifecycle, exact evidence contract and credential aggregates verified.')


if __name__=='__main__':
    main()
