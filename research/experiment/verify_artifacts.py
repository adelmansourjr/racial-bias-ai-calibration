"""CPU-only checks of preserved inputs, actual captures and final audit artifacts.

Writes machine-verification.json; never reruns models or alters their results.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from independent_qwen_validation import fit_basis,project
from qwen_audit import probability,metrics

ROOT=Path(__file__).resolve().parent
CHECKS={}

def check(name,condition,evidence=None):
    CHECKS[name]={'passed':bool(condition),'evidence':evidence}
    if not condition:
        raise AssertionError(name)

def finite(value,path='root'):
    if isinstance(value,float) and not math.isfinite(value):
        raise AssertionError('Nonfinite JSON numeric value: '+path)
    if isinstance(value,dict):
        for key,v in value.items():finite(v,path+'.'+key)
    if isinstance(value,list):
        for i,v in enumerate(value):finite(v,path+f'[{i}]')

def main():
    doc=json.loads((ROOT/'cases.json').read_text());cases=doc['cases']
    case_by_id={c['case_id']:c for c in cases}
    check('unique_case_and_resume_ids',len(case_by_id)==len(cases)==len({c['resume_id'] for c in cases}),{'cases':len(cases)})
    names=json.loads((ROOT/'name_bank.json').read_text())
    for c in cases:
        check('qualifications_preserved_'+c['case_id'],all(v['text']==f"NAME: {v['name']}\n{c['resume_text']}" for v in c['variants']))
        check('name_family_fixed_'+c['case_id'],[{k:v[k] for k in ('name','cue_group')} for v in c['variants']]==names)
        check('literal_evidence_label_'+c['case_id'],c['evidence_label']==int(c['requirement_keyword'].lower() in c['resume_text'].lower()))
        check('hiring_label_not_fabricated_'+c['case_id'],c['qualification_label'] is None)
    sources={};missing_source_checks=[]
    manifest_path=ROOT.parent/'data'/'source-manifest.json'
    if not manifest_path.exists():
        manifest_path=ROOT.parents[1]/'source-manifest.json'
    source_manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    manifest_sources={s['filename']:s for s in source_manifest.get('source_files',[])} if source_manifest else {}
    for filename,expected in doc['metadata']['source_file_sha256'].items():
        source=ROOT.parent/'data'/filename
        if source.exists():
            actual=hashlib.sha256(source.read_bytes()).hexdigest()
            check('source_sha256_'+filename,actual==expected,{'sha256':actual})
            sources[filename]=str(source)
        else:
            missing_source_checks.append({'filename':filename,'expected_sha256':expected,'status':'not locally verified; raw source file absent','reason':'Full resume CSV may be omitted from portable package; preserved excerpts remain in cases.json.'})
        if filename in manifest_sources:
            check('source_manifest_hash_matches_case_metadata_'+filename,manifest_sources[filename]['sha256']==expected)
    check('jobs_snapshot_present_and_hashed','jobs-first-rows.json' in sources,sources)
    if source_manifest:
        for item in source_manifest.get('exact_input_files',[]):
            source=ROOT.parents[1]/item['path']
            check('package_exact_input_hash_'+item['path'],source.exists() and hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256'])
    for manifest_path in [ROOT/'ranking/pool-manifest.json',ROOT/'ranking_secondary_full_pools/pool-manifest.json']:
        if not manifest_path.exists():continue
        manifest=json.loads(manifest_path.read_text());seed=manifest['seed'];all_ids=[]
        for pool in manifest['pools']:
            ids=[c['candidate_id'] for c in pool['candidates']];all_ids+=ids
            check(f'{manifest_path.parent.name}_unique_pool_{pool["job_id"]}',len(ids)==len(set(ids)))
            for c in pool['candidates']:
                expected='cand-'+hashlib.sha256((seed+':'+c['case_id']).encode()).hexdigest()[:10]
                check(f'{manifest_path.parent.name}_opaque_id_{c["case_id"]}',c['candidate_id']==expected and str(case_by_id[c['case_id']]['job_id'])==str(pool['job_id']))
            check(f'{manifest_path.parent.name}_valid_top_k_{pool["job_id"]}',0<manifest['top_k']<=len(ids))
        check(manifest_path.parent.name+'_global_id_uniqueness',len(all_ids)==len(set(all_ids)))
    split=json.loads((ROOT/'results/qwen/probe-results.json').read_text())['fresh_probe_split']
    fit=set(split['correction_fit_case_ids']);train=set(split['probe_train_case_ids']);test=set(split['probe_test_case_ids'])
    check('three_way_applicant_split_disjoint',len(fit)==len(train)==len(test)==8 and not(fit&train or fit&test or train&test))
    score_rows=json.loads((ROOT/'results/qwen/scores.json').read_text())
    old=np.load(ROOT/'results/qwen/activations.npz');h=old['hidden'].astype(float);contrast=old['contrast'].astype(float)
    cal=np.array([c['split']=='calibration' for c in cases]);u=fit_basis(h,cal)
    variants={'raw_p_A':h,'projected_p_A':project(h,u),'activation_average_p_A':np.broadcast_to(h.mean(2,keepdims=True),h.shape)}
    lookup={(r['case_id'],r['task'],r['name']):r for r in score_rows}
    max_error=0.;least_constant_error=0.
    for ci,c in enumerate(cases):
        for ti,task in enumerate(['shortlist','evidence']):
            raw=probability(h[ci,ti],contrast);mean=raw.mean()
            for vi,v in enumerate(c['variants']):
                row=lookup[(c['case_id'],task,v['name'])]
                for field,state in variants.items():max_error=max(max_error,abs(float(probability(state[ci,ti,vi],contrast))-row[field]))
                max_error=max(max_error,abs(row['score_average_p_A']-mean))
            # Weighted least-constant identity at several nonoptimal replacements.
            for constant in [0.,.5,1.,float(mean)+.123]:
                lhs=np.mean((raw-constant)**2)
                rhs=np.mean((raw-mean)**2)+(constant-mean)**2
                least_constant_error=max(least_constant_error,abs(lhs-rhs))
    check('qwen_saved_readout_reconstruction',max_error<1e-10,{'maximum_absolute_error':max_error})
    check('score_mean_least_constant_squared_error_identity',least_constant_error<1e-12,{'maximum_absolute_error':least_constant_error})
    qa=np.load(ROOT/'results/qwen_qlora/activations.npz');qc=np.load(ROOT/'results/qwen_qlora/contrast.npy').astype(float)
    qr=json.loads((ROOT/'results/qwen_qlora/results.json').read_text());qe=0.;labels=np.array([c['evidence_label'] for c in cases]);groups=[n['cue_group']for n in names]
    for array,name in [('before','quantized_base'),('after','quantized_base_plus_lora')]:
        states=qa[array].astype(float)
        for ti,task in enumerate(['shortlist','evidence']):
            m,_=metrics(states[~cal,ti],probability(states[~cal,ti],qc),groups,labels[~cal] if task=='evidence' else None)
            qe=max(qe,max(abs(m[k]-v) for k,v in qr['methods'][name][task].items()))
    check('same_quantized_baseline_lora_reconstruction',qe<1e-10,{'maximum_absolute_error':qe,'canonical_NF4':qr['canonical_QLoRA_NF4']})
    token_report={}
    for model in ['qwen','clef']:
        result=json.loads((ROOT/f'heldout_names/{model}-results.json').read_text())
        newnames={n['name'] for n in result['names']}
        check(model+'_names_not_in_fit_bank',not newnames&{n['name'] for n in names},sorted(newnames))
        check(model+'_newname_cases_equal_final_probe_cases',set(result['cases'])==test)
        check(model+'_newname_sources_documented',bool(result['name_source']),result['name_source'])
        by_case={}
        for row in result['records']:
            by_case.setdefault(row['case_id'],{}).setdefault(row['task'],{})[row['name']]=row['tokens']
        unequal=sum(len(set(x.values()))>1 for tasks in by_case.values()for x in tasks.values())
        check(model+'_newname_unequal_lengths_documented',unequal>0,{'case_task_pairs_with_unequal_lengths':unequal})
        token_report[model]=by_case
        for field in ['fixed_bank_score_mean','fixed_bank_activation_mean_score']:
            ranges=[]
            for case_id in result['cases']:
                for task in ['shortlist','evidence']:
                    ps=[r[field]for r in result['records']if r['case_id']==case_id and r['task']==task]
                    ranges.append(float(np.ptp(ps)))
            check(model+'_'+field+'_newcue_exact_invariance',max(ranges)==0.,{'maximum_pair_range':max(ranges),'interpretation':'canonical original fixed-bank output; raw branches not erased'})
        if model=='qwen':
            ps=[]
            for case_id in result['cases']:
                for task in ['shortlist','evidence']:
                    ps.append(float(np.ptp([r['projection_score']for r in result['records']if r['case_id']==case_id and r['task']==task])))
            check('qwen_projection_unseen_cue_failure_retained',max(ps)>1e-6,{'maximum_pair_score_range':max(ps)})
            newh=np.load(ROOT/'heldout_names/qwen-activations.npz')['hidden'].astype(float)
            newz=project(newh,u)
            residual=float(np.linalg.norm(newz[:,:,0]-newz[:,:,1],axis=-1).max())
            check('qwen_actual_newcue_projected_state_residual_nonzero',residual>1e-6,{'maximum_pair_hidden_distance':residual})
            newerror=0.;mu=h[cal].mean(axis=(0,2))
            for ci,case_id in enumerate(result['cases']):
                for ti,task in enumerate(['shortlist','evidence']):
                    for vi,name in enumerate(result['names']):
                        row=next(r for r in result['records']if r['case_id']==case_id and r['task']==task and r['name']==name['name'])
                        centered=mu[ti]+project(newh[ci,ti,vi]-mu[ti],u)
                        for field,state in [('raw_score',newh[ci,ti,vi]),('projection_score',newz[ci,ti,vi]),('centered_projection_score_exploratory',centered)]:
                            newerror=max(newerror,abs(row[field]-float(probability(state,contrast))))
            check('qwen_actual_newcue_readout_reconstructed',newerror<1e-10,{'maximum_absolute_error':newerror})
        else:
            check('clef_unaligned_newcue_projection_unavailable',all(r['projection_score'] is None and r['projection_reason'] for r in result['records']))
    json_count=0;jsonl_count=0
    for path in ROOT.rglob('*.json'):
        if path.name=='machine-verification.json':continue
        finite(json.loads(path.read_text()),str(path));json_count+=1
    for path in ROOT.rglob('*.jsonl'):
        for line in path.read_text().splitlines():
            if line.strip():finite(json.loads(line),str(path));jsonl_count+=1
    check('all_saved_json_numbers_finite_nulls_allowed',True,{'json_files':json_count,'jsonl_records':jsonl_count})
    array_count=0
    for path in ROOT.rglob('*.npz'):
        with np.load(path) as data:
            for key in data.files:
                x=data[key]
                if np.issubdtype(x.dtype,np.number):check('finite_array_'+str(path.relative_to(ROOT))+'_'+key,np.isfinite(x).all());array_count+=1
    output={'status':'all available CPU artifact checks passed' if missing_source_checks else 'all CPU artifact checks passed','no_model_inference_or_training_rerun':True,'checks':CHECKS,'new_name_token_lengths':token_report,
            'missing_source_checks':missing_source_checks,'raw_source_availability':{'locally_hashed':sources,'all_manifest_raw_sources_locally_verified':not missing_source_checks},
            'limits':['Passing artifact integrity does not prove hiring suitability or population fairness.','Name-pool census covers declared measured inputs only.','Clef new-name projection is unavailable because sequence alignment differs.']}
    (ROOT/'machine-verification.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':output['status'],'check_count':len(CHECKS),'json_files':json_count,'numeric_arrays':array_count,'qwen_readout_max_error':max_error,'qlora_reconstruction_max_error':qe},indent=2))

if __name__=='__main__':main()
