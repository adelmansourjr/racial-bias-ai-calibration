"""CPU validation of actual affine4bit Qwen LoRA captures and fresh probes."""
from pathlib import Path
import json
import numpy as np
from independent_qwen_validation import presence, run_probe
from qwen_audit import metrics, probability

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'qwen_qlora'

def main():
    cases=json.loads((ROOT/'cases.json').read_text())['cases']
    ref=json.loads((OUT/'results.json').read_text())
    arrays=np.load(OUT/'activations.npz')
    contrast=np.load(OUT/'contrast.npy').astype(float)
    cal=np.array([c['split']=='calibration' for c in cases])
    behavior_test=~cal
    idx=np.flatnonzero(behavior_test)
    probe_train=np.zeros(len(cases),bool);probe_train[idx[::2]]=True
    probe_test=np.zeros(len(cases),bool);probe_test[idx[1::2]]=True
    labels=np.array([c['evidence_label'] for c in cases])
    groups=[v['cue_group'] for v in cases[0]['variants']]
    cue=np.array([g=='white_associated' for g in groups],int)
    assert ref['trainable_parameters']==19456 and ref['steps']==24
    assert not np.any(cal&probe_train) and not np.any(cal&probe_test) and not np.any(probe_train&probe_test)
    cal_ids={c['case_id'] for c in cases if c['split']=='calibration'}
    assert all(row['case_id'] in cal_ids for row in ref['training'])
    out={'source':'actual Qwen MLX affine4-bit group64 frozen-base last-block q/v rank4 LoRA captures',
         'canonical_NF4_QLoRA':False,'trainable_parameters':19456,'steps':24,
         'baseline_comparison':'same quantized 4-bit model before versus after adapter; not original BF16',
         'probe_protocol':{'ridge':.01,'rbf_bandwidth':'median positive squared distances in probe-train states only',
                           'preprocessing':'probe-train mean and scalar RMS normalization','threshold':0,'numeric_zero_tolerance':1e-10,
                           'correction_train_count':8,'fresh_probe_train_count':8,'fresh_probe_test_count':8,
                           'selection_status':'exploratory protocol following earlier projection-probe diagnostic; no prospective population certificate'},
         'probe_train_ids':[c['case_id'] for i,c in enumerate(cases) if probe_train[i]],
         'probe_test_ids':[c['case_id'] for i,c in enumerate(cases) if probe_test[i]],
         'methods':{},'aggregate_reconstruction_max_error':0.0}
    for key,name in [('before','quantized_base'),('after','quantized_base_plus_lora')]:
        h=arrays[key].astype(float);assert h.shape==(24,2,4,1536)
        tasks={}
        for ti,task in enumerate(['shortlist','evidence']):
            score=probability(h[:,ti],contrast)
            m,v=metrics(h[behavior_test,ti],score[behavior_test],groups,labels[behavior_test] if task=='evidence' else None)
            error=max(abs(m[k]-value) for k,value in ref['methods'][name][task].items())
            assert error<1e-10,(name,task,error)
            out['aggregate_reconstruction_max_error']=max(out['aggregate_reconstruction_max_error'],error)
            pmat,pv=presence(h[behavior_test,ti],cue)
            d=score[behavior_test]>=.5
            m.update(pmat)
            m.update({'variant_acceptance_rate':float(d.mean()),'accepted_variant_count':int(d.sum()),
                      'decision_total':int(d.size),'score_min':float(score[behavior_test].min()),'score_max':float(score[behavior_test].max()),
                      'fresh_probes':{kind:run_probe(h[:,ti],probe_train,cue,kind,evaluation=probe_test) for kind in ['linear_ridge','rbf_kernel_ridge']},
                      'calibration_to_test_probes':{kind:run_probe(h[:,ti],cal,cue,kind) for kind in ['linear_ridge','rbf_kernel_ridge']}})
            if task=='evidence':
                dc=score[behavior_test].mean(1)>=.5
                m['evidence_case_tpr']=float(dc[labels[behavior_test]==1].mean())
                m['evidence_case_tnr']=float((~dc[labels[behavior_test]==0]).mean())
            tasks[task]=m
        out['methods'][name]=tasks
    (OUT/'probe-results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({name:{t:{'presence':m['normalized_presence'],'acceptance':m['variant_acceptance_rate'],'flips':m['any_variant_decision_flip_fraction'],'probes':{p:v['test_balanced_accuracy'] for p,v in m['fresh_probes'].items()}} for t,m in tasks.items()} for name,tasks in out['methods'].items()},indent=2))

if __name__=='__main__':main()
