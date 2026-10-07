"""Analyze an additional affine repair from the actual saved Qwen activations.

Chosen after observing uncentered projection collapse: exploratory, not a new
independent confirmation set. All fitted quantities still use calibration only.
"""
from pathlib import Path
import json
import numpy as np
from qwen_audit import projector_basis, project, probability, metrics

root=Path(__file__).resolve().parent
out=root/'results'/'qwen'
data=np.load(out/'activations.npz')
h=data['hidden'].astype(np.float64)
cases=json.loads((root/'cases.json').read_text())['cases']
cal=np.array([c['split']=='calibration' for c in cases]); test=~cal
d=(h[cal,:,1:]-h[cal,:,:1]).reshape(-1,h.shape[-1])
basis,_=projector_basis(d)
mu=h[cal].mean(axis=(0,2),keepdims=True)
centered=mu+project(h-mu,basis)
assert np.max(np.linalg.norm(centered[cal,:,1:]-centered[cal,:,:1],axis=-1))<1e-8
labels=np.array([c['evidence_label'] for c in cases])
groups=[v['cue_group'] for v in cases[0]['variants']]
contrast=data['contrast'].astype(np.float64)
results={'status':'exploratory repair after original held-out collapse',
         'formula':'z=mu_task+(I-UU^T)(h-mu_task)',
         'calibration_only_mean_and_basis':True,'rank':basis.shape[1], 'test':{}}
for ti,task in enumerate(['shortlist','evidence']):
    scores=probability(centered[test,ti],contrast)
    stats,_=metrics(centered[test,ti],scores,groups,labels[test] if ti==1 else None)
    stats['total_acceptance_fraction']=float((scores>=.5).mean())
    stats['paired_rho_to_reference_mean']=float(np.linalg.norm(centered[test,ti,1:]-centered[test,ti,:1],axis=-1).mean()/np.linalg.norm(centered[test,ti],axis=-1).mean())
    results['test'][task]=stats
np.savez_compressed(out/'centered-repair.npz',hidden=centered.astype(np.float32),mean=mu.astype(np.float32))
(out/'centered-results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
