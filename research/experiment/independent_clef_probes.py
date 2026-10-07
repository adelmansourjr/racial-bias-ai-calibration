"""Fresh disjoint CPU probes of actual Clef saved activation summaries."""
from pathlib import Path
import json
import numpy as np
from independent_qwen_validation import presence,run_probe

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'clef_results'

def main():
    cases=json.loads((ROOT/'cases.json').read_text())['cases']
    metadata=json.loads((OUT/'activation-summary-metadata.json').read_text())
    data=np.load(OUT/'activation-summaries.npz')
    expected=[(c['case_id'],v['name']) for c in cases for v in c['variants']]
    lookup={(m['case_id'],m['name']):i for i,m in enumerate(metadata)}
    assert set(expected)==set(lookup)
    order=np.array([lookup[key] for key in expected])
    cal=np.array([c['split']=='calibration' for c in cases])
    idx=np.flatnonzero(~cal);train=np.zeros(24,bool);test=np.zeros(24,bool)
    train[idx[::2]]=True;test[idx[1::2]]=True
    cue=np.array([v['cue_group']=='white_associated' for v in cases[0]['variants']],int)
    out={'source':'actual Clef-flash9B affine8bit final sequence summary captures; CPU probe only',
         'scope':'pooled and last-token summaries do not equal complete trained-head input; raw full tensor intervention was evaluated separately',
         'splits':{'correction_fit':8,'probe_train':8,'probe_test':8},
         'protocol':'same fixed ridge=.01, train-mean/RMS scaling, train-median RBF bandwidth, zero threshold; no test tuning',
         'status':'exploratory diagnostic; eight probe-test applicants, four reused names, no population certificate',
         'results':{}}
    for mode in ['raw','projected','orbit_mean']:
        out['results'][mode]={}
        for pooling in ['pooled','last']:
            h=data[f'{mode}_{pooling}'][order].astype(float).reshape(24,4,4096)
            m,_=presence(h[~cal],cue)
            m['fresh_probes']={kind:run_probe(h,train,cue,kind,evaluation=test) for kind in ['linear_ridge','rbf_kernel_ridge']}
            out['results'][mode][pooling]=m
    (OUT/'fresh-probe-results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({m:{p:{k:v['test_balanced_accuracy'] for k,v in r['fresh_probes'].items()} for p,r in pools.items()} for m,pools in out['results'].items()},indent=2))

if __name__=='__main__':main()
