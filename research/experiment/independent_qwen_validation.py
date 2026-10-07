"""Independent CPU audit of actual saved Qwen model captures; no model rerun.

Protected-cue probes fit only on eight calibration applicants, with fixed
regularization and fixed kernel rule. Held-out results are descriptive.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'qwen'


def fit_basis(h, calibration):
    differences = (h[calibration, :, 1:] - h[calibration, :, :1]).reshape(-1, h.shape[-1])
    _, s, vt = np.linalg.svd(differences, full_matrices=False)
    tol = max(differences.shape) * np.finfo(float).eps * s[0]
    return vt[s > tol].T


def project(h, u):
    return h - (h @ u) @ u.T


def presence(h, groups):
    a = groups == 1
    b = ~a
    v = h[:, a].mean((0, 1)) - h[:, b].mean((0, 1))
    numerator = float(np.linalg.norm(v))
    denominator = float(np.linalg.norm(h, axis=-1).mean())
    cross = np.linalg.norm(h[:, a, None] - h[:, None, b], axis=-1)
    return {
        'numerator': numerator,
        'mean_individual_norm': denominator,
        'normalized_presence': numerator / denominator if denominator > 0 else None,
        'paired_cross_group_mean_distance': float(cross.mean()),
        'paired_cross_group_normalized_distance': float(cross.mean()) / denominator if denominator > 0 else None,
    }, v


def sqdist(x, y):
    return np.maximum(np.sum(x*x, 1)[:, None] + np.sum(y*y, 1)[None] - 2*x@y.T, 0)


def run_probe(x, calibration, labels, kind, evaluation=None):
    if evaluation is None:
        evaluation=~calibration
    train = x[calibration].reshape(-1, x.shape[-1])
    test = x[evaluation].reshape(-1, x.shape[-1])
    y_train = np.tile(labels, int(calibration.sum())).astype(float) * 2 - 1
    y_test = np.tile(labels, int(evaluation.sum())).astype(float) * 2 - 1
    mu = train.mean(0)
    train = train - mu
    test = test - mu
    scale = float(np.sqrt(np.mean(np.sum(train*train, axis=1))))
    if scale > 0:
        train /= scale
        test /= scale
    ridge = .01
    if kind == 'linear_ridge':
        k_train = train @ train.T
        k_test = test @ train.T
        bandwidth_squared = None
    else:
        distances = sqdist(train, train)
        positive = distances[np.triu_indices(len(train), 1)]
        positive = positive[positive > 1e-12]
        bandwidth_squared = float(np.median(positive)) if len(positive) else 1.0
        k_train = np.exp(-distances / (2*bandwidth_squared))
        k_test = np.exp(-sqdist(test, train) / (2*bandwidth_squared))
    coefficients = np.linalg.solve(k_train + ridge*np.eye(len(train)), y_train)
    train_margin = k_train @ coefficients
    test_margin = k_test @ coefficients
    # Erased calibration labels yield exact zero scores mathematically.
    # Do not classify roundoff-sized residuals as learned protected information.
    tolerance = 1e-10
    def predictions(m):
        return np.where(np.abs(m) <= tolerance, 1, np.where(m > 0, 1, -1))
    pred = predictions(test_margin)
    per_case = (pred == y_test).reshape(-1, len(labels)).mean(1)
    a, b = y_test == 1, y_test == -1
    return {
        'test_balanced_accuracy': float(.5*((pred[a]==1).mean() + (pred[b]==-1).mean())),
        'test_accuracy': float((pred == y_test).mean()),
        'calibration_accuracy': float((predictions(train_margin) == y_train).mean()),
        'test_per_applicant_accuracy': per_case.tolist(),
        'calibration_margin_max_abs': float(np.abs(train_margin).max()),
        'test_margin_max_abs': float(np.abs(test_margin).max()),
        'training_center_norm': float(np.linalg.norm(mu)),
        'training_rms_centered_state_norm': scale,
        'ridge': ridge,
        'rbf_bandwidth_squared': bandwidth_squared,
        'numeric_zero_margin_tolerance': tolerance,
    }


def main():
    cases = json.loads((ROOT/'cases.json').read_text())['cases']
    rows = json.loads((OUT/'scores.json').read_text())
    reference = json.loads((OUT/'results.json').read_text())
    data = np.load(OUT/'activations.npz')
    h = data['hidden'].astype(float)
    layers = data['layer_states'].astype(float)
    calibration = np.array([c['split'] == 'calibration' for c in cases])
    cue = np.array([v['cue_group']=='white_associated' for v in cases[0]['variants']], dtype=int)
    assert h.shape == (24, 2, 4, 1536)
    assert len({c['resume_id'] for c in cases}) == 24
    assert int(calibration.sum()) == 8 and int((~calibration).sum()) == 16
    assert all([v['cue_group']=='white_associated' for v in c['variants']] == cue.tolist() for c in cases)
    lookup = {(r['case_id'], r['task'], r['name']): r for r in rows}
    assert len(lookup) == 24*2*4
    labels = np.array([c['evidence_label'] for c in cases])[~calibration]
    heldout_indices=np.flatnonzero(~calibration)
    fresh_train=np.zeros(len(cases),dtype=bool)
    fresh_test=np.zeros(len(cases),dtype=bool)
    fresh_train[heldout_indices[::2]]=True
    fresh_test[heldout_indices[1::2]]=True
    assert not np.any(calibration & (fresh_train|fresh_test))
    out = {
        'source': 'Actual captured Qwen2.5-1.5B-Instruct states; CPU postprocessing only',
        'model': reference['model'],
        'calibration_applicants': 8,
        'test_applicants': 16,
        'variants_per_applicant': 4,
        'names_held_out': False,
        'label': 'experimental White-associated versus Black-associated name treatment; not actual race',
        'probe_protocol': {
            'features': 'one last-token state per applicant/task/name',
            'fit_split': 'eight calibration applicants; no held-out applicant state or label used to fit',
            'preprocessing': 'calibration mean subtraction and one scalar calibration RMS normalization',
            'ridge': .01,
            'rbf_rule': 'bandwidth squared is median positive squared calibration distance; no test tuning',
            'classification': 'fixed zero threshold with 1e-10 numeric zero tolerance and White-associated constant tie',
            'inference': 'dual 32-by-32 solve; no 1536-dimensional inverse',
            'limitations': 'two finite decoder families; empirical results on a small convenience sample',
            'projection_probe_training_limitation': 'Projection fit and probe training reuse the same calibration applicants; their projected cue states are identical, so chance probes do not show that held-out states lack protected information. Fresh probe-training applicants are needed.',
        },
        'centered_projection_status': 'Exploratory correction chosen after observing uncentered held-out acceptance collapse; no untouched confirmatory claim',
        'fresh_probe_split': {
            'status':'diagnostic selected after observing original projection-probe training vacuity; not prospectively preregistered',
            'correction_fit_case_ids':[c['case_id'] for i,c in enumerate(cases) if calibration[i]],
            'probe_train_case_ids':[c['case_id'] for i,c in enumerate(cases) if fresh_train[i]],
            'probe_test_case_ids':[c['case_id'] for i,c in enumerate(cases) if fresh_test[i]],
            'train_rule':'alternating behavioral-heldout applicants, beginning with first',
            'test_rule':'remaining alternating behavioral-heldout applicants',
            'counts':{'correction_fit':8,'probe_train':8,'probe_test':8},
            'hyperparameters':'same fixed ridge and kernel rule; preprocessing uses fresh probe-train states only',
        },
        'generation_scope': 'conditional A/B next-token readout only; not unrestricted sequence-generation fairness',
        'final': {},
        'layers': {},
        'independent_score_checks': {},
        'readout_reconstruction_max_absolute_error': {},
    }
    tasks = ['shortlist', 'evidence']
    bases = {}
    for layer_name, states in [('final',h)] + [(str(li),layers[:,:,:,index]) for index,li in enumerate([1,8,16,24,28])]:
        basis = fit_basis(states, calibration)
        bases[layer_name] = basis
        variants = {
            'raw': states,
            'projection': project(states,basis),
            'orbit_mean': np.broadcast_to(states.mean(2,keepdims=True),states.shape).copy(),
        }
        calibration_task_mean=states[calibration].mean(axis=(0,2),keepdims=True)
        variants['centered_projection_exploratory']=calibration_task_mean+project(states-calibration_task_mean,basis)
        residual = float(np.linalg.norm(variants['projection'][calibration,:,1:] - variants['projection'][calibration,:,:1],axis=-1).max())
        block = {'projection_rank': int(basis.shape[1]),'calibration_projection_max_residual':residual,'tasks':{}}
        for ti,task in enumerate(tasks):
            block['tasks'][task] = {}
            for name,transformed in variants.items():
                metrics, vector = presence(transformed[~calibration,ti],cue)
                _,raw_vector = presence(states[~calibration,ti],cue)
                denom = float(np.linalg.norm(vector)*np.linalg.norm(raw_vector))
                metrics['cosine_to_raw'] = float(np.clip(vector@raw_vector/denom,-1,1)) if denom > 1e-12 else None
                if metrics['normalized_presence'] is not None:
                    assert 0 <= metrics['normalized_presence'] <= 2+1e-10
                metrics['probes'] = {kind:run_probe(transformed[:,ti],calibration,cue,kind) for kind in ['linear_ridge','rbf_kernel_ridge']}
                metrics['fresh_split_probes'] = {kind:run_probe(transformed[:,ti],fresh_train,cue,kind,evaluation=fresh_test) for kind in ['linear_ridge','rbf_kernel_ridge']}
                block['tasks'][task][name] = metrics
        if layer_name == 'final':
            out['final'] = block
        else:
            out['layers'][layer_name] = block
    for task in tasks:
        ti = tasks.index(task)
        selected = [c for c in cases if c['split']=='test']
        probabilities = {key:np.array([[lookup[(c['case_id'],task,v['name'])][key] for v in c['variants']] for c in selected]) for key in ['raw_p_A','projected_p_A','activation_average_p_A','score_average_p_A']}
        mu=h[calibration].mean(axis=(0,2),keepdims=True)
        centered=mu+project(h-mu,bases['final'])
        logits=centered[~calibration,ti]@data['contrast'].astype(float)
        probabilities['centered_projected_p_A_exploratory']=1/(1+np.exp(-np.clip(logits,-700,700)))
        out['independent_score_checks'][task] = {}
        for key,p in probabilities.items():
            d = p>=.5
            m = {'mean_score':float(p.mean()), 'score_min':float(p.min()), 'score_max':float(p.max()),
                 'variant_acceptance_rate':float(d.mean()),'any_name_flip_applicants':int((np.ptp(d.astype(int),axis=1)>0).sum()),
                 'any_name_flip_fraction':float((np.ptp(d.astype(int),axis=1)>0).mean()),
                 'group_acceptance_gap':float(abs(d[:,cue==1].mean()-d[:,cue==0].mean())),
                 'mean_name_score_range':float(np.ptp(p,axis=1).mean())}
            if task=='evidence':
                d_case=p.mean(1)>=.5
                m.update({'evidence_positive_applicants':int(labels.sum()),'evidence_negative_applicants':int((1-labels).sum()),
                          'evidence_case_tpr':float(d_case[labels==1].mean()),'evidence_case_tnr':float((~d_case[labels==0]).mean()),
                          'evidence_case_balanced_accuracy':float(.5*(d_case[labels==1].mean()+(~d_case[labels==0]).mean()))})
            out['independent_score_checks'][task][key]=m
        for key,method in [('raw_p_A','raw'),('projected_p_A','learned_full_difference_projection'),('activation_average_p_A','fixed_orbit_activation_average')]:
            assert abs(out['independent_score_checks'][task][key]['mean_score']-reference['methods'][method][task]['mean_score'])<1e-12
            name={'raw':'raw','learned_full_difference_projection':'projection','fixed_orbit_activation_average':'orbit_mean'}[method]
            assert abs(out['final']['tasks'][task][name]['normalized_presence']-reference['methods'][method][task]['thesis_normalized_presence'])<1e-10
            reconstructed_state={'raw_p_A':h,'projected_p_A':project(h,bases['final']),'activation_average_p_A':np.broadcast_to(h.mean(2,keepdims=True),h.shape)}[key]
            reconstructed_logit=reconstructed_state[~calibration,ti]@data['contrast'].astype(float)
            reconstructed_score=1/(1+np.exp(-np.clip(reconstructed_logit,-700,700)))
            score_error=float(np.abs(reconstructed_score-probabilities[key]).max())
            assert score_error<1e-10
            out['readout_reconstruction_max_absolute_error'][task+'/'+key]=score_error
    u=bases['final']
    saved_u=data['basis'].astype(float)
    out['precision_checks']={
        'saved_basis_float32_orthogonality_max_error':float(np.abs(saved_u.T@saved_u-np.eye(saved_u.shape[1])).max()),
        'recomputed_basis_float64_orthogonality_max_error':float(np.abs(u.T@u-np.eye(u.shape[1])).max()),
        'saved_basis_calibration_residual':float(np.linalg.norm(project(h,saved_u)[calibration,:,1:]-project(h,saved_u)[calibration,:,:1],axis=-1).max()),
        'exactly_zero_orbit_presence':all(out['final']['tasks'][t]['orbit_mean']['normalized_presence']==0 for t in tasks),
        'zero_orbit_cosine_is_undefined':all(out['final']['tasks'][t]['orbit_mean']['cosine_to_raw'] is None for t in tasks),
    }
    out['conditional_zero_failure_bound_if_iid_16_clusters'] = 1-.05**(1/16)
    (OUT/'probe-results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'final_probe_accuracy': {t:{m:{k:v['test_balanced_accuracy'] for k,v in x['probes'].items()} for m,x in out['final']['tasks'][t].items()} for t in tasks}, 'score_checks':out['independent_score_checks'],'precision':out['precision_checks']},indent=2))


if __name__ == '__main__':
    main()
