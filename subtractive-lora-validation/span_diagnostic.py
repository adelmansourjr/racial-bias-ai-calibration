"""Check whether exact cancellation on earlier fitting states transfers to earlier held-out CVs."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def contrasts(states, rows, name_key='context_id'):
    centered = []
    for context in sorted({row[name_key] for row in rows}):
        ix = [i for i, row in enumerate(rows) if row[name_key]==context]
        block = states[ix].astype(np.float64)
        centered.extend(block-block.mean(0))
    return np.asarray(centered)


def main():
    before = ROOT.parent/'subtractive-lora-pilot-v2'
    after = ROOT.parent/'subtractive-lora-score-aware'
    rows = json.loads((before/'results/raw-rows.json').read_text())
    all_states = np.load(before/'local-private/raw-decision-states.npy',allow_pickle=False)
    fit = [i for i,row in enumerate(rows) if row['split']=='fit']
    fit_rows = [rows[i] for i in fit]
    fit_states = all_states[fit].astype(np.float64)
    test_rows = json.loads((after/'results/final.json').read_text())['rows']
    with np.load(after/'local-private/confirmation-states.npz',allow_pickle=False) as archive:
        test_states = archive['raw'].astype(np.float64)
    if len(test_rows)!=len(test_states):
        raise ValueError('Rows and states differ')
    if {row['cv_cluster'] for row in fit_rows}&{row['cv_cluster'] for row in test_rows}:
        raise ValueError('Fit and held-out CV overlap')
    delta = contrasts(fit_states,fit_rows)
    unseen = contrasts(test_states,test_rows)
    _, spectrum, axes = np.linalg.svd(delta,full_matrices=False)
    # Standard double-precision SVD rank tolerance for this fixed matrix.
    cutoff = spectrum[0]*max(delta.shape)*np.finfo(np.float64).eps
    basis = axes[spectrum>cutoff].T
    residual = lambda values: values-(values@basis)@basis.T
    fit_residual = residual(delta)
    heldout_residual = residual(unseen)

    # Enforce exact fitting margins across names and preserve every fitting context mean.
    # All quantities here are diagnostic replays of cached model states and scores.
    mean = fit_states.mean(0)
    d_margin = []
    mean_rows = []
    for context in sorted({row['context_id'] for row in fit_rows}):
        ix = [i for i,row in enumerate(fit_rows) if row['context_id']==context]
        margins = np.array([fit_rows[i]['raw_margin'] for i in ix])
        d_margin.extend(margins-margins.mean())
        mean_rows.append(fit_states[ix].mean(0)-mean)
    matrix = np.vstack([delta,mean_rows])
    correction, _, rank, _ = np.linalg.lstsq(matrix,np.r_[d_margin,np.zeros(len(mean_rows))],rcond=None)
    fit_constraint_error = np.max(np.abs(matrix@correction-np.r_[d_margin,np.zeros(len(mean_rows))]))
    old = np.array([row['raw_margin'] for row in test_rows])
    new = old-(test_states-mean)@correction
    groups = [[i for i,row in enumerate(test_rows) if row['context_id']==context]
              for context in sorted({row['context_id'] for row in test_rows})]
    scores = lambda x: {'mean_logit_range':float(np.mean([np.ptp(x[ix]) for ix in groups])),
                        'flip_contexts':int(sum(min(x[ix])<0<=max(x[ix]) for ix in groups))}
    result = {
        'scope':'Retrospective cached-state geometry diagnostic from earlier studies, separate from the frozen current test.',
        'fit_CVs':len({row['cv_cluster'] for row in fit_rows}),
        'heldout_CVs':len({row['cv_cluster'] for row in test_rows}),
        'span_rank':basis.shape[1],
        'maximum_fit_centered_state_residual_L2':float(np.max(np.linalg.norm(fit_residual,axis=1))),
        'heldout_contrast_energy_outside_fitted_span':float(np.linalg.norm(heldout_residual)**2/np.linalg.norm(unseen)**2),
        'hard_fit_margin_constraints_rank':int(rank),
        'maximum_fit_margin_constraint_error':float(fit_constraint_error),
        'heldout_native':scores(old),'heldout_hard_fit_cancellation':scores(new),
        'proof_scope':'Finite fitted differences only. Numerical least squares and an affine margin replay.'
    }
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
