"""Quantify the unavoidable native-fidelity cost of replacing varying scores by one score."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parent


def audit(values,constant):
    values=tuple(values)
    if not values:raise ValueError('Nonempty score family required')
    mean=statistics.mean(values)
    variance=statistics.mean((v-mean)**2 for v in values)
    mse=statistics.mean((v-constant)**2 for v in values)
    positives=sum(v>=0 for v in values)
    return {'range':max(values)-min(values),'minimum_uniform_error':(max(values)-min(values))/2,
            'minimum_mse':variance,'measured_mse':mse,'constant_offset_squared':(constant-mean)**2,
            'minimum_binary_decision_changes':min(positives,len(values)-positives),
            'measured_binary_decision_changes':sum((v>=0)!=(constant>=0) for v in values)}


def main():
    rows=json.loads((ROOT/'results/gpu/synthetic-rows.json').read_text())
    groups=defaultdict(list)
    for r in rows:groups[r['context']].append(r)
    records=[]
    for cid,block in groups.items():
        accepted=[r for r in block if r['compiled_admitted']]
        if len(accepted)!=len(block):raise ValueError('Full-family admission required for this audit')
        constants={r['compiled_margin'] for r in accepted}
        if len(constants)!=1:raise ValueError('Not a name-invariant score')
        records.append({'context':cid,**audit([r['native_margin'] for r in block],next(iter(constants)))})
    result={'contexts':len(records),'native_nonconstant_contexts':sum(r['range']>0 for r in records),
            'minimum_binary_changes_any_invariant_rule':sum(r['minimum_binary_decision_changes'] for r in records),
            'measured_binary_changes':sum(r['measured_binary_decision_changes'] for r in records),
            'mean_best_possible_uniform_logit_error':statistics.mean(r['minimum_uniform_error'] for r in records),
            'mse_decomposition_verified':all(abs(r['measured_mse']-r['minimum_mse']-r['constant_offset_squared'])<1e-12 for r in records),
            'records':records,'scope':'Fidelity to existing native scores, not prediction accuracy. Accuracy loss is not mathematically required by name invariance.'}
    target=ROOT/'results/fidelity.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
