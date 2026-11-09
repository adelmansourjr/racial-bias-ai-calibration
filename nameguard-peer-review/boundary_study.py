"""Fit name-panel contracts on tokenizer-only data; test disjoint names without refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from compiler import enumerate_candidates,select_contract,match_contract,ContractError
from fixtures import FIT_NAMES,TEST_NAMES,LAYOUTS,rendered
from bootstrap import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/boundary.json')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve completed results; choose a new output')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    summary={}
    records=[]
    for label,(prefix,suffix) in LAYOUTS.items():
        by_split={}
        for split,names in (('fit',FIT_NAMES),('test',TEST_NAMES)):
            by_split[split]=[enumerate_candidates(*rendered(tok,name,prefix,suffix+'Evidence: active license.'),tok)
                             for name in names]
        fit=by_split['fit']
        compiled=select_contract(fit)
        # The anchor baseline freezes the first least-cost witness from the same
        # fitting panel. Empty-panel cases are reported, never excluded silently.
        first=next((member[0] for member in fit if member),None)
        anchor=first.evidence if first else None
        row={'fit_panel':len(fit),'fit_optimal_support':compiled.supported,
             'fit_total_formatting_cost':compiled.total_formatting_cost,
             'evidence_ids':compiled.evidence,'test_panel':len(TEST_NAMES)}
        counts={'test_token_compatible':0,'test_strict':0,'test_greedy_anchor':0,
                'test_compiled':0,'test_compiled_formatting_cost':0}
        for name,candidates in zip(TEST_NAMES,by_split['test']):
            strict=next((c for c in candidates if c.formatting_cost==0),None)
            greedy=candidates[0] if candidates else None
            matched=next((c for c in candidates if c.evidence==compiled.evidence),None)
            counts['test_token_compatible']+=bool(candidates)
            counts['test_strict']+=strict is not None
            counts['test_greedy_anchor']+=bool(greedy and greedy.evidence==anchor)
            counts['test_compiled']+=matched is not None
            if matched:counts['test_compiled_formatting_cost']+=matched.formatting_cost
            records.append({'layout':label,'name':name,'strict':strict is not None,
                            'token_compatible':bool(candidates),
                            'greedy_anchor':bool(greedy and greedy.evidence==anchor),
                            'compiled':matched is not None,'candidate_count':len(candidates)})
        row.update(counts)
        # Diagnostic only: selecting on test names is an explicitly labeled oracle,
        # never the primary held-out contract.
        oracle=select_contract(by_split['test'])
        row['test_oracle_support_descriptive_only']=oracle.supported
        summary[label]=row
    result={'fit_names':FIT_NAMES,'test_names':TEST_NAMES,'layouts':summary,
            'records':records,'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (Path(__file__),ROOT/'src/compiler.py',ROOT/'src/fixtures.py')},
            'scope':'Generator-defined roles; no model scores used. Frozen fit-panel contract applied to disjoint test names. Script variants are not verified racial groups.'}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='evidence_ids'} for k,v in summary.items()},indent=2),flush=True)


if __name__=='__main__':main()
