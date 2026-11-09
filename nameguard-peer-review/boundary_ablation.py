"""Separate frozen contract selection from inference-time witness-search coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from bootstrap import ROOT
from compiler import enumerate_candidates,select_contract
from fixtures import FIT_NAMES,TEST_NAMES,LAYOUTS,rendered


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);args=p.parse_args()
    target=ROOT/'results/boundary-ablation.json'
    if target.exists():raise FileExistsError(target)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    layouts={}
    for name,(prefix,suffix) in LAYOUTS.items():
        candidates=lambda names:[enumerate_candidates(*rendered(tok,n,prefix,suffix+'Evidence: active license.'),tok) for n in names]
        fit=candidates(FIT_NAMES);test=candidates(TEST_NAMES)
        anchor=next(member[0].evidence for member in fit if member)
        optimal=select_contract(fit)
        layouts[name]={
         'greedy_contract_greedy_witness':sum(bool(member and member[0].evidence==anchor) for member in test),
         'greedy_contract_all_witnesses':sum(any(c.evidence==anchor for c in member) for member in test),
         'compiled_contract_greedy_witness':sum(bool(member and member[0].evidence==optimal.evidence) for member in test),
         'compiled_contract_all_witnesses':sum(any(c.evidence==optimal.evidence for c in member) for member in test),
         'fit_anchor_all_witness_support':sum(any(c.evidence==anchor for c in member) for member in fit),
         'fit_compiled_all_witness_support':optimal.supported,
         'test_panel':len(test)}
    keys=tuple(next(iter(layouts.values())))
    result={'adaptive_followup':True,'reason':'Technical review identified confounded contract and witness search.',
            'layouts':layouts,'totals':{k:sum(row[k] for row in layouts.values()) for k in keys},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Tokenizer-only component ablation. Contracts use fit names only. No test-outcome fitting.'}
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
