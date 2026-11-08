"""Audit compatibility and protected evidence equivalence separately on synthetic cases."""
import json
import argparse
import hashlib
from collections import Counter
from pathlib import Path
import secrets
import sys

from gate import WORKSPACE, native_layout, ContractError
from evidence_contract import EvidenceContract, evidence_digest, BoundEvidenceScorer
from prepare_review import propose_whitespace_repair
from nameguard.annotations import DualReviewAuthority, endorse, packet
sys.path.insert(0,str(WORKSPACE/'native-token-activation-gate'))
from tokenizer_stress import NAMES, LAYOUTS
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/boundary-contract-exact-audit.json')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        WORKSPACE/'qwen-safe-runtime/models/frozen-qwen-1.5b-4bit',local_files_only=True)
    keys = {k:secrets.token_bytes(32) for k in ('a','b')}
    authority = DualReviewAuthority(keys)
    summary = {}
    for label,(prefix,suffix) in LAYOUTS.items():
        records = []
        for name in NAMES:
            actual = ' '+name if label=='space_inside_identity' else name
            source = tok.apply_chat_template([{'role':'user','content':prefix+actual+suffix}],
                                             tokenize=False,add_generation_prompt=True)
            start = source.index(prefix+actual+suffix)+len(prefix)
            span = {'start':start,'end':start+len(actual),'text':actual}
            try:
                proposal = propose_whitespace_repair(
                    source,[span],[actual],tok,allow_formatting=True)
                proposal.payload['resolved'] = True  # generator-defined role truth in this test
                wire = packet(proposal.payload,[endorse(proposal.payload,reviewer=k,key=v) for k,v in keys.items()])
                layout = native_layout(authority.verify(wire),tok)
                records.append((wire,layout))
            except ContractError:
                continue
        anchor = EvidenceContract.from_reviewed(records[0][0],authority,tok)
        counts = Counter()
        for _,layout in records:
            try:
                anchor.verify(layout)
                counts['matches_approved_evidence'] += 1
            except ContractError:
                counts['changed_evidence_rejected'] += 1
        summary[label] = dict(counts,token_compatible_variants=len(records))
    result = {'synthetic_variants':120,'layouts':summary,
              'first_reviewed_variant_is_approved_anchor':True,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'contract_source_sha256':hashlib.sha256((ROOT/'evidence_contract.py').read_bytes()).hexdigest(),
              'review_entry_source_sha256':hashlib.sha256((ROOT/'prepare_review.py').read_bytes()).hexdigest(),
              'comparison':'Exact immutable token tuple equality; digest is reporting only.',
              'scope':'Boundary admission and evidence identity are distinct checks. Anchor compatibility depends on the approved role choice.'}
    out = args.output
    if out.exists():
        if json.loads(out.read_text())!=result:
            raise ValueError('Boundary audit drift')
    else:
        out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
