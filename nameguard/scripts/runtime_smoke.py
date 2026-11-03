"""Exercise the final serialized factory and model-drift fail-closed path."""
import argparse
import json
from pathlib import Path
import secrets
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nameguard.contracts import ReviewAuthority,sign_review
from nameguard.encoding import ScalarEncoder
from nameguard.runtime import ScoringService,RuntimeRejected
from nameguard.serialized_backend import SerialQwenBackend,create_service,_MODEL_GATE
from nameguard.mlx_backend import load_pinned,QwenBackend


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);args=p.parse_args()
    output=ROOT/'results/runtime-smoke.json'
    if output.exists():raise FileExistsError(output)
    key=secrets.token_bytes(32)
    nodes=[{'role':'evidence','text':'Applicant: '},{'role':'identity','text':'An Unseen 名'},
           {'role':'evidence','text':'\nFour years paid Python work. A yes; B no. Answer:'}]
    wire=sign_review(nodes,reviewer='smoke',key=key)
    service,close=create_service(args.checkpoint,args.manifest,{'smoke':key})
    first=service.score(wire)
    _MODEL_GATE.acquire()
    try:
        try:service.score(wire)
        except RuntimeRejected:cross_service_gate_blocks=True
        else:raise AssertionError('Process-wide lock bypass')
    finally:_MODEL_GATE.release()
    # A rejected backend call intentionally poisons that service.
    try:service.score(wire)
    except RuntimeRejected:poisoned=True
    else:raise AssertionError('Poisoned engine reused')
    integrity=close()
    model,tok,verify_files=load_pinned(args.checkpoint,args.manifest)
    backend=SerialQwenBackend(model,tok)
    another=ScoringService(ReviewAuthority({'smoke':key}),ScalarEncoder(tok),backend)
    original=model.model.layers[0].self_attn.q_proj
    import mlx.nn as nn
    class UnexpectedWrapper(nn.Module):
        def __init__(self):super().__init__();self.original=original
        def __call__(self,x):return self.original(x)
    model.model.layers[0].self_attn.q_proj=UnexpectedWrapper()
    try:
        try:another.score(wire)
        except RuntimeRejected:module_drift_blocked=True
        else:raise AssertionError('Changed graph accepted')
    finally:model.model.layers[0].self_attn.q_proj=original
    final_digest=backend.verify_parameters();verify_files();another.close()
    output.write_text(json.dumps({'initial_score':first.as_dict(),'process_wide_busy_blocked':cross_service_gate_blocks,
          'poisoned_reuse_blocked':poisoned,'module_drift_blocked':module_drift_blocked,
          'final_parameter_digest':final_digest,'factory_close_integrity':integrity},indent=2)+'\n')
    print('PASS: final factory, process-wide lock, poisoning, module-drift rejection and base integrity.')


if __name__=='__main__':main()
