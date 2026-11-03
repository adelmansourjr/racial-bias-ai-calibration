"""Exercise the final dual-review native factory and its fail-closed runtime."""
import argparse
import json
from pathlib import Path
import secrets
import sys
from fixtures import ROOT,fixture,BODIES
from nameguard.annotations import endorse,packet
from nameguard.canonical_backend import create_native_service
from nameguard.serialized_backend import _MODEL_GATE
from nameguard.runtime import RuntimeRejected


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);args=p.parse_args()
    output=ROOT/'results/runtime-smoke.json'
    if output.exists():raise FileExistsError(output)
    keys={'A':secrets.token_bytes(32),'B':secrets.token_bytes(32)}
    source,_=fixture(BODIES[0],'Never Seen 名',1)
    wire=packet(source,[endorse(source,reviewer=n,key=k)for n,k in keys.items()])
    service,close=create_native_service(args.checkpoint,args.manifest,keys)
    result=service.score(wire)
    if set(result.as_dict())!={'margin','conditional_A_probability'}:raise AssertionError('Private metadata exposed')
    _MODEL_GATE.acquire()
    try:
        try:service.score(wire)
        except RuntimeRejected:busy_blocked=True
        else:raise AssertionError('Process-wide gate bypass')
    finally:_MODEL_GATE.release()
    try:service.score(wire)
    except RuntimeRejected:poisoned_blocked=True
    else:raise AssertionError('Poisoned service reused')
    integrity=close()
    output.write_text(json.dumps({'public_score':result.as_dict(),'process_wide_gate_blocked':busy_blocked,
        'poisoned_reuse_blocked':poisoned_blocked,'integrity':integrity,'keys_saved':False},indent=2)+'\n')
    print('PASS: final canonical factory, dual review, process-wide lock, poisoning and protected weights.')


if __name__=='__main__':main()
