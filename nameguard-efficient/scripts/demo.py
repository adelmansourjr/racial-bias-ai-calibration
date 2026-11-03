"""Synthetic demo, no human review represented: two generated test credentials."""
import argparse
import json
import secrets
from fixtures import ROOT,fixture,BODIES
from nameguard.annotations import endorse,packet
from nameguard.canonical_backend import create_native_service


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--names',nargs='+',default=['John Miller','Mohammed Hassan','Previously Unseen 名'])
    args=p.parse_args();keys={'synthetic_A':secrets.token_bytes(32),'synthetic_B':secrets.token_bytes(32)}
    service,close=create_native_service(args.checkpoint,args.manifest,keys);rows=[]
    try:
        for name in args.names:
            source,_=fixture(BODIES[0],name,0)
            wire=packet(source,[endorse(source,reviewer=n,key=k)for n,k in keys.items()])
            score=service.score(wire)
            rows.append({'synthetic_name':name,'score':score.as_dict()})
    finally:integrity=close()
    if any(r['score']!=rows[0]['score']for r in rows):raise AssertionError('Matched-name inequality')
    print(json.dumps({'rows':rows,'integrity':integrity,
         'scope':'Generated exact-role CV demo with two synthetic credentials. Not evidence of two independent human reviewers or automatic identity discovery.'},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
