"""Offline candidate/endorsement tooling. Never auto-resolves name semantics."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nameguard.annotations import propose_candidates,endorse,packet,DualReviewAuthority
from nameguard.agreements import compare_reviews
from nameguard.contracts import strict_loads, MAX_SOURCE_BYTES, ContractError


def read_json(path):return strict_loads(Path(path).read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    c=commands.add_parser('candidates');c.add_argument('--source',type=Path,required=True);c.add_argument('--identity',nargs='+',required=True);c.add_argument('--output',type=Path,required=True)
    c=commands.add_parser('compare');c.add_argument('--first',type=Path,required=True);c.add_argument('--second',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    c=commands.add_parser('endorse');c.add_argument('--proposal',type=Path,required=True);c.add_argument('--reviewer',required=True);c.add_argument('--key-file',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    c=commands.add_parser('assemble');c.add_argument('--proposal',type=Path,required=True);c.add_argument('--approvals',nargs=2,type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous review artifact')
    if args.command=='candidates':
        if args.source.stat().st_size>MAX_SOURCE_BYTES:raise ContractError('Source exceeds supported review size')
        source=args.source.read_text(encoding='utf-8')
        # This queue is NOT a resolved proposal; a reviewer must create exact
        # roles, retained meanings and adjudicate all ambiguous references.
        result={'source':source,'source_sha256':hashlib.sha256(source.encode('utf-8')).hexdigest(),
                'registry':args.identity,'candidates':propose_candidates(source,args.identity),'resolved':False}
    elif args.command=='compare':result=compare_reviews(read_json(args.first),read_json(args.second))
    elif args.command=='endorse':
        # Raw random key bytes stay local; never print them or include them in
        # an output packet. Review signing is trusted operator work.
        result=endorse(read_json(args.proposal),reviewer=args.reviewer,key=args.key_file.read_bytes())
    else:
        wire=packet(read_json(args.proposal),[read_json(p)for p in args.approvals])
        args.output.write_bytes(wire);return
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
