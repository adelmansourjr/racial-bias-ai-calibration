"""Score an intact prompt using reviewed spans and an explicit task profile."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets

from gate import SlotScorer, native_layout
from service import CredentialScoringService, load_credential_calibration
from evidence_contract import evidence_digest
from nameguard.annotations import DualReviewAuthority, endorse, packet, proposal_from_spans
from nameguard.mlx_backend import load_pinned

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--prompt-file',type=Path,required=True)
    parser.add_argument('--span',action='append',required=True,metavar='START:END')
    parser.add_argument('--credential-profile',action='store_true',
                        help='Use the published synthetic-active-credential-v1 readout calibration')
    args = parser.parse_args()
    content = args.prompt_file.read_bytes().decode('utf-8')
    intervals = []
    for item in args.span:
        start,end = map(int,item.split(':'))
        if not 0 <= start < end <= len(content):
            parser.error('Invalid Unicode character identity span')
        intervals.append((start,end))
    intervals.sort()
    model,tok,verify_files = load_pinned(args.checkpoint,ROOT/'audit/checkpoint-manifest.json')
    text = tok.apply_chat_template([{'role':'user','content':content}],
                                  tokenize=False,add_generation_prompt=True)
    if text.count(content) != 1:
        raise ValueError('Chat template must preserve one exact prompt copy')
    offset = text.index(content)
    spans = [{'start':offset+a,'end':offset+b,'text':content[a:b]} for a,b in intervals]
    registry = sorted({span['text'] for span in spans})
    proposal = proposal_from_spans(text,spans,registry,resolved=True)
    keys = {key:secrets.token_bytes(32) for key in ('demo_a','demo_b')}
    authority = DualReviewAuthority(keys)
    wire = packet(proposal,[endorse(proposal,reviewer=k,key=v) for k,v in keys.items()])
    layout = native_layout(authority.verify(wire),tok)
    raw_service = SlotScorer(model,tok,authority)
    if args.credential_profile:
        manifest = json.loads((ROOT/'audit/public-manifest.json').read_text())
        calibration = load_credential_calibration(
            ROOT/'audit/calibration.json',manifest['sha256']['audit/calibration.json'])
        service = CredentialScoringService(raw_service,calibration)
    else:
        service = raw_service
    result = service.score(wire)
    service.close()
    raw_service.verify_parameters()
    verify_files()
    print(json.dumps({
        'score':result.as_dict(),
        'profile':'synthetic-active-credential-v1' if args.credential_profile else 'original-readout',
        'original_native_tokens':len(layout.ids),
        'evidence_tokens':len(layout.public),
        'evidence_ids_sha256':evidence_digest(layout.public_ids),
        'original_input_ids_sha256':hashlib.sha256(
            json.dumps(layout.ids,separators=(',',':')).encode()).hexdigest(),
    },indent=2))


if __name__=='__main__':
    main()
