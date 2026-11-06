"""Score a raw named prompt with manually reviewed Unicode character spans."""
import argparse
import json
from pathlib import Path
import secrets

from experiment import ROOT
from gate import SlotScorer, native_layout
from nameguard.annotations import DualReviewAuthority, endorse, packet, proposal_from_spans
from nameguard.mlx_backend import load_pinned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--span', action='append', required=True, metavar='START:END',
                        help='Reviewed identity span in the raw prompt, measured in Unicode characters')
    args = parser.parse_args()
    raw = args.prompt_file.read_bytes().decode('utf-8')
    ranges = []
    for value in args.span:
        start, end = map(int, value.split(':'))
        if not 0 <= start < end <= len(raw):
            parser.error('Invalid Unicode character span')
        ranges.append((start, end))
    ranges.sort()
    model, tokenizer, verify_files = load_pinned(args.checkpoint, ROOT/'audit/checkpoint-manifest.json')
    text = tokenizer.apply_chat_template([{'role':'user','content':raw}],
                                         tokenize=False,add_generation_prompt=True)
    if text.count(raw) != 1:
        raise ValueError('Chat template must contain the original prompt once')
    offset = text.index(raw)
    spans = [{'start':offset+a,'end':offset+b,'text':raw[a:b]} for a,b in ranges]
    registry = sorted({row['text'] for row in spans})
    # The CLI is a local test harness. User-supplied spans stand in for human review.
    keys = {label:secrets.token_bytes(32) for label in ('demo_a','demo_b')}
    authority = DualReviewAuthority(keys)
    proposal = proposal_from_spans(text,spans,registry,resolved=True)
    wire = packet(proposal,[endorse(proposal,reviewer=key,key=value) for key,value in keys.items()])
    scorer = SlotScorer(model,tokenizer,authority)
    layout = native_layout(authority.verify(wire),tokenizer)
    score = scorer.score(wire)
    scorer.verify_parameters()
    verify_files()
    print(json.dumps({'score':score.as_dict(),'original_native_tokens':len(layout.ids),
                      'evidence_compute_tokens':len(layout.public),
                      'identity_slots_cleared':len(layout.ids)-len(layout.public)},indent=2))


if __name__=='__main__':
    main()
