"""Audit native tokenizer role boundaries on entirely synthetic prompts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from gate import ConsensusDocument, ContractError, native_layout
from nameguard.contracts import Node

ROOT = Path(__file__).resolve().parent
NAMES = (
    'Ai','Bo','José Álvarez','Jose\u0301 A\u0301lvarez','Jean-Luc Moreau',"D’Andre Scott",
    'Fatou Ndiaye','أمينة بنت يوسف','ليلى منصور','陳美玲','山田 太郎','김민준',
    'सोनिया शर्मा','Nguyễn Thị Mai','Ｊｏｈｎ Ｍｉｌｌｅｒ','Ольга Волкова',
    'A\u200bmina','Name\u00a0Surname','Inês de Albuquerque-Silva','𐐷 Persona',
    ' Ava','Ava ','Ava\n','\nAva',
)
LAYOUTS = {
    'newlines':('Applicant name:\n','\nEvaluate the evidence.'),
    'space_outside_identity':('Applicant name: ','\nEvaluate the evidence.'),
    'space_inside_identity':('Applicant name:','\nEvaluate the evidence.'),
    'punctuation_seam':('Name=','; Evidence: active license.'),
    'attached_letters':('Candidate','Identifier meets the rule.'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    manifest = json.loads((ROOT/'audit/checkpoint-manifest.json').read_text())['files']
    for key in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json'):
        actual = hashlib.sha256((args.checkpoint/key).read_bytes()).hexdigest()
        if actual != manifest[key]['sha256']:
            raise ValueError('Unexpected tokenizer files')
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    report = {}
    for label,(prefix,suffix) in LAYOUTS.items():
        result = Counter()
        ordered_evidence = set()
        for name in NAMES:
            rendered_name = ' '+name if label=='space_inside_identity' else name
            raw = prefix+rendered_name+suffix
            source = tokenizer.apply_chat_template([{'role':'user','content':raw}],
                                                   tokenize=False,add_generation_prompt=True)
            start = source.index(raw)+len(prefix)
            end = start+len(rendered_name)
            nodes = (Node('evidence',source[:start]),Node('identity',source[start:end]),
                     Node('evidence',source[end:]))
            doc = ConsensusDocument(nodes,('fixture_a','fixture_b'),'synthetic',('synthetic',))
            try:
                layout = native_layout(doc,tokenizer)
            except ContractError as error:
                result[str(error)] += 1
                continue
            if list(layout.ids)!=tokenizer.encode(source,add_special_tokens=False):
                raise ValueError('Original token IDs changed')
            ordered_evidence.add(layout.public_ids)
            result['accepted'] += 1
        if len(ordered_evidence)>1:
            raise ValueError('Accepted layout leaked identity through evidence IDs')
        report[label] = dict(result, distinct_accepted_evidence_sequences=len(ordered_evidence))
    print(json.dumps({'planned_tokenizer_checks':len(NAMES)*len(LAYOUTS),
                      'layouts':report,
                      'scope':'Synthetic tokenizer factorization only. Authentication and model activations are exercised separately.'},indent=2))


if __name__=='__main__':
    main()
