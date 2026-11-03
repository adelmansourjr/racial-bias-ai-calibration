"""Generate references and lint local citation integrity (not semantic truth)."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def references():
    return json.loads((ROOT / 'literature/references.json').read_text())


def reference_text(refs):
    rows = []
    for r in refs:
        year = r['year'] if r['year'] is not None else 'n.d.'
        doi = f" DOI: {r['doi']}." if r.get('doi') else ''
        rows.append(f"{r['id']}. {r['authors'].replace(' and ', ', ')}. {year}. "
                    f"{r['title']}. {r['venue']}.{doi} {r['url']} "
                    f"Accessed {r['accessed']}.")
    return '\n\n'.join(rows) + '\n'


def bibtex(refs):
    rows = []
    for r in refs:
        # UTF-8 BibTeX, title case protected; n.d. explicitly avoids invented date.
        fields = {'author': '{{Hugging Face}}' if r['id']==2 else r['authors'],
                  'title': '{'+r['title']+'}', 'year': str(r['year'] or 'n.d.'),
                  'howpublished': r['venue'], 'url': r['url'],
                  'note': 'Accessed '+r['accessed']+'; '+r.get('version_note',r['verified'])}
        if r.get('doi'): fields['doi'] = r['doi']
        kind = 'misc' if r['id']==2 or r['id']>=26 else ('article' if r['id'] in (5,6,12,16,24) else 'inproceedings')
        fields['author']=fields['author'].replace('Hal Daumé III','Daumé, III, Hal')
        fields['journal' if kind=='article' else 'booktitle' if kind=='inproceedings' else 'howpublished'] = fields.pop('howpublished')
        rows.append('@'+kind+'{'+r['key']+',\n'+',\n'.join(
            '  '+key+' = {'+value+'}' for key,value in fields.items())+'\n}')
    return '\n\n'.join(rows)+'\n'


def cited_ids(text):
    result = set()
    for group in re.findall(r'\[([0-9, -]+)\]',text):
        for part in group.replace(' ','').split(','):
            if '-' in part:
                a,b=map(int,part.split('-'));result.update(range(a,b+1))
            else:result.add(int(part))
    return result


def generate():
    refs=references(); text=reference_text(refs)
    (ROOT/'literature/references.bib').write_text(bibtex(refs))
    (ROOT/'literature/reference-list.md').write_text('# References\n\n'+text)
    for path in ['paper/manuscript.md']:
        p=ROOT/path
        p.write_text(p.read_text().split('## References')[0].rstrip()+'\n\n## References\n\n'+text)


def check():
    refs=references(); ids={r['id'] for r in refs}
    if ids!=set(range(1,32)) or len(refs)!=31:raise ValueError('Reference ID drift')
    for field in ('key','url'):
        if len({r[field] for r in refs})!=len(refs):raise ValueError('Duplicate '+field)
    dois=[r['doi'].lower() for r in refs if r.get('doi')]
    if len(dois)!=len(set(dois)):raise ValueError('Duplicate DOI')
    for r in refs:
        for field in ('authors','title','venue','url','locator','verified','accessed'):
            if not r[field]:raise ValueError('Incomplete metadata')
        if not r['url'].startswith('https://'):raise ValueError('Bad URL')
    if (ROOT/'literature/references.bib').read_text()!=bibtex(refs):raise ValueError('Stale BibTeX')
    for path in ['paper/manuscript.md']:
        text=(ROOT/path).read_text();body,tail=text.split('## References\n\n')
        if cited_ids(body)!=ids:raise ValueError('Uncited or unknown reference in '+path)
        if tail!=reference_text(refs):raise ValueError('Stale reference list in '+path)
    return {'references':len(refs),'all_citations_resolve':True,'all_references_cited_in_manuscript':True,
            'bibtex_and_reference_lists_synchronized':True,
            'scope':'Local metadata/citation integrity only; not automatic validation of source claims or exhaustive coverage.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true')
    args=parser.parse_args()
    if args.generate:generate()
    print(json.dumps(check(),indent=2))
