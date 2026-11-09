"""Validate and compare the sealed blinded AI annotations, with explicit AI provenance."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def validate(path,cases):
    data=json.loads(path.read_text())
    if data['reviewer_kind']!='AI agent' or data['blinded_to_methods_and_results'] is not True:
        raise ValueError('Explicit blinded AI provenance required')
    rows=data['cases']
    if len(rows)!=len(cases) or {r['case_id'] for r in rows}!=set(cases):raise ValueError('Complete unique case coverage required')
    for r in rows:
        text=cases[r['case_id']]['text']
        if r['source_sha256']!=hashlib.sha256(text.encode()).hexdigest():raise ValueError('Stale source hash')
        used=[]
        for field in ('identity_spans','retained_spans'):
            previous=0
            for pair in r[field]:
                if type(pair) is not list or len(pair)!=2 or any(type(v) is not int for v in pair):raise ValueError('Invalid span')
                a,b=pair
                if not 0<=previous<=a<b<=len(text):raise ValueError('Invalid span ordering')
                previous=b
                chars=set(range(a,b))
                if any(chars&other for other in used):raise ValueError('Overlapping role assignments')
                used.append(chars)
        if r['status'] not in ('clear','uncertain') or r['eligible'] not in ('yes','no','uncertain'):
            raise ValueError('Invalid annotation decision')
    return data,{r['case_id']:r for r in rows}


def main():
    cases={c['case_id']:c for c in json.loads((ROOT/'cases.json').read_text())}
    ap=ROOT/'ai-reviewer-a.json';bp=ROOT/'ai-reviewer-b.json'
    ad,a=validate(ap,cases);bd,b=validate(bp,cases)
    if ad['reviewer_id']==bd['reviewer_id']:raise ValueError('Different agent IDs required')
    result={'review_type':'Blinded AI-agent review, not independent human validation',
            'reviewers':[ad['reviewer_id'],bd['reviewer_id']],'cases':len(cases),
            'exact_identity_span_agreement':sum(a[c]['identity_spans']==b[c]['identity_spans'] for c in cases),
            'exact_retained_span_agreement':sum(a[c]['retained_spans']==b[c]['retained_spans'] for c in cases),
            'qualification_agreement':sum(a[c]['eligible']==b[c]['eligible'] for c in cases),
            'status_agreement':sum(a[c]['status']==b[c]['status'] for c in cases),
            'uncertain_per_reviewer':[sum(r['status']=='uncertain' for r in x.values()) for x in (a,b)],
            'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ap,bp)},
            'limitations':'Fresh contexts and restricted instructions provide procedural blinding. Agents share the provider/model family; no human reliability, population accuracy or independent-lab replication follows.'}
    p=ROOT/'ai-review-summary.json'
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
