"""Compare actual independent submissions; reject blank or unbound response files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def read_review(path,cases):
    rows=list(csv.DictReader(Path(path).open(newline='')))
    if len(rows)!=len(cases) or {r['case_id'] for r in rows}!=set(cases):raise ValueError('Complete unique case coverage required')
    names={r['reviewer_id'].strip() for r in rows}
    if len(names)!=1 or not next(iter(names)):raise ValueError('One nonempty reviewer ID per submission required')
    output={}
    for row in rows:
        text=cases[row['case_id']]['text']
        if row['source_sha256']!=hashlib.sha256(text.encode()).hexdigest():raise ValueError('Stale source binding')
        if row['independent_review_attested']!='yes':raise ValueError('Actual independent-review attestation required')
        if row['status'] not in ('clear','uncertain') or row['eligible'] not in ('yes','no','uncertain'):raise ValueError('Incomplete review decisions')
        for field in ('identity_spans_json','retained_spans_json'):
            spans=json.loads(row[field])
            if type(spans) is not list:raise ValueError('Span list required')
            previous=0
            for pair in spans:
                if type(pair) is not list or len(pair)!=2 or any(type(x) is not int for x in pair):raise ValueError('Exact integer intervals required')
                a,b=pair
                if not 0<=previous<=a<b<=len(text):raise ValueError('Unsorted, overlapping or invalid span')
                previous=b
            row[field]=spans
        identity={i for a,b in row['identity_spans_json'] for i in range(a,b)}
        retained={i for a,b in row['retained_spans_json'] for i in range(a,b)}
        if identity&retained:raise ValueError('Conflicting role spans')
        output[row['case_id']]=row
    return next(iter(names)),output


def main():
    p=argparse.ArgumentParser();p.add_argument('first',type=Path);p.add_argument('second',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    cases={c['case_id']:c for c in json.loads((ROOT/'cases.json').read_text())}
    an,a=read_review(args.first,cases);bn,b=read_review(args.second,cases)
    if an==bn:raise ValueError('Distinct independently attesting reviewers required')
    exact=eligible=status=overlap=total=0
    for cid in cases:
        ar,br=a[cid],b[cid]
        ac={i for start,end in ar['identity_spans_json'] for i in range(start,end)}
        bc={i for start,end in br['identity_spans_json'] for i in range(start,end)}
        exact+=ar['identity_spans_json']==br['identity_spans_json']
        eligible+=ar['eligible']==br['eligible'];status+=ar['status']==br['status']
        overlap+=len(ac&bc);total+=len(ac)+len(bc)
    result={'cases':len(cases),'reviewer_ids':[an,bn],'identity_exact_span_agreement':exact/len(cases),
            'identity_character_dice':2*overlap/total if total else None,
            'qualification_agreement':eligible/len(cases),'status_agreement':status/len(cases),
            'uncertain_counts':[sum(r['status']=='uncertain' for r in x.values()) for x in (a,b)],
            'source_files_sha256':[hashlib.sha256(x.read_bytes()).hexdigest() for x in (args.first,args.second)],
            'scope':'Agreement only, not accuracy against independent gold or verified job outcomes. Independence is attested, not independently verified.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
