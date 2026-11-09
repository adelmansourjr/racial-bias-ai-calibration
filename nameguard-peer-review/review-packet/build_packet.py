"""Generate original blinded synthetic annotation cases and blank response forms."""
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAMES=('Jordan Lee','Rose Baker','سلمى الخطيب','王小明','Zoë Kovač','Ayaan Qureshi')


def main():
    cases=[]
    for i,name in enumerate(NAMES):
        variants=(
          (f'Applicant: {name}\nCV: Active translation credential. Reference: Jordan Blake, former supervisor.\nJob: Translator; active translation credential required.','explicit applicant, independent reference'),
          (f'Applicant: {name}\nCV: Uses Django and Java. Translation credential expired.\nJob: Translator; active translation credential required.','skills remain evidence'),
          (f'Applicant: {name}\nCV: Completed Prince2 training. Current translation credential is not held.\nJob: Translator; active translation credential required.','training title remains evidence'),
          (f'{name} supplied the credential evidence. Jordan approved it.\nJob: Translator; active translation credential required.','context-limited ambiguity'),
        )
        for j,(text,note) in enumerate(variants):
            cases.append({'case_id':f'case-{i:02d}-{j}','text':text,
                          'source_sha256':hashlib.sha256(text.encode()).hexdigest()})
    target=ROOT/'cases.json'
    if target.exists():raise FileExistsError('Preserve the completed packet')
    target.write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'blank-response.csv').open('w',newline='') as f:
        fields=('reviewer_id','independent_review_attested','case_id','source_sha256',
                'identity_spans_json','retained_spans_json','status','eligible','notes')
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in cases:w.writerow({'case_id':c['case_id'],'source_sha256':c['source_sha256']})

if __name__=='__main__':main()
