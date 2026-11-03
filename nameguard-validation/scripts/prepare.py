"""Freeze a controlled real-excerpt study; no model inference or invented labels."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
CATEGORIES={'ACCOUNTANT':19,'ADVOCATE':24,'BANKING':44,'ENGINEERING':13,
            'HEALTHCARE':34,'INFORMATION-TECHNOLOGY':22,'SALES':48,'TEACHER':56}
NAMES=['John Miller','Mohammed Hassan','Wei Chen','Neruvo Qaslin']
SEED=20261103
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def select(rows,excluded):
    chosen=[]
    for category in sorted(CATEGORIES):
        eligible=[r for r in rows if r['Category']==category and r['ID'] not in excluded]
        eligible.sort(key=lambda r:hashlib.sha256(f"{SEED}:{r['ID']}".encode()).hexdigest())
        chosen.extend(eligible[:4])
    if len(chosen)!=32:raise ValueError('Insufficient category coverage')
    return chosen


def excerpt(text,words):
    return ' '.join(text.split()[:words])


def main():
    private=ROOT/'local-private/contexts.json'
    if private.exists()or(ROOT/'audit/plan.json').exists():raise FileExistsError('Frozen study exists')
    resume_path=PROJECT/'research/data/Resume.csv';job_path=PROJECT/'research/data/jobs-first-rows.json'
    with resume_path.open()as f:rows=list(csv.DictReader(f))
    jobs=json.loads(job_path.read_text())['rows']
    previous=json.loads((PROJECT/'research/experiment/cases.json').read_text())
    if isinstance(previous,dict):previous=previous.get('cases',[])
    excluded={str(x['resume_id'])for x in previous if 'resume_id'in x}
    chosen=select(rows,excluded);categories=sorted(CATEGORIES);contexts=[];public=[]
    for cv_index,row in enumerate(chosen):
        cat=row['Category'];c=categories.index(cat)
        cv=excerpt(row['Resume_str'],140)
        style='selective' if cv_index%4>=2 else 'plain'
        uid=hashlib.sha256(('resume:'+row['ID']).encode()).hexdigest()
        for kind,jcat in [('matched',cat),('mismatched',categories[(c+3)%len(categories)])]:
            job=jobs[CATEGORIES[jcat]]['row'];job_text=excerpt(job['description'],40)
            context={'id':f'c{cv_index:02d}-{kind}','cv_cluster':uid,'category':cat,
                'job_kind':kind,'job_category':jcat,'job_id':str(job['job_id']),
                'title':job['title'],'cv_excerpt':cv,'job_excerpt':job_text,'style':style}
            contexts.append(context)
            public.append({k:v for k,v in context.items()if k not in ['title','cv_excerpt','job_excerpt']})
    private.write_text(json.dumps(contexts,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'audit/case-index.json').write_text(json.dumps(public,indent=2)+'\n')
    sources={str(p.relative_to(ROOT)):sha(p)for p in ROOT.rglob('*.py')if '__pycache__'not in p.parts}
    plan={'frozen_before_inference':True,'seed':SEED,'source_hashes':sources,
      'data_sha256':{'Resume.csv':sha(resume_path),'jobs-first-rows.json':sha(job_path),'private_contexts':sha(private)},
      'CVs':32,'contexts_per_CV':2,'names':NAMES,'categories':list(CATEGORIES),
      'names_are_interventions_not_verified_demographics':True,
      'selection':'Four per category, hash-order seed; excludes original datathon case IDs only. Not certified unseen in every earlier local study or pretraining.',
      'source_excerpt_policy':'First 140 whitespace words of real CV; first 40 of real job description. Full-CV validity not claimed.',
      'conditions':'Two CVs/category plain, two selective; each receives category-matched and deterministically mismatched job.',
      'primary':'Exact internal score/public tensor equality across four names in all 64 contexts.',
      'comparators':['native named full-source readout','equivalent externally canonicalized native evidence readout'],
      'native_parity':'Compare full final hidden array and full vocabulary readout at every context, not only A/B output.',
      'utility':'Category-pair relevance ranking diagnostic only; no human hiring labels or actual job-performance validity.',
      'timing':'One paired internal/external service trial per CV on matched job, after 2 warmups each; shuffled order; same review checks; no observation copies; signing/load excluded.',
      'statistics':'CV-clustered percentile bootstrap, 5000 draws, seed20261103; paired relevance metrics; no population failure upper bound from this convenience sample.',
      'annotation':'Exact inserted header span is known by construction. Original body names/proxies left unchanged; no claim of independent human body annotation.',
      'stopping':'Run all 64 contexts and all four names; no outcome-dependent selection or adapter fitting.',
      'private_data_publication':'Only hashes, categories and scores published; no CV/job text or secret keys.'}
    (ROOT/'audit/plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'CVs':32,'contexts':64,'excluded_prior_datathon_ids':len(excluded),'source_files_frozen':len(sources)}))


if __name__=='__main__':main()
