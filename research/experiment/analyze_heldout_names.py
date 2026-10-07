"""CPU analysis of newly inferred naming cues in the original fixed pools."""
import json
from pathlib import Path
import numpy as np
from ranking_audit import ranking

ROOT=Path(__file__).resolve().parent

def main():
    ranking_results=json.loads((ROOT/'ranking/results.json').read_text())
    manifest=ranking_results['design']['pools']
    result={}
    for kind,model in [('qwen','Qwen1.5B'),('clef','Clef9B')]:
        raw=json.loads((ROOT/f'heldout_names/{kind}-results.json').read_text())
        records=raw['records'];rows=[]
        mapped={'raw_score':'original','projection_score':'uncentered_projection',
            'centered_projection_score_exploratory':'centered_projection_exploratory',
            'fixed_bank_score_mean':'score_orbit_mean','fixed_bank_activation_mean_score':'activation_orbit_mean'}
        for field,method in mapped.items():
            if method not in ranking_results['models'][model]: continue
            for pool in manifest:
                ids=[c['candidate_id'] for c in pool['candidates']]
                cases=[c['case_id'] for c in pool['candidates']]
                stats=next(p for p in ranking_results['models'][model][method]['shortlist']['pools'] if p['job_id']==pool['job_id'])
                base=stats['baseline_scores']; reference=ranking(base,ids,1)
                for i,case in enumerate(cases):
                    options=[r for r in records if r['case_id']==case and r['task']=='shortlist' and r.get(field) is not None]
                    for row in options:
                        scores=list(base);scores[i]=row[field]
                        order,ranks,selected=ranking(scores,ids,1)
                        rows.append({'method':method,'job_id':pool['job_id'],'case_id':case,
                            'changed_id':ids[i],'new_name':row['name'],'score':row[field],
                            'baseline_score':base[i],'selected_ids':sorted(selected),
                            'baseline_selected_ids':sorted(reference[2]),
                            'selected_set_changed':selected!=reference[2],
                            'changed_rank_displacement':ranks[ids[i]]-reference[1][ids[i]],
                            'unchanged_max_rank_displacement':max(abs(ranks[c]-reference[1][c]) for c in ids if c!=ids[i])})
        summary={}
        for method in sorted({r['method'] for r in rows}):
            rs=[r for r in rows if r['method']==method]
            summary[method]={'one_at_a_time_interventions':len(rs),
                'interventions_changing_top1_set':sum(r['selected_set_changed'] for r in rs),
                'pools_with_any_top1_change':len({r['job_id'] for r in rs if r['selected_set_changed']}),
                'max_changed_applicant_rank_displacement':max(abs(r['changed_rank_displacement']) for r in rs)}
        result[kind]={'summary':summary,'rows':rows,
            'scope':'Eight fresh probe-test applicants receive two new names one at a time inside six original heldout pools; other applicants retain documented baseline names. Each method uses its own baseline ranking. Exploratory.'}
    (ROOT/'heldout_names/ranking-extension.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:r['summary'] for k,r in result.items()},indent=2))

if __name__=='__main__': main()
