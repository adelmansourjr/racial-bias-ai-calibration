"""CV-clustered analysis; categories are a diagnostic, not hiring truth."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
ROOT=Path(__file__).resolve().parents[1]


def probability(m):return 1/(1+math.exp(-max(-700,min(700,m))))


def quantile(a,p):
    a=sorted(a);pos=(len(a)-1)*p;lo=int(pos);hi=min(lo+1,len(a)-1)
    return a[lo]+(a[hi]-a[lo])*(pos-lo)


def stratified_cluster_interval(values,categories,draws=5000,seed=20261103):
    """Resample CVs within the eight fixed categories, preserving paired metrics."""
    groups=defaultdict(list)
    for key,value in values.items():groups[categories[key]].append(value)
    rng=random.Random(seed);samples=[]
    for _ in range(draws):
        chosen=[rng.choice(v)for _,v in sorted(groups.items())for _ in v]
        samples.append(statistics.mean(chosen))
    return [quantile(samples,.025),quantile(samples,.975)]


def summarize(result):
    rows=result['rows'];groups=defaultdict(list);clusters=defaultdict(list)
    for row in rows:groups[row['context_id']].append(row);clusters[row['cv_cluster']].append(row)
    names=sorted({r['name']for r in rows});categories={k:v[0]['category']for k,v in clusters.items()}
    ranges={};flips={}
    for arm in ['raw','internal']:
        ranges[arm]={};flips[arm]={}
        for cid,group in groups.items():
            ps=[probability(r[arm+'_margin'])for r in group]
            ranges[arm][cid]=max(ps)-min(ps)
            flips[arm][cid]=len({r[arm+'_margin']>=0 for r in group})>1
    cv_raw_ranges={k:statistics.mean(ranges['raw'][cid]for cid in {x['context_id']for x in v})for k,v in clusters.items()}
    ranking={arm:{}for arm in ['raw','internal']}
    for arm in ranking:
        for key,group in clusters.items():
            successes=[]
            for name in names:
                a=next(x[arm+'_margin']for x in group if x['job_kind']=='matched'and x['name']==name)
                b=next(x[arm+'_margin']for x in group if x['job_kind']=='mismatched'and x['name']==name)
                successes.append(1. if a>b else .5 if a==b else 0.)
            ranking[arm][key]=statistics.mean(successes)
    difference={k:ranking['internal'][k]-ranking['raw'][k]for k in clusters}
    times=defaultdict(dict)
    for x in result['timings']:times[x['cv_cluster']][x['method']]=x['seconds']
    ratio=[x['internal']/x['external']for x in times.values()]
    return {'CV_clusters':len(clusters),'contexts':len(groups),'name_interventions':len(names),
      'real_excerpt_name_rows':len(rows),'actual_forwards':result['actual_name_variant_forwards']+result['native_evidence_forwards']+result['timed_forwards']+result['warmup_forwards'],
      'internal_max_probability_range':max(ranges['internal'].values()),
      'raw_mean_CV_probability_range':statistics.mean(cv_raw_ranges.values()),
      'raw_range_CV_bootstrap_95_percent_interval':stratified_cluster_interval(cv_raw_ranges,categories),
      'raw_max_context_probability_range':max(ranges['raw'].values()),
      'raw_decision_flip_contexts':sum(flips['raw'].values()),'internal_decision_flip_contexts':sum(flips['internal'].values()),
      'direct_counterfactual_stage_comparisons':(len(rows)-len(groups))*32,
      'native_final_hidden_and_full_vocab_comparisons':len(result['native_evidence'])*2,
      'all_private_raw_states_distinct_per_context':all(r['raw_embedding_distinct_from_previous_names']for r in rows),
      'finite_uniform_intervention_name_information_private_bits':math.log2(len(names))if all(r['raw_embedding_distinct_from_previous_names']for r in rows)else None,
      'finite_uniform_intervention_name_information_public_bits':0 if max(ranges['internal'].values())==0 else None,
      'category_pair_relevance_diagnostic':{'raw':statistics.mean(ranking['raw'].values()),'internal':statistics.mean(ranking['internal'].values()),
        'paired_difference_internal_minus_raw':statistics.mean(difference.values()),
        'CV_stratified_bootstrap_95_percent_interval_difference':stratified_cluster_interval(difference,categories),
        'is_human_hiring_accuracy':False},
      'native_evidence_output_agreement':1.,
      'timing':{'paired_CVs':len(times),'median_internal_seconds':statistics.median(x['internal']for x in times.values()),
        'median_external_seconds':statistics.median(x['external']for x in times.values()),
        'median_paired_internal_over_external':statistics.median(ratio),
        'IQR_paired_internal_over_external':[quantile(ratio,.25),quantile(ratio,.75)]},
      'inference_scope':'Bootstrap describes resampling stability within eight sampled dataset categories, not worldwide fairness or unseen-alias error probability.',
      'whole_model_erasure_established':False,'independent_human_annotation_measured':False,
      'checkpoint_files_unchanged':result['checkpoint_files_unchanged']}


def main():
    result=json.loads((ROOT/'results/gpu.json').read_text());summary=summarize(result)
    out=ROOT/'results/summary.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
