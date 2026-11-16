"""Replay article-held-out NameGuard verification without loading a model."""
import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import random

RISK_CAP = Fraction(1, 10)
COVERAGE = Fraction(1, 5)


def read_inputs(queue, results):
    paths = (queue / 'rows.json', queue / 'labels.json', results / 'observed.json', results / 'baseline.json')
    rows, labels, observed, baseline = [json.loads(p.read_bytes()) for p in paths]
    assert rows.keys() == labels.keys() == observed.keys() == baseline.keys()
    articles = defaultdict(list)
    for key, record in rows.items():
        assert baseline[key]['admitted'] and observed[key]['admitted'], key
        articles[record['article']].append((key, baseline[key]['mean_margin'], observed[key]['accepted'], labels[key]))
    article_rows = [articles[key] for key in sorted(articles)]
    assert len(article_rows) == 10 and sum(map(len, article_rows)) == 320
    return article_rows, {p.name: sha256(p.read_bytes()).hexdigest() for p in paths}


def train(rows, extracted_only):
    valid = [r for r in rows if r[2] or not extracted_only]
    scores = sorted({r[1] for r in valid})
    eligible = []
    for threshold in scores:
        selected = [r for r in valid if r[1] >= threshold]
        accepted = len(selected)
        false = sum(not r[3] for r in selected)
        if Fraction(accepted, len(rows)) >= COVERAGE and Fraction(false, accepted) <= RISK_CAP:
            eligible.append((accepted, -false, threshold))
    return max(eligible)[2] if eligible else None


def scored(rows):
    return {'confirmed': len(rows), 'false_supports': sum(not r[3] for r in rows)}


def evaluate(articles):
    fold_results = []
    for i, holdout in enumerate(articles):
        training = [record for j, article in enumerate(articles) if i != j for record in article]
        result = {'cases': len(holdout)}
        for label, extracted_only in (('score_threshold', False), ('extraction_then_threshold', True)):
            threshold = train(training, extracted_only)
            result[label] = {'threshold': threshold, **scored([
                r for r in holdout if threshold is not None and r[1] >= threshold and (r[2] or not extracted_only)
            ])}
        count = len(holdout) // 5
        result['top_score'] = scored(sorted(holdout, key=lambda r: (-r[1], r[0]))[:count])
        result['top_extraction_first'] = scored(sorted(holdout, key=lambda r: (-r[2], -r[1], r[0]))[:count])
        result['extraction_only'] = scored([r for r in holdout if r[2]])
        fold_results.append(result)
    return fold_results


def aggregate(folds, method):
    totals = {key: sum(r[method][key] for r in folds) for key in ('confirmed', 'false_supports')}
    count = totals['confirmed']
    risk = Fraction(totals['false_supports'], count) if count else None
    coverage = Fraction(count, sum(r['cases'] for r in folds))
    return {**totals, 'risk': float(risk) if risk is not None else None, 'coverage': float(coverage),
            'clears_gate': risk is not None and risk <= RISK_CAP and coverage >= COVERAGE}


def conditional_interval(folds):
    rng = random.Random(3750)
    draws = []
    for _ in range(10_000):
        sample = rng.choices(folds, k=len(folds))
        one = aggregate(sample, 'top_score')
        two = aggregate(sample, 'top_extraction_first')
        assert one['confirmed'] == two['confirmed']
        draws.append((one['false_supports'] - two['false_supports']) / one['confirmed'])
    ordered = sorted(draws)
    return [ordered[249], ordered[9749]]


def study(queue, results):
    articles, hashes = read_inputs(queue, results)
    folds = evaluate(articles)
    totals = {method: aggregate(folds, method) for method in folds[0] if method != 'cases'}
    counts = totals['extraction_only']
    assert (counts['confirmed'], counts['false_supports']) == (55, 19)
    tp = counts['confirmed'] - counts['false_supports']
    maximum = max(tp + k for k in range(counts['false_supports'] + 1) if Fraction(k, tp + k) <= RISK_CAP)
    return {'protocol_sha256': sha256(Path(__file__).with_name('ARTICLE_HOLDOUT_PROTOCOL.md').read_bytes()).hexdigest(),
            'inputs_sha256': hashes, 'folds': folds, 'aggregate': totals,
            'infeasible_training_folds': {name: sum(r[name]['threshold'] is None for r in folds)
                                          for name in ('score_threshold', 'extraction_then_threshold')},
            'paired_article_interval_fixed_capacity': conditional_interval(folds),
            'extraction_subset_oracle_maximum_confirmations': maximum}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--queue', required=True, type=Path)
    cli.add_argument('--results', required=True, type=Path)
    output = cli.add_mutually_exclusive_group(required=True)
    output.add_argument('--output', type=Path)
    output.add_argument('--verify', type=Path)
    args = cli.parse_args()
    result = study(args.queue, args.results)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(rendered)
    else:
        assert rendered == args.verify.read_text(), 'Article holdout replay differs'
    print(json.dumps({k: result[k] for k in ('aggregate', 'infeasible_training_folds',
                                         'paired_article_interval_fixed_capacity',
                                         'extraction_subset_oracle_maximum_confirmations')}, indent=2))
