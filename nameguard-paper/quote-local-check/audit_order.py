"""Exploratory comparison of both single prompt orders and their mean."""
import argparse
import json
from pathlib import Path

import numpy as np

from quote_local import CAPACITIES, counts, load


def effects(cohort, reverse, capacity):
    def quota(method):
        return cohort["methods"][method]["quotas"][str(capacity)]
    first = quota("single_score")["by_article_accepted_unsupported"]
    dual = quota("dual_mean")["by_article_accepted_unsupported"]
    other = reverse["quotas"][str(capacity)]["by_article_accepted_unsupported"]
    first, dual, other = map(np.array, (first, dual, other))
    assert np.all(first[:, 0] == dual[:, 0]) and np.all(first[:, 0] == other[:, 0])
    return first[:, 0], first[:, 1], dual[:, 1], other[:, 1]


def summarize(*panels):
    kept, first, dual, reversed_order = [np.concatenate(column) for column in zip(*panels, strict=True)]
    paired = dual - first
    total = int(kept.sum())
    rng = np.random.default_rng(51230)
    sample = rng.integers(0, len(kept), size=(9999, len(kept)))
    denominator = kept[sample].sum(axis=1)
    bootstrap = paired[sample].sum(axis=1)[denominator > 0] / denominator[denominator > 0]
    return {
        "accepted_each": total,
        "unsupported_first_order": int(first.sum()),
        "unsupported_dual_order": int(dual.sum()),
        "unsupported_reversed_order": int(reversed_order.sum()),
        "difference_first_vs_dual_percentage_points": float(100 * paired.sum() / total),
        "article_bootstrap_95_first_vs_dual_points": (100 * np.quantile(bootstrap, [.025, .975])).tolist(),
        "articles_first_better_than_dual": int((paired > 0).sum()),
        "articles_tied": int((paired == 0).sum()),
        "articles_dual_better_than_first": int((paired < 0).sum()),
    }


def audit(source, held_rows, reserve_rows):
    method = {name: counts(rows, np.array([row.margins[1] for row in rows]))
              for name, rows in (("holdout", held_rows), ("reserve", reserve_rows))}
    results = {}
    for capacity in CAPACITIES:
        panels = {name: effects(source[name], method[name], capacity) for name in method}
        results[str(capacity)] = {
            **{name: summarize(panel) for name, panel in panels.items()},
            "pooled": summarize(*panels.values()),
        }
    results["reverse_pair_accuracy"] = {name: record["pair_accuracy"] for name, record in method.items()}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--archived-results", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--output", type=Path)
    modes.add_argument("--verify", type=Path)
    args = parser.parse_args()
    holdout = load(args.queue / "queue/holdout-inputs.json", args.queue / "queue/holdout-labels.json",
                   args.archived_results / "registered/holdout-rows.json")
    reserve = load(args.queue / "reserve/rows.json", args.queue / "reserve/labels.json",
                   args.archived_results / "reserve/baseline.json")
    findings = audit(json.loads(args.results.read_text()), holdout, reserve)
    serial = json.dumps(findings, sort_keys=True, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serial)
    else:
        assert args.verify.read_text() == serial
    print(json.dumps(findings["20"], indent=2))
