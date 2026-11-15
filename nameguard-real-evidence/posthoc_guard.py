"""Study two additional actor guards on already inspected material."""
from collections import Counter
import json
from analyze import RUNS
from sample import ROOT, read_json
from evidence import check_provenance
from actor_guard import verify_excerpt


def summarize(private=None):
    private = ROOT / "local-private" if private is None else private
    results = {}
    for name, (cohort_path, labels_path) in RUNS.items():
        seen = read_json(private / cohort_path)
        if isinstance(seen, dict):
            seen = seen["test"]
        labels = read_json(private / labels_path)
        score = Counter(correct=0, unsupported=0, missed=0, review=0)
        for case in seen:
            result = verify_excerpt(case["excerpt"], case["tool"])
            truth = bool(labels[case["case"]])
            sent = result.decision == "supported"
            if not check_provenance(case["excerpt"], result):
                raise ValueError("Untraceable report")
            score["correct"] += sent and truth
            score["unsupported"] += sent and not truth
            score["missed"] += not sent and truth
            score["review"] += not sent
        results[name] = dict(score, cases=len(seen))
    return {"scope":"Post-hoc on inspected excerpts. Source guard only; no independent test or new GPU run.", "results":results}


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
