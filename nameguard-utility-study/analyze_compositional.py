"""Verify compositional rows, measure matched-pair accuracy and family uncertainty."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from analyze import read, interval, WORKSPACE
from compositional_fixtures import AUDIT_NAMES, challenge
from experiment import digest
from methods import decision_report, classify

ARMS = ("native", "legacy", "plain", "fewshot", "ridge")


def summary(directory):
    plan = read(directory / "plan.json")
    for path, expected in plan["sources"].items():
        if digest(WORKSPACE / path) != expected:
            raise ValueError(f"Changed experiment dependency: {path}")
    cases = tuple(challenge("test"))
    if [case.key for case in cases] != plan["test_cases"]:
        raise ValueError("Fixture does not match plan")
    rows = read(directory / "rows.json")
    given = {(r["case"], r["name"]): r for r in rows}
    required = {(c.key, name) for c in cases for name in (AUDIT_NAMES if c.family % 4 == 0 else AUDIT_NAMES[:1])}
    if len(given) != len(rows) or set(given) != required:
        raise ValueError("Missing, duplicate, or extra case/name rows")
    primary = []
    equality = Counter()
    for case in cases:
        block = [given[case.key, name] for name in AUDIT_NAMES if (case.key, name) in given]
        for index, row in enumerate(block):
            if (row["family"], row["kind"], row["status"], row["primary"]) != (case.family, case.kind, case.status, index == 0):
                raise ValueError("Case metadata does not match fixture")
            if any(row[k] != classify(row[k + ("_scores" if k == "ridge" else "_logits")]) for k in ("ridge","plain","fewshot")):
                raise ValueError("Stored predictions are inconsistent")
            if row["report"] != decision_report(case.task.credential, row["ridge"]).as_bytes().decode():
                raise ValueError("Stored explanation differs from the report renderer")
        primary.append(block[0])
        if len(block) > 1:
            equality["contexts_checked"] += 1
            for field in ("hidden_sha256", "ridge_scores", "fewshot_logits", "plain_logits", "report"):
                equality[field] += len({json.dumps(row[field], sort_keys=True) for row in block}) == 1
    row_by_case = {r["case"]: r for r in primary}
    score = {}
    for arm in ARMS:
        correct = [r for r in primary if r[arm] == r["status"]]
        differences = defaultdict(list)
        for r in primary:
            differences[r["family"]].append(int(r[arm] == r["status"]) - int(r["legacy"] == r["status"]))
        outcome_pairs = {}
        for kind in ("subject","time","negation"):
            matched = [(row_by_case[f"{i}-{kind}_0"], row_by_case[f"{i}-{kind}_1"])
                       for i in range(plan["test_families"])]
            outcome_pairs[kind] = {"both_correct": sum(a[arm] == a["status"] and b[arm] == b["status"] for a,b in matched),
                                   "same_decision_on_reversal": sum(a[arm] == b[arm] for a,b in matched)}
        score[arm] = {"correct": len(correct), "known_correct": sum(r["status"] != "review" for r in correct),
                      "correct_referrals": sum(r["status"] == "review" for r in correct),
                      "unresolved_false_confirmations": sum(r["status"] == "review" and r[arm] == "confirmed" for r in primary),
                      "by_kind": dict(Counter(r["kind"] for r in correct)),
                      "pairs": outcome_pairs,
                      "paired_accuracy_change_vs_legacy": interval([sum(f) / len(f) for f in differences.values()])}
    return {"families": plan["test_families"], "primary_cases": len(primary), "known_cases": sum(r["status"] != "review" for r in primary),
            "scored_name_variants": len(rows), "exact_across_names": dict(equality), "utility": score,
            "fit": read(directory / "fit.json"), "runtime": read(directory / "runtime.json")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    output = json.dumps(summary(a.directory), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if a.output:
        with a.output.open("x", encoding="utf-8") as stream:
            stream.write(output)
    else:
        print(output, end="")
