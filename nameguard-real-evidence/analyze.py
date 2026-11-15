"""Audit local ground truth and exact GPU results. Emit aggregates only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import struct

from sample import ROOT, WORKSPACE, NAMES, read_json
from evidence import check_provenance

RUNS = {
    "initial": ("cohort.json", "test-labels.json"),
    "fresh": ("fresh-cohort.json", "fresh-labels.json"),
    "confirm": ("confirm-cohort.json", "confirm-labels.json"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_model(scores):
    return scores is not None and scores[0] > scores[1]


def analyze(which, private=None):
    private = ROOT / "local-private" if private is None else private
    cohort_file, label_file = RUNS[which]
    cohort = read_json(private / cohort_file)
    if isinstance(cohort, dict):
        cohort = cohort["test"]
    labels = read_json(private / label_file)
    run = private / (which + "-run")
    plan = read_json(run / "plan.json")
    if plan["cohort_sha256"] != sha(private / cohort_file) or plan["labels_sha256"] != sha(private / label_file):
        raise ValueError("Frozen cohort or label drift")
    for rel, expected in plan["sources"].items():
        if sha(WORKSPACE / rel) != expected:
            raise ValueError(f"Frozen source changed: {rel}")
    cases = {c["case"]: c for c in cohort}
    if len(cases) != len(cohort) or set(cases) != set(labels):
        raise ValueError("Reference coverage differs from cohort")
    if len(set(c["source_id"] for c in cohort)) != len(cohort):
        raise ValueError("CV repeats in primary cohort")
    rows = read_json(run / "rows.json")
    records = {(r["case"], r["name"]): r for r in rows}
    if len(records) != len(rows) or set(records) != {(case, name) for case in cases for name in NAMES}:
        raise ValueError("Duplicate or incomplete case/name records")
    checker = importlib.import_module(plan["verifier"]).verify_excerpt
    counts = Counter()
    utility = defaultdict(Counter)
    by_tool = defaultdict(Counter)
    for index, (key, item) in enumerate(cases.items()):
        gold = labels[key]
        if gold and gold not in item["excerpt"]:
            raise ValueError("Gold citation lost its source")
        group = [records[key, name] for name in NAMES]
        for rank,row in enumerate(group):
            if row["primary"] != (rank == 0) or row["tool"] != item["tool"] or row["band"] != item["band"]:
                raise ValueError("Cohort metadata drift")
        complete = all(row["protected"] is not None for row in group)
        counts["all_variants_admitted"] += complete
        counts["admitted_requests"] += sum(row["protected"] is not None for row in group)
        counts["native_decision_variation"] += len({label_model(row["native"]) for row in group}) > 1
        counts["native_logit_variation"] += len({tuple(row["native"]) for row in group}) > 1
        if complete:
            bits = [struct.pack("!2f", *row["protected"]).hex() for row in group]
            if bits != [row["protected_bits"] for row in group]:
                raise ValueError("Recorded score bytes are inconsistent")
            counts["exact_scores"] += len(set(bits)) == 1
            counts["exact_reports"] += len({r["report"] for r in group}) == 1
        if index % 8 == 0:
            counts["planned_direct_array_checks"] += 3 * 31
            counts["zero_identity_checks"] += sum(r["zero_checks"] for r in group)
            for row in group[1:]:
                checked = row["comparisons"] or []
                if checked and len(checked) != 31:
                    raise ValueError("Incomplete activation comparisons")
                counts["direct_arrays_compared"] += len(checked)
                counts["exact_direct_arrays"] += sum(checked)
        finding = checker(item["excerpt"],item["tool"])
        if not check_provenance(item["excerpt"],finding):
            raise ValueError("Evidence quote cannot be reconstructed")
        if any(r["report"] != finding.bytes().decode() for r in group if r["protected"] is not None):
            raise ValueError("Released report differs from source verification")
        gold_positive = bool(gold)
        verified = finding.decision == "supported"
        counts["labeled_support"] += gold_positive
        counts["verbatim_verified_spans"] += verified
        if verified and gold_positive:
            a = item["excerpt"].index(gold)
            b = a + len(gold)
            counts["true_positive_span_overlap"] += finding.evidence.start < b and a < finding.evidence.end
        primary = group[0]
        named = label_model(primary["native"])
        isolated = label_model(primary["protected"])
        outputs = {"mention_only": True, "native_model": named, "protected_model": isolated,
                   "protected_verified": verified and primary["protected"] is not None,
                   "verified_and_model_agrees": verified and isolated,
                   "verifier_only": verified}
        for method,predicted in outputs.items():
            u=utility[method]
            u["definitive"] += predicted
            u["correct_support"] += predicted and gold_positive
            u["unsupported_definitive"] += predicted and not gold_positive
            u["correct_review"] += not predicted and not gold_positive
            u["missed_support"] += not predicted and gold_positive
        by_tool[item["tool"]]["cases"] += 1
        by_tool[item["tool"]]["labeled_support"] += gold_positive
        by_tool[item["tool"]]["correct_verified"] += gold_positive and outputs["protected_verified"]
        by_tool[item["tool"]]["unsupported_verified"] += not gold_positive and outputs["protected_verified"]
    runtime = read_json(run / "runtime.json")
    if not runtime["checkpoint_verified"] or runtime["parameter_sha256"] != read_json(ROOT / "audit/checkpoint-manifest.json")["parameter_digest"]:
        raise ValueError("Model parameter integrity failed")
    denom = counts["labeled_support"]
    for metric in utility.values():
        metric["positive_recall"] = metric["correct_support"] / denom if denom else None
        metric["definitive_coverage"] = metric["definitive"] / len(cohort)
    return {"run": which, "verifier": plan["verifier"], "independent_cv_excerpts":len(cohort),
            "planned_name_requests":len(rows), "names":NAMES, "audit":dict(counts),
            "decisions":{a:dict(b) for a,b in utility.items()},
            "by_tool":{a:dict(b) for a,b in by_tool.items()},
            "gates":{
                "exact_logit_and_report_equality_with_full_admission": counts["admitted_requests"] == len(rows)
                    and counts["exact_scores"] == counts["exact_reports"] == len(cohort),
                "all_planned_arrays_match": counts["exact_direct_arrays"] ==
                                            counts["direct_arrays_compared"] == counts["planned_direct_array_checks"],
                "zero_unsupported_and_minimum_60_percent_recall":
                     utility["protected_verified"]["unsupported_definitive"] == 0 and
                     bool(denom) and 5 * utility["protected_verified"]["correct_support"] >= 3 * denom},
            "runtime":runtime,
            "reference":"Single output-blinded AI annotator. CV documentary claim only. Not verified experience or a hiring label."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", choices=tuple(RUNS))
    p.add_argument("--private", type=Path)
    a=p.parse_args()
    print(json.dumps(analyze(a.run,a.private), indent=2, allow_nan=False))
