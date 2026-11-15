"""Recompute utility, conformance, budget, and paired effects from local records."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import struct

from dependencies import ROOT, WORKSPACE
from coverage_cases import cases, DEV_NAMES, TEST_NAMES
from coverage_policy import ARMS, review
from witness import oracle_from_registry, prompt

STAGES = ("boundary", *(f"block_{i + 1}" for i in range(28)), "final_hidden", "vocabulary")


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_bits(logits):
    return struct.pack("!5f", *logits)


def trace_case(case, split):
    return split == "test" and case.family % 3 == 0 and case.kind in ("valid", "same_date_conflict")


def paired_interval(values):
    rng = random.Random(61116)
    estimates = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(8000))
    return {"mean": statistics.mean(values), "lower_95": estimates[200], "upper_95": estimates[7799],
            "families": len(values), "repetitions": len(estimates)}


def measure(directory):
    plan = read(directory / "plan.json")
    for relative, expected in plan["sha256"].items():
        if file_hash(WORKSPACE / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    cohort = tuple(cases(plan["split"]))
    names = DEV_NAMES if plan["split"] == "dev" else TEST_NAMES
    keys = {(case.key, name) for case in cohort for name in names}
    if list(names) != plan["names"] or [c.key for c in cohort] != plan["cases"]:
        raise ValueError("Case plan mismatch")
    rows = read(directory / "rows.json")
    case_rows = {(row["case"], row["name"]): row for row in rows}
    if len(case_rows) != len(rows) or set(case_rows) != keys:
        raise ValueError("Extra, missing, or duplicate observations")

    utility = defaultdict(Counter)
    traces = Counter()
    effects = Counter()
    family = defaultdict(list)
    case_results = defaultdict(dict)
    for case in cohort:
        expected = (case.expected_status, case.expected_ids)
        oracle = oracle_from_registry(case.target, case.entries)
        if (oracle.status, oracle.record_ids) != expected:
            raise ValueError(f"Authored outcome and complete rule disagree: {case.key}")
        grouped = [case_rows[case.key, name] for name in names]
        for index, row in enumerate(grouped):
            if row["primary"] != (index == 0) or row["family"] != case.family or row["kind"] != case.kind:
                raise ValueError(f"Malformed result metadata: {case.key}")
            if row["visible_prompt_sha256"] != hashlib.sha256(prompt(case).encode()).hexdigest():
                raise ValueError("Prompt provenance mismatch")
            if row["logits"] is not None and len(row["logits"]) != 5:
                raise ValueError("Expected five logits")
        complete = all(row["logits"] is not None for row in grouped)
        traces["admitted_name_requests"] += sum(r["logits"] is not None for r in grouped)
        traces["all_names_admitted_conditions"] += complete
        traces["bit_exact_logit_conditions"] += complete and len({score_bits(r["logits"]) for r in grouped}) == 1
        if trace_case(case, plan["split"]):
            traces["planned_direct_comparisons"] += (len(names) - 1) * len(STAGES)
            for row in grouped:
                traces["zero_identity_checks"] += row["zero_identity_checks"]
            for row in grouped[1:]:
                if row["direct_comparisons"] is not None:
                    if len(row["direct_comparisons"]) != len(STAGES):
                        raise ValueError("Incomplete activation observation")
                    traces["compared_aligned_arrays"] += len(row["direct_comparisons"])
                    traces["bit_exact_aligned_arrays"] += sum(row["direct_comparisons"])
        correctness = {}
        for arm in ARMS:
            reports = [(oracle, 0) if arm == "registry_only" else review(case, row["logits"], arm) for row in grouped]
            outcome, used = reports[0]
            result = (outcome.status, outcome.record_ids) if outcome is not None else None
            correct = result == expected
            correctness[arm] = correct
            case_results[arm][case.family, case.kind] = correct
            counters = utility[arm]
            counters["correct_status_and_ids"] += correct
            counters["correct_definitive"] += correct and expected[0] != "review"
            counters["correct_review"] += correct and expected[0] == "review"
            counters["verified_confirmations"] += correct and expected[0] == "confirmed"
            counters["verified_contradictions"] += correct and expected[0] == "contradicted"
            counters["wrong_definitive"] += outcome is not None and outcome.status != "review" and not correct
            counters["inspection_budget_max"] = max(counters["inspection_budget_max"], used)
            counters["proposals_total"] += used
            counters["all_reports_exact"] += all(report is not None for report, _ in reports) and (
                len({report.serialize() for report, _ in reports}) == 1)
        if expected[0] != "review":
            baseline = correctness["model_top_two"]
            repaired = correctness["latest_top_one"]
            effects["recovered"] += repaired and not baseline
            effects["lost"] += baseline and not repaired
            ids = [r.record_id for r in case.entries]
            effects["display_order_correct"] += bool(set(ids[:2]) & set(expected[1]))
            effects["sorted_id_correct"] += bool(set(sorted(ids)[:2]) & set(expected[1]))
            family[case.family].append(int(repaired) - int(baseline))
    definitive = sum(c.expected_status != "review" for c in cohort)
    runtime = read(directory / "runtime.json")
    if not runtime["checkpoint_files_verified"] or runtime["parameter_sha256"] != read(ROOT / "audit/checkpoint-manifest.json")["parameter_digest"]:
        raise ValueError("Checkpoint verification failed")
    fixed = utility["latest_top_one"]
    baseline = utility["model_top_two"]
    contrast = {key: {kind: sum(hit and decisions[family, "valid"]
                        for (family, name), hit in decisions.items() if name == kind)
                        for kind in sorted({c.kind for c in cohort} - {"valid"})}
                for key, decisions in case_results.items()}
    return {"split": plan["split"], "conditions": len(cohort), "decisive": definitive,
            "unresolved": len(cohort) - definitive, "families": len(family), "planned_requests": len(rows),
            "utility": {arm: dict(counts) for arm, counts in utility.items()},
            "invariance": dict(traces), "paired_rescues_controls": dict(effects),
            "uniform_top_two_expected_correct": definitive / 2,
            "family_paired_coverage_change": paired_interval([statistics.mean(v) for v in family.values()]),
            "both_correct_contrasts_with_valid": contrast,
            "gates": {
                "complete_coverage_with_gain": fixed["correct_definitive"] == definitive and
                                               fixed["correct_definitive"] > baseline["correct_definitive"],
                "no_wrong_definitive": fixed["wrong_definitive"] == 0,
                "all_reports_match_oracle": fixed["correct_status_and_ids"] == len(cohort),
                "exact_name_outcomes_and_scores": traces["admitted_name_requests"] == len(rows)
                    and traces["bit_exact_logit_conditions"] == fixed["all_reports_exact"] == len(cohort),
                "all_direct_comparisons_exact": traces["planned_direct_comparisons"] ==
                    traces["compared_aligned_arrays"] == traces["bit_exact_aligned_arrays"]},
            "runtime": runtime}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(json.dumps(measure(args.result), indent=2, allow_nan=False))
