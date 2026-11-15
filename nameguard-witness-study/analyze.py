"""Rebuild planned metrics from logged model outputs and written ground truth."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

from study_cases import cases, DEV_NAMES, NAMES, rendered
from witness import LETTERS, checked, unchecked, oracle_from_registry

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def pick(logits):
    if logits is None:
        return None
    if len(logits) != len(LETTERS):
        raise ValueError("Unexpected logit dimension")
    values = [i for i, value in enumerate(logits) if value == max(logits)]
    return LETTERS[values[0]] if len(values) == 1 else "E"


def pair_ci(values):
    generator = random.Random(260630)
    sample = sorted(statistics.mean(generator.choices(values, k=len(values))) for _ in range(8000))
    return {"mean": statistics.mean(values), "lower_95": sample[200], "upper_95": sample[7799],
            "units": len(values), "resamples": len(sample)}


def correct(result, expected, exact_ids=False):
    return result is not None and result.status == expected.expected_status and (
        not exact_ids or result.record_ids == expected.expected_ids)


def analyze(out):
    plan = read(out / "plan.json")
    for path, digest in plan["sources"].items():
        if hashlib.sha256((ROOT.parent / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source changed: {path}")
    cohort = tuple(cases(plan["split"]))
    names = NAMES if plan["split"] == "test" else DEV_NAMES
    if plan["case_ids"] != [case.key for case in cohort] or plan["names"] != list(names):
        raise ValueError("Fixture plan mismatch")
    rows = read(out / "rows.json")
    by_key = {(row["case"], row["name"]): row for row in rows}
    if len(rows) != len(by_key) or set(by_key) != {(c.key, n) for c in cohort for n in names}:
        raise ValueError("Unexpected or missing case/name rows")
    count = Counter()
    utility = defaultdict(Counter)
    by_kind = defaultdict(Counter)
    reports_by_family = {}
    changes = defaultdict(list)
    for case in cohort:
        registry = oracle_from_registry(case.target, case.entries)
        if not correct(registry, case, exact_ids=True):
            raise ValueError(f"Checker disagrees with intervention oracle: {case.key}")
        group = [by_key[case.key, name] for name in names]
        for j, row in enumerate(group):
            if (row["primary"], row["family"], row["kind"], row["visible_records_sha256"]) != (
                j == 0, case.family, case.kind, hashlib.sha256(rendered(case).encode()).hexdigest()
            ):
                raise ValueError("Row metadata or document binding mismatch")
        complete = all(r["protected_logits"] is not None for r in group)
        count["planned_name_rows"] += len(group)
        count["admitted_name_rows"] += sum(r["protected_logits"] is not None for r in group)
        count["complete_conditions"] += complete
        count["native_logit_variation"] += len({tuple(r["native_logits"]) for r in group}) > 1
        count["native_witness_change"] += len({pick(r["native_logits"]) for r in group}) > 1
        count["native_unchecked_status_change"] += len({unchecked(case, pick(r["native_logits"])).status for r in group}) > 1
        if complete:
            count["protected_logit_equality"] += all(r["protected_logits"] == group[0]["protected_logits"] for r in group)
            rendered_reports = [checked(case, pick(r["protected_logits"])).serialize() for r in group]
            count["checked_report_byte_equality"] += len(set(rendered_reports)) == 1
        if case.family in plan["trace_families"]:
            count["direct_activation_conditions"] += 1
            count["identical_direct_activation_conditions"] += all(r["stage_equality"] == [True] * 31 for r in group[1:])
            count["direct_stage_comparisons"] += sum(len(r["stage_equality"] or ()) for r in group[1:])
            count["exact_direct_stage_comparisons"] += sum(sum(r["stage_equality"] or ()) for r in group[1:])
            count["zero_identity_stage_checks"] += sum(r["zero_identity_checks"] for r in group)
        primary = group[0]
        named = pick(primary["native_logits"])
        isolated = pick(primary["protected_logits"])
        brief = pick(primary["brief_logits"])
        arms = {"native_unchecked": unchecked(case, named), "native_checked": checked(case, named),
                "protected_unchecked": unchecked(case, isolated) if isolated else None,
                "protected_checked": checked(case, isolated) if isolated else None,
                "brief_checked": checked(case, brief) if brief else None, "registry_only": registry}
        reports_by_family[case.family, case.kind] = (case, arms)
        for arm, result in arms.items():
            u = utility[arm]
            u["status_correct"] += correct(result, case)
            u["status_and_evidence_correct"] += correct(result, case, exact_ids=True)
            u["definitive"] += result is not None and result.status != "review"
            u["wrong_definitive"] += result is not None and result.status != "review" and not correct(result, case)
            u["false_confirmations"] += result is not None and result.status == "confirmed" and case.expected_status != "confirmed"
            u["verifiable_covered"] += case.expected_status != "review" and correct(result, case)
            u["unresolved_referred"] += case.expected_status == "review" and correct(result, case)
            by_kind[case.kind][arm] += correct(result, case)
        changes[case.family].append(int(correct(arms["protected_checked"], case)) - int(correct(arms["protected_unchecked"], case)))
    contrast = defaultdict(Counter)
    for (family, kind), (right, arms_right) in reports_by_family.items():
        if kind == "valid":
            continue
        left, arms_left = reports_by_family[family, "valid"]
        for arm in arms_right:
            contrast[kind][arm] += correct(arms_left[arm], left) and correct(arms_right[arm], right)
    decision = utility["protected_checked"]
    verifiable = sum(c.expected_status != "review" for c in cohort)
    runtime = read(out / "runtime.json")
    checkpoint_digest = read(ROOT / "audit/checkpoint-manifest.json")["parameter_digest"]
    if not runtime["checkpoint_files_verified"] or runtime["parameter_sha256"] != checkpoint_digest:
        raise ValueError("Checkpoint integrity failed")
    return {"split": plan["split"], "conditions": len(cohort), "verifiable": verifiable,
            "unresolved": len(cohort) - verifiable, "invariance": dict(count),
            "utility": {a: dict(u) for a, u in utility.items()}, "correct_by_kind": dict(by_kind),
            "contrasts_both_correct_vs_valid": dict(contrast),
            "paired_status_change_checked_vs_unchecked": pair_ci([statistics.mean(v) for v in changes.values()]),
            "gates": {"all_names_admit_and_equal": count["complete_conditions"] == count["protected_logit_equality"] ==
                      count["checked_report_byte_equality"] == len(cohort),
                      "no_wrong_definitive": decision["wrong_definitive"] == 0,
                      "at_least_half_definitive_coverage": 2 * decision["verifiable_covered"] >= verifiable},
            "runtime": runtime}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.out), indent=2, allow_nan=False))
