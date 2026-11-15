"""Reveal reserve labels after freezing observed accepts and the finite audit sample."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
from pathlib import Path

from analyze import PARAMETERS, metrics, verify_rows
from prepare import ROOT, digest, read, save
from risk import decisions, ranking, sample, upper_error_count
from run import evidence_bytes


def audit_contrasts(panel, observed, plan):
    if sorted(panel) != sorted(plan["conformance"]):
        raise ValueError("Unexpected conformance cohort")
    exact = checks = 0
    fields = ("admitted", "accepted", "stopped", "generated_count",
              "generated_sha256", "report_sha256", "step_sha256", "public_prefix_sha256")
    for key, rows in panel.items():
        if rows[0] != observed[key] or len(rows) != len(plan["names"]):
            raise ValueError("Anchor row or panel size differs")
        all_exact = all(r["admitted"] for r in rows)
        for row in rows[1:]:
            if row["admitted"] and rows[0]["admitted"]:
                recorded = all(row[field] == rows[0][field] for field in fields)
                if recorded != row["direct_exact"]:
                    raise ValueError("Direct and digest comparisons disagree")
                if recorded:
                    checks += len(row["step_sha256"])
            all_exact = all_exact and row["direct_exact"]
        exact += all_exact
    return {"cases": len(panel), "exact_orbits": exact, "variants_per_case": len(plan["names"]),
            "direct_full_vocabulary_step_matches": checks}


def analyze(plan_path, result_dir, queue, thresholds_path, output):
    plan = read(plan_path)
    data = read(result_dir / "observed.json")
    baseline = read(result_dir / "baseline.json")
    panel = read(result_dir / "name-panel.json")
    runtime = read(result_dir / "runtime.json")
    if (not runtime["checkpoint_verified"] or runtime["parameters"] != PARAMETERS
            or runtime["plan_sha256"] != digest(plan_path)
            or runtime["thresholds_sha256"] != digest(thresholds_path)):
        raise ValueError("Runtime binding mismatch")
    if sorted(data) != plan["cases"] or sorted(plan["audit_order"]) != plan["cases"]:
        raise ValueError("Incorrect reserve IDs")
    verify_rows(baseline, plan["cases"])
    audited_contrasts = audit_contrasts(panel, data, plan)
    accepted = sorted(key for key, row in data.items() if row["accepted"])
    if any(not data[key]["admitted"] or not data[key]["stopped"] for key in accepted):
        raise ValueError("Acceptance without an admitted terminated answer")
    choice = sample(accepted, plan["audit_order"], plan["design"]["audit_max"])
    selection = dict(accepted_ids=accepted, audit_ids=choice, plan_sha256=digest(plan_path),
                     observations_sha256=digest(result_dir / "observed.json"))
    output.mkdir(parents=True, exist_ok=True)
    seal = output / "reserve-audit-selection.json"
    if not seal.exists():
        save(seal, selection, freeze=True)
    elif read(seal) != selection:
        raise ValueError("Reserve seal mismatch")

    label_path, row_path = queue / "labels.json", queue / "rows.json"
    if digest(label_path) != plan["labels_sha256"] or digest(row_path) != plan["inputs_sha256"]:
        raise ValueError("Reserve source manifest mismatch")
    labels, inputs = read(label_path), read(row_path)
    if (sorted(labels) != plan["cases"] or sum(labels.values()) * 2 != len(labels)
            or any(type(label) is not bool for label in labels.values())):
        raise ValueError("Invalid balanced labels")
    for key, row in data.items():
        report = evidence_bytes(inputs[key], row["accepted"])
        if row["report_sha256"] != hashlib.sha256(report).hexdigest():
            raise ValueError("Evidence report does not reconstruct from the official source")
    strata = {article: Counter() for article in plan["articles"]}
    for key, classification in data.items():
        bucket = strata[inputs[key]["article"]]
        bucket.update(cases=1, supported=int(labels[key]), accepted=int(classification["accepted"]),
                      false_support=int(classification["accepted"] and not labels[key]))
    for key, counter in strata.items():
        strata[key] = dict(counter)
    n, m = len(accepted), len(choice)
    x = sum(not labels[key] for key in choice)
    upper = upper_error_count(n, m, x, Fraction(plan["design"]["alpha"])) if n else None
    stats = metrics(data, labels, None, accepted)
    del stats["auc_among_admitted"]
    threshold = read(thresholds_path)["rules"]["mean"]["threshold"]
    base_ids = decisions(baseline, "mean", threshold)
    if base_ids != sorted(key for key, row in baseline.items() if row["candidate_aware_accept"]):
        raise ValueError("Inconsistent baseline decisions")
    at_matched_coverage = sorted(key for _, key in ranking(baseline, "mean")[:n])
    true_positive_rate = Fraction(stats["correct"], sum(labels.values()))
    false_positive_rate = Fraction(stats["false_support"], len(labels) - sum(labels.values()))
    risk_target = Fraction(str(plan["design"]["risk_upper"]))
    numerator = (1 - risk_target) * false_positive_rate
    denominator = numerator + risk_target * true_positive_rate
    sensitivity = {}
    for prevalence in (Fraction(1, 10), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
        bad = (1 - prevalence) * false_positive_rate
        total = bad + prevalence * true_positive_rate
        sensitivity[str(prevalence)] = float(bad / total) if total else None
    tests = dict(coverage=stats["coverage"] >= plan["design"]["coverage_min"],
                 finite_audit=upper is not None and Fraction(upper, n) <= Fraction(str(plan["design"]["risk_upper"])),
                 exact_name_orbits=audited_contrasts["exact_orbits"] == len(plan["conformance"]))
    results = dict(
        fit_check=tests, gate_pass=all(tests.values()), panel=audited_contrasts,
        cases=len(data), admitted=sum(row["admitted"] for row in data.values()),
        terminated=sum(row.get("stopped", False) for row in data.values()),
        extraction=stats, candidate_aware=metrics(baseline, labels, "mean", base_ids),
        literal=metrics(data, labels, None, sorted(data)), by_article=strata,
        posthoc_matched_count_candidate_aware=metrics(baseline, labels, "mean", at_matched_coverage),
        prevalence_sensitivity=dict(tpr=float(true_positive_rate), fpr=float(false_positive_rate),
                                    error_by_prevalence=sensitivity,
                                    minimum_for_target=float(numerator / denominator) if denominator else None),
        sample=dict(N=n, m=m, x=x, upper_count=upper, upper_fraction=upper / n if n else None),
        audit_covers_census=upper is not None and stats["false_support"] <= upper,
    )
    save(output / "reserve-metrics.json", results)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=ROOT / "audit/reserve-plan.json")
    p.add_argument("--results", type=Path, default=ROOT / "results/reserve")
    p.add_argument("--queue", type=Path, default=ROOT / "data/reserve")
    p.add_argument("--thresholds", type=Path, default=ROOT / "results/registered/thresholds.json")
    p.add_argument("--output", type=Path, default=ROOT / "results")
    a = p.parse_args()
    print(analyze(a.plan, a.results, a.queue, a.thresholds, a.output))
