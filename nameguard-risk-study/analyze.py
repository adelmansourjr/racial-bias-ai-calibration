"""Verify registered observations, freeze accepted-set audit, then reveal labels."""
import argparse
from fractions import Fraction
from pathlib import Path

from prepare import ROOT, digest, read, save
from risk import MODES, auc, decisions, sample, upper_error_count
from run import STAGES, bits, make_thresholds

PARAMETERS = "c6c622310d66489a5a9de5f438c23063bf784b3031f2e8af64bfd60da0394b40"


def verify_rows(rows, ids):
    if sorted(rows) != ids:
        raise ValueError("Measured IDs differ from frozen list")
    for row in rows.values():
        if row["admitted"] and (len(row["logits"]) != 2 or len(row["logit_bits"]) != 2):
            raise ValueError("Incomplete admitted result")
        if [bits(pair) for pair in row["logits"]] != row["logit_bits"]:
            raise ValueError("Logit values and bits disagree")


def metrics(rows, labels, mode, accepted):
    errors = sum(not labels[key] for key in accepted)
    correct = len(accepted) - errors
    return {
        "accepted": len(accepted), "correct": correct, "false_support": errors,
        "coverage": len(accepted) / len(rows),
        "false_support_fraction": errors / len(accepted) if accepted else None,
        "recall_among_supported": correct / sum(labels.values()),
        "auc_among_admitted": auc(rows, labels, mode) if mode else 0.5,
    }


def audit_inputs(results, holdout, expected, mode, rule, acceptance):
    panel = read(results / "name-panel.json")
    if set(panel) != set(expected["conformance"]):
        raise ValueError("Conformance IDs do not match")
    stages = tuple(sorted(STAGES))
    exact = stage_checks = native_flips = native_logit_changes = 0
    name_count = len(expected["names"])
    for key, rows in panel.items():
        if len(rows) != name_count:
            raise ValueError("Wrong number of conformance names")
        for fields in ("logits", "admitted", "logit_bits", "report_sha256", "accepted"):
            if rows[0][fields] != holdout[key][fields]:
                raise ValueError("Primary measurement differs from name-panel anchor")
        for row in rows:
            if len(row["native"]) != 2 or len(row["report_sha256"]) != 64:
                raise ValueError("Native or report result missing")
            if row["admitted"] and (len(row["traces"]) != 2 or any(
                tuple(sorted(trace)) != stages or any(len(h) != 64 for h in trace.values())
                for trace in row["traces"]
            )):
                raise ValueError("Activation traces missing")
            verify_rows({key: row}, [key])
            required = bool(decisions({key: row}, mode, rule["threshold"]))
            if row["accepted"] != required:
                raise ValueError("Name-panel report used an unexpected decision")
        if all(row["admitted"] for row in rows):
            stage_checks += sum(x[s] == y[s] for row in rows[1:]
                                for x, y in zip(rows[0]["traces"], row["traces"], strict=True)
                                for s in STAGES)
            if all(all(row[field] == rows[0][field] for field in
                       ("logit_bits", "traces", "report_sha256", "accepted")) for row in rows[1:]):
                exact += 1
        native_flips += len({r["native"][0] >= r["native"][1] for r in rows}) > 1
        native_logit_changes += len({bits(r["native"]) for r in rows}) > 1
    if sorted(key for key, row in holdout.items() if row["accepted"]) != acceptance:
        raise ValueError("Holdout predictions do not match frozen threshold")
    return {"cases": len(panel), "names_per_case": name_count, "exact_orbits": exact,
            "stage_digest_matches": stage_checks, "native_unthresholded_flips": native_flips,
            "native_logit_changes": native_logit_changes}


def checked_labels(path, plan, split):
    labels = read(path)
    if (digest(path) != plan["files"][f"{split}-labels.json"]
            or sorted(labels) != plan["cases"][split]
            or any(type(y) is not bool for y in labels.values())
            or sum(labels.values()) * 2 != len(labels)):
        raise ValueError("Labels differ from frozen balanced cases")
    return labels


def analyze(plan_path, results, development_labels, holdout_labels, output):
    plan = read(plan_path)
    runtime = read(results / "runtime.json")
    if (runtime["parameter_sha256"] != PARAMETERS or not runtime["checkpoint_verified"]
            or runtime["plan_sha256"] != digest(plan_path)):
        raise ValueError("Checkpoint or plan verification failed")
    thresholds = read(results / "thresholds.json")
    development = read(results / "development-rows.json")
    holdout = read(results / "holdout-rows.json")
    verify_rows(development, plan["cases"]["development"])
    verify_rows(holdout, plan["cases"]["holdout"])
    dev_labels = checked_labels(development_labels, plan, "development")
    if make_thresholds(development, dev_labels) != thresholds:
        raise ValueError("Development thresholds do not reproduce")

    mode = thresholds["primary"]
    rule = thresholds["rules"][mode]
    accepted = decisions(holdout, mode, rule["threshold"])
    panel = audit_inputs(results, holdout, plan, mode, rule, accepted)
    random_order = plan["audit_order"]
    if sorted(random_order) != sorted(holdout):
        raise ValueError("Audit order is not a permutation of holdout IDs")
    audit = sample(accepted, random_order, plan["design"]["audit_max"])
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "audit-selection.json"
    selection = {"rule": rule, "accepted_ids": accepted, "audit_ids": audit,
                 "plan_sha256": digest(plan_path), "observations_sha256": digest(results / "holdout-rows.json")}
    if selection_path.exists():
        if read(selection_path) != selection:
            raise ValueError("Existing audit selection differs")
    else:
        save(selection_path, selection, freeze=True)

    labels = checked_labels(holdout_labels, plan, "holdout")
    n, m = len(accepted), len(audit)
    x = sum(not labels[key] for key in audit)
    upper = upper_error_count(n, m, x, Fraction(plan["design"]["alpha"])) if n else None
    study = {name: metrics(holdout, labels, name, decisions(holdout, name, thresholds["rules"][name]["threshold"]))
             for name in MODES}
    study["literal_span"] = metrics(holdout, labels, None, sorted(holdout))
    checks = {
        "coverage": study[mode]["coverage"] >= plan["design"]["coverage_min"],
        "finite_audit": upper is not None and Fraction(upper, n) <= Fraction(str(plan["design"]["risk_upper"])),
        "exact_name_orbits": panel["exact_orbits"] == len(plan["conformance"]),
    }
    record = dict(primary=mode, dev_rules=thresholds["rules"], holdout=study, panel=panel,
                  holdout_cases=len(holdout), holdout_admitted=sum(r["admitted"] for r in holdout.values()),
                  development_cases=len(development), fit_check=checks, gate_pass=all(checks.values()),
                  sample=dict(N=n, m=m, x=x, upper_count=upper, upper_fraction=upper / n if n else None,
                              alpha=plan["design"]["alpha"]),
                  audit_covers_census=upper is not None and study[mode]["false_support"] <= upper)
    save(output / "metrics.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "audit/plan.json")
    parser.add_argument("--results", type=Path, default=ROOT / "results/registered")
    parser.add_argument("--development-labels", type=Path, default=ROOT / "data/queue/development-labels.json")
    parser.add_argument("--holdout-labels", type=Path, default=ROOT / "data/queue/holdout-labels.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    a = parser.parse_args()
    print(analyze(a.plan, a.results, a.development_labels, a.holdout_labels, a.output))
