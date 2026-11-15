"""Paired family-level analysis and exact conformance verification."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

from fixtures import DEV_NAMES, TEST_NAMES, scenarios
from methods import decision_report

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
BINARY = ("native", "legacy", "binary_forward", "binary_reverse", "paired", "deleted_paired")
THREE_WAY = ("single", "cyclic")


def read(path):
    return json.loads(path.read_text())


def prediction(row, arm):
    if arm == "native":
        return row["native_prediction"]
    if arm in THREE_WAY:
        return row.get(f"{arm}_status")
    margin = row.get(f"{arm}_margin")
    if margin is None:
        return None
    return margin < 0 if arm == "binary_reverse" else margin >= 0


def outcome(row, arm):
    if arm in THREE_WAY:
        return prediction(row, arm)
    value = prediction(row, arm)
    return ("confirmed" if value else "contradicted") if value is not None else None


def interval(differences):
    rng = random.Random(20261116)
    averages = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(8000))
    return {"mean": statistics.mean(differences), "lower_95": averages[200], "upper_95": averages[7799],
            "families": len(differences), "resamples": len(averages)}


def paired(rows, treatment, baseline):
    counts = Counter()
    families = defaultdict(list)
    for row in rows:
        t = outcome(row, treatment) == row["status"]
        b = outcome(row, baseline) == row["status"]
        counts["improved"] += t and not b
        counts["worsened"] += b and not t
        families[row["family"]].append(int(t) - int(b))
    return {**counts, "family_bootstrap": interval([statistics.mean(v) for v in families.values()])}


def analyze(directory):
    plan = read(directory / "plan.json")
    split = plan["split"]
    for relative, expected in plan["sources"].items():
        if hashlib.sha256((WORKSPACE / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source differs: {relative}")
    rows = read(directory / "rows.json")
    actual = {(row["case"], row["name"]): row for row in rows}
    cases = tuple(scenarios(split))
    names = TEST_NAMES if split == "test" else DEV_NAMES
    expected = {(case.key, name) for case in cases
                for name in (names if split == "test" and case.family % 4 == 0 else names[:1])}
    if len(rows) != len(actual) or set(actual) != expected or plan["case_ids"] != [case.key for case in cases]:
        raise ValueError("Incomplete or duplicate planned cohort")
    primary = []
    exact = Counter()
    names_with_trace = []
    for case in cases:
        block = [actual[case.key, name] for name in names if (case.key, name) in actual]
        if any((row["status"], row["kind"], row["family"], row["primary"]) !=
               (case.status, case.kind, case.family, i == 0) for i, row in enumerate(block)):
            raise ValueError(f"Invalid fixture mapping: {case.key}")
        primary.append(block[0])
        if len(block) == 1:
            continue
        exact["contexts"] += 1
        exact["complete_admission"] += all(row["admitted"] for row in block)
        if all(row["admitted"] for row in block):
            expected_stages = ["boundary"] + [f"block_{i + 1}" for i in range(28)] + ["final_hidden", "vocabulary"]
            def stage_names_valid(row):
                arms = {"legacy", "binary_0", "binary_1", "cycle_0", "cycle_1", "cycle_2"}
                return set(row["stages"]) == arms and all(
                    [stage[0] for stage in stages] == expected_stages for stages in row["stages"].values())
            if not all(stage_names_valid(row) for row in block):
                raise ValueError("Incomplete ordered activation observations")
            exact["exact_aligned_stages"] += all(row["stages"] == block[0]["stages"] for row in block)
            exact["exact_paired_scores"] += all(row["paired_margin"] == block[0]["paired_margin"] for row in block)
            exact["exact_cyclic_scores"] += all(row["cyclic_scores"] == block[0]["cyclic_scores"] for row in block)
            exact["exact_report_bytes"] += all(row["report"] == block[0]["report"] ==
                decision_report(case.task.credential, row["cyclic_status"]).as_bytes().decode() for row in block)
            exact["stage_fingerprint_comparisons"] += (len(block) - 1) * 6 * len(expected_stages)
            names_with_trace.extend(row["name"] for row in block)

    known = [row for row in primary if row["status"] != "review"]
    unknown = [row for row in primary if row["status"] == "review"]
    utility = {}
    for arm in BINARY + THREE_WAY:
        breakdown = defaultdict(lambda: {"correct": 0, "total": 0})
        for row in primary:
            breakdown[row["kind"]]["correct"] += outcome(row, arm) == row["status"]
            breakdown[row["kind"]]["total"] += 1
        confusion = Counter((row["status"], outcome(row, arm) or "rejected") for row in primary)
        utility[arm] = {
            "known_correct": sum(outcome(row, arm) == row["status"] for row in known),
            "all_correct": sum(outcome(row, arm) == row["status"] for row in primary),
            "correct_referrals": sum(outcome(row, arm) == "review" for row in unknown),
            "unresolved_false_confirmations": sum(outcome(row, arm) == "confirmed" for row in unknown),
            "wrong_terminal_known_decisions": sum(outcome(row, arm) not in (row["status"], "review", None) for row in known),
            "known_referred": sum(outcome(row, arm) == "review" for row in known),
            "by_kind": dict(breakdown),
            "confusion": {f"{truth} -> {decision}": count for (truth, decision), count in sorted(confusion.items())},
        }
    runtime = read(directory / "runtime.json")
    if not runtime["checkpoint_files_verified"]:
        raise ValueError("Checkpoint integrity failed")
    return {
        "split": split, "scored_name_variants": len(rows), "all_name_variants_admitted": all(r["admitted"] for r in rows),
        "primary_contexts": len(primary), "known_contexts": len(known), "unresolved_contexts": len(unknown),
        "families": len({r["family"] for r in primary}), "invariance": dict(exact), "utility": utility,
        "paired_binary_vs_legacy_protected": paired(known, "paired", "legacy"),
        "paired_binary_vs_native": paired(known, "paired", "native"),
        "paired_binary_vs_forward": paired(known, "paired", "binary_forward"),
        "paired_binary_vs_external_deletion": paired(known, "paired", "deleted_paired"),
        "cyclic_vs_single_all_contexts": paired(primary, "cyclic", "single"),
        "runtime": runtime,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(analyze(args.directory), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(result)
    else:
        print(result, end="")
