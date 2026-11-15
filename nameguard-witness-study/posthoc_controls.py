"""Expose fixed-ID and budget shortcuts using the already logged results."""
from collections import Counter, defaultdict
from fractions import Fraction
import json
from analyze import ROOT, read
from opaque_replay import rekey
from proposal_budget import checked_rank
from study_cases import cases
from witness import choices


def compute():
    known = {c.key: c for c in cases("test") if c.expected_status != "review"}
    opaque_rows = [row for row in read(ROOT / "results/opaque/rows.json") if row["case"] in known]
    all_original = read(ROOT / "results/test/rows.json")
    original_rows = [r for r in all_original if r["primary"] and r["case"] in known]
    result = {"scope": "Post-hoc budget curve. The original held-out gate and rekeyed top-two comparison retain their frozen definitions."}
    for label, rows in (("original", original_rows), ("opaque", opaque_rows)):
        curve = []
        for budget in range(1, 5):
            actual = first = by_id = 0
            uniform = Fraction()
            for row in rows:
                case = known[row["case"]]
                if label == "opaque":
                    case = rekey(case, row["replicate"])
                options = list(choices(case).values())
                correct = set(case.expected_ids)
                scores = row["logits"] if label == "opaque" else row["protected_logits"]
                report = checked_rank(case, scores, budget)
                actual += report is not None and bool(correct & set(report.record_ids)) and report.status == case.expected_status
                first += bool(correct & set(options[:budget]))
                by_id += bool(correct & set(sorted(options)[:budget]))
                uniform += Fraction(min(budget, len(options)), len(options))
            curve.append({"budget": budget, "cases": len(rows), "model": actual,
                          "first_displayed": first, "id_order": by_id, "uniform_expected": float(uniform)})
        result[label] = curve
    grouped = defaultdict(list)
    for row in all_original:
        if row["case"] in known:
            grouped[row["case"]].append(row)
    status_changes = {}
    for channel in ("native", "protected"):
        changes = Counter()
        for key, block in grouped.items():
            case = known[key]
            outcomes = [checked_rank(case, row[f"{channel}_logits"], 2) for row in block]
            distinct = {r.status if r is not None else "rejected" for r in outcomes}
            if len(distinct) > 1:
                changes[case.expected_status] += 1
        status_changes[channel] = dict(changes)
    result["two_proposal_name_dependence_by_true_status"] = status_changes
    return result


if __name__ == "__main__":
    print(json.dumps(compute(), indent=2))
