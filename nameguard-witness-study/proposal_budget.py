"""Validate eligible-record ranking budgets chosen on development."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
from analyze import analyze, read, pair_ci
from study_cases import cases, DEV_NAMES, NAMES
from witness import LETTERS, Report, choices, oracle_from_registry

DEFER = Report("review", "proposal_budget_exhausted", ())


def rank(case, logits):
    return sorted(choices(case), key=lambda letter: (-logits[LETTERS.index(letter)], letter))


def checked_rank(case, logits, budget):
    if logits is None:
        return None
    if type(budget) is not int or not 1 <= budget <= 4:
        raise ValueError("Proposal budget outside 1..4")
    oracle = oracle_from_registry(case.target, case.entries)
    if oracle.status == "review":
        return oracle
    options = choices(case)
    return oracle if any(options[p] in oracle.record_ids for p in rank(case, logits)[:budget]) else DEFER


def analyze_budget(out):
    original = analyze(out)
    names = NAMES if original["split"] == "test" else DEV_NAMES
    rows = {(r["case"], r["name"]): r for r in read(out / "rows.json")}
    outcomes = defaultdict(Counter)
    variants = defaultdict(dict)
    family_scores = defaultdict(lambda: defaultdict(list))
    for c in cases(original["split"]):
        for budget in range(1, 5):
            for channel in ("protected", "native"):
                reports = [checked_rank(c, rows[c.key, n][f"{channel}_logits"], budget) for n in names]
                result = reports[0]
                valid = result is not None and (result.status, result.record_ids) == (c.expected_status, c.expected_ids)
                entry = outcomes[f"{channel}_{budget}"]
                entry["correct_status_and_ids"] += valid
                entry["definitive_cases"] += result is not None and result.status != "review"
                entry["verified_confirmations"] += valid and c.expected_status == "confirmed"
                entry["verified_contradictions"] += valid and c.expected_status == "contradicted"
                entry["wrong_definitive"] += result is not None and result.status != "review" and not valid
                entry["all_name_reports_equal_and_present"] += all(report is not None for report in reports) and (
                    len({report.serialize() for report in reports}) == 1)
                variants[budget, channel][c.family, c.kind] = valid
                family_scores[budget, channel][c.family].append(int(valid))
    contrasts = defaultdict(dict)
    for key, results in variants.items():
        budget, channel = key
        for kind in {kind for _, kind in results} - {"valid"}:
            contrasts[f"{channel}_{budget}"][kind] = sum(
                results[family, "valid"] and correct for (family, right), correct in results.items() if right == kind)
    primary = outcomes["protected_2"]
    invariance = original["invariance"]
    traced = invariance.get("direct_activation_conditions", 0)
    frame = family_scores[2, "protected"]
    baseline = family_scores[1, "protected"]
    return {"split": original["split"], "conditions": original["conditions"], "definite": original["verifiable"],
            "budgets": {arm: dict(result) for arm, result in outcomes.items()},
            "contrasts_both_correct": {arm: dict(sorted(c.items())) for arm, c in sorted(contrasts.items())},
            "paired_accuracy_difference_2_vs_1": pair_ci([
                (sum(records) - sum(baseline[i])) / len(records) for i, records in frame.items()]),
            "gates": {
                "activation_and_logit_equality": original["gates"]["all_names_admit_and_equal"]
                   and invariance.get("identical_direct_activation_conditions", 0) == traced
                   and invariance.get("direct_stage_comparisons", 0) == traced * (len(names) - 1) * 31,
                "all_checked_report_bytes_identical": primary["all_name_reports_equal_and_present"] == original["conditions"],
                "no_wrong_definitive": primary["wrong_definitive"] == 0,
                "at_least_75_percent_definitive": 4 * primary["definitive_cases"] >= 3 * original["verifiable"]}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze_budget(args.out), indent=2, allow_nan=False))
