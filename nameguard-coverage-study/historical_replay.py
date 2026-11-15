"""Apply the new proposal policy to the earlier logged, synthetic witness benchmark."""
from collections import defaultdict
import json

from dependencies import WORKSPACE
from measure import read, file_hash, score_bits
from coverage_policy import ARMS, review
from study_cases import NAMES, cases


def replay():
    source = WORKSPACE / "nameguard-witness-study/results/test/rows.json"
    cohort = tuple(cases("test"))
    rows = read(source)
    observations = {(r["case"], r["name"]): r for r in rows}
    if len(rows) != len(observations) or set(observations) != {(case.key, name) for case in cohort for name in NAMES}:
        raise ValueError("Historical cohort is incomplete")
    data = defaultdict(lambda: {"correct_definitive": 0, "correct_review": 0,
                                 "wrong_definitive": 0, "exact_report_conditions": 0})
    equal_scores = 0
    for case in cohort:
        logits = [observations[case.key, name]["protected_logits"] for name in NAMES]
        if any(value is None for value in logits):
            raise ValueError("Archived inference fails the expected admission gate")
        equal_scores += len({score_bits(value) for value in logits}) == 1
        for arm in ARMS:
            outcomes = [review(case, value, arm)[0] for value in logits]
            actual = outcomes[0]
            correct = (actual.status, actual.record_ids) == (case.expected_status, case.expected_ids)
            data[arm]["correct_definitive"] += correct and actual.status != "review"
            data[arm]["correct_review"] += correct and actual.status == "review"
            data[arm]["wrong_definitive"] += not correct and actual.status != "review"
            data[arm]["exact_report_conditions"] += len({result.serialize() for result in outcomes}) == 1
    return {"source_sha256": file_hash(source), "planned_requests": len(rows), "conditions": len(cohort),
            "decisive": sum(c.expected_status != "review" for c in cohort),
            "exact_score_conditions": equal_scores, "methods": dict(data),
            "scope": "Post-hoc archived logits. No new model inference and no independent generalization claim."}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
