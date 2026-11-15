"""Exhaustively check the reviewed-record policy on independent finite oracle cases."""
from dataclasses import asdict
from datetime import date, timedelta
from itertools import product
import json

from evidence_policy import Observation, evaluate

VALUES = ("missing", "valid", "invalid", "conflict")
ALIASES = ("Io", "Ngọc Anh Phạm", "李明哲", "مريم السعدي")
DATE = date(2026, 11, 16)


def observations(bucket, holder, credential, effective, status):
    if status == "missing":
        return ()
    findings = ("valid", "invalid") if status == "conflict" else (status,)
    return tuple(Observation(f"{bucket}-{index}", holder, credential, effective, finding)
                 for index, finding in enumerate(findings))


def audit_policy():
    cases = checks = with_citations = 0
    decision_counts = {"confirmed": 0, "contradicted": 0, "review": 0}
    for old, current, third_party, unrelated, future in product(VALUES, repeat=5):
        evidence = (*observations("old", "applicant", "required", DATE - timedelta(days=1), old),
                    *observations("current", "applicant", "required", DATE, current),
                    *observations("reference", "third_party", "required", DATE, third_party),
                    *observations("unrelated", "applicant", "other", DATE, unrelated),
                    *observations("future", "applicant", "required", DATE + timedelta(days=1), future))
        expected_state = old if current == "missing" else current
        expected = {"missing": "review", "conflict": "review", "valid": "confirmed",
                    "invalid": "contradicted"}[expected_state]
        expected_bucket = ("old" if current == "missing" else "current") if expected_state != "missing" else None
        reference = evaluate("required", DATE, evidence)
        if reference.status != expected:
            raise ValueError("Reviewed-record oracle disagreement")
        expected_ids = tuple(o.record_id for o in evidence if expected_bucket and
                             o.record_id.startswith(expected_bucket + "-"))
        if reference.records != expected_ids:
            raise ValueError("Incorrect controlling record identifiers")
        for index, alias in enumerate(ALIASES):
            # Intake supplies a name alongside already reviewed records. The
            # policy interface accepts only the records and exact rule parameters.
            form = {"applicant_name": alias, "reviewed_records": evidence[index:] + evidence[:index]}
            result = evaluate("required", DATE, form["reviewed_records"])
            if result.serialize() != reference.serialize():
                raise ValueError("Irrelevant identity or record ordering changed the report")
            checks += 1
        cases += 1
        decision_counts[reference.status] += 1
        with_citations += bool(reference.records)
    return {"tested_structured_cases": cases, "identical_report_comparisons": checks,
            "case_status_counts": decision_counts, "reports_with_controlling_records": with_citations,
            "scope": "Finite exhaustive conformance with supplied reviewed observations. This is independent of free-text model accuracy."}


if __name__ == "__main__":
    print(json.dumps(audit_policy(), indent=2))
