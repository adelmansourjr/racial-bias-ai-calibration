"""Separate protected ranking from authoritative record selection and verification."""
import math

import dependencies
from study_cases import AS_OF
from witness import LETTERS, Report, choices, oracle_from_registry

NO_PROPOSAL = Report("review", "proposal_budget_exhausted", ())
ARMS = ("model_top_two", "applicable_top_two", "latest_top_one", "registry_only")


def ranked_records(case, logits):
    if len(logits) != len(LETTERS) or not all(math.isfinite(value) for value in logits):
        raise ValueError("Finite answer logits required")
    options = choices(case)
    if len(options) != len(case.entries) or len({r.record_id for r in case.entries}) != len(case.entries):
        raise ValueError("Expected one unique selectable ID per record")
    letters = sorted(options, key=lambda letter: (-logits[LETTERS.index(letter)], letter))
    by_id = {row.record_id: row for row in case.entries}
    return tuple(by_id[options[letter]] for letter in letters)


def proposals(case, logits, arm):
    ranked = ranked_records(case, logits)
    if arm == "model_top_two":
        return ranked[:2]
    if arm not in ("applicable_top_two", "latest_top_one"):
        raise ValueError("Unrecognized proposal mode")
    applicable = tuple(row for row in ranked if row.holder == "applicant" and row.issuer == "registry"
                       and row.credential == case.target and row.effective <= AS_OF)
    if arm == "applicable_top_two":
        return applicable[:2]
    if not applicable:
        return ()
    return (max(applicable, key=lambda row: row.effective),)


def review(case, logits, arm):
    """Return the certificate plus proposal count. Review always checks the full ledger."""
    if logits is None:
        return None, 0
    checked = oracle_from_registry(case.target, case.entries)
    if arm == "registry_only":
        return checked, 0
    if checked.status == "review":
        return checked, 0
    shortlist = proposals(case, logits, arm)
    verified = any(record.record_id in checked.record_ids for record in shortlist)
    return (checked if verified else NO_PROPOSAL), len(shortlist)
