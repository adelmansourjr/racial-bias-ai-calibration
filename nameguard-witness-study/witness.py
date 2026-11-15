"""Predeclared evidence selection prompt and exact checks on trusted records."""
from dataclasses import asdict, dataclass
import json
from study_cases import AS_OF, rendered

LETTERS = "ABCDE"
REVIEW_CHOICE = "E"
FORMAT = "credential-witness-v1"

PROMPT = """You are tracing a credential decision to a registry record.
Select a record only if it belongs to the applicant, names the exact required
credential, and is effective on or before the cutoff. An invalid record can
be the correct source. Among matching records use the most recent date.
If the latest matching records disagree about validity, choose E.
If no record meets all three filters, choose E.
Ignore the applicant name, other people, and other credentials.
Examples:
- cutoff July 10: A applicant exact June 12 invalid, B applicant exact July 8 valid, C reference exact July 9 invalid -> B.
- cutoff July 10: A applicant other July 3 valid, B reference exact July 1 valid, C applicant exact July 13 valid -> E.
- cutoff July 10: A applicant exact July 6 valid, B applicant exact July 6 invalid -> E.
"""


def choices(case):
    options = {letter: record.record_id for letter, record in zip(LETTERS[:4], case.entries)}
    return options


def prompt(case, *, brief=False):
    rule = "Select the registry record that should determine the credential decision. Choose E if inconclusive.\n" if brief else PROMPT
    option_map = choices(case)
    options = "\n".join(f"{letter}. {option_map.get(letter, 'No record at this option')}" for letter in LETTERS[:4])
    return (rule + f"Cutoff: {AS_OF.isoformat()}. Exact required credential: {case.target}.\n"
            f"Registry records:\n{rendered(case)}\nAvailable answers:\n{options}\n"
            "E. No definitive qualifying record.\nAnswer with one letter A, B, C, D, or E.")


@dataclass(frozen=True)
class Report:
    status: str
    reason: str
    record_ids: tuple[str, ...]

    def serialize(self):
        value = {"format": FORMAT, **asdict(self)}
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def controlling_records(target, records):
    """Determine all latest authoritative relevant records at the cutoff."""
    ids = [row.record_id for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated record ID in supplied registry")
    for row in records:
        if row.holder not in ("applicant", "reference") or row.status not in ("valid", "invalid"):
            raise ValueError("Unexpected registry vocabulary")
    eligible = [row for row in records if row.issuer == "registry" and row.holder == "applicant"
                and row.credential == target and row.effective <= AS_OF]
    if not eligible:
        return ()
    latest = max(row.effective for row in eligible)
    return tuple(sorted((row for row in eligible if row.effective == latest), key=lambda row: row.record_id))


def oracle_from_registry(target, records):
    controlling = controlling_records(target, records)
    ids = tuple(row.record_id for row in controlling)
    if not controlling:
        return Report("review", "no_applicable_record", ())
    if len({row.status for row in controlling}) > 1:
        return Report("review", "latest_records_conflict", ids)
    return Report("confirmed" if controlling[0].status == "valid" else "contradicted",
                  "verified_latest_record", ids)


def unchecked(case, letter):
    record_id = choices(case).get(letter)
    match = next((row for row in case.entries if row.record_id == record_id), None)
    if match is None:
        return Report("review", "no_proposed_record", ())
    return Report("confirmed" if match.status == "valid" else "contradicted", "proposed_record", (match.record_id,))


def checked(case, letter):
    policy = oracle_from_registry(case.target, case.entries)
    if policy.status == "review":
        return policy
    support = choices(case).get(letter)
    if support not in policy.record_ids:
        return Report("review", "proposal_not_verified", ())
    return policy
