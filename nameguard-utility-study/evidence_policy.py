"""Evaluate a declared credential rule on reviewed structured observations.

Source review must establish the holder role, credential ID, authority and
observation date. This module checks the policy consequences of those assertions.
"""
from dataclasses import asdict, dataclass
from datetime import date
import json


@dataclass(frozen=True)
class Observation:
    record_id: str
    holder_role: str
    credential_id: str
    effective_on: date
    finding: str

    def __post_init__(self):
        for value in (self.record_id, self.credential_id):
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError("Nonempty canonical record and credential IDs required")
        if self.holder_role not in ("applicant", "third_party") or self.finding not in ("valid", "invalid"):
            raise ValueError("Unsupported declared evidence role or finding")
        if type(self.effective_on) is not date:
            raise ValueError("Exact observation date required")


@dataclass(frozen=True)
class EvidenceFinding:
    credential_id: str
    status: str
    records: tuple[str, ...]
    explanation: str

    def serialize(self):
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def evaluate(credential_id, as_of, observations):
    """Evaluate latest relevant effective records with equal source precedence."""
    if type(credential_id) is not str or not credential_id or credential_id.strip() != credential_id:
        raise ValueError("Canonical required credential ID required")
    if type(as_of) is not date:
        raise ValueError("Exact evaluation date required")
    by_id = {}
    for observation in observations:
        if type(observation) is not Observation:
            raise ValueError("Reviewed observations required")
        if observation.record_id in by_id and by_id[observation.record_id] != observation:
            raise ValueError("One record ID refers to conflicting observations")
        by_id[observation.record_id] = observation
    relevant = [o for o in by_id.values()
                if o.holder_role == "applicant" and o.credential_id == credential_id and o.effective_on <= as_of]
    if not relevant:
        return EvidenceFinding(credential_id, "review", (), "No relevant record is effective on or before the evaluation date.")
    latest = max(o.effective_on for o in relevant)
    current = [o for o in relevant if o.effective_on == latest]
    ids = tuple(sorted(o.record_id for o in current))
    values = {o.finding for o in current}
    if len(values) > 1:
        return EvidenceFinding(credential_id, "review", ids, "The latest applicable records conflict.")
    if values == {"valid"}:
        return EvidenceFinding(credential_id, "confirmed", ids, "The latest applicable records agree that the credential is valid.")
    return EvidenceFinding(credential_id, "contradicted", ids, "The latest applicable records agree that the credential is invalid.")
