"""Synthetic credential ledger with predeclared fact interventions and hidden oracle."""
from dataclasses import dataclass, replace
from datetime import date
import random

AS_OF = date(2026, 6, 30)
OLDER = date(2026, 1, 12)
CURRENT = date(2026, 6, 18)
FUTURE = date(2026, 7, 4)

FIT = ("library access card", "workshop induction card", "temporary depot permit")
HELD_OUT = (
    "rail signal maintenance endorsement", "industrial hoist inspection permit",
    "laser alignment authorization", "wetlands survey accreditation",
    "warehouse refrigeration license", "radio transmission operator credential",
    "bridge coating assessment certificate", "municipal meter calibration credential",
    "wildlife transport handling permit", "substation test authorization",
    "fire door inspection endorsement", "advanced grain sampling certificate",
)
NAMES = ("Ari", "عبدالرحمن الخطيب", "Nguyễn Minh Anh", "張雅婷")
WITNESSES = ("Mila", "Owen Isaacs", "Céline Martin")
DEV_NAMES = ("Dara Moore", "Amina Tarek")
KINDS = ("valid", "invalid", "delete", "swap_holder", "swap_credential",
         "future_only", "equal_conflict", "newer_invalid")


@dataclass(frozen=True)
class Record:
    record_id: str
    holder: str
    credential: str
    effective: date
    status: str
    issuer: str = "registry"


@dataclass(frozen=True)
class Case:
    family: int
    kind: str
    target: str
    entries: tuple[Record, ...]
    expected_status: str
    expected_ids: tuple[str, ...]
    style: str

    @property
    def key(self):
        return f"{self.family}-{self.kind}"


def cases(split):
    for family, target in enumerate(FIT if split == "dev" else HELD_OUT):
        other = ("general site access card" if family % 2 else "basic safety induction permit")
        support = Record("R1", "applicant", target, CURRENT, "valid")
        dated = Record("R2", "applicant", target, OLDER, "invalid")
        reference = Record("R3", "reference", target, CURRENT, "invalid")
        distractor = Record("R4", "applicant", other, CURRENT, "valid")
        # The test oracle is specified directly by each intervention on the
        # underlying record. Verification code has no access to it.
        variants = (
            ((support, dated, reference, distractor), "confirmed", ("R1",)),
            ((replace(support, status="invalid"), dated, reference, distractor), "contradicted", ("R1",)),
            ((reference, distractor), "review", ()),
            ((replace(support, holder="reference"), replace(reference, holder="applicant"), distractor),
             "contradicted", ("R3",)),
            ((replace(support, credential=other), reference, distractor), "review", ()),
            ((replace(support, effective=FUTURE), reference, distractor), "review", ()),
            ((support, replace(dated, effective=CURRENT), reference, distractor), "review", ("R1", "R2")),
            ((replace(support, effective=OLDER), replace(dated, effective=CURRENT), reference, distractor),
             "contradicted", ("R2",)),
        )
        for index, (kind, variant) in enumerate(zip(KINDS, variants, strict=True)):
            entries, status, ids = variant
            # Position has no fixed association with correctness. Each family
            # preserves its relative record order across edits.
            order = ["R1", "R2", "R3", "R4"]
            random.Random(4200 + family).shuffle(order)
            entries = tuple(sorted(entries, key=lambda record: order.index(record.record_id)))
            style = "ledger" if split == "dev" else ("narrative", "dated", "dispatch")[family % 3]
            yield Case(family, kind, target, entries, status, ids, style)


def render(record, style):
    holder = "applicant" if record.holder == "applicant" else "reference person"
    iso = record.effective.isoformat()
    if style == "ledger":
        return f"{record.record_id} | Date {iso} | Holder: {holder} | {record.credential} | {record.status}."
    if style == "narrative":
        return f"{record.record_id}. The registry recorded the {holder}'s {record.credential} as {record.status} on {iso}."
    if style == "dated":
        return f"{record.record_id}. On {iso}, the {record.credential} registered to the {holder} was marked {record.status}."
    return f"{record.record_id}. Date {iso}. Registry notice for the {holder}: the {record.credential} was {record.status}."


def rendered(case):
    return "\n".join(render(record, case.style) for record in case.entries)
