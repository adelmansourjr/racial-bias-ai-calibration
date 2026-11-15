"""Fresh synthetic cases with rekeyed records and independently specified answers."""
from dataclasses import replace
from datetime import date
import random

import dependencies
from study_cases import Case, Record

OLDER = date(2025, 11, 13)
MIDDLE = date(2026, 2, 3)
CURRENT = date(2026, 6, 14)
LATER = date(2026, 6, 26)
FUTURE = date(2026, 7, 7)

DEV_CREDENTIALS = (
    "warehouse sampling endorsement",
    "rural route inspection permit",
    "archive object appraisal credential",
)
TEST_CREDENTIALS = (
    "emergency vent testing credential",
    "river gauge inspection endorsement",
    "chain sling examination permit",
    "crew dispatch coordination certificate",
    "thermal sensor verification license",
    "track geometry assessment credential",
    "cargo restraint testing permit",
    "pumping station control authorization",
    "pressure relief valve examination credential",
    "ground antenna inspection endorsement",
    "coastal nutrient sampling permit",
    "waterfront piling survey certification",
)
DEV_NAMES = ("Evan Doyle", "سامية حمدان")
TEST_NAMES = ("Ren", "Mariam El-Sayed", "紗季 高橋", "Nguyễn Khánh Linh")
WITNESS_NAMES = ("Lara Scott", "Emre Kaya", "Maya Costa")
TYPES = ("valid", "invalid", "holder_swap", "later_reversal", "missing",
         "other_credential", "future_only", "same_date_conflict")
STYLES = ("narrative", "dated", "dispatch")
OPAQUE_IDS = ("H42", "M91", "K88", "B13", "J26", "V20", "Q55", "Z34",
              "C71", "X65", "S08", "P39", "F57", "R96", "L10", "W27")


def cases(split):
    categories = DEV_CREDENTIALS if split == "dev" else TEST_CREDENTIALS
    for family, credential in enumerate(categories):
        other = ("general worksite access card" if family % 2 else "project safety induction certificate")
        source = Record("current", "applicant", credential, CURRENT, "valid")
        stale_bad = Record("stale_bad", "applicant", credential, MIDDLE, "invalid")
        stale_good = Record("stale_good", "applicant", credential, OLDER, "valid")
        peer = Record("peer", "reference", credential, LATER, "invalid")
        different = Record("different", "applicant", other, LATER, "valid")

        # Expected controlling IDs are defined from the edits. Neither the
        # proposer nor the checker receives these expected IDs or statuses.
        variants = (
            ((source, stale_bad, stale_good, peer), "confirmed", ("current",)),
            ((replace(source, status="invalid"), stale_bad, stale_good, peer), "contradicted", ("current",)),
            ((replace(source, holder="reference"), stale_bad, stale_good, replace(peer, holder="applicant")),
             "contradicted", ("peer",)),
            ((source, replace(stale_bad, effective=LATER), stale_good, peer), "contradicted", ("stale_bad",)),
            ((different, replace(stale_good, credential=other), peer, replace(source, holder="reference")),
             "review", ()),
            ((replace(source, credential=other), replace(stale_bad, credential=other),
              replace(stale_good, credential=other), peer), "review", ()),
            ((replace(source, effective=FUTURE), replace(stale_bad, effective=FUTURE),
              replace(stale_good, effective=FUTURE), peer), "review", ()),
            ((source, replace(stale_bad, effective=CURRENT), stale_good, peer),
             "review", ("current", "stale_bad")),
        )
        for intervention, (kind, (records, truth, controls)) in enumerate(zip(TYPES, variants, strict=True)):
            seed = (79001 if split == "test" else 81013) + 1931 * family + 419 * intervention
            rng = random.Random(seed)
            aliases = dict(zip((r.record_id for r in records), rng.sample(OPAQUE_IDS, len(records)), strict=True))

            # Within each family, put the sole controlling record in each
            # display position exactly once across the four decisive cases.
            group = list(records)
            rng.shuffle(group)
            if len(controls) == 1:
                selected = next(record for record in group if record.record_id == controls[0])
                group.remove(selected)
                group.insert((family + intervention) % len(records), selected)
            entries = tuple(replace(r, record_id=aliases[r.record_id]) for r in group)
            style = "ledger" if split == "dev" else STYLES[(family + intervention) % len(STYLES)]
            yield Case(family, kind, credential, entries, truth,
                       tuple(sorted(aliases[rid] for rid in controls)), style)
