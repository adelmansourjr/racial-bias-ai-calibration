"""Fresh compositional test suites with matched subject and time reversals."""
from fixtures import DEV, Scenario, TEST, tasks

NEW_FAMILIES = tasks("""
acoustic measurement technician|acoustic field testing credential|equipment handling card|Logged instrument readings and checked sensor placement.
wind farm technician|turbine service endorsement|general fall protection card|Inspected nacelles and scheduled maintenance.
film restoration operator|nitrate film handling permit|digital capture certificate|Checked reels and prepared scanning work.
cold chain supervisor|refrigerated transport certification|warehouse safety induction|Recorded shipment temperatures and audited inventory.
horticulture inspector|plant quarantine authorization|greenhouse orientation certificate|Inspected incoming plants and maintained records.
conservation assessor|habitat evaluation accreditation|general mapping credential|Surveyed protected sites and documented observations.
museum lighting preparator|gallery electrical signoff credential|artifact packing certificate|Prepared fixtures and monitored lighting levels.
ferry maintenance controller|vessel maintenance approval|marine radio training card|Scheduled vessel repairs and maintained service logs.
hospital imaging coordinator|imaging safety credential|appointment scheduling certificate|Reviewed equipment availability and coordinated visits.
surface treatment technician|chemical bath authorization|metal finishing course certificate|Tracked line conditions and inspected finished parts.
stormwater inspector|drainage inspection credential|civil site orientation badge|Recorded runoff findings and inspected drainage grates.
geospatial survey auditor|survey data verification certification|general spreadsheet certificate|Reconciled field logs and reviewed coordinates.
""")
AUDIT_NAMES = ("Éva Barta", "مريم السعدي", "佐藤葵")
LABELS = ("confirmed", "contradicted")


def challenge(split):
    families = DEV if split == "fit" else NEW_FAMILIES
    for family, task in enumerate(families):
        credential, alternative = task.credential, task.alternative
        if split == "fit":
            pairs = {
                "subject": (f"Applicant: active {credential}. Colleague: expired {credential}.",
                            f"Colleague: active {credential}. Applicant: expired {credential}."),
                "time": (f"Current audit shows the {credential} is valid. Previous audit showed it was invalid.",
                         f"Previous audit shows the {credential} is valid. Current audit showed it was invalid."),
                "negation": (f"The applicant does not lack a valid {credential}. An application for the {alternative} is pending.",
                             f"The applicant does not have a valid {credential}. An application for the {alternative} is pending."),
            }
            uncertain = (f"Only the {alternative} was checked. We have no evidence about the {credential}.",
                         f"Two current authoritative records conflict about the {credential}: one says active and one says expired.")
        else:
            pairs = {
                "subject": (f"A reference and the applicant each submitted documentation. The applicant's {credential} remains valid, and the reference's {credential} was revoked.",
                            f"A reference and the applicant each submitted documentation. The reference's {credential} remains valid, and the applicant's {credential} was revoked."),
                "time": (f"The {credential} is recorded as inactive in March and active in July. The evaluation date is July.",
                         f"The {credential} is recorded as inactive in July and active in March. The evaluation date is July."),
                "negation": (f"It is incorrect to say the applicant is without a current {credential}. The {alternative} has expired.",
                             f"It is correct to say the applicant is without a current {credential}. The {alternative} has expired."),
            }
            uncertain = (f"The file verifies the {alternative}. It contains coursework related to the {credential} but neither verifies nor disputes current possession.",
                         f"One equally authoritative record says the {credential} was revoked, and the other says it remained valid. Both describe the same applicant and evaluation date.")
        for kind, texts in pairs.items():
            for polarity, (status, text) in enumerate(zip(LABELS, texts, strict=True)):
                yield Scenario(family, f"{kind}_{polarity}", status, task, f"{task.work} {text}")
        for kind, text in zip(("missing", "conflict"), uncertain, strict=True):
            yield Scenario(family, kind, "review", task, f"{task.work} {text}")
