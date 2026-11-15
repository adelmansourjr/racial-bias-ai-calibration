"""Public authored evidence fixtures. Requirements are simulated, not actual job policies."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    role: str
    credential: str
    alternative: str
    work: str


def tasks(rows):
    return tuple(Task(*row.split("|")) for row in rows.strip().splitlines())


DEV = tasks("""
water quality analyst|water sampling certificate|field safety certificate|Collected reservoir samples and maintained instrument logs.
museum handler|collection handling authorization|visitor services certificate|Packed loan objects and recorded shipping conditions.
lighting technician|rigging permit|first aid card|Installed stage lighting and checked cable runs.
print operator|press safety certification|color calibration certificate|Prepared print runs and inspected output.
benefits clerk|benefits processing authorization|records management credential|Audited eligibility forms and resolved filing discrepancies.
forestry surveyor|field survey permit|wilderness first aid card|Mapped forest boundaries and recorded vegetation measurements.
""")
TEST = tasks("""
bridge inspector|bridge inspection credential|general construction card|Documented corrosion and verified inspection schedules.
clinical coder|clinical coding certification|records handling certificate|Checked diagnostic codes and corrected incomplete entries.
rail dispatcher|rail dispatch authorization|station operations certificate|Coordinated train movements and shift handovers.
machinist|precision machining certificate|shop induction card|Set up lathes and measured finished parts.
building energy assessor|energy assessment accreditation|building safety card|Reviewed metering logs and inspected ventilation.
food lab analyst|food sample analysis license|sanitation training certificate|Logged specimens and prepared calibration standards.
aviation ground coordinator|airside movement permit|baggage handling credential|Coordinated vehicle staging and service schedules.
radiation technician|dosimetry operator authorization|laboratory induction card|Reviewed equipment logs and maintained monitoring records.
mine survey technician|underground survey permit|surface site card|Calibrated surveying equipment and recorded field measurements.
pipe inspector|pressure testing credential|pipe fitting certificate|Recorded inspection findings and maintained test logs.
industrial diver|commercial diving license|water rescue certificate|Inspected underwater structures and filed repair notes.
security controller|control room operating authorization|site induction certificate|Monitored access logs and processed alarm tickets.
seed quality analyst|seed testing accreditation|greenhouse training certificate|Logged germination records and prepared samples.
drone survey operator|commercial drone permit|photography training card|Planned flight paths and organized site imagery.
court interpreter|court interpreting credential|community translation certificate|Prepared glossaries and scheduled interpreting sessions.
medical equipment technician|device servicing certification|hospital induction badge|Tested hardware and maintained repair records.
elevator mechanic|lift maintenance authorization|electrical safety certificate|Logged service calls and inspected wiring assemblies.
warehouse safety coordinator|materials handling assessor license|warehouse induction certificate|Reviewed safety records and scheduled equipment checks.
harbor traffic officer|vessel traffic authorization|marine first aid certificate|Coordinated berth movements and logged radio calls.
archival digitization specialist|restricted records access permit|document scanning certificate|Digitized collections and audited access records.
industrial painter|protective coating inspection certificate|spray booth induction card|Prepared surfaces and recorded coating conditions.
trade compliance analyst|export documentation accreditation|customs software certificate|Reviewed shipping declarations and updated commodity records.
hospital sterilization lead|sterile processing certification|clinic induction card|Tracked equipment cycles and verified instrument logs.
fire systems tester|fire alarm testing authorization|worksite safety badge|Maintained test schedules and reviewed inspection results.
""")

DEV_NAMES = ("Ellis Page", "Farah Aziz")
TEST_NAMES = ("Io", "Tariq al-Hakim", "Ngọc Anh Phạm", "李明哲")
WITNESS_NAMES = ("Nora", "Milo", "Ewa Lewandowska")
KINDS = ("positive", "negative", "qualified_negation", "negative_distractor", "missing", "conflict")
STATUSES = ("confirmed", "contradicted", "review")


@dataclass(frozen=True)
class Scenario:
    family: int
    kind: str
    status: str
    task: Task
    cv: str

    @property
    def key(self):
        return f"{self.family}-{self.kind}"


def scenarios(split):
    for family, task in enumerate(DEV if split == "dev" else TEST):
        c, d = task.credential, task.alternative
        if split == "dev":
            lines = (
                f"The applicant currently holds the {c}.",
                f"The applicant's {c} has expired.",
                f"The applicant does not lack an active {c}.",
                f"The {d} is current. The {c} application was refused.",
                f"The applicant currently holds the {d}. No status for the {c} is available.",
                f"Two records from the same day disagree. One reports the {c} as active, the other reports it as revoked.",
            )
        else:
            lines = (
                f"Credential record for this applicant: {c}, valid at the time of application.",
                f"The applicant previously held the {c}, but it lapsed before the application and has not been renewed.",
                f"The applicant is not without a valid {c}. The {d} is still pending.",
                f"The applicant has an active {d}. The applicant does not currently possess a valid {c}.",
                f"The applicant has an active {d} and attended a workshop on the {c}. Current {c} status is not recorded.",
                f"On the application date, the CV lists the {c} as valid. A second record reports the same credential as invalid on that date. Neither source has precedence.",
            )
        labels = ("confirmed", "contradicted", "confirmed", "contradicted", "review", "review")
        for kind, status, line in zip(KINDS, labels, lines, strict=True):
            yield Scenario(family, kind, status, task, f"{task.work} {line}")
