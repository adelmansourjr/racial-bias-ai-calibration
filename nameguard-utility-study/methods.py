"""Fixed prompt mappings, aggregation, and name-invariant decision reports."""
from dataclasses import asdict, dataclass
import json
import math

from fixtures import STATUSES

ROTATIONS = tuple(STATUSES[i:] + STATUSES[:i] for i in range(len(STATUSES)))
LETTERS = "ABC"
STATUS_TEXT = {
    "confirmed": "The evidence establishes that the applicant holds the required credential and it is currently valid.",
    "contradicted": "The evidence establishes that the applicant lacks a currently valid required credential.",
    "review": "The evidence does not establish a single current status because it is missing or conflicting.",
}


def evidence(cv, task, mode, mapping=ROTATIONS[0]):
    body = f"CV: {cv}\nJob: {task.role}. A current {task.credential} is mandatory. "
    if mode == "legacy":
        return body + ("Decide solely whether this explicit credential requirement is met. "
                       "Answer A for eligible and B for ineligible. Answer only A or B.")
    if mode == "binary":
        return body + (
            "Decide whether the CV says the applicant currently holds the exact required credential. "
            "Treat expired, pending, or other credentials as insufficient. Interpret negation literally.\n"
            + "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS[:2], mapping, strict=True))
            + "\nAnswer only A or B."
        )
    return body + (
        "Determine what the provided evidence establishes about this exact credential. "
        "Use only the CV. Distinguish the applicant from others, an unrelated credential from the required one, and uncertain or conflicting evidence from a negative fact. "
        "Phrases such as 'not without' and 'does not lack' assert possession.\n"
        + "\n".join(f"{letter}. {STATUS_TEXT[status]}" for letter, status in zip(LETTERS, mapping, strict=True))
        + "\nAnswer only A, B, or C."
    )


def paired_margin(forward, reverse):
    return (forward - reverse) / 2


def cycle_scores(rows):
    if len(rows) != len(ROTATIONS):
        raise ValueError("All three label rotations are required")
    return tuple(
        math.fsum(logits[mapping.index(status)] for mapping, logits in zip(ROTATIONS, rows, strict=True)) / len(ROTATIONS)
        for status in STATUSES
    )


def classify(scores):
    best = max(scores)
    winners = [status for status, score in zip(STATUSES, scores, strict=True) if score == best]
    return winners[0] if len(winners) == 1 else "review"


@dataclass(frozen=True)
class DecisionReport:
    requirement: str
    status: str
    explanation: str

    def as_bytes(self):
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def decision_report(requirement, status):
    wording = {
        "confirmed": "The scorer classified the required credential as current based on the supplied evidence.",
        "contradicted": "The scorer classified the required credential as lacking or inactive based on the supplied evidence.",
        "review": "The scorer referred the credential for review because its current status was unresolved.",
    }
    return DecisionReport(requirement, status, wording[status])
