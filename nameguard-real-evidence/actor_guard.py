"""Conservative post-hoc guards for passive or generic experience evidence."""
import re
from evidence import Finding
from evidence_v3 import verify_excerpt as broad

GENERIC_EXPERIENCE = re.compile(r"\bexperience\s+with\b", re.I)
PASSIVE_VOICE_AT_WITNESS = re.compile(r"\b(?:was|were|is|are|been|be)\s+$", re.I)
PASSIVE_VOICE_WITHIN_WITNESS = re.compile(
    r"\b(?:was|were|is|are|been|be)\s+(?:created|built|developed|written|designed|prepared|implemented)\b", re.I)


def verify_excerpt(excerpt, tool):
    report = broad(excerpt, tool)
    if report.evidence is None:
        return report
    span = report.evidence
    quote = excerpt[span.start:span.end]
    preceding = excerpt[max(0, span.start - 30):span.start]
    if GENERIC_EXPERIENCE.search(quote):
        return Finding("review", "generic_experience_without_explicit_tool_use", None)
    if PASSIVE_VOICE_AT_WITNESS.search(preceding) or PASSIVE_VOICE_WITHIN_WITNESS.search(quote):
        return Finding("review", "tool_use_actor_requires_review", None)
    return report
