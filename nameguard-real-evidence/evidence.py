"""Conservative documentary evidence extraction with exact source spans."""
from dataclasses import asdict, dataclass
import hashlib
import json
import re

TERM_PREFIX = r"(?:(?:Microsoft|MS|Oracle)\s+)?"
ACTIVITY = re.compile(r"\b(?:responsible for|develop\w*|generat\w*|design\w*|creat\w*|build\w*|built|prepar\w*|manag\w*|maintain\w*|analy[sz]\w*|implement\w*|updat\w*|writ\w*|wrote|produc\w*|extract\w*|automat\w*|perform\w*|process\w*|run\w*|ran|report\w*|experience|monitor\w*|track\w*|calculat\w*|model\w*|support\w*|conduct\w*|coordinat\w*|draft\w*)\b", re.I)
DIRECT = re.compile(r"\b(?:used|utilized|utilised|leveraged|employed|worked with)\s+$", re.I)
USING = re.compile(r"\b(?:using|utilizing|utilising|through mastery of|via|with|in)\s+$", re.I)
AMBIGUOUS = re.compile(r"\b(?:no|never|not|without|will|would|should|could|can|plan(?:ned)? to|knowledge of|familiar with|proficient in|learning|certification|training|coursework|requirement[s]?|instruct\w*|teach\w*|demonstrat\w*|help\w*|assisted|team members|students|vendors|providers)\b", re.I)
BARRIER = re.compile(r"\b(?:Skills|Skills Summary|Software|Technologies|Education|Core Qualifications)\s*:", re.I)
SEPARATOR = re.compile(r"[.!?;•*]\s*")
UPPER_ANCHOR = re.compile(r"\s+(?=[A-Z][a-z]+(?:ed|ing)\s|(?:Skills|Education|Certifications|Qualifications)\b)")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    sha256: str


@dataclass(frozen=True)
class Finding:
    decision: str
    reason: str
    evidence: Span | None

    def bytes(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()


def _bound(text, stop):
    # Extract a bounded local clause, never bridge bullets, sentences, or
    # explicit section introductions.
    left = max(0, stop - 125)
    for delimiter in (SEPARATOR, BARRIER):
        left = max([left, *(m.end() for m in delimiter.finditer(text, 0, stop))])
    if left and not text[left - 1].isspace() and text[left].isalnum():
        space = text.find(" ", left, stop)
        if space != -1:
            left = space + 1
    return left


def verify_excerpt(excerpt, tool):
    if not isinstance(excerpt, str) or not isinstance(tool, str) or not tool:
        raise ValueError("Expected excerpt text and exact tool name")
    pattern = re.compile(r"(?<![\w-])" + TERM_PREFIX + re.escape(tool) + r"(?![\w-])", re.I)
    for found in pattern.finditer(excerpt):
        left = _bound(excerpt, found.start())
        before = excerpt[left:found.start()]
        # A branded tool can be followed by prose explicitly assigning use as
        # a skill, but a simple lexical match is never decisive.
        action = ACTIVITY.search(before)
        direct = DIRECT.search(before)
        connector = USING.search(before)
        if not (direct or (action and connector)):
            continue
        start = left + (action.start() if action else direct.start())
        if AMBIGUOUS.search(before) or BARRIER.search(before):
            continue
        tail = excerpt[found.end():found.end() + 85]
        local_tail = SEPARATOR.split(tail, maxsplit=1)[0]
        if re.match(r"\s+(?:would|could|will|was not|is not|training|knowledge|courses?)\b", local_tail, re.I):
            continue
        # Source span ends at the matched tool, avoiding unrelated PII in
        # the rest of the section. Keep original excerpt coordinates.
        text = excerpt[start:found.end()]
        if not text or not match_exact(text, tool):
            continue
        return Finding("supported", "explicit_documented_tool_use",
                       Span(start, found.end(), hashlib.sha256(text.encode()).hexdigest()))
    return Finding("review", "no_verified_explicit_tool_use_span", None)


def match_exact(span, tool):
    return re.search(r"(?<![\w-])" + re.escape(tool) + r"(?![\w-])", span, re.I) is not None


def check_provenance(excerpt, finding):
    span = finding.evidence
    if span is None:
        return finding.decision == "review"
    return (finding.decision == "supported" and 0 <= span.start < span.end <= len(excerpt)
            and hashlib.sha256(excerpt[span.start:span.end].encode()).hexdigest() == span.sha256)
