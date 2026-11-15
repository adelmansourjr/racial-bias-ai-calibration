"""Expand the documented-use grammar while preserving exact source provenance."""
import hashlib
import re
from evidence import Span, Finding, verify_excerpt as conservative, match_exact

ACTIVE = r"(?:developed|designed|drafted|created|built|implemented|wrote|maintained|generated|prepared|constructed|automated|managed|processed|produced|tested|analyzed|analysed|performed|troubleshot|configur(?:ed|ing)|deploy(?:ed|ing))"
EXPLICIT = r"(?:used|utilized|utilised|leveraged|operated)"
SECTION = re.compile(r"[.!?;•*]|\b(?:technical skills|skills|qualifications|education|certifications)\s*:", re.I)
RISK = re.compile(
    r"\b(?:no|not|never|without|would|could|should|will|planned to|seeking to|knowledge of|"
    r"familiar with|proficient in|awareness of|training (?!and test data|data)|"
    r"job requirements?|responsible for training|taught|students?|colleagues?|"
    r"client(?:'s|s used| used)|customer(?:'s|s used| used)|team members?|"
    r"assisted|tutorial|coursework)\b", re.I)
BRAND = r"(?:(?:MS|Microsoft|Oracle|SAP)\s+)?"
LIST_MEMBER = r"[A-Za-z0-9#_+&/() -]{1,38}"
LIST_CONNECTOR = r"(?:,\s*(?:and\s+)?|\s+and\s+|\s*&\s*|\s+plus\s+)"


def _boundaries(text, start, end):
    prior = [m.end() for m in SECTION.finditer(text, max(0, start - 100), start)]
    later = SECTION.search(text, end, min(len(text), end + 100))
    return max(prior, default=max(0, start - 80)), later.start() if later else min(len(text), end + 90)


def _result(excerpt, start, end, reason):
    return Finding("supported", reason,
                   Span(start, end, hashlib.sha256(excerpt[start:end].encode()).hexdigest()))


def verify_excerpt(excerpt, tool):
    previous = conservative(excerpt, tool)
    if previous.evidence:
        s = previous.evidence
        left, right = _boundaries(excerpt, s.start, s.end)
        if not RISK.search(excerpt[left:right]):
            return previous

    token = r"(?<![\w-])" + BRAND + re.escape(tool) + r"(?![\w-])"
    phrases = (
        ("explicit_tool_before_activity",
         r"\busing\s+" + token + r"\s*,\s*(?:I\s+)?" + ACTIVE + r"\s+[^.!?;•*]{3,85}"),
        ("explicit_use_within_technology_list",
         r"\b" + EXPLICIT + r"\s+(?:" + LIST_MEMBER + LIST_CONNECTOR + r"){0,5}" + token),
        ("stated_activity_using_technology",
         r"\b" + ACTIVE + r"\b[^.!?;•*]{0,115}?\b(?:using|utilizing|with|in|via)\s+(?:"
         + LIST_MEMBER + LIST_CONNECTOR + r"){0,5}" + token),
        ("named_tool_linked_to_own_work_role",
         r"\b(?:languages|tools|software|technologies)\s+used\s+(?:in\s+(?:this|the|my)\s+(?:position|role|project)\s+)?"
         r"(?:were\s+|was\s+|:\s*)?(?:" + LIST_MEMBER + LIST_CONNECTOR + r"){0,5}" + token),
        ("explicit_tool_artifact_activity",
         r"\b" + ACTIVE + r"\s+(?:(?:a|an|the|several|various|multiple|custom|new|reusable)\s+){0,3}"
         + token + r"\s+(?:scripts?|utilities|applications?|programs?|models?|APIs?|macros?|modules?|services?)\b"),
    )
    for reason, pattern in phrases:
        for candidate in re.finditer(pattern, excerpt, re.I):
            left, right = _boundaries(excerpt, candidate.start(), candidate.end())
            if RISK.search(excerpt[left:right]) or not match_exact(candidate.group(), tool):
                continue
            return _result(excerpt, candidate.start(), candidate.end(), reason)
    return Finding("review", "no_verified_explicit_tool_use_span", None)
