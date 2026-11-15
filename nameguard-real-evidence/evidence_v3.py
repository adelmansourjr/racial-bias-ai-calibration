"""Reject uncertain attribution and recognize explicit work-use clauses."""
import re
from evidence import Finding, match_exact
from evidence_v2 import verify_excerpt as expanded, _boundaries, _result, RISK, LIST_MEMBER, LIST_CONNECTOR

UNCERTAIN_ATTRIBUTION = re.compile(
    r"\b(?:suggested\s*/|proposed\s*/|proposed for|intended to|"
    r"(?:support(?:ed)?|assisted)\b[^.!?;•*]{0,100}\b(?:system|application|software)\b"
    r"[^.!?;•*]{0,45}\b(?:written|built|coded)\s+in)\b", re.I)
WORK = (
    r"(?:performance\s+tuning|web\s+(?:development|design)|(?:web\s+)?scraping|"
    r"(?:responsible\s+for\s+)?(?:entering|processing|tracking|recording|"
    r"developing|drafting|preparing|designing|maintaining|creating|building|monitoring|"
    r"administering|automating|reconciling|analy[sz]ing))"
)


def permitted(excerpt, start, end):
    a,b = _boundaries(excerpt, start, end)
    return not UNCERTAIN_ATTRIBUTION.search(excerpt[a:b]) and not RISK.search(excerpt[a:b])


def verify_excerpt(excerpt, tool):
    result = expanded(excerpt, tool)
    if result.evidence and permitted(excerpt, result.evidence.start, result.evidence.end):
        return result
    token = r"(?<![\w-])(?:(?:Microsoft|MS|Oracle)\s+)?" + re.escape(tool) + r"(?![\w-])"
    frames = (
        ("stated_work_activity_with_tool",
         r"\b"+WORK+r"\b[^.!?;•*]{0,110}?\b(?:using|with|in|via)\s+(?:the\s+)?(?:"
         +LIST_MEMBER+LIST_CONNECTOR+r"){0,5}"+token),
        ("created_tool_specific_artifact",
         r"\b(?:created|developed|built|completed|prepared|designed|drafted|wrote|automated|"
         r"implemented|completing|preparing|prepare)\s+(?:(?:a|an|the|several|new|"
         r"reusable|custom|as-built|detailed|fully functional)\s+){0,5}"+token+
         r"\s+(?:GUI|plans?|drawings?|spreadsheets?|dashboards?|scripts?|automation|models?|app(?:s|lications?)?)\b"),
    )
    for reason, frame in frames:
        for m in re.finditer(frame, excerpt, re.I):
            if match_exact(m.group(), tool) and permitted(excerpt, m.start(), m.end()):
                return _result(excerpt, m.start(), m.end(), reason)
    return Finding("review", "no_verified_explicit_tool_use_span", None)
