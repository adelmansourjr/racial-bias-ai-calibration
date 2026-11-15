"""Draw a distinct benchmark from six tool families absent in calibration."""
import argparse
import csv
import json
from pathlib import Path
from sample import ROOT, WORKSPACE, NEARBY, match_skill, norm, prior_keys, read_json, save, sha, snippet

NEW_TOOLS = ("Tableau", "MATLAB", "QuickBooks", "WordPress", "Photoshop", "Salesforce")


def select(source, prior_paths):
    blocked, old_prefixes = prior_keys()
    for path in prior_paths:
        record = read_json(path)
        items = [x for split in record.values() for x in split] if isinstance(record,dict) else record
        for entry in items:
            blocked.add(entry["source_id"])
            old_prefixes.add(entry["document_prefix"][:100])
    with source.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    seen_ids, seen_prefixes, seen_excerpts = set(), set(), set()
    output = []
    for tool in NEW_TOOLS:
        for band in ("near_verb","other_mention"):
            candidates = []
            for record in source_rows:
                id_ = record["ID"]
                text = " ".join(record["Resume_str"].split())
                prefix = norm(text)[:120]
                if id_ in blocked or sha("resume:" + id_) in blocked or prefix[:100] in old_prefixes:
                    continue
                windows = []
                for m in match_skill(tool).finditer(text):
                    visible = snippet(text, m.start(), m.end())
                    near = "near_verb" if NEARBY.search(text[max(0,m.start()-65):m.end()+65]) else "other_mention"
                    if near == band and len(visible)>140 and not any(c in visible for c in ("<|","@", "http","www.")):
                        windows.append(visible)
                if windows:
                    window = min(windows,key=sha)
                    candidates.append((sha(f"unseen-tools:61116:{tool}:{band}:{id_}"), id_, prefix, window))
            n = 0
            for _, id_, prefix, window in sorted(candidates):
                if id_ in seen_ids or prefix in seen_prefixes or norm(window) in seen_excerpts:
                    continue
                output.append({"case": f"confirm-{len(output):02}", "source_id": id_, "document_prefix": prefix,
                               "excerpt": window, "tool": tool, "band": band})
                seen_ids.add(id_)
                seen_prefixes.add(prefix)
                seen_excerpts.add(norm(window))
                n += 1
                if n == 3:
                    break
            if n != 3:
                raise ValueError(f"Undersampled tool/stratum: {tool}, {band}, {n}")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=WORKSPACE / "research/data/Resume.csv")
    p.add_argument("--output", type=Path, default=ROOT / "local-private/confirm-cohort.json")
    p.add_argument("--exclude", type=Path, action="append",
                   default=[ROOT / "local-private/cohort.json", ROOT / "local-private/fresh-cohort.json"])
    a = p.parse_args()
    rows = select(a.source, a.exclude)
    save(a.output, rows)
    print(json.dumps({"tools": NEW_TOOLS, "new_cases": len(rows)}, indent=2))
