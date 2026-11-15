"""Sample fresh documents, with three excerpts per tool and lexical stratum."""
import argparse
import csv
import json
from pathlib import Path
from sample import ROOT, WORKSPACE, SKILLS, NEARBY, match_skill, norm, prior_keys, read_json, save, sha, snippet


def select(source, previous):
    excluded_ids, excluded_prefixes = prior_keys()
    old = [r for rows in read_json(previous).values() for r in rows]
    excluded_ids.update(r["source_id"] for r in old)
    excluded_prefixes.update(r["document_prefix"][:100] for r in old)
    used_ids, prefixes, windows = set(), set(), set()
    with source.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    result = []
    for tool in SKILLS:
        for band in ("near_verb", "other_mention"):
            candidates = []
            for row in source_rows:
                id_ = row["ID"]
                text = " ".join(row["Resume_str"].split())
                prefix = norm(text)[:120]
                if id_ in excluded_ids or sha("resume:" + id_) in excluded_ids or prefix[:100] in excluded_prefixes:
                    continue
                spans = []
                for m in match_skill(tool).finditer(text):
                    local = text[max(0,m.start()-65):m.end()+65]
                    observed = "near_verb" if NEARBY.search(local) else "other_mention"
                    excerpt = snippet(text, m.start(), m.end())
                    if observed == band and len(excerpt)>140 and not any(x in excerpt for x in ("<|","@","http","www.")):
                        spans.append(excerpt)
                if spans:
                    excerpt = min(spans, key=lambda s: (len(list(match_skill(tool).finditer(s))),sha(s)))
                    candidates.append((sha(f"fresh-evidence:61116:{tool}:{band}:{id_}"), id_, prefix, excerpt))
            added = 0
            for _, id_, prefix, excerpt in sorted(candidates):
                hashed_window = sha(norm(excerpt))
                if id_ in used_ids or prefix in prefixes or hashed_window in windows:
                    continue
                result.append({"case": f"fresh-{len(result):02}", "source_id": id_, "document_prefix": prefix,
                               "excerpt": excerpt, "tool": tool, "band": band})
                used_ids.add(id_)
                prefixes.add(prefix)
                windows.add(hashed_window)
                added += 1
                if added == 3:
                    break
            if added < 3:
                raise ValueError(f"Insufficient fresh documents for {tool} / {band}: {added}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=WORKSPACE / "research/data/Resume.csv")
    p.add_argument("--initial", type=Path, default=ROOT / "local-private/cohort.json")
    p.add_argument("--output", type=Path, default=ROOT / "local-private/fresh-cohort.json")
    args = p.parse_args()
    result = select(args.source, args.initial)
    save(args.output, result)
    print(json.dumps({"fresh_cases": len(result), "calibration_cases_excluded": sum(map(len,read_json(args.initial).values())),
                      "cases_per_tool_and_stratum": 3}, indent=2))
