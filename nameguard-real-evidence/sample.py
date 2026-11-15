"""Draw distinct real CV snippets without consulting the model or verifier."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SKILLS = ("Excel", "SQL", "Python", "AutoCAD", "Java", "SAP")
NAMES = ("Ivo", "مروان القيسي", "Linh Trần", "中村 美咲")
WITNESSES = ("Jules", "Tomas Vale", "Elena Petrova")
DEV_NAMES = ("Amina Bakri", "Theo Moss")
NEARBY = re.compile(r"\b(?:using|used|utilized|developed|created|built|implemented|worked with|experience with|proficient in|knowledge of|designed|prepared)\b", re.I)
PRIOR = (
    "nameguard-validation/local-private/contexts.json",
    "native-token-activation-gate/local-private/contexts.json",
    "subtractive-lora-validation/local-private/contexts.json",
    "subtractive-lora-score-aware/local-private/confirmation-contexts.json",
)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def match_skill(skill):
    return re.compile(r"(?<![\w-])" + re.escape(skill) + r"(?![\w-])", re.I)


def norm(text):
    return " ".join(text.casefold().split())


def read_json(path):
    return json.loads(path.read_text())


def save(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def prior_keys():
    old_ids, prefixes = set(), set()
    def walk(node):
        if isinstance(node, list):
            for entry in node:
                walk(entry)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key in ("cv", "cv_cluster", "resume_id") and isinstance(value, (str, int)):
                    old_ids.add(str(value))
                if key in ("text", "cv_excerpt") and isinstance(value, str) and len(value) > 80:
                    prefixes.add(norm(value)[:100])
                if isinstance(value, (list, dict)):
                    walk(value)
    for relative in PRIOR:
        path = WORKSPACE / relative
        if path.exists():
            walk(read_json(path))
    return old_ids, prefixes


def snippet(text, start, end):
    first = max(0, start - 165)
    last = min(len(text), end + 150)
    if first:
        space = text.find(" ", first, start)
        if space != -1:
            first = space + 1
    if last < len(text):
        space = text.rfind(" ", end, last + 1)
        if space != -1:
            last = space
    return text[first:last].strip()


def sample(source):
    with source.open(encoding="utf-8", newline="") as stream:
        cvs = list(csv.DictReader(stream))
    old_ids, old_prefixes = prior_keys()
    candidates = {}
    for tool in SKILLS:
        groups = {"near_verb": [], "other_mention": []}
        for row in cvs:
            key = row["ID"]
            if key in old_ids or sha("resume:" + key) in old_ids or norm(row["Resume_str"])[:100] in old_prefixes:
                continue
            text = " ".join(row["Resume_str"].split())
            choices = {"near_verb": [], "other_mention": []}
            for found in match_skill(tool).finditer(text):
                window = snippet(text, found.start(), found.end())
                # The matcher prioritizes mentions but never reads outcome labels.
                band = "near_verb" if NEARBY.search(text[max(0, found.start()-65) : found.end()+65]) else "other_mention"
                if len(window) > 140 and not any(ch in window for ch in ("<|", "@", "http", "www.")):
                    choices[band].append(window)
            # Use at most one candidate span per input CV and tool.
            for band, windows in choices.items():
                if windows:
                    # Fix position independently of the checker. Favor windows
                    # containing a single target mention to simplify adjudication.
                    window = min(windows, key=lambda s: (len(list(match_skill(tool).finditer(s))), sha(s)))
                    groups[band].append({"tool": tool, "source_id": key, "excerpt": window, "band": band,
                                         "document_prefix": norm(text)[:120], "category": row["Category"]})
        for band, rows in groups.items():
            candidates[tool, band] = sorted(rows, key=lambda r: sha(f"real-evidence:20261116:{tool}:{band}:{r['source_id']}"))
    used = set()
    used_prefixes = set()
    excerpts = set()
    split_rows = {}
    for split, count in (("dev", 1), ("test", 4)):
        chosen = []
        for tool in SKILLS:
            for band in ("near_verb", "other_mention"):
                added = 0
                for row in candidates[tool, band]:
                    item_hash = sha(norm(row["excerpt"]))
                    if row["source_id"] in used or row["document_prefix"] in used_prefixes or item_hash in excerpts:
                        continue
                    chosen.append({"case": f"{split}-{len(chosen):02}", **row})
                    used.add(row["source_id"])
                    used_prefixes.add(row["document_prefix"])
                    excerpts.add(item_hash)
                    added += 1
                    if added == count:
                        break
                if added != count:
                    raise ValueError(f"Underfilled stratum: {split}, {tool}, {band}")
        split_rows[split] = chosen
    return split_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=WORKSPACE / "research/data/Resume.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "local-private/cohort.json")
    args = parser.parse_args()
    cohorts = sample(args.source)
    save(args.output, cohorts)
    print(json.dumps({"dataset_file_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      "split_sizes": {k: len(v) for k, v in cohorts.items()},
                      "prior_local_files_found": sum((WORKSPACE/p).exists() for p in PRIOR)}, indent=2))
