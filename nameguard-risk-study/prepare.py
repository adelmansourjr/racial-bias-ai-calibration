"""Construct fixed matched article-disjoint evidence cohorts from official SQuAD 2.0."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SOURCE_SHA256 = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"
TAG = "nameguard-risk-study-v1"
NAMES = ("Elina Vaher", "Tariq Al-Mansouri", "Nkemdilim Okoro", "Ánh Nguyễn")
WITNESSES = ("Lena Frost", "Sam Bello")
SIZES = {"development": (8, 72), "holdout": (17, 156), "reserve": (10, 0)}
SOURCES = (
    "nameguard-risk-study/prepare.py",
    "nameguard-risk-study/run.py",
    "nameguard-risk-study/risk.py",
    "nameguard-risk-study/PROTOCOL.md",
    "nameguard-risk-study/audit/checkpoint-manifest.json",
    "nameguard-utility-study/experiment.py",
    "native-token-gate-hardening/gate.py",
    "native-token-gate-hardening/evidence_contract.py",
    "nameguard-peer-review/src/compiler.py",
)


def source_paths():
    return sorted([WORKSPACE / p for p in SOURCES] + list(
        (WORKSPACE / "nameguard-validation/src/nameguard").glob("*.py")))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(*parts):
    return hashlib.sha256(json.dumps([TAG, *parts], ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value, *, freeze=False):
    with path.open("x" if freeze else "w", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def candidate(passage, qa):
    question = qa["question"]
    if not question.strip() or "\n" in question or "<|" in question:
        return None
    answers = qa["plausible_answers"] if qa["is_impossible"] else qa["answers"]
    options = Counter(
        (a["answer_start"], a["text"]) for a in answers
        if a["text"].strip() and "\n" not in a["text"] and "<|" not in a["text"]
        and a["answer_start"] >= 0
        and passage[a["answer_start"]:a["answer_start"] + len(a["text"])] == a["text"]
    )
    if not options:
        return None
    start, answer = min(options, key=lambda key: (-options[key], key))
    return dict(id=qa["id"], question=question, answer=answer, start=start)


def build(source):
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Expected the byte-exact official SQuAD 2.0 development JSON")
    articles = sorted(read(source)["data"], key=lambda a: rank("article", a["title"]))
    if len(articles) != sum(size for size, _ in SIZES.values()):
        raise ValueError("Unexpected number of source articles")
    partitions = {}
    used = set()
    for split, (count, target) in SIZES.items():
        selected, articles = articles[:count], articles[count:]
        pairs = []
        for article in selected:
            title = article["title"]
            for index, paragraph in enumerate(article["paragraphs"]):
                passage = paragraph["context"]
                if not 0 < len(passage) <= 1600 or "<|" in passage:
                    continue
                matching = {True: [], False: []}
                for qa in paragraph["qas"]:
                    row = candidate(passage, qa)
                    if row:
                        matching[not qa["is_impossible"]].append(row)
                if not all(matching.values()):
                    continue
                pair = rank("paragraph", title, index)
                rows = []
                for truth, options in matching.items():
                    chosen = min(options, key=lambda r: rank("candidate", r["id"]))
                    rows.append((chosen["id"], truth, dict(chosen, article=title, pair=pair, passage=passage)))
                pairs.append((pair, rows))
        rows, labels = {}, {}
        for _, matched in sorted(pairs)[:target]:
            for key, truth, row in matched:
                if key in used:
                    raise ValueError("Question ID repeats across the benchmark")
                used.add(key)
                rows[key] = row
                labels[key] = truth
        if len(rows) != 2 * target:
            raise ValueError("Insufficient distinct matched paragraphs")
        partitions[split] = dict(articles=[a["title"] for a in selected], rows=rows, labels=labels)
    return partitions


def prepare(source, queue, plan_path):
    if queue.exists() or plan_path.exists():
        raise FileExistsError("Use fresh run paths to preserve the original freeze")
    splits = build(source)
    queue.mkdir(parents=True)
    plan = dict(dataset_sha256=SOURCE_SHA256, names=NAMES, witnesses=WITNESSES,
                sources={str(p.relative_to(WORKSPACE)): digest(p) for p in source_paths()},
                design=dict(risk_upper=0.10, coverage_min=0.20, alpha="1/20", audit_max=160,
                            dev_empirical_risk_max="1/25", minimum_dev_accepts=12),
                articles={}, cases={}, files={})
    for split, data in splits.items():
        plan["articles"][split] = data["articles"]
        if not data["rows"]:
            continue
        for filename, payload in ((f"{split}-inputs.json", data["rows"]),
                                  (f"{split}-labels.json", data["labels"])):
            path = queue / filename
            save(path, payload, freeze=True)
            plan["files"][filename] = digest(path)
        plan["cases"][split] = sorted(data["rows"])
    plan["conformance"] = sorted(plan["cases"]["holdout"], key=lambda key: rank("conformance", key))[:20]
    audit_order = list(plan["cases"]["holdout"])
    secrets.SystemRandom().shuffle(audit_order)
    plan["audit_order"] = audit_order
    save(plan_path, plan, freeze=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/dev-v2.0.json")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.source, arguments.queue, arguments.plan)
