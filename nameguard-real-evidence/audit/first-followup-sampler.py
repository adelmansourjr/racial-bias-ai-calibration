"""Reserve a fresh document split after removing the initial cohort."""
import argparse
import csv
import json
from pathlib import Path
import tempfile

from sample import ROOT, WORKSPACE, sample, read_json, norm, save


def main(source, initial, out):
    old = [item for split in read_json(initial).values() for item in split]
    ids = {item["source_id"] for item in old}
    prefixes = {item["document_prefix"] for item in old}
    with tempfile.TemporaryDirectory(prefix="nameguard-real-sample-") as tmp:
        filtered = Path(tmp) / "candidate-source.csv"
        with source.open(encoding="utf-8", newline="") as f, filtered.open("w", encoding="utf-8", newline="") as g:
            reader = csv.DictReader(f)
            writer = csv.DictWriter(g, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row["ID"] not in ids and norm(row["Resume_str"])[:120] not in prefixes:
                    writer.writerow(row)
        result = sample(filtered)
    selected = result["test"]
    if any(r["source_id"] in ids or r["document_prefix"] in prefixes for r in selected):
        raise ValueError("Fresh cohort overlaps calibration")
    save(out, [{**row, "case": row["case"].replace("test-", "fresh-")} for row in selected])
    print(json.dumps({"fresh_cases": len(selected), "calibration_cases_excluded": len(old)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=WORKSPACE / "research/data/Resume.csv")
    p.add_argument("--initial", type=Path, default=ROOT / "local-private/cohort.json")
    p.add_argument("--output", type=Path, default=ROOT / "local-private/fresh-cohort.json")
    a = p.parse_args()
    main(a.source, a.initial, a.output)
