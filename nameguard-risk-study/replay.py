"""Verify the archived benchmark plans and rebuild their local input/label queues."""
import argparse
from pathlib import Path

from prepare import ROOT, WORKSPACE, build, digest, read, save
import prepare


def replay(source, plan_path, reserve_path, output):
    initial, reserve = read(plan_path), read(reserve_path)
    if reserve["original_plan_sha256"] != digest(plan_path):
        raise ValueError("Reserve protocol does not bind the original plan")
    for plan in (initial, reserve):
        if digest(source) != plan["dataset_sha256"]:
            raise ValueError("Dataset hash mismatch")
        for path, expected in plan["sources"].items():
            if digest(WORKSPACE / path) != expected:
                raise ValueError(f"Archived plan source differs: {path}")
    prepare.SIZES["reserve"] = (10, 160)
    try:
        splits = build(source)
    finally:
        prepare.SIZES["reserve"] = (10, 0)
    reserve_split = splits.pop("reserve")
    output.mkdir(parents=True)
    queue = output / "queue"
    queue.mkdir()
    for name, split in splits.items():
        if sorted(split["rows"]) != initial["cases"][name] or split["articles"] != initial["articles"][name]:
            raise ValueError("Original registered cohort differs")
        for suffix, field in (("inputs", "rows"), ("labels", "labels")):
            path = queue / f"{name}-{suffix}.json"
            save(path, split[field], freeze=True)
            if digest(path) != initial["files"][path.name]:
                raise ValueError("Original registered queue hash differs")
    path = output / "reserve"
    path.mkdir()
    if sorted(reserve_split["rows"]) != reserve["cases"] or reserve_split["articles"] != reserve["articles"]:
        raise ValueError("Reserve cohort differs")
    for field, digest_name in (("rows", "inputs"), ("labels", "labels")):
        file = path / f"{field}.json"
        save(file, reserve_split[field], freeze=True)
        if digest(file) != reserve[f"{digest_name}_sha256"]:
            raise ValueError("Reserve queue hash differs")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/dev-v2.0.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "audit/plan.json")
    parser.add_argument("--reserve-plan", type=Path, default=ROOT / "audit/reserve-plan.json")
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    print(replay(a.source, a.plan, a.reserve_plan, a.output))
