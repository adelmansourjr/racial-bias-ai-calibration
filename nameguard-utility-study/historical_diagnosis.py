"""Reconstruct the published utility discrepancy without changing its records."""
from collections import defaultdict
import json
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "nameguard-peer-review/results/gpu/synthetic-rows.json"


def diagnosis():
    raw = SOURCE.read_bytes()
    rows = json.loads(raw)
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["style"], row["label"])].append(row)
    cells = []
    for (style, label), group in sorted(by_group.items()):
        cells.append({"wording": style, "truth": label, "rows": len(group),
                      "cv_conditions": len({row["context"] for row in group}),
                      "native_correct": sum((r["native_margin"] >= 0) == label for r in group),
                      "protected_correct": sum((r["compiled_margin"] >= 0) == label for r in group)})
    return {"post_hoc_diagnosis": True, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "cells": cells, "native_total": sum(c["native_correct"] for c in cells),
            "protected_total": sum(c["protected_correct"] for c in cells),
            "scope": "Repeated names share CV conditions. This localizes historical errors, without identifying a unique causal mechanism."}


if __name__ == "__main__":
    print(json.dumps(diagnosis(), indent=2))
