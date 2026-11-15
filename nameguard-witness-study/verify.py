"""Verify output summaries and artifact hashes using only standard-library code."""
import hashlib
import json
from pathlib import Path
from analyze import analyze, read
from proposal_budget import analyze_budget
from posthoc_controls import compute
from opaque_replay import summarize

ROOT = Path(__file__).resolve().parent


def verify_hashes(expected):
    for path, digest in expected.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Artifact differs: {path}")


def main():
    verify_hashes(read(ROOT / "audit/heldout-freeze.json")["sha256"])
    for split in ("dev", "test"):
        folder = ROOT / f"results/{split}"
        for suffix, recomputed in (("summary", analyze(folder)), ("budget", analyze_budget(folder))):
            if recomputed != read(ROOT / f"results/{split}-{suffix}.json"):
                raise ValueError(f"Recomputed results differ: {split}-{suffix}")
    opaque = ROOT / "results/opaque"
    for path, digest in read(opaque / "plan.json")["sources"].items():
        if hashlib.sha256((ROOT.parent / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Opaque replay source differs: {path}")
    result = read(opaque / "summary.json")
    expected_param = read(ROOT / "audit/checkpoint-manifest.json")["parameter_digest"]
    if not result["checkpoint_files_verified"] or result["parameter_sha256"] != expected_param:
        raise ValueError("Opaque replay checkpoint integrity failure")
    recomputed_opaque = json.loads(json.dumps(summarize(read(opaque / "rows.json"))))
    if recomputed_opaque != result["metrics"]:
        raise ValueError("Opaque replay results differ")
    if compute() != read(ROOT / "results/posthoc-controls.json"):
        raise ValueError("Budget or fixed ranking controls differ")
    manifest = ROOT / "audit/public-manifest.json"
    if manifest.exists():
        verify_hashes(read(manifest)["sha256"])
    print("Verified frozen protocol, full case/name coverage, activation comparison logs, held-out results and artifact hashes.")


if __name__ == "__main__":
    main()
