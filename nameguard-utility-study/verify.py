"""Read-only reproduction of result summaries and frozen artifact hashes."""
import hashlib
from collections import defaultdict
from pathlib import Path
from analyze import analyze, read
from analyze_compositional import summary
from historical_diagnosis import diagnosis
from structured_policy_audit import audit_policy
from methods import classify

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(actual, expected, artifact):
    if actual != expected:
        raise ValueError(f"Artifact failed verification: {artifact}")


def main():
    for path, expected in read(ROOT / "audit/pretest.json")["pre_test_source_sha256"].items():
        check(digest(ROOT / path), expected, path)
    for path, expected in read(ROOT / "audit/compositional-pretest.json")["pre_inference_sha256"].items():
        check(digest(ROOT / path), expected, path)
    for folder in ("development", "test"):
        check(analyze(ROOT / "results" / folder), read(ROOT / "results" / f"{folder}-summary.json"), folder)
    check(summary(ROOT / "results/compositional"), read(ROOT / "results/compositional-summary.json"), "compositional")
    check(diagnosis(), read(ROOT / "results/historical-diagnosis.json"), "historical")
    check(audit_policy(), read(ROOT / "results/structured-policy-audit.json"), "structured policy")
    replay_dir = ROOT / "results/historical-replay"
    for path, expected in read(replay_dir / "plan.json")["sources"].items():
        check(digest(WORKSPACE / path), expected, path)
    replay = read(replay_dir / "rows.json")
    baseline = read(WORKSPACE / "nameguard-peer-review/results/gpu/synthetic-rows.json")
    by_key = {(r["context"], r["name"]): r for r in replay}
    check(len(replay), len(by_key), "replay uniqueness")
    check(set(by_key), {(r["context"], r["name"]) for r in baseline}, "replay row coverage")
    grouped = defaultdict(list)
    for prior in baseline:
        row = by_key[prior["context"], prior["name"]]
        check(row["fewshot"], classify(row["logits"]), "replay prediction")
        check(row["native_correct"], (prior["native_margin"] >= 0) == prior["label"], "native replay reference")
        check(row["old_gate_correct"], (prior["compiled_margin"] >= 0) == prior["label"], "old gate replay reference")
        check(row["status"], "confirmed" if prior["label"] else "contradicted", "replay label")
        grouped[row["context"]].append(row)
    scores = read(replay_dir / "summary.json")
    check(sum(row["fewshot"] == row["status"] for row in replay), scores["new_fewshot_correct"], "replay accuracy")
    check(sum(len({(tuple(row["logits"]), row["report"]) for row in block}) == 1 for block in grouped.values()),
          scores["identical_logits_and_reports_contexts"], "replay equality")
    manifest = ROOT / "audit/public-manifest.json"
    if manifest.exists():
        for path, expected in read(manifest)["sha256"].items():
            check(digest(ROOT / path), expected, path)
    print("Frozen source, row completeness, historical diagnosis, reproducible utility summaries and artifact hashes verified.")


if __name__ == "__main__":
    main()
