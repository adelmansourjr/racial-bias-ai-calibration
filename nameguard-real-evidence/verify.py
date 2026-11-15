"""Verify aggregate accounting, frozen code, and local real-CV output when present."""
import hashlib
import json

from analyze import analyze, RUNS
from posthoc_guard import summarize
from sample import ROOT, WORKSPACE, read_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def validate_freeze(filename, field, archived=None):
    aliases = archived or {}
    for key, expected in read_json(ROOT / "audit" / filename)[field].items():
        require(digest(ROOT / aliases.get(key, key)) == expected, f"Frozen source drift: {key}")


def main():
    validate_freeze("reference-freeze.json", "sources")
    validate_freeze("fresh-freeze.json", "sources", {"followup_sample.py": "audit/first-followup-sampler.py"})
    validate_freeze("confirm-freeze.json", "source_sha256", {"confirm_sample.py": "audit/first-confirm-sampler.py"})
    audit = ROOT / "audit"
    for filename, field, path in (
        ("fresh-sampling-revision.json", "sha256", "audit/second-followup-sampler.py"),
        ("fresh-sampling-final.json", "sampler_sha256", "followup_sample.py"),
        ("confirm-sampling-revision.json", "sha256", "confirm_sample.py"),
    ):
        require(digest(ROOT / path) == read_json(audit / filename)[field], "Sampler provenance differs")
    expected_parameters = read_json(audit / "checkpoint-manifest.json")["parameter_digest"]
    attached = read_json(audit / "gpu-plans.json")
    completed = []
    for name in RUNS:
        for path, expected in attached[name]["sources"].items():
            require(digest(WORKSPACE / path) == expected, f"GPU source drift: {path}")
        result = read_json(ROOT / "results" / f"{name}-summary.json")
        n = result["independent_cv_excerpts"]
        yes = result["audit"]["labeled_support"]
        a = result["audit"]
        require(n == attached[name]["cases"] and result["planned_name_requests"] == 4 * n == a["admitted_requests"],
                "Incorrect coverage denominator")
        require(a["exact_scores"] == a["exact_reports"] == n, "Recorded conformance differs")
        require(a["planned_direct_array_checks"] == a["exact_direct_arrays"] == ((n + 7) // 8) * 3 * 31,
                "Direct array comparison count differs")
        for arm in result["decisions"].values():
            require(arm["correct_support"] + arm["missed_support"] == yes, "Positive accounting differs")
            require(arm["correct_review"] + arm["unsupported_definitive"] == n - yes, "Review accounting differs")
            require(arm["correct_support"] + arm["unsupported_definitive"] == arm["definitive"],
                    "Definitive coverage accounting differs")
        require(result["runtime"]["parameter_sha256"] == expected_parameters, "Parameter digest mismatch")
        local = ROOT / "local-private" / f"{name}-run" / "rows.json"
        if local.exists():
            require(json.loads(json.dumps(analyze(name))) == result, f"Local results drift: {name}")
            completed.append(name)
    if len(completed) == len(RUNS):
        require(summarize() == read_json(ROOT / "results/posthoc-actor-guard.json"), "Post-hoc guard results drift")
    manifest = audit / "public-manifest.json"
    if manifest.exists():
        for path, expected in read_json(manifest)["sha256"].items():
            require(digest(ROOT / path) == expected, f"Release differs: {path}")
    print(f"Verified aggregate accounting, frozen code and evidence integrity. Recomputed {len(completed)} private cohorts when available.")


if __name__ == "__main__":
    main()
