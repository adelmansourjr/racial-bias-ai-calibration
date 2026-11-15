"""Verify the rerun artifacts, archived replay, and exact recorded scores."""
from dependencies import ROOT
from measure import measure, read, file_hash, score_bits
from historical_replay import replay


def main():
    for split in ("dev", "test"):
        result = ROOT / "results" / split
        if measure(result) != read(ROOT / "results" / f"{split}-summary.json"):
            raise ValueError(f"Result differs on recomputation: {split}")
        for row in read(result / "rows.json"):
            if row["logits"] is not None and score_bits(row["logits"]).hex() != row["score_bits"]:
                raise ValueError("Score bytes disagree with the recorded bits")
    if replay() != read(ROOT / "results/historical-replay.json"):
        raise ValueError("Historical replay drift")
    manifest = ROOT / "audit/public-manifest.json"
    if manifest.exists():
        for relative, expected in read(manifest)["sha256"].items():
            if file_hash(ROOT / relative) != expected:
                raise ValueError(f"Artifact differs: {relative}")
    print("Verified frozen sources, fresh and archived coverage, score bytes, activation comparisons, and public artifacts.")


if __name__ == "__main__":
    main()
