"""Read-only final publication-artifact check, without GPU or NumPy."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'audit/artifact-manifest.json').read_text())
    for rel,digest in manifest['files'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Artifact drift: '+rel)
    print(f"Verified {len(manifest['files'])} publication artifacts.")


if __name__=='__main__':main()
