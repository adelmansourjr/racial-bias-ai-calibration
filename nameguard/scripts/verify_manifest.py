"""Read-only check of the publication artifact manifest."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'audit/artifact-manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Artifact drift: '+name)
    print(f"Verified {len(manifest['files'])} publication artifacts.")


if __name__=='__main__':main()
