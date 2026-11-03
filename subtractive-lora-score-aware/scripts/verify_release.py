"""Public two-study manifest check; no model or private source data required."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[2]


def main():
    m=json.loads((ROOT/'subtractive-lora-score-aware/audit/release-manifest.json').read_text())
    for rel,h in m['files'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('Release drift: '+rel)
    print('Verified '+str(len(m['files']))+' public artifacts.')


if __name__=='__main__':main()
