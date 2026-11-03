"""Read-only public release hash validation."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    m=json.loads((ROOT/'audit/release-manifest.json').read_text())
    for rel,h in m['files'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('Release drift: '+rel)
    print('Verified '+str(len(m['files']))+' public artifacts; no raw CV text or model weights required.')


if __name__=='__main__':main()
