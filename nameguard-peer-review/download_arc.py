"""Fetch the immutable official ARC source and verify all 100 measured rows."""
import hashlib
import io
import json
from pathlib import Path
import ssl
from urllib.request import urlopen
import certifi
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent


def main():
    path=ROOT/'local-private/arc-challenge-200-299.json'
    if path.exists():raise FileExistsError('Preserve the existing benchmark snapshot')
    pin=json.loads((ROOT/'audit/arc-pinned-source.json').read_text())
    context=ssl.create_default_context(cafile=certifi.where())
    with urlopen(pin['url'],timeout=45,context=context) as response:raw=response.read()
    if hashlib.sha256(raw).hexdigest()!=pin['source_parquet_sha256']:raise ValueError('Pinned parquet digest mismatch')
    rows=pq.read_table(io.BytesIO(raw)).to_pylist()[pin['offset']:pin['offset']+pin['length']]
    canonical=json.dumps(rows,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    if hashlib.sha256(canonical).hexdigest()!=pin['selected_rows_canonical_sha256']:raise ValueError('Pinned row digest mismatch')
    if len(rows)!=100:raise ValueError('Incomplete benchmark')
    data=json.dumps({'rows':[{'row_idx':200+i,'row':r} for i,r in enumerate(rows)]},ensure_ascii=False).encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    report={'url':pin['url'],'snapshot_sha256':hashlib.sha256(data).hexdigest(),
            'selected_rows_canonical_sha256':pin['selected_rows_canonical_sha256'],
            'scope':'Transport JSON can differ from the historical viewer snapshot; every question/choice/answer row is verified exactly.'}
    (path.parent/'arc-download-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
