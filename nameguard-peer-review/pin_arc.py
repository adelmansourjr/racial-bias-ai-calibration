"""Pin the official dataset revision and verify its rows equal the measured snapshot."""
import io,json,hashlib,ssl
from pathlib import Path
from urllib.request import urlopen
import certifi
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent

def main():
    target=ROOT/'audit/arc-pinned-source.json'
    if target.exists():raise FileExistsError(target)
    ctx=ssl.create_default_context(cafile=certifi.where())
    with urlopen('https://huggingface.co/api/datasets/allenai/ai2_arc',context=ctx,timeout=40) as r:metadata=json.load(r)
    revision=metadata['sha']
    url=f'https://huggingface.co/datasets/allenai/ai2_arc/resolve/{revision}/ARC-Challenge/test-00000-of-00001.parquet'
    with urlopen(url,context=ctx,timeout=40) as r:raw=r.read()
    selected=pq.read_table(io.BytesIO(raw)).to_pylist()[200:300]
    measured=[x['row'] for x in json.loads((ROOT/'local-private/arc-challenge-200-299.json').read_text())['rows']]
    canonical=lambda rows:json.dumps(rows,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    if canonical(selected)!=canonical(measured):raise ValueError('Pinned public source differs from measured benchmark rows')
    result={'dataset_revision':revision,'url':url,'source_parquet_sha256':hashlib.sha256(raw).hexdigest(),
            'selected_rows_canonical_sha256':hashlib.sha256(canonical(selected)).hexdigest(),
            'exactly_matches_measured_rows':True,'offset':200,'length':100,
            'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
