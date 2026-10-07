"""Download the supplied public resume CSV and small job-viewer sample."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

root=Path(__file__).resolve().parent.parent/'data'
root.mkdir(exist_ok=True)
urls={
    'resume-metadata.json':'https://www.kaggle.com/api/v1/datasets/list?search=snehaanbhawal%2Fresume-dataset',
    'jobs-first-rows.json':'https://datasets-server.huggingface.co/first-rows?dataset=datastax%2Flinkedin_job_listings&config=default&split=train',
}
for filename,url in urls.items():
    with urlopen(url,timeout=60) as response:
        (root/filename).write_bytes(response.read())
archive=root/'resume-download.zip'
with urlopen('https://www.kaggle.com/api/v1/datasets/download/snehaanbhawal/resume-dataset',timeout=60) as response,archive.open('wb') as stream:
    while block:=response.read(1024*1024):stream.write(block)
with ZipFile(archive) as z:
    names=[n for n in z.namelist() if n.lower().endswith('.csv')]
    if names!=['Resume/Resume.csv']:raise ValueError(f'Unexpected CSV layout: {names}')
    (root/'Resume.csv').write_bytes(z.read(names[0]))
archive.unlink()
print('Downloaded source CSV and public job sample; no model weights downloaded.')
