"""Build a model-free, key-free research archive from explicit artifact types."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DIRECTORIES=['src','scripts','tests','audit','results','reference','data','paper','docs','output/pdf']
ROOT_FILES=['README.md','SECURITY.md','CITATION.cff','pyproject.toml','requirements-mlx.txt','requirements-paper.txt','.gitignore','ci-workflow.yml']
EXTENSIONS={'.py','.md','.json','.smt2','.txt','.log','.cff','.toml','.yml','.pdf'}


def selected():
    files=[ROOT/name for name in ROOT_FILES]
    for directory in DIRECTORIES:
        for path in (ROOT/directory).rglob('*'):
            if path.is_file()and'__pycache__'not in path.parts and path.suffix in EXTENSIONS and str(path.relative_to(ROOT))!='audit/artifact-manifest.json':files.append(path)
    return sorted(set(files))


def main():
    files=selected();digests={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
    manifest=ROOT/'audit/artifact-manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['files']!=digests:raise ValueError('Completed manifest differs; create a new release rather than overwriting')
    else:
        manifest.write_text(json.dumps({'files':digests,'contains_real_applicant_data':False,'contains_model_weights':False,
          'reviewer_secret_keys_saved':False,'scope':'Synthetic typed-identity research artifact and preprint draft, not universal raw-CV fairness.'},indent=2)+'\n')
    destination=ROOT/'dist/nameguard-research.zip';destination.parent.mkdir(exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED)as archive:
        for p in files+[manifest]:archive.write(p,Path('nameguard')/p.relative_to(ROOT))
    print(json.dumps({'artifact_files':len(files)+1,'archive_bytes':destination.stat().st_size,'archive_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
