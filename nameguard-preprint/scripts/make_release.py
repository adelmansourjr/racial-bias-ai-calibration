"""Build a small immutable publication archive with no keys or model weights."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
DIRS=['src','scripts','tests','audit','results','reference','data','paper','docs','output/pdf']
ROOT_FILES=['README.md','LICENSE','SECURITY.md','CITATION.cff','pyproject.toml','requirements-mlx.txt','requirements-paper.txt','.gitignore','ci-workflow.yml']
EXTS={'.py','.md','.json','.smt2','.txt','.log','.cff','.toml','.yml','.pdf'}


def main():
    paths=[ROOT/p for p in ROOT_FILES]
    paths += [p for directory in DIRS for p in (ROOT/directory).rglob('*')if p.is_file()and'__pycache__'not in p.parts and p.suffix in EXTS and str(p.relative_to(ROOT))!='audit/artifact-manifest.json']
    paths=sorted(set(paths));files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
    manifest=ROOT/'audit/artifact-manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['files']!=files:raise ValueError('Frozen release differs; create another release instead')
    else:manifest.write_text(json.dumps({'files':files,'real_applicant_data_included':False,'model_weights_included':False,'review_secret_keys_saved':False},indent=2)+'\n')
    out=ROOT/'dist/nameguard-efficient-research.zip';out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED)as archive:
        for path in paths+[manifest]:archive.write(path,Path('nameguard-efficient')/path.relative_to(ROOT))
    print(json.dumps({'files':len(paths)+1,'archive_bytes':out.stat().st_size,'archive_sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
