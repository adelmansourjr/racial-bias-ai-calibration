"""Seal only explicitly whitelisted public artifacts, never raw CVs or keys."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
DIRS=['src','scripts','tests','audit','results','docs','literature','paper','output/pdf']
FILES=['README.md','LICENSE','CITATION.cff','pyproject.toml','.gitignore','requirements-mlx.txt','requirements-paper.txt','ci-workflow.yml']
EXTS={'.py','.md','.json','.jsonl','.log','.txt','.cff','.toml','.yml','.bib','.pdf'}


def main():
    paths=[ROOT/x for x in FILES]
    paths.extend(p for d in DIRS for p in (ROOT/d).rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix in EXTS and p.name!='release-manifest.json')
    paths=sorted(set(paths))
    if any('local-private'in p.parts or p.suffix in {'.key','.pem'}for p in paths):raise ValueError('Private path in release')
    manifest={'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},'raw_CV_or_job_text_included':False,'review_keys_included':False,'model_weights_included':False}
    target=ROOT/'audit/release-manifest.json'
    if target.exists():raise FileExistsError('Release is already sealed')
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    out=ROOT/'dist/nameguard-validation-research.zip';out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED)as z:
        for p in paths+[target]:z.write(p,Path('nameguard-validation')/p.relative_to(ROOT))
    print(json.dumps({'files':len(paths)+1,'bytes':out.stat().st_size,'archive_sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
