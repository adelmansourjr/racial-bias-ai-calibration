"""Seal the two subtractive-adapter studies, excluding raw data/hidden caches."""
from pathlib import Path
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[2]
STUDIES=['subtractive-lora-pilot-v2','subtractive-lora-score-aware']
DIRS=['scripts','src','tests','results','audit','adapters']
FILES=['README.md','METHOD.md','LICENSE','.gitignore']
ALLOWED={'.py','.md','.json','.jsonl','.log','.npz','.yml'}


def main():
    paths=[]
    for study in STUDIES:
        base=ROOT/study
        paths.extend(base/f for f in FILES)
        if (base/'ci-workflow.yml').exists():paths.append(base/'ci-workflow.yml')
        for directory in DIRS:
            paths.extend(p for p in (base/directory).rglob('*')if p.is_file()and'__pycache__'not in p.parts and p.suffix in ALLOWED and p.name!='release-manifest.json')
    paths=sorted(set(paths))
    if any('local-private'in p.parts for p in paths):raise ValueError('Private path in release')
    data={'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},
      'original_model_weights_included':False,'raw_CV_job_text_or_activation_caches_included':False,
      'small_learned_adapter_parameters_included':True,'successful_universal_erasure_claim':False}
    target=ROOT/STUDIES[1]/'audit/release-manifest.json'
    if target.exists():raise FileExistsError('Release sealed; use another version')
    target.write_text(json.dumps(data,indent=2)+'\n')
    out=ROOT/STUDIES[1]/'dist/subtractive-lora-experiments.zip';out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED)as archive:
        for p in paths+[target]:archive.write(p,p.relative_to(ROOT))
    print(json.dumps({'files':len(paths)+1,'archive_bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
