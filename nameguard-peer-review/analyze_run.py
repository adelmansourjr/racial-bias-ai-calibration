"""Analyze a fresh GPU output directory without touching the archived study."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import analyze
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--boundary',type=Path,default=ROOT/'results/boundary.json')
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError('New analysis output directory required')
    with tempfile.TemporaryDirectory(prefix='nameguard-analysis-') as work:
        work=Path(work);(work/'results').mkdir()
        shutil.copytree(args.run,work/'results/gpu')
        shutil.copyfile(args.boundary,work/'results/boundary.json')
        original=analyze.ROOT
        try:analyze.ROOT=work;analyze.main()
        finally:analyze.ROOT=original
        args.output_dir.mkdir(parents=True)
        for name in ('summary.json','context-analysis.json'):
            shutil.copyfile(work/'results'/name,args.output_dir/name)
    (args.output_dir/'analysis-provenance.json').write_text(json.dumps({
        'wrapper_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'frozen_analysis_source_sha256':hashlib.sha256((ROOT/'analyze.py').read_bytes()).hexdigest(),
        'run_plan_sha256':hashlib.sha256((args.run/'plan.json').read_bytes()).hexdigest(),
        'boundary_record_sha256':hashlib.sha256(args.boundary.read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
