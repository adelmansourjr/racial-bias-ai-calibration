"""Check conditional factorization and the independent-piece counterexample."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1]


def main():
    executable=shutil.which('z3')
    if executable is None:raise RuntimeError('A Z3 executable is required; tests are not a proof substitute')
    rows=[]
    for name,expected in [('canonical-factorization','unsat'),('independent-piece-tokenizer-counterexample','sat')]:
        path=ROOT/'audit'/f'{name}.smt2'
        r=subprocess.run([executable,str(path)],check=True,capture_output=True,text=True,timeout=30)
        if r.stdout.splitlines()[0]!=expected:raise AssertionError('Unexpected symbolic result')
        (ROOT/'results'/f'{name}-solver.txt').write_text(r.stdout)
        rows.append({'spec':str(path.relative_to(ROOT)),'spec_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'result':expected})
    summary={'checks':rows,'scope':'Conditional uninterpreted-function congruence and tokenizer non-homomorphism counterexample; not verified annotation correctness, GPU semantics, token minimality or universal raw-CV fairness.'}
    (ROOT/'results/proofs.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
