"""Check abstract noninterference lemmas and deliberately unsafe countermodels."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    executable=shutil.which('z3')
    if executable is None:raise RuntimeError('Install Z3 to check proofs; CPU tests are not a proof substitute')
    version=subprocess.run([executable,'-version'],check=True,capture_output=True,text=True).stdout.strip()
    outcomes=[]
    for name,expected in [('gather-noninterference','unsat'),('two-identity-blocks','unsat'),
                          ('finite-tests-not-universal','sat'),('position-leak-counterexample','sat')]:
        spec=ROOT/'audit'/f'{name}.smt2'
        run=subprocess.run([executable,str(spec)],check=True,capture_output=True,text=True,timeout=30)
        if run.stdout.splitlines()[0]!=expected:raise AssertionError(name+': '+run.stdout)
        (ROOT/'results'/f'{name}-solver.txt').write_text(run.stdout)
        outcomes.append({'spec':str(spec.relative_to(ROOT)),'spec_sha256':hashlib.sha256(spec.read_bytes()).hexdigest(),'result':expected})
    summary={'solver':version,'outcomes':outcomes,
         'scope':'Solver-checked integer-array abstractions of pointwise embedding/gather and unsafe variants, not end-to-end MLX/Python verification or semantic-role correctness.'}
    (ROOT/'results/proofs.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
