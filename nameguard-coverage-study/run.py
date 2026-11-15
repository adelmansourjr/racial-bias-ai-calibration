"""Collect the shared protected scores and exact activation comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import time

from dependencies import ROOT, WORKSPACE
from coverage_cases import cases, DEV_NAMES, TEST_NAMES, WITNESS_NAMES
from measure import file_hash, score_bits, STAGES, trace_case
from witness import LETTERS, prompt
from experiment import prompt_source
from compiler import enumerate_candidates, select_contract, match_contract, unresolved_payload
from gate import SlotScorer, native_layout, ContractError
from evidence_contract import EvidenceContract
from nameguard.annotations import DualReviewAuthority, endorse, packet
from nameguard.mlx_backend import load_pinned


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def run(checkpoint, split, out):
    import mlx.core as mx
    import numpy as np
    cohort = tuple(cases(split))
    names = DEV_NAMES if split == "dev" else TEST_NAMES
    manifest = ROOT / "audit/checkpoint-manifest.json"
    sources = [*ROOT.glob("*.py"), ROOT / "PROTOCOL.md", manifest]
    sources += [WORKSPACE / p for p in (
        "nameguard-witness-study/witness.py", "nameguard-witness-study/study_cases.py",
        "nameguard-utility-study/experiment.py", "nameguard-peer-review/src/compiler.py",
        "nameguard-peer-review/src/bootstrap.py", "native-token-gate-hardening/gate.py",
        "native-token-gate-hardening/evidence_contract.py")]
    sources += sorted((WORKSPACE / "nameguard-validation/src/nameguard").glob("*.py"))
    out.mkdir(parents=True, exist_ok=True)
    save(out / "plan.json", {"split": split, "cases": [c.key for c in cohort], "names": names,
         "before_inference": True, "sha256": {str(p.relative_to(WORKSPACE)): file_hash(p) for p in sources}})
    model, tokenizer, verify_files = load_pinned(checkpoint, manifest)
    keys = {reviewer: secrets.token_bytes(32) for reviewer in ("synthetic_a", "synthetic_b")}
    authority = DualReviewAuthority(keys)
    answer_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(ids) != 1 for ids in answer_ids):
        raise ValueError("The answer symbols require singleton tokens")
    indices = mx.array([ids[0] for ids in answer_ids])

    class TraceGate(SlotScorer):
        def _infer_layout(self, layout, mode, observe):
            self.logits = None
            self.arrays = {}
            self.zero_checks = 0
            private = sorted(set(range(len(layout.ids))) - set(layout.public))

            def inspect(stage, output):
                if stage == "vocabulary":
                    self.logits = mx.take(output[0, 0], indices).tolist()
                if not self.capture or stage not in STAGES:
                    return
                array = np.array(output.astype(mx.float32))
                if not np.isfinite(array).all():
                    raise ValueError("Nonfinite protected state")
                bits = array.view(np.uint32)
                if stage != "vocabulary":
                    if bits.shape[1] != len(layout.ids) or np.any(bits[:, private, :]):
                        raise ValueError("Physical slot geometry or zero identity checks failed")
                    self.zero_checks += 1
                    bits = bits[:, layout.public, :]
                self.arrays[stage] = np.ascontiguousarray(bits)

            score = super()._infer_layout(layout, mode, inspect)
            if self.logits is None or (self.capture and tuple(self.arrays) != STAGES):
                raise ValueError("Incomplete observations")
            return score

    scorer = TraceGate(model, tokenizer, authority)
    rows = []
    elapsed = time.perf_counter()
    for i, case in enumerate(cohort):
        body = prompt(case)
        contract = select_contract([enumerate_candidates(*prompt_source(tokenizer, name, body), tokenizer)
                                    for name in WITNESS_NAMES])
        captured = trace_case(case, split)
        reference = None
        body_hash = hashlib.sha256(body.encode()).hexdigest()
        for index, name in enumerate(names):
            source, start, end = prompt_source(tokenizer, name, body)
            row = {"case": case.key, "family": case.family, "kind": case.kind, "name": name, "primary": index == 0,
                   "logits": None, "zero_identity_checks": 0, "direct_comparisons": None,
                   "visible_prompt_sha256": body_hash}
            try:
                witness = match_contract(enumerate_candidates(source, start, end, tokenizer), contract)
                payload = unresolved_payload(source, witness, [name])
                payload["resolved"] = True
                wire = packet(payload, [endorse(payload, reviewer=r, key=k) for r, k in keys.items()])
                layout = native_layout(authority.verify(wire), tokenizer)
                EvidenceContract(contract.evidence).verify(layout)
            except ContractError:
                rows.append(row)
                continue
            if layout.ids != tuple(tokenizer.encode(source, add_special_tokens=False)):
                raise ValueError("Original tokenizer sequence changed")
            scorer.capture = captured
            scorer.score(wire)
            row.update(logits=scorer.logits, score_bits=score_bits(scorer.logits).hex(),
                       original_token_count=len(layout.ids), evidence_token_count=len(layout.public),
                       zero_identity_checks=scorer.zero_checks)
            if captured:
                if index == 0:
                    reference = scorer.arrays
                elif reference is not None:
                    row["direct_comparisons"] = [bool(np.array_equal(reference[s], scorer.arrays[s])) for s in STAGES]
            rows.append(row)
        if (i + 1) % 8 == 0:
            print(f"{split}: {i + 1}/{len(cohort)} cases", flush=True)
    save(out / "rows.json", rows)
    param = scorer.verify_parameters()
    verify_files()
    scorer.close()
    save(out / "runtime.json", {"seconds": time.perf_counter() - elapsed, "checkpoint_files_verified": True,
                               "parameter_sha256": param, "forwards": sum(r["logits"] is not None for r in rows),
                               "stages": STAGES})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.split, args.output)
