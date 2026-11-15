"""Run the frozen witness-prompt, activation isolation, and verification comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys
import time

from study_cases import DEV_NAMES, NAMES, WITNESSES, cases, rendered
from witness import LETTERS, prompt

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE / "nameguard-utility-study"))
from experiment import prompt_source
from compiler import enumerate_candidates, select_contract, match_contract, unresolved_payload
from gate import SlotScorer, native_layout, ContractError
from evidence_contract import EvidenceContract
from nameguard.annotations import DualReviewAuthority, endorse, packet
from nameguard.mlx_backend import load_pinned

TRACE_FAMILIES = (0, 3, 6, 9)
STAGES = ("boundary", *(f"block_{i + 1}" for i in range(28)), "final_hidden", "vocabulary")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def make_plan(split, cohort, names):
    sources = [ROOT / path for path in (
        "run.py", "study_cases.py", "witness.py", "PROTOCOL.md", "analyze.py", "audit/checkpoint-manifest.json")]
    sources += [WORKSPACE / path for path in (
        "nameguard-utility-study/experiment.py", "nameguard-peer-review/src/compiler.py",
        "nameguard-peer-review/src/bootstrap.py", "native-token-gate-hardening/gate.py",
        "native-token-gate-hardening/evidence_contract.py")]
    sources += sorted((WORKSPACE / "nameguard-validation/src/nameguard").glob("*.py"))
    return {"split": split, "sources": {str(p.relative_to(WORKSPACE)): digest(p) for p in sources},
            "before_this_run_inference": True, "case_ids": [c.key for c in cohort],
            "names": names, "trace_families": TRACE_FAMILIES if split == "test" else (),
            "primary_name": names[0]}


def run(checkpoint, split, out):
    import mlx.core as mx
    import numpy as np

    names = NAMES if split == "test" else DEV_NAMES
    cohort = tuple(cases(split))
    out.mkdir(parents=True, exist_ok=True)
    write(out / "plan.json", make_plan(split, cohort, names))
    model, tokenizer, verify_files = load_pinned(checkpoint, ROOT / "audit/checkpoint-manifest.json")
    keys = {label: secrets.token_bytes(32) for label in ("synthetic_role_a", "synthetic_role_b")}
    authority = DualReviewAuthority(keys)
    encoded = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(tokens) != 1 for tokens in encoded):
        raise ValueError("Readout requires singleton answer tokens")
    indices = mx.array([tokens[0] for tokens in encoded])
    forwards = 0

    class TracedGate(SlotScorer):
        def _infer_layout(self, layout, mode, observe):
            self.logits = None
            self.arrays = {}
            self.zero_checks = 0
            private = sorted(set(range(len(layout.ids))) - set(layout.public))

            def record(stage, value):
                if stage == "vocabulary":
                    selected = mx.take(value[0, 0], indices)
                    mx.eval(selected)
                    self.logits = np.array(selected).tolist()
                if self.capture and stage in STAGES:
                    if stage != "vocabulary" and value.shape[1] != len(layout.ids):
                        raise ValueError("Native physical slot shape changed")
                    array = np.array(value.astype(mx.float32))
                    if not np.isfinite(array).all():
                        raise ValueError("Nonfinite stage")
                    bits = array.view(np.uint32)
                    if stage != "vocabulary":
                        if np.any(bits[:, private, :]):
                            raise ValueError("Nonzero designated identity row")
                        self.zero_checks += 1
                        bits = bits[:, layout.public, :]
                    self.arrays[stage] = np.ascontiguousarray(bits)
            result = super()._infer_layout(layout, mode, record)
            if self.logits is None or self.capture and tuple(self.arrays) != STAGES:
                raise ValueError("Incomplete activation observation")
            return result

    gate = TracedGate(model, tokenizer, authority)

    def native(source):
        nonlocal forwards
        ids = tokenizer.encode(source, add_special_tokens=False)
        hidden = model.model(mx.array(ids)[None])
        logits = model.model.embed_tokens.as_linear(hidden[:, -1:].astype(mx.float32)).astype(mx.float32)
        selected = mx.take(logits[0, 0], indices)
        mx.eval(selected)
        forwards += 1
        return len(ids), np.array(selected).tolist()

    def contract(body):
        panels = [enumerate_candidates(*prompt_source(tokenizer, name, body), tokenizer) for name in WITNESSES]
        return select_contract(panels)

    def protected(body, expected, name, capture):
        nonlocal forwards
        source, start, end = prompt_source(tokenizer, name, body)
        try:
            selected = match_contract(enumerate_candidates(source, start, end, tokenizer), expected)
            payload = unresolved_payload(source, selected, [name])
            payload["resolved"] = True
            wire = packet(payload, [endorse(payload, reviewer=reviewer, key=key) for reviewer, key in keys.items()])
            layout = native_layout(authority.verify(wire), tokenizer)
            EvidenceContract(expected.evidence).verify(layout)
        except ContractError:
            return None, {}, 0, None
        if layout.ids != tuple(tokenizer.encode(source, add_special_tokens=False)):
            raise ValueError("Protected path changed the source tokens")
        gate.capture = capture
        gate.score(wire)
        forwards += 1
        return gate.logits, gate.arrays, gate.zero_checks, len(layout.public)

    rows = []
    start_time = time.perf_counter()
    for i, case in enumerate(cohort):
        full, brief = prompt(case), prompt(case, brief=True)
        full_contract, brief_contract = contract(full), contract(brief)
        trace = split == "test" and case.family in TRACE_FAMILIES
        reference = {}
        for j, name in enumerate(names):
            source, _, _ = prompt_source(tokenizer, name, full)
            full_length, named = native(source)
            values, arrays, zero_checks, evidence_length = protected(full, full_contract, name, trace)
            comparisons = None
            if trace and j == 0:
                reference = arrays
            if trace and j > 0 and tuple(reference) == tuple(arrays) == STAGES:
                comparisons = [bool(np.array_equal(reference[s], arrays[s])) for s in STAGES]
            row = {"case": case.key, "family": case.family, "kind": case.kind, "name": name,
                   "primary": j == 0, "native_logits": named, "protected_logits": values,
                   "native_tokens": full_length, "evidence_tokens": evidence_length,
                   "stage_equality": comparisons, "zero_identity_checks": zero_checks,
                   "visible_records_sha256": hashlib.sha256(rendered(case).encode()).hexdigest()}
            if j == 0:
                row["brief_logits"] = protected(brief, brief_contract, name, False)[0]
            rows.append(row)
        if (i + 1) % 8 == 0:
            print(f"{split} {i + 1}/{len(cohort)} conditions", flush=True)
    write(out / "rows.json", rows)
    parameter_hash = gate.verify_parameters()
    verify_files()
    gate.close()
    write(out / "runtime.json", {"forwards": forwards, "seconds": time.perf_counter() - start_time,
                                 "parameter_sha256": parameter_hash, "checkpoint_files_verified": True,
                                 "stage_names": STAGES})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.split, args.output)
