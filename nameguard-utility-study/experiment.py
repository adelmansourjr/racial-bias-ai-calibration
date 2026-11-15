"""Frozen Qwen comparison under reviewed native-token name isolation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import secrets
import sys
from time import perf_counter

from fixtures import DEV_NAMES, TEST_NAMES, WITNESS_NAMES, STATUSES, scenarios
from methods import ROTATIONS, evidence, cycle_scores, paired_margin, classify, decision_report

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE / "nameguard-peer-review/src"))
from compiler import enumerate_candidates, select_contract, match_contract, unresolved_payload
from gate import SlotScorer, native_layout, ContractError
from evidence_contract import EvidenceContract
from nameguard.annotations import DualReviewAuthority, endorse, packet
from nameguard.mlx_backend import load_pinned


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def prompt_source(tokenizer, name, body):
    prefix = "Applicant name:\n"
    content = prefix + name + "\n" + body
    source = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                           tokenize=False, add_generation_prompt=True)
    if source.count(content) != 1:
        raise ValueError("Chat template must preserve exact input")
    start = source.index(content) + len(prefix)
    return source, start, start + len(name)


def plan_for(split, cases):
    paths = [ROOT / f for f in ("experiment.py", "fixtures.py", "methods.py", "audit/checkpoint-manifest.json")]
    paths += [WORKSPACE / f for f in ("nameguard-peer-review/src/compiler.py",
                                      "native-token-gate-hardening/gate.py",
                                      "native-token-gate-hardening/evidence_contract.py")]
    paths += sorted((WORKSPACE / "nameguard-validation/src/nameguard").glob("*.py"))
    return {
        "split": split, "frozen_before_this_run_inference": True,
        "sources": {str(path.relative_to(WORKSPACE)): digest(path) for path in paths},
        "case_ids": [c.key for c in cases],
        "primary": "Compare paired binary versus legacy protected and native on four known-status conditions per family.",
        "secondary": "Compare cyclic three-way and same-prompt single rotation on all cases. Report unknown false passes and referrals.",
        "rules": "Threshold zero binary. No parameters, thresholds, or examples fit on test. Compare all arms whether development improves or not.",
        "name_panel": list(TEST_NAMES if split == "test" else DEV_NAMES),
        "primary_name": (TEST_NAMES if split == "test" else DEV_NAMES)[0],
        "invariance_subset": "All six cases of each test family with zero-based index divisible by four. All four test names on these cases.",
        "unit": "Family, with six dependent scenarios. Name variants are conformance reruns, not extra utility observations.",
        "review": "Generator-defined identity and credential status. Original text passes through native tokenizer and embedding.",
        "deletion_control": "Remove designated name from original prompt and retokenize; use the same binary pair on one name per case.",
    }


def run(args):
    import mlx.core as mx
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    cases = tuple(scenarios(args.split))
    write(args.output / "plan.json", plan_for(args.split, cases))
    model, tokenizer, verify_files = load_pinned(args.checkpoint, ROOT / "audit/checkpoint-manifest.json")
    keys = {reviewer: secrets.token_bytes(32) for reviewer in ("generator_a", "generator_b")}
    authority = DualReviewAuthority(keys)
    letters = [tokenizer.encode(character, add_special_tokens=False) for character in "ABC"]
    if any(len(ids) != 1 for ids in letters):
        raise ValueError("Letter readout requires one token per label")
    labels = [ids[0] for ids in letters]
    stats = defaultdict(int)
    start_time = perf_counter()

    class Observer(SlotScorer):
        def _infer_layout(self, layout, mode, unused):
            self.trace = []
            self.logits = None
            def record(stage, value):
                if stage in ("native_ids", "raw_embedding"):
                    return
                if stage == "vocabulary":
                    mx.eval(value)
                    self.logits = tuple(float(value[0, 0, label].item()) for label in labels)
                if not self.trace_enabled:
                    return
                mx.eval(value)
                array = np.array(value.astype(mx.float32))
                if stage != "vocabulary":
                    private = np.ones(len(layout.ids), dtype=bool)
                    private[list(layout.public)] = False
                    if array.shape[1] != len(layout.ids) or np.any(array[:, private, :].view(np.uint32)):
                        raise AssertionError("Nonzero private state or lost original slot")
                    array = array[:, layout.public, :]
                    stats["zero_identity_checks"] += 1
                if not np.isfinite(array).all():
                    raise AssertionError("Nonfinite scoring state")
                self.trace.append((stage, list(array.shape),
                    hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()))
            result = super()._infer_layout(layout, mode, record)
            if self.logits is None or self.trace_enabled and len(self.trace) != 31:
                raise AssertionError("Incomplete activation trace")
            return result

    scorer = Observer(model, tokenizer, authority)

    def score_native(source):
        ids = tokenizer.encode(source, add_special_tokens=False)
        hidden = model.model(mx.array(ids)[None])
        logits = model.model.embed_tokens.as_linear(hidden[:, -1:].astype(mx.float32)).astype(mx.float32)
        mx.eval(logits)
        stats["native_forwards"] += 1
        return tuple(float(logits[0, 0, label].item()) for label in labels)

    def prepare(body):
        panels = []
        for name in WITNESS_NAMES:
            source, first, last = prompt_source(tokenizer, name, body)
            panels.append(enumerate_candidates(source, first, last, tokenizer))
        return select_contract(panels)

    def score_protected(body, contract, name, trace):
        source, first, last = prompt_source(tokenizer, name, body)
        try:
            witness = match_contract(enumerate_candidates(source, first, last, tokenizer), contract)
            payload = unresolved_payload(source, witness, [name])
            payload["resolved"] = True
            wire = packet(payload, [endorse(payload, reviewer=r, key=k) for r, k in keys.items()])
            layout = native_layout(authority.verify(wire), tokenizer)
            EvidenceContract(contract.evidence).verify(layout)
        except ContractError:
            stats["protected_rejections"] += 1
            return None
        scorer.trace_enabled = trace
        margin = scorer.score(wire).margin
        stats["protected_forwards"] += 1
        return margin, scorer.logits, scorer.trace

    rows = []
    names = TEST_NAMES if args.split == "test" else DEV_NAMES
    for i, case in enumerate(cases):
        trace = args.split == "test" and case.family % 4 == 0
        case_names = names if trace else names[:1]
        bodies = {"legacy": evidence(case.cv, case.task, "legacy")}
        binary = ("The applicant holds the required credential and it is current.",
                  "The applicant lacks a current required credential.")
        bodies.update({f"binary_{j}": evidence(case.cv, case.task, "binary", mapping)
                       for j, mapping in enumerate((binary, binary[::-1]))})
        bodies.update({f"cycle_{j}": evidence(case.cv, case.task, "cycle", mapping)
                       for j, mapping in enumerate(ROTATIONS)})
        contracts = {arm: prepare(body) for arm, body in bodies.items()}
        for j, name in enumerate(case_names):
            raw, _, _ = prompt_source(tokenizer, name, bodies["legacy"])
            native = score_native(raw)
            outputs = {arm: score_protected(body, contracts[arm], name, trace) for arm, body in bodies.items()}
            row = {"case": case.key, "family": case.family, "kind": case.kind, "status": case.status,
                   "name": name, "primary": j == 0, "native_prediction": native[0] >= native[1],
                   "admitted": all(output is not None for output in outputs.values())}
            if row["admitted"]:
                legacy, forward, reverse = (outputs[arm][0] for arm in ("legacy", "binary_0", "binary_1"))
                cycle = [outputs[f"cycle_{k}"][1] for k in range(3)]
                paired = paired_margin(forward, reverse)
                status = classify(cycle_scores(cycle))
                row.update(legacy_margin=legacy, binary_forward_margin=forward, binary_reverse_margin=reverse,
                           paired_margin=paired, cyclic_status=status, cyclic_scores=cycle_scores(cycle),
                           single_status=classify(cycle[0]), report=decision_report(case.task.credential, status).as_bytes().decode())
                if trace:
                    row["stages"] = {arm: output[2] for arm, output in outputs.items()}
            if j == 0:
                external = []
                for key in ("binary_0", "binary_1"):
                    source, first, last = prompt_source(tokenizer, name, bodies[key])
                    logits = score_native(source[:first] + source[last:])
                    external.append(logits[0] - logits[1])
                row["deleted_paired_margin"] = paired_margin(*external)
            rows.append(row)
        if (i + 1) % 12 == 0:
            print(f"{args.split} {i + 1}/{len(cases)} contexts", flush=True)
    write(args.output / "rows.json", rows)
    parameter_hash = scorer.verify_parameters()
    verify_files()
    scorer.close()
    write(args.output / "runtime.json", {**stats, "seconds": perf_counter() - start_time,
                                          "parameter_sha256": parameter_hash, "checkpoint_files_verified": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
