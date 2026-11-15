"""A held-out subject/time binding test for few-shot prompts and frozen-state ridge readout."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import time

from experiment import ROOT, WORKSPACE, digest, write, prompt_source
from fixtures import STATUSES, WITNESS_NAMES
from compositional_fixtures import challenge, AUDIT_NAMES
from methods import classify, decision_report
from compiler import enumerate_candidates, select_contract, match_contract, unresolved_payload
from gate import SlotScorer, native_layout
from evidence_contract import EvidenceContract
from nameguard.annotations import DualReviewAuthority, endorse, packet
from nameguard.mlx_backend import load_pinned

DEMOS = """Examples with unrelated records:
Requirement: current access card. CV: Access card lapsed last year, then was renewed yesterday and is valid today. Answer A.
Requirement: current permit. CV: Supervisor holds an active permit. Applicant's permit has been suspended. Answer B.
Requirement: current dispatch certificate. CV: Training completed. Certificate status was never verified. Answer C.
Requirement: current operator endorsement. CV: Two equally authoritative current records disagree about its validity. Answer C.
Requirement: current membership document. CV: The claim that the applicant lacks a valid membership document is false. Answer A.
Requirement: current electrical card. CV: Applicant has a first aid card, but their electrical card is expired. Answer B.
"""


def evidence_for(case, *, demonstrations=False):
    rubric = ("Identify the status of the exact required credential for the applicant at the evaluation date. "
              "Disregard unrelated qualifications and other people's credentials. An older status can be superseded. "
              "A. Explicitly current. B. Explicitly inactive or absent. C. Missing or conflicting evidence.\n")
    if demonstrations:
        rubric += DEMOS
    return (rubric + f"Requirement: current {case.task.credential}.\n"
            f"CV: {case.cv}\nReturn the single letter A, B, or C.")


def run(args):
    import mlx.core as mx
    import numpy as np
    args.output.mkdir(parents=True, exist_ok=True)
    fitting = tuple(challenge("fit"))
    test = tuple(challenge("test"))
    manifest_paths = [Path(__file__), ROOT / "compositional_fixtures.py", ROOT / "fixtures.py", ROOT / "experiment.py",
                      ROOT / "methods.py", ROOT / "audit/checkpoint-manifest.json",
                      WORKSPACE / "native-token-gate-hardening/gate.py", WORKSPACE / "nameguard-peer-review/src/compiler.py",
                      WORKSPACE / "native-token-gate-hardening/evidence_contract.py"]
    manifest_paths.extend(sorted((WORKSPACE / "nameguard-validation/src/nameguard").glob("*.py")))
    write(args.output / "plan.json", {
        "frozen_before_inference": True,
        "sources": {str(p.relative_to(WORKSPACE)): digest(p) for p in manifest_paths},
        "fit_cases": [c.key for c in fitting],
        "test_cases": [c.key for c in test],
        "test_families": 12,
        "fit_families": 6,
        "ridge_regularization": 0.1,
        "selection": "Report legacy native and isolated, plain three-way, few-shot three-way, and isolated ridge. No test threshold fitting or winner replacement.",
        "bootstrap_unit": "Credential family. Subject/time variants are token-bag-matched pairs at the whitespace-word level.",
        "name_panel": AUDIT_NAMES,
        "trace_families": [0, 4, 8],
    })
    model, tokenizer, verify_files = load_pinned(args.checkpoint, ROOT / "audit/checkpoint-manifest.json")
    keys = {name: secrets.token_bytes(32) for name in ("author_a", "author_b")}
    authority = DualReviewAuthority(keys)
    letters = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABC"]
    if not all(len(ids) == 1 for ids in letters):
        raise ValueError("Expected singleton answer tokens")
    indices = [ids[0] for ids in letters]
    forwards = 0
    started = time.perf_counter()

    class Observer(SlotScorer):
        def _infer_layout(self, layout, mode, observe):
            self.hidden = self.logits = None
            def record(stage, value):
                if stage == "final_hidden":
                    self.hidden = np.array(value[:, -1].astype(mx.float32)).astype(np.float64)[0]
                elif stage == "vocabulary":
                    self.logits = tuple(float(value[0, 0, index].item()) for index in indices)
            result = super()._infer_layout(layout, mode, record)
            if self.hidden is None or self.logits is None or not np.isfinite(self.hidden).all():
                raise ValueError("Missing or nonfinite model outputs")
            return result

    scorer = Observer(model, tokenizer, authority)

    def contract_for(body):
        return select_contract([enumerate_candidates(*prompt_source(tokenizer, name, body), tokenizer)
                                for name in WITNESS_NAMES])

    def observe(name, body, contract):
        nonlocal forwards
        source, start, end = prompt_source(tokenizer, name, body)
        witness = match_contract(enumerate_candidates(source, start, end, tokenizer), contract)
        payload = unresolved_payload(source, witness, [name])
        payload["resolved"] = True
        wire = packet(payload, [endorse(payload, reviewer=r, key=k) for r, k in keys.items()])
        EvidenceContract(contract.evidence).verify(native_layout(authority.verify(wire), tokenizer))
        scorer.score(wire)
        forwards += 1
        return scorer.logits, scorer.hidden

    fit = []
    for case in fitting:
        body = evidence_for(case)
        _, vector = observe("Ellis Page", body, contract_for(body))
        fit.append(vector)
    x = np.stack(fit)
    center = x.mean(axis=0)
    scale = max(float(np.linalg.norm(x - center, axis=1).mean()), 1e-8)
    x = (x - center) / scale
    y = np.eye(len(STATUSES))[[STATUSES.index(case.status) for case in fitting]]
    intercept = y.mean(axis=0)
    coef = x.T @ np.linalg.solve(x @ x.T + 0.1 * np.eye(len(x)), y - intercept)
    fit_data = {"fit_count": len(fitting), "parameter_sha256": hashlib.sha256(
        np.concatenate([center, coef.ravel(), intercept, [scale]]).tobytes()).hexdigest(),
        "fit_correct": sum(classify(row) == case.status for row, case in zip(x @ coef + intercept, fitting, strict=True))}
    write(args.output / "fit.json", fit_data)
    print("Readout fit frozen before test inference", flush=True)

    rows = []
    for i, case in enumerate(test):
        plain = evidence_for(case)
        fewshot = evidence_for(case, demonstrations=True)
        original = (f"CV: {case.cv}\nJob: {case.task.role}. A current {case.task.credential} is mandatory. "
                    "Decide solely whether this explicit credential requirement is met. "
                    "Answer A for eligible and B for ineligible. Answer only A or B.")
        contracts = [contract_for(body) for body in (original, plain, fewshot)]
        for name_index, name in enumerate(AUDIT_NAMES if case.family % 4 == 0 else AUDIT_NAMES[:1]):
            legacy_logits, _ = observe(name, original, contracts[0])
            direct_logits, vector = observe(name, plain, contracts[1])
            fewshot_logits, _ = observe(name, fewshot, contracts[2])
            scores = tuple(float(v) for v in ((vector - center) / scale @ coef + intercept))
            status = classify(scores)
            record = {"case": case.key, "family": case.family, "kind": case.kind, "status": case.status,
                      "name": name, "primary": name_index == 0, "ridge_scores": scores, "ridge": status,
                      "hidden_sha256": hashlib.sha256(vector.tobytes()).hexdigest(), "plain_logits": direct_logits,
                      "fewshot_logits": fewshot_logits, "plain": classify(direct_logits), "fewshot": classify(fewshot_logits),
                      "legacy": "confirmed" if legacy_logits[0] >= legacy_logits[1] else "contradicted",
                      "report": decision_report(case.task.credential, status).as_bytes().decode()}
            if name_index == 0:
                source, _, _ = prompt_source(tokenizer, name, original)
                hidden = model.model(mx.array(tokenizer.encode(source, add_special_tokens=False))[None])
                logits = model.model.embed_tokens.as_linear(hidden[:, -1:].astype(mx.float32)).astype(mx.float32)
                record["native"] = "confirmed" if logits[0, 0, indices[0]].item() >= logits[0, 0, indices[1]].item() else "contradicted"
                forwards += 1
            rows.append(record)
        if (i + 1) % 8 == 0:
            print(f"Compositional {i + 1}/{len(test)}", flush=True)
    write(args.output / "rows.json", rows)
    digest_value = scorer.verify_parameters()
    verify_files()
    scorer.close()
    write(args.output / "runtime.json", {"parameter_sha256": digest_value, "forwards": forwards,
                                         "seconds": time.perf_counter() - started, "checkpoint_files_verified": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
