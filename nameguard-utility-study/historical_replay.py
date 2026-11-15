"""Post-hoc execution of new methods on the published 179/192 versus 168/192 cohort."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import secrets

from experiment import ROOT, WORKSPACE, digest, write, prompt_source
from fixtures import Scenario, Task, WITNESS_NAMES
from methods import classify, decision_report
from compositional import evidence_for
from compiler import enumerate_candidates, select_contract, match_contract, unresolved_payload
from gate import SlotScorer, native_layout
from evidence_contract import EvidenceContract
from nameguard.annotations import DualReviewAuthority, endorse, packet
from nameguard.mlx_backend import load_pinned


def run(args):
    import mlx.core as mx
    prior = WORKSPACE / "nameguard-peer-review"
    fixtures_path = prior / "src/fixtures.py"
    rows_path = prior / "results/gpu/synthetic-rows.json"
    spec = importlib.util.spec_from_file_location("historical_fixtures", fixtures_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    cases = {row["id"]: row for row in old.contexts()}
    records = json.loads(rows_path.read_text())
    if len(records) != 192 or len({(r["context"], r["name"]) for r in records}) != 192:
        raise ValueError("Previous planned rows missing or duplicated")
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "plan.json", {"adaptation": "Post-hoc replay. Historical errors already examined.",
          "frozen_before_replay": True, "sources": {str(p.relative_to(WORKSPACE)): digest(p) for p in
          (Path(__file__), fixtures_path, rows_path, ROOT / "compositional.py", ROOT / "fixtures.py",
           ROOT / "experiment.py", ROOT / "methods.py", WORKSPACE / "native-token-gate-hardening/gate.py",
           WORKSPACE / "nameguard-peer-review/src/compiler.py")},
          "arms": ["published native", "published internal gate", "new few-shot three-way on the same CV body"]})
    model, tokenizer, verify_files = load_pinned(args.checkpoint, ROOT / "audit/checkpoint-manifest.json")
    keys = {name: secrets.token_bytes(32) for name in ("review_a", "review_b")}
    authority = DualReviewAuthority(keys)
    IDs = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABC"]
    if any(len(ids) != 1 for ids in IDs):
        raise ValueError("Expected singleton answer tokens")
    indices = [ids[0] for ids in IDs]

    class Observer(SlotScorer):
        def _infer_layout(self, layout, mode, unused):
            def record(stage, value):
                if stage == "vocabulary":
                    self.readout = tuple(float(value[0, 0, label].item()) for label in indices)
            self.readout = None
            output = super()._infer_layout(layout, mode, record)
            if self.readout is None:
                raise ValueError("Missing projected readout")
            return output

    scorer = Observer(model, tokenizer, authority)
    grouped = defaultdict(list)
    for row in records:
        grouped[row["context"]].append(row)
    output_rows = []
    for i, (context, block) in enumerate(grouped.items()):
        item = cases[context]
        role, credential, work = old.TASKS[item["task_cluster"]]
        # Reuse the exact historic CV body. Only the instruction/readout protocol changes.
        body = item["evidence"].split("\nJob:", 1)[0]
        if not body.startswith("CV: "):
            raise ValueError("Unexpected historical prompt schema")
        case = Scenario(item["task_cluster"], item["style"], "confirmed" if item["label"] else "contradicted",
                        Task(role, credential, "", work), body[4:])
        prompt = evidence_for(case, demonstrations=True)
        contract = select_contract([enumerate_candidates(*prompt_source(tokenizer, name, prompt), tokenizer)
                                    for name in WITNESS_NAMES])
        for row in block:
            source, a, b = prompt_source(tokenizer, row["name"], prompt)
            candidate = match_contract(enumerate_candidates(source, a, b, tokenizer), contract)
            payload = unresolved_payload(source, candidate, [row["name"]])
            payload["resolved"] = True
            wire = packet(payload, [endorse(payload, reviewer=r, key=k) for r, k in keys.items()])
            EvidenceContract(contract.evidence).verify(native_layout(authority.verify(wire), tokenizer))
            scorer.score(wire)
            prediction = classify(scorer.readout)
            output_rows.append({"context": context, "name": row["name"], "status": case.status,
                                "wording": case.kind, "fewshot": prediction, "logits": scorer.readout,
                                "report": decision_report(credential, prediction).as_bytes().decode(),
                                "native_correct": (row["native_margin"] >= 0) == item["label"],
                                "old_gate_correct": (row["compiled_margin"] >= 0) == item["label"]})
        if (i + 1) % 8 == 0:
            print(f"Replay {i + 1}/{len(grouped)} contexts", flush=True)
    result_groups = defaultdict(list)
    for row in output_rows:
        result_groups[row["context"]].append(row)
    write(args.output / "rows.json", output_rows)
    sha = scorer.verify_parameters()
    verify_files()
    scorer.close()
    write(args.output / "summary.json", {
        "rows": len(output_rows), "contexts": len(result_groups),
        "native_correct": sum(r["native_correct"] for r in output_rows),
        "old_gate_correct": sum(r["old_gate_correct"] for r in output_rows),
        "new_fewshot_correct": sum(r["fewshot"] == r["status"] for r in output_rows),
        "positive_double_negation_new_correct": sum(r["status"] == "confirmed" and r["wording"] == "negation"
                                                    and r["fewshot"] == r["status"] for r in output_rows),
        "identical_logits_and_reports_contexts": sum(len({(r["logits"], r["report"]) for r in block}) == 1
                                                     for block in result_groups.values()),
        "review_count": sum(r["fewshot"] == "review" for r in output_rows),
        "parameter_sha256": sha, "checkpoint_files_verified": True,
        "scope": "Post-hoc historical replay. Unchanged CV body and names; protocol changes also alter prompt layout. Fresh comparative evidence is in the held-out studies."})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
