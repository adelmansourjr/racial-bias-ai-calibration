"""Rekey source records and reorder options to expose label/position shortcuts."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import secrets
import time

from analyze import pair_ci
from proposal_budget import checked_rank
from study_cases import cases, NAMES, WITNESSES
from witness import LETTERS, choices, prompt
from run import (ROOT, WORKSPACE, digest, write, prompt_source, SlotScorer, native_layout,
                 EvidenceContract, DualReviewAuthority, endorse, packet, load_pinned,
                 enumerate_candidates, select_contract, match_contract, unresolved_payload)

KEYS = ("K7", "Q2", "M9", "P4")


def rekey(case, replicate):
    generator = random.Random(83411 + 1009 * case.family + 83 * replicate)
    assigned = generator.sample(KEYS, k=len(KEYS))
    lookup = dict(zip(("R1", "R2", "R3", "R4"), assigned, strict=True))
    entries = [replace(row, record_id=lookup[row.record_id]) for row in case.entries]
    generator.shuffle(entries)
    return replace(case, entries=tuple(entries),
                   expected_ids=tuple(sorted(lookup[record_id] for record_id in case.expected_ids)))


def summarize(rows):
    expected = {(c.key, replicate): rekey(c, replicate) for c in cases("test") for replicate in (0, 1)}
    if len(rows) != len(expected) or len({(r["case"], r["replicate"]) for r in rows}) != len(expected):
        raise ValueError("Missing or duplicated opaque-ID trials")
    eligible = 0
    model = position = opaque_sort = unsafe = 0
    uniform_expected = 0.0
    differences = defaultdict(list)
    both_correct = defaultdict(list)
    constructed = defaultdict(list)
    for row in rows:
        case = expected[row["case"], row["replicate"]]
        scores = row["logits"]
        report = checked_rank(case, scores, 2)
        truth = (case.expected_status, case.expected_ids)
        success = report is not None and (report.status, report.record_ids) == truth
        unsafe += report is not None and report.status != "review" and not success
        constructed[row["replicate"]].append(success)
        if case.expected_status == "review":
            continue
        eligible += 1
        options = choices(case)
        first_two = list(options.values())[:2]
        by_id = sorted(options.values())[:2]
        position_hit = bool(set(first_two) & set(case.expected_ids))
        opaque_hit = bool(set(by_id) & set(case.expected_ids))
        position += position_hit
        opaque_sort += opaque_hit
        model += success
        if len(case.expected_ids) != 1:
            raise ValueError("Uniform baseline is for unique controlling IDs")
        uniform_expected += min(2, len(options)) / len(options)
        differences[case.family].append(int(success) - int(position_hit))
        both_correct[case.key].append(success)
    return {"condition_layout_pairs": len(rows), "verifiable_condition_layout_pairs": eligible,
            "protected_top_two_verified": model, "first_two_displayed_verified": position,
            "first_two_sorted_opaque_ids_verified": opaque_sort, "uniform_top_two_expected_verified": uniform_expected,
            "both_rekeyed_layouts_definite_correct": sum(all(x) for x in both_correct.values()),
            "correct_regardless_of_state_by_layout": {i: sum(u) for i, u in constructed.items()},
            "wrong_definitive": unsafe,
            "paired_coverage_change_vs_first_two_displayed": pair_ci([sum(u) / len(u) for u in differences.values()])}


def run(checkpoint, out):
    import mlx.core as mx
    out.mkdir(parents=True, exist_ok=True)
    write(out / "plan.json", {"before_opaque_inference": True,
                            "scope": "Adaptive design-check follow-up. Two rekeyed layouts per case; every case included. Compare top-two coverage with deterministic and uniform rank baselines. One primary name. No parameters refitted.",
                            "sources": {str(path.relative_to(WORKSPACE)): digest(path) for path in (
                                Path(__file__), ROOT / "run.py", ROOT / "analyze.py",
                                ROOT / "study_cases.py", ROOT / "witness.py", ROOT / "proposal_budget.py",
                                ROOT / "audit/checkpoint-manifest.json")}})
    model, tokenizer, verify_files = load_pinned(checkpoint, ROOT / "audit/checkpoint-manifest.json")
    ids = [tokenizer.encode(char, add_special_tokens=False) for char in LETTERS]
    if any(len(item) != 1 for item in ids):
        raise ValueError("Expected singleton readout tokens")
    select = mx.array([item[0] for item in ids])
    keys = {reviewer: secrets.token_bytes(32) for reviewer in ("author_a", "author_b")}
    authority = DualReviewAuthority(keys)

    class Scorer(SlotScorer):
        def _infer_layout(self, layout, mode, observe):
            self.logits = None

            def record(stage, output):
                if stage == "vocabulary":
                    self.logits = mx.take(output[0, 0], select).tolist()

            super()._infer_layout(layout, mode, record)
            if self.logits is None:
                raise ValueError("Missing witness logits")

    scorer = Scorer(model, tokenizer, authority)
    results = []
    started = time.perf_counter()
    for i, case in enumerate(cases("test")):
        for replicate in (0, 1):
            body = prompt(rekey(case, replicate))
            contract = select_contract([enumerate_candidates(*prompt_source(tokenizer, n, body), tokenizer)
                                        for n in WITNESSES])
            name = NAMES[0]
            source, a, b = prompt_source(tokenizer, name, body)
            candidate = match_contract(enumerate_candidates(source, a, b, tokenizer), contract)
            payload = unresolved_payload(source, candidate, [name])
            payload["resolved"] = True
            wire = packet(payload, [endorse(payload, reviewer=r, key=k) for r, k in keys.items()])
            EvidenceContract(contract.evidence).verify(native_layout(authority.verify(wire), tokenizer))
            scorer.score(wire)
            results.append({"case": case.key, "replicate": replicate, "logits": scorer.logits})
        if (i + 1) % 12 == 0:
            print(f"Rekeyed {i + 1}/96", flush=True)
    param = scorer.verify_parameters()
    verify_files()
    scorer.close()
    write(out / "rows.json", results)
    write(out / "summary.json", {"metrics": summarize(results), "seconds": time.perf_counter() - started,
                                  "parameter_sha256": param, "checkpoint_files_verified": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
