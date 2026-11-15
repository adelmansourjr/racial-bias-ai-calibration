"""Audit same-ID generation equivalence and measure evidence dependence."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
from pathlib import Path
import sys
from time import perf_counter

from analyze import PARAMETERS
from prepare import ROOT, WORKSPACE, digest, read, save
from reserve import extract_prompt, witness_matches

PLAN = ROOT / "audit/diagnostic-plan.json"
RESERVE = ROOT / "results/reserve"
QUEUE = ROOT / "data/reserve"


def freeze(path, reserve_plan, reserve_results):
    save(path, {
        "condition": "Passage blanked; compare exact generated answer before checking original offsets",
        "control": "Same retained IDs through native cached Qwen; original named IDs as unrestricted control",
        "reserve_plan_sha256": digest(reserve_plan),
        "reserve_observed_sha256": digest(reserve_results / "observed.json"),
        "reserve_panel_sha256": digest(reserve_results / "name-panel.json"),
        "program_sha256": digest(Path(__file__)),
        "selected_after_reserve_results": True,
    }, freeze=True)


def assert_binding(plan, reserve_plan, results_path):
    if (plan["reserve_plan_sha256"] != digest(reserve_plan)
            or plan["reserve_panel_sha256"] != digest(results_path / "name-panel.json")
            or plan["reserve_observed_sha256"] != digest(results_path / "observed.json")
            or plan["program_sha256"] != digest(Path(__file__))):
        raise ValueError("Diagnostic freeze differs from input files")


def run(checkpoint, diagnostic_plan, reserve_plan, reserve_results, queue, out):
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache

    sys.path.append(str(WORKSPACE / "nameguard-utility-study"))
    from experiment import prompt_source
    from compiler import enumerate_candidates, select_contract
    from nameguard.mlx_backend import load_pinned, parameter_digest

    plan = read(reserve_plan)
    assert_binding(read(diagnostic_plan), reserve_plan, reserve_results)
    if digest(queue / "rows.json") != plan["inputs_sha256"]:
        raise ValueError("Input source mismatch")
    cohort = read(queue / "rows.json")
    model, tokenizer, verify = load_pinned(checkpoint, ROOT / "audit/checkpoint-manifest.json")
    eos = set(tokenizer.eos_token_ids)
    out.mkdir(parents=True)
    counts = Counter()
    began = perf_counter()

    def ids(name, body):
        text, _, _ = prompt_source(tokenizer, name, body)
        return tokenizer.encode(text, add_special_tokens=False)

    def kept(body):
        return select_contract([enumerate_candidates(*prompt_source(tokenizer, n, body), tokenizer)
                                for n in plan["witnesses"]]).evidence

    def generate(tokens, capture=False):
        cache = make_prompt_cache(model)
        initial = mx.array(tokens)[None]
        prefix = model.model.embed_tokens(initial)
        # Plain native call starts directly from retained or full original IDs.
        hidden = model.model(initial, cache=cache)[:, -1:]
        generated, trace = [], []
        done = False
        for _ in range(plan["max_tokens"]):
            logits = model.model.embed_tokens.as_linear(hidden.astype(mx.float32)).astype(mx.float32)
            if capture:
                data = np.array(logits)
                if not np.isfinite(data).all():
                    raise ValueError("Nonfinite reference logits")
                trace.append(hashlib.sha256(data.tobytes()).hexdigest())
            token = int(mx.argmax(logits[0, 0]).item())
            if token in eos:
                done = True
                break
            generated.append(token)
            text = tokenizer.decode(generated)
            if "\n" in text.strip():
                break
            if text.strip().upper() == "UNKNOWN":
                done = True
                break
            hidden = model.model(mx.array([[token]]), cache=cache)
        counts["forwards"] += 1
        record = {"generated_sha256": hashlib.sha256(str(generated).encode()).hexdigest(),
                  "stopped": done, "generated_count": len(generated)}
        if capture:
            record.update(step_sha256=trace, public_prefix_sha256=hashlib.sha256(
                np.array(prefix.astype(mx.float32)).tobytes()).hexdigest())
        return record, tokenizer.decode(generated)

    matches, panel = {}, {}
    cases = set(plan["conformance"])
    for index, (key, item) in enumerate(cohort.items(), 1):
        blank = extract_prompt(dict(item, passage=""))
        row, text = generate(kept(blank))
        matches[key] = dict(row, matched=witness_matches(text, item["answer"], row["stopped"]))
        if key in cases:
            body = extract_prompt(item)
            reference, _ = generate(kept(body), capture=True)
            native = []
            for name in plan["names"]:
                record, text = generate(ids(name, body))
                native.append(dict(record, matched=witness_matches(text, item["answer"], record["stopped"])))
            panel[key] = dict(reference=reference, native=native)
        if index % 40 == 0:
            print(f"diagnostic {index}/{len(cohort)}", flush=True)
    verification = parameter_digest(model)
    if verification != PARAMETERS:
        raise ValueError("Frozen parameter digest mismatch")
    verify()
    save(out / "blank.json", matches, freeze=True)
    save(out / "reference.json", panel, freeze=True)
    save(out / "runtime.json", dict(**counts, runtime=version("mlx"), parameters=verification,
                                   plan_sha256=digest(diagnostic_plan), seconds=perf_counter() - began),
         freeze=True)


def analyze(diagnostic_plan, reserve_plan, reserve_results, queue, out, summary):
    plan = read(reserve_plan)
    assert_binding(read(diagnostic_plan), reserve_plan, reserve_results)
    runtime = read(out / "runtime.json")
    if runtime["parameters"] != PARAMETERS or runtime["plan_sha256"] != digest(diagnostic_plan):
        raise ValueError("Runtime binding mismatch")
    blank, reference = read(out / "blank.json"), read(out / "reference.json")
    observed, panel = read(reserve_results / "observed.json"), read(reserve_results / "name-panel.json")
    labels_path = queue / "labels.json"
    if digest(labels_path) != plan["labels_sha256"]:
        raise ValueError("Reference labels differ from freeze")
    labels = read(labels_path)
    if sorted(blank) != plan["cases"] or sorted(reference) != sorted(plan["conformance"]):
        raise ValueError("Diagnostic IDs do not match")
    equiv = checks = different_native = native_match_flips = 0
    for key, control in reference.items():
        a = control["reference"]
        variants = panel[key]
        equal = all(all(row[field] == value for field, value in a.items()) for row in variants)
        equiv += equal
        checks += len(a["step_sha256"]) * len(variants) if equal else 0
        different_native += len({row["generated_sha256"] for row in control["native"]}) > 1
        native_match_flips += len({row["matched"] for row in control["native"]}) > 1
    counts = {state: Counter() for state in ("supported", "unsupported")}
    for key, originally in observed.items():
        both = (originally["accepted"], blank[key]["matched"])
        count = counts["supported" if labels[key] else "unsupported"]
        count[("source_" if both[0] else "no_source_")
              + ("blank_match" if both[1] else "blank_no_match")] += 1
    data = dict(cases=len(blank), same_id_panels=equiv, panel_cases=len(panel), vocabulary_matches=checks,
                native_generation_varies=different_native, native_confirmation_varies=native_match_flips,
                blank_passage_comparison=counts)
    save(summary, data)
    print(data)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "run", "analyze"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--reserve-plan", type=Path, default=ROOT / "audit/reserve-plan.json")
    p.add_argument("--reserve-results", type=Path, default=RESERVE)
    p.add_argument("--queue", type=Path, default=QUEUE)
    p.add_argument("--out", type=Path, default=ROOT / "results/diagnostic")
    p.add_argument("--summary", type=Path, default=ROOT / "results/diagnostic-metrics.json")
    a = p.parse_args()
    if a.stage == "freeze":
        freeze(a.plan, a.reserve_plan, a.reserve_results)
    elif a.stage == "run":
        if a.checkpoint is None:
            p.error("--checkpoint is required")
        run(a.checkpoint, a.plan, a.reserve_plan, a.reserve_results, a.queue, a.out)
    else:
        analyze(a.plan, a.reserve_plan, a.reserve_results, a.queue, a.out, a.summary)
