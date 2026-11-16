"""Check manuscript results against public records and inspect a clean LaTeX build."""
import argparse
from fractions import Fraction
from itertools import product
import json
from math import comb
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def check(label, actual, expected):
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def lower_tail(total, errors, sample, observed):
    """Exact audit CDF by counting the supported and unsupported draws."""
    supported = total - errors
    numerator = sum(comb(errors, x) * comb(supported, sample - x)
                    for x in range(sample + 1)
                    if x <= min(observed, errors) and sample - x <= supported)
    return Fraction(numerator, comb(total, sample))


def verify_math():
    n, m, x = 55, 44, 15
    alpha = Fraction(1, 20)
    feasible = [k for k in range(x, n - m + x + 1) if lower_tail(n, k, m, x) > alpha]
    check("Exact audited error bound", max(feasible), 22)
    if not lower_tail(n, 23, m, x) <= alpha < lower_tail(n, 22, m, x):
        raise ValueError("Error bound fails the adjacent-count tail check")

    # Sum the exact probability of under-coverage over each feasible observation.
    for n in range(1, 13):
        for m, k in product(range(n + 1), repeat=2):
            missed = Fraction(0)
            previous = Fraction(0)
            for x in range(m + 1):
                cdf = lower_tail(n, k, m, x)
                if cdf <= alpha:
                    missed += cdf - previous
                previous = cdf
            if missed > alpha:
                raise ValueError(f"Audit under-coverage at {n=}, {m=}, {k=}")

    for values in product(range(-2, 3), repeat=3):
        mean = Fraction(sum(values), len(values))
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        midpoint = Fraction(min(values) + max(values), 2)
        radius = Fraction(max(values) - min(values), 2)
        check("Minimax radius at midpoint", max(abs(v - midpoint) for v in values), radius)
        for constant in range(-3, 4):
            mse = Fraction(sum((v - constant) ** 2 for v in values), len(values))
            check("Fidelity decomposition", mse, variance + (mean - constant) ** 2)
            if max(abs(v - constant) for v in values) < radius:
                raise ValueError("Candidate contradicts the minimax lower bound")
    print("Exact audit inversion, small-population coverage and fidelity identities checked.")


def verify_records():
    cv = [read(f"nameguard-real-evidence/results/{run}-summary.json")
          for run in ("initial", "fresh", "confirm")]
    check("Résumé excerpt count", sum(r["independent_cv_excerpts"] for r in cv), 119)
    for key, expected in {"admitted_requests": 476, "exact_scores": 119, "exact_reports": 119,
                          "native_decision_variation": 18, "exact_direct_arrays": 1488}.items():
        check(key, sum(r["audit"][key] for r in cv), expected)
    for row, expected in zip(cv, [(48, 17, 7, 0), (35, 12, 5, 2), (36, 7, 4, 2)], strict=True):
        rule = row["decisions"]["protected_verified"]
        check("Résumé evidence table",
              (row["independent_cv_excerpts"], row["audit"]["labeled_support"],
               rule["correct_support"], rule["unsupported_definitive"]), expected)
    agreement = cv[-1]["decisions"]["verified_and_model_agrees"]
    check("Final résumé agreement", (agreement["correct_support"], agreement["unsupported_definitive"]), (2, 0))

    initial = read("nameguard-risk-study/results/metrics.json")
    check("Development and initial test counts", (initial["development_cases"], initial["holdout_cases"]), (144, 312))
    check("Initial operating point", [initial["holdout"][mode]["accepted"]
                                      for mode in ("single", "mean", "minimum")], [0, 0, 0])
    check("Initial AUROCs", [round(initial["holdout"][mode]["auc_among_admitted"], 3)
                              for mode in ("single", "mean", "minimum")], [.790, .761, .769])
    check("Initial name conformance",
          tuple(initial["panel"][key] for key in ("exact_orbits", "stage_digest_matches", "native_unthresholded_flips")),
          (20, 3720, 1))

    reserve = read("nameguard-risk-study/results/reserve-metrics.json")
    check("Reserve size", reserve["cases"], 320)
    for rule, expected in {
        "candidate_aware": (0, 0, 0), "extraction": (55, 36, 19), "literal": (320, 160, 160),
        "posthoc_matched_count_candidate_aware": (55, 44, 11),
    }.items():
        check(rule, tuple(reserve[rule][k] for k in ("accepted", "correct", "false_support")), expected)
    check("Reserve audit", tuple(reserve["sample"][k] for k in ("N", "m", "x", "upper_count")), (55, 44, 15, 22))
    check("Generator conformance",
          (reserve["panel"]["exact_orbits"], reserve["panel"]["direct_full_vocabulary_step_matches"]), (20, 375))

    diag = read("nameguard-risk-study/results/diagnostic-metrics.json")
    check("Same-ID and original-name references",
          tuple(diag[k] for k in ("same_id_panels", "vocabulary_matches", "native_generation_varies",
                                 "native_confirmation_varies")), (20, 500, 7, 0))
    check("Blank-passage result",
          (diag["blank_passage_comparison"]["supported"]["source_blank_match"],
           diag["blank_passage_comparison"]["unsupported"]["source_blank_no_match"]), (1, 19))

    baseline = read("nameguard-peer-review/results/summary.json")
    check("ARC and synthetic capability controls",
          (baseline["arc_challenge"]["full_vocabulary_exact"], baseline["arc_challenge"]["native_correct"],
           baseline["arc_challenge"]["gated_correct"], baseline["planned_rows"], baseline["task_native_correct"],
           baseline["task_compiled_correct_intent_to_score"]), (100, 68, 68, 192, 179, 168))
    matrix = read("nameguard-peer-review/results/boundary-ablation.json")["totals"]
    check("Admission figure",
          tuple(matrix[k] for k in ("greedy_contract_greedy_witness", "greedy_contract_all_witnesses",
                                    "compiled_contract_greedy_witness", "compiled_contract_all_witnesses")),
          (158, 160, 154, 170))
    print("Manuscript result counts match the published study records.")


def verify_build(directory):
    aux = (directory / "manuscript.aux").read_text(encoding="utf-8")
    log = (directory / "manuscript.log").read_text(encoding="utf-8")
    if re.search(r"(?:Overfull|(?:LaTeX|Package \S+) Warning:)", log):
        raise ValueError("Inspect layout, citation or navigation warnings in manuscript.log")
    for kind, count in (("fig:", 5), ("tab:", 5), ("prop:", 1)):
        check(kind, aux.count(r"\newlabel{" + kind), count)
    check("Bibliography entries", aux.count(r"\bibcite{"), 53)
    from pypdf import PdfReader

    pdf = PdfReader(directory / "manuscript.pdf")
    text = "\n".join(page.extract_text() for page in pdf.pages)
    flat = " ".join(text.split())
    for term in ("Abstract", "Table of Contents", "List of Figures", "List of Tables",
                 "Abbreviations and notation", "Vocabulary", "References",
                 "1,488", "3,720", "55 of 320", "40.0%", "Protected-trace equality"):
        if term not in flat:
            raise ValueError(f"Published manuscript is missing {term!r}")
    if "??" in text or "Hannah Liu" in text:
        raise ValueError("Manuscript has an unresolved marker or an excluded reference")
    def bookmark_titles(outline):
        for entry in outline:
            if isinstance(entry, list):
                yield from bookmark_titles(entry)
            else:
                yield entry.title

    bookmarks = set(bookmark_titles(pdf.outline))
    required = {"Title page", "Abstract", "Table of Contents", "List of Figures", "List of Tables",
                "Abbreviations and notation", "Vocabulary", "Introduction", "References"}
    if required - bookmarks:
        raise ValueError(f"Missing PDF bookmarks: {required - bookmarks}")
    check("Front-matter page labels", pdf.page_labels[:7], ["Cover", "i", "ii", "iii", "iv", "v", "1"])

    destinations = pdf.named_destinations
    external_links = set()
    for page in pdf.pages:
        for ref in page.get("/Annots", []):
            annotation = ref.get_object()
            if annotation.get("/Subtype") != "/Link":
                continue
            action = annotation.get("/A", {})
            if action.get("/S") == "/URI":
                external_links.add(action["/URI"])
                continue
            dest = annotation.get("/Dest", action.get("/D"))
            if isinstance(dest, str) and dest not in destinations:
                raise ValueError(f"Broken internal PDF link: {dest}")
    if not any(url.endswith("/tree/main/nameguard-paper") for url in external_links):
        raise ValueError("The PDF does not link to the manuscript package")
    print("PDF front matter, page labels, named links and manuscript result markers checked.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, help="Also inspect the PDF, aux and log from a LaTeX build")
    options = parser.parse_args()
    verify_math()
    verify_records()
    if options.build_dir:
        verify_build(options.build_dir)
