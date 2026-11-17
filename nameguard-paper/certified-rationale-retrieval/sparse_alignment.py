"""Validate and benchmark the sparse exact weighted alignment kernel."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics
from time import process_time_ns

from replay import digest, read_cases, select, weighted_lcs


AUDIT_SHA = "a8cdeb7373a0dba371aa375ac8184b581916d38bdffd0cd577aa5f1da0e7ad38"
EXACT = ("exhaustive", "weighted_bag", "ordered_pairs")
KERNELS = ("dense", "sparse")


def sparse_lcs(query, sentence, meter=None):
    positions = defaultdict(list)
    for i, word in enumerate(sentence, start=1):
        positions[word].append(i)
    tree = [0] * (len(sentence) + 1)
    best = 0
    for word, weight in query:
        for pos in reversed(positions.get(word, ())):
            prev = 0
            i = pos - 1
            while i:
                prev = max(prev, tree[i])
                i &= i - 1
            score = prev + weight
            best = max(best, score)
            i = pos
            steps = 0
            while i < len(tree):
                tree[i] = max(tree[i], score)
                i += i & -i
                steps += 1
            if meter is not None:
                meter["matching_pairs"] += 1
                meter["fenwick_query_steps"] += (pos - 1).bit_count()
                meter["fenwick_update_steps"] += steps
    return best


def dag_lcs(query, sentence):
    nodes = [(i, j, weight) for i, (word, weight) in enumerate(query)
             for j, token in enumerate(sentence) if token == word]
    scores = []
    for i, j, weight in nodes:
        previous = max((score for (a, b, _), score in zip(nodes, scores) if a < i and b < j),
                       default=0)
        scores.append(previous + weight)
    return max(scores, default=0)


def audit_kernel():
    rng = random.Random(95201)
    weights = dict(enumerate((2, 1, 8, 6)))
    for _ in range(1000):
        query = [(term, weights[term]) for term in rng.choices(range(4), k=rng.randrange(13))]
        sentence = rng.choices(range(4), k=rng.randrange(13))
        assert dag_lcs(query, sentence) == weighted_lcs(query, sentence) == sparse_lcs(query, sentence)


def study(path):
    protocol_hash = digest(Path(__file__).with_name("SPARSE_PROTOCOL.md"))
    if protocol_hash != AUDIT_SHA:
        raise ValueError("Sparse verification protocol differs from its original digest")
    audit_kernel()
    cases, weights, unseen, average_length, _ = read_cases(path)
    checked_alignments = 0
    work = {f"{method}_{kernel}": Counter() for method in EXACT for kernel in KERNELS}
    for case in cases:
        query = [(token, weights.get(token, unseen)) for token in case.claim]
        for sentence in case.sentences:
            assert weighted_lcs(query, sentence) == sparse_lcs(query, sentence)
            checked_alignments += 1
        target, *_ = select(case, weights, unseen, average_length, "exhaustive")
        for method in EXACT:
            for kernel in KERNELS:
                key = f"{method}_{kernel}"
                counts = work[key]
                if kernel == "dense":
                    align = weighted_lcs
                else:
                    def align(query, sentence):
                        return sparse_lcs(query, sentence, meter=counts)
                chosen, calls, cells = select(case, weights, unseen, average_length, method, align=align)
                assert chosen == target
                counts["alignment_calls"] += calls
                if kernel == "dense":
                    counts["dense_cells"] += cells
    return {"protocol_sha256": protocol_hash, "random_oracle_pairs": 1000,
            "corpus_alignment_checks": checked_alignments, "exact_rankings_per_arm": len(cases), "work": work}


def benchmark(path):
    rng = random.Random(74124)
    modes = [(method, kernel) for method in EXACT for kernel in KERNELS]
    timings = {f"{method}_{kernel}": [] for method, kernel in modes}
    preprocessing = []
    for _ in range(9):
        start = process_time_ns()
        cases, weights, unseen, average_length, _ = read_cases(path)
        preprocessing.append((process_time_ns() - start) / 1e6)
        rng.shuffle(modes)
        for method, kernel in modes:
            align = weighted_lcs if kernel == "dense" else sparse_lcs
            start = process_time_ns()
            for _ in range(12):
                for case in cases:
                    select(case, weights, unseen, average_length, method, align=align)
            duration = (process_time_ns() - start) / (12 * 1e6)
            timings[f"{method}_{kernel}"].append(duration)

    def summary(samples):
        return {"median": statistics.median(samples), "min": min(samples), "max": max(samples)}

    baseline = timings["weighted_bag_dense"]
    return {
        "preprocessing_cpu_ms": preprocessing,
        "ranking_cpu_ms_by_block": timings,
        "ranking_cpu_ms": {key: summary(measurements) for key, measurements in timings.items()},
        "estimated_cold_cpu_ms": {key: summary([a + b for a, b in zip(measurements, preprocessing, strict=True)])
                                  for key, measurements in timings.items()},
        "cpu_ratio_vs_weighted_bag_dense": {
            key: summary([a / b for a, b in zip(measurements, baseline, strict=True)])
            for key, measurements in timings.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--output", type=Path)
    options.add_argument("--verify", type=Path)
    options.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    result = benchmark(args.archive) if args.timing else study(args.archive)
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as file:
            file.write(rendered)
    elif args.verify and args.verify.read_text() != rendered:
        raise SystemExit("Sparse validation does not match the recorded result")
    print(rendered, end="")
