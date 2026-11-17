"""Certify exact two-sentence SciFact rationale rankings with a CPU-only replay."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from math import log1p
from pathlib import Path
import random
import re
import tarfile


ARCHIVE_SHA = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
PROTOCOL_SHA = "8c814acc43ffba65043f1c78cfa37ecba910be36ef7bd56d7ba9438eb9257a63"
WORD = re.compile(r"[^\W_]+", re.UNICODE)
STOP = frozenset(
    "a an and are as at be been by for from had has have in is it of on or that the to was were with".split()
)
METHODS = ("exhaustive", "weighted_bag", "ordered_pairs", "matched_bm25_first", "bm25")


@dataclass(frozen=True)
class Case:
    claim_id: int
    doc_id: int
    claim: tuple
    sentences: tuple
    gold: tuple
    label: str


def tokenize(text):
    return tuple(word for word in WORD.findall(text.casefold()) if word not in STOP)


def digest(path):
    with path.open("rb") as file:
        return sha256(file.read()).hexdigest()


def read_cases(path):
    if digest(path) != ARCHIVE_SHA:
        raise ValueError("The SciFact archive digest differs")
    with tarfile.open(path, "r:gz") as archive:
        claims = [json.loads(line) for line in archive.extractfile("data/claims_dev.jsonl")]
        ids = {int(doc) for claim in claims for doc in claim["evidence"]}
        frequency = Counter()
        docs = {}
        total_words = total_sentences = 0
        for line in archive.extractfile("data/corpus.jsonl"):
            doc = json.loads(line)
            sentences = tuple(tokenize(sentence) for sentence in doc["abstract"])
            total_sentences += len(sentences)
            for sentence in sentences:
                frequency.update(set(sentence))
                total_words += len(sentence)
            if doc["doc_id"] in ids:
                docs[doc["doc_id"]] = sentences
    weights = {
        word: max(1, round(100 * log1p((total_sentences - count + .5) / (count + .5))))
        for word, count in frequency.items()
    }
    unseen = round(100 * log1p((total_sentences + .5) / .5))
    cases = []
    cases_without_label = 0
    for row in claims:
        claim = tokenize(row["claim"])
        if not row["evidence"]:
            cases_without_label += 1
        for doc, labels in sorted(row["evidence"].items(), key=lambda item: int(item[0])):
            doc_id = int(doc)
            abstract = docs[doc_id]
            gold = tuple(frozenset(item["sentences"]) for item in labels)
            statuses = {item["label"] for item in labels}
            if not abstract or not claim or any(not item or min(item) < 0 or max(item) >= len(abstract) for item in gold):
                raise ValueError("Invalid SciFact rationale or empty input")
            status = next(iter(statuses)) if len(statuses) == 1 else "MIXED"
            cases.append(Case(row["id"], doc_id, claim, abstract, gold, status))
    corpus = {"claims": len(claims), "without_labelled_rationale": cases_without_label,
              "index_sentences": total_sentences, "labelled_pairs": len(cases)}
    return cases, weights, unseen, total_words / total_sentences, corpus


def weighted_lcs(query, sentence):
    previous = [0] * (len(sentence) + 1)
    for token, weight in query:
        current = [0]
        for j, other in enumerate(sentence):
            if token == other:
                current.append(previous[j] + weight)
            else:
                current.append(max(previous[j + 1], current[-1]))
        previous = current
    return previous[-1]


def bag_bound(query_counts, counts, weights):
    return sum(min(query_count, counts[token]) * weight
               for (token, query_count), weight in zip(query_counts.items(), weights, strict=True))


def order_bound(query, sentence, bag):
    first, last = {}, {}
    for pos, term in enumerate(sentence):
        first.setdefault(term, pos)
        last[term] = pos
    # An alignment induces a legal alignment within each disjoint query block.
    def tiling(offset):
        bound = query[0][1] if offset and query[0][0] in first else 0
        for i in range(offset, len(query) - 1, 2):
            term, weight = query[i]
            other, other_weight = query[i + 1]
            if term in first and other in last and first[term] < last[other]:
                bound += weight + other_weight
            else:
                bound += max(weight if term in first else 0, other_weight if other in first else 0)
        if (len(query) - offset) % 2 and query[-1][0] in first:
            bound += query[-1][1]
        return bound
    return min(bag, tiling(0), tiling(1))


def audit_bounds():
    rng = random.Random(821123)
    weights = dict(enumerate((3, 2, 5, 11)))
    for _ in range(500):
        terms = rng.choices(range(4), k=rng.randint(1, 12))
        query = [(term, weights[term]) for term in terms]
        sentence = rng.choices(range(4), k=rng.randint(0, 15))
        counts = Counter(terms)
        upper = bag_bound(counts, Counter(sentence), (weights[term] for term in counts))
        assert 0 <= weighted_lcs(query, sentence) <= order_bound(query, sentence, upper) <= upper
    assert order_bound([("a", 1), ("b", 2)], ["b", "a"], 3) == 2


def lexical_scores(query_counts, sentence_counts, sentences, term_weights, average_length):
    maximum = sum(term_weights) * 2.2
    scores = []
    for counts, sentence in zip(sentence_counts, sentences, strict=True):
        norm = 1.2 * (.25 + .75 * len(sentence) / average_length)
        score = 0.
        for term, weight in zip(query_counts, term_weights, strict=True):
            tf = counts[term]
            score += weight * (tf * 2.2) / (tf + norm)
        scores.append(int(1000 * score / maximum))
    return scores


def ranked(scores):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def select(case, weights, unseen, average_length, method, alignment_budget=None, *, align=weighted_lcs):
    query = [(term, weights.get(term, unseen)) for term in case.claim]
    total_weight = sum(weight for _, weight in query)
    counts = Counter(case.claim)
    term_weights = [weights.get(term, unseen) for term in counts]
    sentence_counts = [Counter(sentence) for sentence in case.sentences]
    lexical = lexical_scores(counts, sentence_counts, case.sentences, term_weights, average_length)
    limit = min(2, len(case.sentences))
    if method == "bm25":
        return ranked(lexical)[:limit], 0, 0

    def hybrid(index, subsequence):
        return 7 * lexical[index] + 3 * ((1000 * subsequence) // total_weight)

    bounds = None
    if method in ("weighted_bag", "ordered_pairs"):
        upper = [bag_bound(counts, words, term_weights) for words in sentence_counts]
        if method == "ordered_pairs":
            upper = [order_bound(query, sentence, bound) for sentence, bound in
                     zip(case.sentences, upper, strict=True)]
        bounds = [hybrid(i, bound) for i, bound in enumerate(upper)]
        candidates = ranked(bounds)
    elif method == "matched_bm25_first":
        if alignment_budget is None or not limit <= alignment_budget <= len(case.sentences):
            raise ValueError("Invalid matched alignment budget")
        candidates = ranked(lexical)[:alignment_budget]
    elif method == "exhaustive":
        candidates = range(len(case.sentences))
    else:
        raise ValueError(f"Unexpected selection method: {method}")

    best = []
    calls = cells = 0
    for i in candidates:
        if bounds is not None and len(best) == limit and (bounds[i], -i) <= best[-1]:
            break
        sentence = case.sentences[i]
        score = hybrid(i, align(query, sentence))
        calls += 1
        cells += len(query) * len(sentence)
        best.append((score, -i))
        best.sort(reverse=True)
        best = best[:limit]
    return [-item[1] for item in best], calls, cells


def is_complete(chosen, case):
    return any(gold.issubset(chosen) for gold in case.gold)


def is_hit(chosen, case):
    return any(not gold.isdisjoint(chosen) for gold in case.gold)


def connected_components(cases):
    roots = {}

    def root(node):
        while node in roots:
            node = roots[node]
        return node

    for case in cases:
        claim, doc = root(("claim", case.claim_id)), root(("doc", case.doc_id))
        if claim != doc:
            roots[claim] = doc
    return [root(("claim", case.claim_id)) for case in cases]


def cluster_intervals(deltas, groups):
    grouped = defaultdict(lambda: [0, 0])
    for key, delta in zip(groups, deltas, strict=True):
        grouped[key][0] += delta
        grouped[key][1] += 1
    values = list(grouped.values())
    rng = random.Random(79921)
    measures = []
    for _ in range(4999):
        draw = rng.choices(values, k=len(values))
        measures.append(sum(v[0] for v in draw) / sum(v[1] for v in draw))
    measures.sort()
    return [measures[int((len(measures) - 1) * q)] for q in (.025, .975)]


def study(path):
    protocol_hash = digest(Path(__file__).with_name("PROTOCOL.md"))
    if protocol_hash != PROTOCOL_SHA:
        raise ValueError("Protocol differs from its original digest")
    audit_bounds()
    cases, weights, unseen, average_length, corpus = read_cases(path)
    by_method = {method: Counter() for method in METHODS}
    by_label = defaultdict(lambda: defaultdict(Counter))
    deltas = {baseline: [] for baseline in ("bm25", "matched_bm25_first")}
    for case in cases:
        observed = {}
        for method in METHODS:
            budget = observed["ordered_pairs"][1] if method == "matched_bm25_first" else None
            observed[method] = select(case, weights, unseen, average_length, method, budget)
        exact = observed["exhaustive"][0]
        assert observed["weighted_bag"][0] == exact == observed["ordered_pairs"][0]
        assert observed["matched_bm25_first"][1] == observed["ordered_pairs"][1]
        for method, (choice, calls, cells) in observed.items():
            assert len(choice) == min(2, len(case.sentences))
            selected = set(choice)
            stats = {"cases": 1, "alignment_calls": calls, "alignment_cells": cells,
                     "complete_rationale": is_complete(selected, case), "rationale_hit": is_hit(selected, case),
                     "agrees_with_exhaustive": choice == exact}
            by_method[method].update(stats)
            by_label[case.label][method].update(stats)
        for baseline, values in deltas.items():
            values.append(int(is_complete(set(exact), case)) -
                          int(is_complete(set(observed[baseline][0]), case)))
    groups = connected_components(cases)
    savings = {
        comparator: 1 - by_method["ordered_pairs"]["alignment_cells"] / by_method[comparator]["alignment_cells"]
        for comparator in ("exhaustive", "weighted_bag")
    }
    return {
        "protocol_sha256": protocol_hash,
        "archive_sha256": ARCHIVE_SHA,
        "corpus": corpus,
        "independent_gold_components": len(set(groups)),
        "stats": by_method,
        "by_label": by_label,
        "ordered_pair_cell_savings": savings,
        "primary_gate_met": savings["exhaustive"] >= .25 and savings["weighted_bag"] >= .10,
        "complete_rationale_rate_comparisons": {
            baseline: {"paired_difference": sum(values) / len(values),
                       "cluster_bootstrap_95": cluster_intervals(values, groups)}
            for baseline, values in deltas.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--output", type=Path)
    modes.add_argument("--verify", type=Path)
    arguments = parser.parse_args()
    value = study(arguments.archive)
    rendered = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if arguments.output:
        with arguments.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    elif arguments.verify.read_text() != rendered:
        raise SystemExit("Replay does not match the recorded result")
    print(json.dumps({k: value[k] for k in ("stats", "ordered_pair_cell_savings", "primary_gate_met",
                                            "complete_rationale_rate_comparisons")}, indent=2))
