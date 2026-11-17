"""Replay a span-local feature intervention on archived NameGuard scores."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from math import log1p
from pathlib import Path
import re

import numpy as np


WORD = re.compile(r"[^\W_]+", re.UNICODE)
STOP = frozenset(
    "a an the this that these those to of in on for about as and or but what which who when where how "
    "why is was were are be been being at under from with by did do does had has have would could should "
    "may can might must it its they their not no".split()
)
FILES = {
    "development-inputs.json": "e31439ad8c90342b4c0b1743f89482bf9d3f0020f242442edbd18cac453d6830",
    "development-labels.json": "7d8bd48b7fb604025320ae6d02ad9ab042085fa9ba2163064b340cd6337060c6",
    "holdout-inputs.json": "b9d719bdc258fe375ec075681e4ef51ed71a8af902212ce033c05efe8bfc0f7a",
    "holdout-labels.json": "b7e31f6f20dae4ec18d7c981e800bf37ef3f36315a891e0cee07d6817a8301a3",
    "rows.json": "a5afabc01c8b21d16b792c13fc09ab82a24c6a2cf10f0afb317a7181578f2fb4",
    "labels.json": "40db3c2cb08136fe96d3f79b0970db25202a3fe496cec458a772d045048da946",
    "development-rows.json": "11c86c51ba9cf67f39fd55f083976db7ef2409b57ecb0a7d7e9ee4a2a4b1abd6",
    "holdout-rows.json": "a27f2e077c933d34ab5663c8cacc97a74fab650516e5632938dc9854e6efb9a3",
    "baseline.json": "ad20321e52bc08777afadc8e987d53fee3dae62325a9776f546f2d1208aea9ce",
}
CAPACITIES = (10, 20, 30, 40)
PROTOCOL_SHA = "e13887ce88f1f772fe3613bacfe6d14677dcff870f53c4362e0c550663b44966"


@dataclass(frozen=True)
class Row:
    key: str
    article: str
    pair: str
    passage: str
    question: str
    quote: str
    start: int
    margins: tuple
    supported: bool


def source(path):
    data = path.read_bytes()
    if sha256(data).hexdigest() != FILES[path.name]:
        raise ValueError(f"Input does not match the frozen archive: {path.name}")
    return json.loads(data)


def load(inputs, labels, scores):
    cases, truth, scores = map(source, (inputs, labels, scores))
    if cases.keys() != truth.keys() or cases.keys() != scores.keys():
        raise ValueError("Input, label and score case IDs differ")
    rows, pairs = [], defaultdict(list)
    for key in sorted(cases):
        record, score = cases[key], scores[key]
        if not score["admitted"] or not isinstance(truth[key], bool):
            raise ValueError("Expected an admitted binary case")
        (a, b), (reverse_b, reverse_a) = score["logits"]
        quote, start, passage = record["answer"], record["start"], record["passage"]
        assert 0 <= start and passage[start:start + len(quote)] == quote
        row = Row(key, record["article"], record["pair"], passage,
                  record["question"], quote, start, (a - b, reverse_a - reverse_b), truth[key])
        rows.append(row)
        pairs[row.pair].append(row)
    for a, b in pairs.values():
        assert a.article == b.article and a.passage == b.passage and a.supported != b.supported
    return rows


def words(text, remove_stop=True):
    terms = WORD.findall(text.casefold())
    if remove_stop:
        terms = [term for term in terms if term not in STOP]
    return terms


def windows(row):
    """Return equal-length disjoint contexts with the supplied quote removed."""
    before, after = [], []
    end = row.start + len(row.quote)
    for token in WORD.finditer(row.passage):
        if token.end() <= row.start:
            before.append(token.group().casefold())
        elif token.start() >= end:
            after.append(token.group().casefold())
    all_terms = before + after
    centre, length = len(before), len(all_terms)
    for size in range(min(24, length // 2), 0, -1):
        near = max(0, min(centre - size // 2, length - size))
        options = []
        if near >= size:
            options.append(0)
        if near + size <= length - size:
            options.append(length - size)
        if options:
            far = min(options, key=lambda p: (-abs(p + size / 2 - centre), p))
            return all_terms, all_terms[near:near + size], all_terms[far:far + size]
    return all_terms, [], []


def lexical(question, context, weights):
    context = [word for word in context if word not in STOP]
    qset, cset = set(question), set(context)
    common = qset & cset
    coverage = sum(sorted(weights[word] for word in common)) / (sum(sorted(weights.values())) or 1)
    precision = sum(term in qset for term in context) / max(1, len(context))
    agreements = set(zip(question, question[1:])) & set(zip(context, context[1:]))
    numeric = {word for word in qset if word.isdigit()}
    return coverage, precision, min(1.0, len(agreements) / 3), len(numeric & cset) / max(1, len(numeric))


def matrices(rows, df, total):
    result = defaultdict(list)
    for row in rows:
        question = words(row.question)
        quote = words(row.quote, remove_stop=False)
        qset = set(question)
        weights = {term: log1p((total - df[term] + .5) / (df[term] + .5)) for term in qset}
        passage, near, far = windows(row)
        assert len(near) == len(far)
        full = lexical(question, passage, weights)
        nuisance = (
            log1p(len(question)), log1p(len(quote)), sum(x in qset for x in quote) / max(1, len(quote)),
            float(any(char.isdigit() for char in row.quote)),
            sum(x.isupper() for x in row.quote) / max(1, len(row.quote)),
        )
        left, right = row.margins
        common = (left,) + nuisance + full
        local = lexical(question, near, weights)
        remote = lexical(question, far, weights)
        result["single_score"].append((left,))
        result["dual_mean"].append(((left + right) / 2,))
        result["dual_fitted"].append((left, right, abs(left - right)) + nuisance + full)
        result["single_nuisance"].append((left,) + nuisance)
        result["single_full"].append(common)
        result["single_distant"].append(common + remote)
        result["single_local"].append(common + local)
        result["text_local"].append(nuisance + full + local)
    return {name: np.asarray(features) for name, features in result.items()}


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def logistic(train, target, test):
    centre, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale < 1e-12] = 1
    fit = np.column_stack((np.ones(len(train)), (train - centre) / scale))
    design = np.column_stack((np.ones(len(test)), (test - centre) / scale))
    penalty = np.eye(fit.shape[1]) * .05
    penalty[0, 0] = 0
    weights = np.zeros(fit.shape[1])
    for _ in range(32):
        p = sigmoid(fit @ weights)
        objective_gradient = fit.T @ (p - target) / len(fit) + penalty @ weights
        hessian = (fit.T * (p * (1 - p))) @ fit / len(fit) + penalty
        step = np.linalg.solve(hessian, objective_gradient)
        weights -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return sigmoid(design @ weights)


def counts(rows, predictions):
    articles, pairs = defaultdict(list), defaultdict(list)
    for row, score in zip(rows, predictions, strict=True):
        articles[row.article].append((row, score))
        pairs[row.pair].append((row.supported, score))
    pair_correct = 0.
    for a, b in pairs.values():
        positive, negative = (a, b) if a[0] else (b, a)
        pair_correct += (positive[1] > negative[1]) + .5 * (positive[1] == negative[1])
    quotas = {}
    for level in CAPACITIES:
        by_article = [(len(group) * level // 100,
                       sum(not row.supported for row, _ in
                           sorted(group, key=lambda x: (-x[1], x[0].key))[:len(group) * level // 100]))
                      for _, group in sorted(articles.items())]
        kept, errors = map(sum, zip(*by_article))
        quotas[str(level)] = {
            "confirmed": kept, "unsupported": errors, "false_support": errors / kept,
            "by_article_accepted_unsupported": by_article,
        }
    truth = np.array([row.supported for row in rows], dtype=float)
    return {"pairs": len(pairs), "pair_accuracy": pair_correct / len(pairs),
            "brier": float(np.mean((truth - predictions) ** 2)), "quotas": quotas}


def comparison(metrics, control):
    def rows(method):
        return np.array(metrics[method]["quotas"]["20"]["by_article_accepted_unsupported"])
    target, baseline = rows("single_local"), rows(control)
    assert np.array_equal(target[:, 0], baseline[:, 0])
    delta = baseline[:, 1] - target[:, 1]
    selected = target[:, 0]
    rng = np.random.default_rng(43571)
    samples = rng.integers(0, len(delta), size=(4999, len(delta)))
    denom = selected[samples].sum(axis=1)
    ratios = delta[samples].sum(axis=1)[denom > 0] / denom[denom > 0]
    return {"unsupported_removed": int(delta.sum()),
            "difference_at_capacity": float(delta.sum() / selected.sum()),
            "article_bootstrap_95": np.quantile(ratios, [.025, .975]).tolist()}


def measure(train, test, train_matrices, df):
    labels = np.array([r.supported for r in train])
    test_matrices = matrices(test, df, len({r.pair for r in train}))
    methods = {}
    for key, features in test_matrices.items():
        probability = logistic(train_matrices[key], labels, features)
        if key in ("single_score", "dual_mean"):
            # Fixed archive margins rank cases. Fitted scalar probabilities are only for Brier.
            raw = counts(test, features[:, 0])
            raw["brier"] = counts(test, probability)["brier"]
            methods[key] = raw
        else:
            methods[key] = counts(test, probability)
    return {
        "cases": len(test), "articles": len({r.article for r in test}),
        "methods": methods, "comparisons": {
            name: comparison(methods, name) for name in methods if name != "single_local"
        },
    }


def run(queue, outputs):
    protocol = sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest()
    assert protocol == PROTOCOL_SHA
    train = load(queue / "queue/development-inputs.json", queue / "queue/development-labels.json",
                 outputs / "registered/development-rows.json")
    held = load(queue / "queue/holdout-inputs.json", queue / "queue/holdout-labels.json",
                outputs / "registered/holdout-rows.json")
    reserve = load(queue / "reserve/rows.json", queue / "reserve/labels.json",
                   outputs / "reserve/baseline.json")
    assert [len(x) for x in (train, held, reserve)] == [144, 312, 320]
    for group in ("article", "key", "pair"):
        a, b, c = [{getattr(r, group) for r in x} for x in (train, held, reserve)]
        assert not (a & b or b & c or c & a)
    passages = {r.pair: r.passage for r in train}
    df = Counter(term for passage in passages.values() for term in set(words(passage)))
    fit = matrices(train, df, len(passages))
    h = measure(train, held, fit, df)
    r = measure(train, reserve, fit, df)
    contrast = h["comparisons"]["single_distant"]
    return {"protocol_sha256": protocol, "development": {"cases": len(train), "articles": len({r.article for r in train})},
            "holdout": h, "reserve": r, "decision": {
                "primary_gate_passes": (contrast["difference_at_capacity"] >= .03
                                         and contrast["article_bootstrap_95"][0] > 0),
                "same_direction_reserve": r["comparisons"]["single_distant"]["difference_at_capacity"] > 0,
                "noninferior_to_dual_mean": h["comparisons"]["dual_mean"]["article_bootstrap_95"][0] > -.02,
            }}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--archived-results", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    cli = parser.parse_args()
    study = run(cli.queue, cli.archived_results)
    serial = json.dumps(study, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if cli.output:
        with cli.output.open("x", encoding="utf-8") as handle:
            handle.write(serial)
    else:
        assert cli.verify.read_text() == serial, "Numerical replay did not reproduce byte for byte"
    for split in ("holdout", "reserve"):
        print(split, [(name, obj["quotas"]["20"]["unsupported"]) for name, obj in study[split]["methods"].items()])
    print(study["decision"])
