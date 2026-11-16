# What generalises across articles in NameGuard evidence verification?

We replayed the archived candidate-aware scores and candidate-blind extraction flags on **320 human-labelled SQuAD cases in ten articles**. In each fold, one whole article was held out and the operating threshold was fitted on the other nine. Neither a score-only threshold nor an extraction-gated score threshold could satisfy the **10% false-support / 20% coverage** criteria on any training fold. Both therefore referred every held-out case. This retrospective result is independent of the project's separate name-invariance guarantee. [Protocol](ARTICLE_HOLDOUT_PROTOCOL.md) · [Fold results](article_holdout_results.json)

| Policy | Confirmed | Unsupported confirmations | Coverage | False support |
|---|---:|---:|---:|---:|
| Score threshold fitted off-article | 0 | 0 | 0% | Undefined |
| Exact extraction with fitted score threshold | 0 | 0 | 0% | Undefined |
| Score ranking at fixed article capacity | 61 | 12 | 19.1% | 19.7% |
| Extraction first, then score at the same capacity | 61 | 20 | 19.1% | 32.8% |
| Original exact extraction | 55 | 19 | 17.2% | 34.5% |

Forcing the same decision budget within each article made the comparison independent of aggregate workload. Prioritising exact extraction produced **eight additional false confirmations** compared with selecting by score alone. The paired article resampling interval for the change in false-support risk was **+9.2 to +18.9 percentage points**. This interval describes sensitivity to the mix of these ten articles. The forced-capacity rule does not meet either operational target.

A stronger counting check does not depend on choosing a better threshold. Exact extraction produced only **36 true positives**. Even a filter with access to every label could keep at most **40 of 320 cases (12.5%)** while limiting false support to 10%. Any policy that keeps only this generator's existing confirmations must therefore miss 20% coverage on this set. A replacement would need to produce additional supported confirmations or use another evidence signal. The audit uses an already inspected corpus, no new inference and no new human labels. It does not validate hiring automation or transfer to new articles.

## Reproduce

Use the original [CPU replay instructions](https://github.com/adelmansourjr/racial-bias-ai-calibration/tree/d51b96f96982bc41a42b368af6d38d459b2a95d5/nameguard-risk-study#reproduce-the-analysis-on-cpu) to reconstruct the reserve queue from the pinned SQuAD archive. Use the archived prediction directory at the same commit. This report publishes aggregate and article-level counts without copying question text, labels or predictions.

```sh
python3 article_holdout.py --queue /path/to/reconstructed/source/reserve \
  --results /path/to/nameguard-risk-study/results/reserve \
  --verify article_holdout_results.json
```
