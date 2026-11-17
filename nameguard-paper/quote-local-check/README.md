# One prompt can beat two in a fixed-budget evidence check

In a retrospective replay of **632 human-labelled cases across 27 SQuAD 2.0 articles**, one fixed Qwen response format produced **15 unsupported confirmations** compared with **24** when averaging both option orders, at the same budget of **120 confirmations**. The other single order produced **58**. This is an exploratory finding about the archived Qwen checkpoint, prompt and data. The prespecified experiment tested a different hypothesis, whether adding an evidence window next to the candidate quote improves ranking at the same budget. That comparison returned equal error counts on both cohorts. [Frozen protocol](PROTOCOL.md) · [Main results](results.json) · [Order diagnostic](order_diagnostic.json)

## Evidence locality at a matched cost

The task is candidate-answer verification. Every candidate is a verbatim passage span, including plausible decoys to questions human annotators marked unanswerable. All fits used the original 144 development cases. The original holdout has 312 cases in 17 articles and the reserve has 320 cases in 10 other articles. The three sets share no article, question ID or labelled pair. Every policy fills the same quota per article, set to the floor of 20% of its cases. The quote span is removed before the four lexical features are computed. The matched location control reads a disjoint, distant window of identical lexical-token length. The model scores and original scope remain unchanged across these two arms.

| Selector | Original holdout unsupported, 59 accepted | Reserve unsupported, 61 accepted | Archived Qwen scores per case |
|---|---:|---:|---:|
| First order alone | 6 | 9 | 1 |
| Both orders, mean signed margin | 12 | 12 | 2 |
| Both orders, fitted score and lexical head | 13 | 13 | 2 |
| First order plus nuisance features | 8 | 11 | 1 |
| First order plus full-passage overlap and nuisance features | 10 | 14 | 1 |
| First order plus full-passage and distant-window features | 8 | 14 | 1 |
| First order plus full-passage and quote-local features | 8 | 14 | 1 |
| Quote-local and full-passage features alone | 29 | 26 | 0 |

At the matched capacity, the main contrast is zero. Its article bootstrap interval is **−8.62 to +7.85 percentage points** on the original holdout and **−4.23 to +4.55 points** on the reserve. That misses both the prespecified practical improvement and the requirement for a positive reserve direction. The one-score local model also missed the prespecified noninferiority criterion against the mean dual-order score. Adding the lexical features costs extra processing and these results give no reason to use them in place of the raw first-order score on this archive.

## Why option order is worth auditing

The original NameGuard protocol fixed the first-order score before collecting the archived outcomes. The first format maps A to Supported and B to Unsupported. The second reverses that mapping, which the scorer sign-corrects before averaging. Both formats receive the same passage, question and proposed answer. The first-order and dual-order baselines were also fixed in this study's protocol. The comparison of these baselines with each other, and the reversed-only diagnostic, were analysed after the main results and are exploratory.

| Cohort | Confirmed by each rule | First order unsupported | Average both unsupported | Reversed order unsupported |
|---|---:|---:|---:|---:|
| Original holdout | 59 | 6 | 12 | 28 |
| Reserve | 61 | 9 | 12 | 30 |
| Pooled | 120 | 15 | 24 | 58 |

Across these articles the first-vs-mean difference is **7.50 percentage points** with a paired 95% article bootstrap interval of **+2.27 to +12.87 points**. The difference is positive in both cohorts, though the reserve interval alone includes zero. The errors with the reversed format show why the result cannot justify picking any single prompt. Relative to the two-order mean, the first order would require one fewer scored Qwen prompt per candidate. This study computed everything from existing scores and measured no inference latency or deployment speedup.

The archive consists of exactly paired positives and plausible decoys within paragraphs. Results apply to that task construction and cannot establish real-world error rates, acceptance thresholds that transfer to new articles, or which prompt to use with another model. A follow-up can lock both mappings, add more than one checkpoint and test prompt selection on independently collected factuality data. This connects to [Simpson and colleagues' guided grounded checking](https://aclanthology.org/2026.acl-long.1468/) through the cost of separate checks and the need to compare the checks themselves.

## Reproduce with public inputs

Use the pinned NameGuard [CPU replay instructions](https://github.com/adelmansourjr/racial-bias-ai-calibration/tree/d51b96f96982bc41a42b368af6d38d459b2a95d5/nameguard-risk-study#reproduce-the-analysis-on-cpu) to reconstruct the original queue and reserve from the official [SQuAD 2.0 source](https://aclanthology.org/P18-2124/). Pass the directory containing `queue/` and `reserve/` to `--queue`. Pass the archived NameGuard `nameguard-risk-study/results/` directory to `--archived-results`. Every input is SHA-256 checked. This folder contains aggregate numbers and no passage text, question IDs or per-example model scores.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 quote_local.py \
  --queue /path/to/rebuilt --archived-results /path/to/nameguard-risk-study/results \
  --verify results.json
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 audit_order.py results.json \
  --queue /path/to/rebuilt --archived-results /path/to/nameguard-risk-study/results \
  --verify order_diagnostic.json
```
